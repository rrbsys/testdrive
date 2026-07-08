"""Parent-side management of worker subprocesses for plugins with a
non-"framework" ``pyenv`` (see ``PluginManifest.pyenv``, ``pyenv.py``,
and ``worker_main.py``'s module docstring for the wire protocol).

One worker subprocess per plugin id, spawned lazily on first use and
kept alive for the rest of this testdrive invocation — so a loop-mode
run over many images pays a plugin's initialize() cost once, not once
per image. Call ``shutdown_all_workers()`` when the CLI invocation is
done (``cli.main()`` does this in a ``finally`` block).

Unlike the framework's own environment (which needs one-time manual
setup — see ``pyenv.py`` — since there's no environment yet to
automate anything from), a plugin's own environment is provisioned
automatically the first time it's needed, gated behind the same
``set_downloads_allowed``/``-T``/``-TT`` discipline as model weight
downloads (see ``_provision_plugin_env`` below): pip-installing a
plugin's dependencies is the same class of "don't do this by surprise
during a plain detect run" action as downloading multi-GB weights.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .detection import Detection

log = logging.getLogger("testdrive.worker_pool")


class WorkerError(Exception):
    """Raised by WorkerHandle.detect() for any failure reported by the
    worker (or in spawning it). ``kind`` matches the wire protocol's
    "kind" field: "missing_dependency", "cache_not_populated", or
    "error" (also used for pool-level failures like a missing venv,
    which have no wire-protocol equivalent since they happen before any
    worker exists to report one).
    """

    def __init__(self, kind: str, message: str, missing: list[str] | None = None):
        self.kind = kind
        self.missing = missing or []
        super().__init__(message)


def _project_root() -> Path:
    """The directory containing pyproject.toml — this file's grandparent
    (testdrive/worker_pool.py -> testdrive/ -> project root).
    """
    return Path(__file__).resolve().parent.parent


def _provision_plugin_env(plugin_id: str, pyenv_name: str, env_directory: Path, python_path: Path) -> None:
    """Automatically create and set up a plugin's own environment.

    Unlike the framework's own environment (which the user must set up
    by hand — see pyenv.py — because there's no environment yet to run
    any automation *from*), a plugin's private environment can safely
    be provisioned by code already running from the verified-working
    framework environment: create the venv with the framework's own
    interpreter (already confirmed to be a working, correct-version
    Python — see pyenv.ensure_framework_env), then pip install testdrive
    itself plus this plugin's extra into it. No manual `python -m venv`
    / `pip install` steps for the user, unlike the framework env.

    Raises WorkerError (kind="error") on any failure, wrapping
    whatever subprocess.CalledProcessError/OSError occurred so callers
    don't need to know this is implemented via subprocess.
    """
    import subprocess
    import sys

    log.info("setting up '%s' environment for plugin '%s' (first use)...", pyenv_name, plugin_id)
    try:
        log.info("  creating venv at %s ...", env_directory)
        subprocess.run(
            [sys.executable, "-m", "venv", str(env_directory)],
            check=True, capture_output=True, text=True,
        )

        project_root = _project_root()
        log.info("  installing testdrive + '%s' extra (this may take a while)...", plugin_id)
        subprocess.run(
            [
                str(python_path), "-m", "pip", "install",
                "-e", str(project_root),
                "-e", f"{project_root}[{plugin_id}]",
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise WorkerError(
            "error",
            f"could not set up '{pyenv_name}' environment for plugin '{plugin_id}': "
            f"{exc.cmd} exited {exc.returncode}\n{exc.stderr}",
        ) from exc
    except OSError as exc:
        raise WorkerError(
            "error", f"could not set up '{pyenv_name}' environment for plugin '{plugin_id}': {exc}",
        ) from exc

    log.info("'%s' environment ready", pyenv_name)


class WorkerHandle:
    """One live worker subprocess for one plugin."""

    def __init__(self, plugin_id: str, python_path: Path):
        from .util import get_downloads_allowed, get_max_parallel_files

        self.plugin_id = plugin_id

        env = os.environ.copy()
        env["TESTDRIVE_WORKER_DOWNLOADS_ALLOWED"] = "1" if get_downloads_allowed() else "0"
        max_parallel = get_max_parallel_files()
        env["TESTDRIVE_WORKER_MAX_PARALLEL_FILES"] = str(max_parallel) if max_parallel is not None else ""

        self._proc = subprocess.Popen(
            [str(python_path), "-m", "testdrive.worker_main", plugin_id],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit — worker-side logging/tracebacks stay visible directly
            text=True,
            bufsize=1,  # line-buffered
            env=env,
        )

    def detect(
        self, image_path: Path, prompt: str, threshold: float, model: str | None,
    ) -> list["Detection"]:
        from .detection import Detection

        if self._proc.stdin is None or self._proc.stdout is None:  # pragma: no cover
            raise WorkerError("error", f"worker for '{self.plugin_id}' has no stdin/stdout pipes")

        request = {
            "cmd": "detect",
            "image_path": str(image_path),
            "prompt": prompt,
            "threshold": threshold,
            "model": model,
        }
        try:
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise WorkerError("error", f"worker for '{self.plugin_id}' is no longer running: {exc}") from exc

        line = self._proc.stdout.readline()
        if not line:
            raise WorkerError("error", f"worker for '{self.plugin_id}' exited unexpectedly")

        response = json.loads(line)
        if response.get("ok"):
            return [
                Detection(label=d["label"], score=d["score"], bbox=tuple(d["bbox"]))
                for d in response["detections"]
            ]

        raise WorkerError(
            response.get("kind", "error"),
            response.get("message", "unknown worker error"),
            missing=response.get("missing"),
        )

    def shutdown(self) -> None:
        if self._proc.poll() is not None:
            return  # already exited
        try:
            if self._proc.stdin:
                self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self._proc.stdin.flush()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self._proc.kill()


class WorkerPool:
    """Keyed by plugin id — one worker per plugin, reused across calls
    within this pool's lifetime (i.e. one testdrive invocation).
    """

    def __init__(self) -> None:
        self._workers: dict[str, WorkerHandle] = {}

    def get(self, plugin_id: str, pyenv_name: str, cd: Path) -> WorkerHandle:
        if plugin_id not in self._workers:
            from .pyenv import env_dir, env_python_path
            from .util import get_downloads_allowed

            python_path = env_python_path(cd, pyenv_name)
            if not python_path.exists():
                if not get_downloads_allowed():
                    # Mirrors the "run -T/-TT first" cache-discipline
                    # message used for model weights: setting up a new
                    # venv + pip installing into it is the same class of
                    # "don't do this by surprise during a plain detect
                    # run" action.
                    raise WorkerError(
                        "env_not_configured",
                        f"plugin '{plugin_id}' needs its '{pyenv_name}' environment set up "
                        f"first. Run `testdrive -T {plugin_id}` (or -TT {plugin_id}) once to "
                        f"set it up automatically, then re-run this command.",
                    )
                _provision_plugin_env(plugin_id, pyenv_name, env_dir(cd, pyenv_name), python_path)
                if not python_path.exists():  # pragma: no cover — defensive
                    raise WorkerError(
                        "error",
                        f"set up '{pyenv_name}' for plugin '{plugin_id}' but the expected "
                        f"interpreter still isn't there ({python_path})",
                    )
            self._workers[plugin_id] = WorkerHandle(plugin_id, python_path)
        return self._workers[plugin_id]

    def shutdown_all(self) -> None:
        for handle in self._workers.values():
            handle.shutdown()
        self._workers.clear()


_pool: WorkerPool | None = None


def get_pool() -> WorkerPool:
    """Process-wide singleton — one pool per testdrive invocation."""
    global _pool
    if _pool is None:
        _pool = WorkerPool()
    return _pool


def shutdown_all_workers() -> None:
    """Tear down every worker spawned so far, if any. Safe to call even
    if no workers were ever spawned (e.g. every plugin used this run
    was "framework"-pyenv, running in-process as usual).
    """
    global _pool
    if _pool is not None:
        _pool.shutdown_all()
        _pool = None
