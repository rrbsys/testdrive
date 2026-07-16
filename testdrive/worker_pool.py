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
    from .detection import Detection, PluginManifest

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


def _provision_marker_path(env_directory: Path) -> Path:
    return env_directory / ".testdrive_plugin_ok"


def _provision_plugin_env(
    plugin_id: str,
    pyenv_name: str,
    env_directory: Path,
    python_path: Path,
    requirements: list[str],
    pip_options: str,
    patches: list[dict[str, str]],
) -> None:
    """Automatically create and set up a plugin's own environment.

    Unlike the framework's own environment (which the user must set up
    by hand — see pyenv.py — because there's no environment yet to run
    any automation *from*), a plugin's private environment can safely
    be provisioned by code already running from the verified-working
    framework environment: create the venv with the framework's own
    interpreter (already confirmed to be a working, correct-version
    Python — see pyenv.ensure_framework_env), optionally upgrade pip in
    it (see get_pyenv_pip_upgrade — on by default; an old bundled pip is
    a common, confusing source of install failures), install testdrive
    itself into it (still needed here even though the plugin's own
    dependencies are handled separately below — the worker subprocess
    spawned into this environment needs to `import testdrive` itself,
    see worker_main.py), then this plugin's own *requirements* (see
    PluginManifest.requirements/pip_options/patches — pip strings with
    real version pins baked in, not a package-manager-agnostic "extra"
    name) via pyenv.run_pip_install. No manual `python -m venv` /
    `pip install` steps for the user, unlike the framework env.

    Progress is printed unconditionally (not gated behind -v/log level)
    — this can take a genuinely long time (multi-GB dependencies like
    torch), and printing nothing in the meantime reads as a hang rather
    than "working on it".

    Writes a small marker file (see _provision_marker_path) on success —
    not a precondition for the environment being considered usable
    (WorkerPool.get() still trusts any environment with the expected
    interpreter present, including ones set up by hand, exactly as
    before), just a breadcrumb recording that our own automation
    completed here specifically, for diagnostics.

    Raises WorkerError (kind="error") on any failure, wrapping
    whatever subprocess.CalledProcessError/OSError/RuntimeError
    occurred so callers don't need to know this is implemented via
    subprocess (and, for a plugin with source patches, direct file
    edits too — see pyenv.apply_source_patches).
    """
    import subprocess
    import sys

    from .pyenv import run_pip_install
    from .util import get_pyenv_pip_upgrade

    def _announce(msg: str) -> None:
        print(f"[testdrive] {msg}", file=sys.stderr)

    _announce(f"setting up '{pyenv_name}' environment for plugin '{plugin_id}' (first use)...")
    try:
        _announce(f"creating venv at {env_directory} ...")
        subprocess.run(
            [sys.executable, "-m", "venv", str(env_directory)],
            check=True, capture_output=True, text=True,
        )

        if get_pyenv_pip_upgrade():
            _announce(f"installing via {python_path} -m pip install --upgrade pip")
            subprocess.run(
                [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
                check=True, capture_output=True, text=True,
            )

        project_root = _project_root()
        _announce(f"installing via {python_path} -m pip install -e {project_root}")
        subprocess.run(
            [str(python_path), "-m", "pip", "install", "-e", str(project_root)],
            check=True, capture_output=True, text=True,
        )

        run_pip_install(python_path, requirements, pip_options, patches, announce=_announce)
    except subprocess.CalledProcessError as exc:
        raise WorkerError(
            "error",
            f"could not set up '{pyenv_name}' environment for plugin '{plugin_id}': "
            f"{exc.cmd} exited {exc.returncode}\n{exc.stderr}",
        ) from exc
    except (OSError, RuntimeError) as exc:
        raise WorkerError(
            "error", f"could not set up '{pyenv_name}' environment for plugin '{plugin_id}': {exc}",
        ) from exc

    try:
        _provision_marker_path(env_directory).write_text(f"{plugin_id}\n")
    except OSError:
        pass  # purely a diagnostic breadcrumb — not worth failing over

    log.info("'%s' environment ready", pyenv_name)


class WorkerHandle:
    """One live worker subprocess for one plugin."""

    def __init__(self, plugin_ref: str, python_path: Path):
        from .util import get_downloads_allowed, get_max_parallel_files

        # plugin_ref is what gets passed to worker_main.py's own
        # load_plugin() call in the child process — for a parked
        # models_inactive/ plugin that must be the original
        # "../models_inactive/<name>"-shaped reference, not the clean
        # manifest id, since load_plugin() only resolves *that* id
        # shape for parked plugins (see pluginloader._parse_inactive_ref).
        # Everything else (pool dict key, pip extras name during
        # provisioning) uses the clean manifest id instead — see
        # WorkerPool.get.
        self.plugin_id = plugin_ref

        env = os.environ.copy()
        env["TESTDRIVE_WORKER_DOWNLOADS_ALLOWED"] = "1" if get_downloads_allowed() else "0"
        max_parallel = get_max_parallel_files()
        env["TESTDRIVE_WORKER_MAX_PARALLEL_FILES"] = str(max_parallel) if max_parallel is not None else ""

        self._proc = subprocess.Popen(
            [str(python_path), "-m", "testdrive.worker_main", plugin_ref],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit — worker-side logging/tracebacks stay visible directly
            text=True,
            bufsize=1,  # line-buffered
            env=env,
        )

    def init(self, model: str | None) -> None:
        """Explicitly run the worker's dependency-check + initialize()
        sequence, without also running a detection. Used by callers
        (self-test) that report initialize() as its own timed step,
        separate from detect() — a plain detect() call also runs this
        implicitly on its first request, so callers that don't care
        about that distinction can just call detect() directly.
        """
        if self._proc.stdin is None or self._proc.stdout is None:  # pragma: no cover
            raise WorkerError("error", f"worker for '{self.plugin_id}' has no stdin/stdout pipes")

        request = {"cmd": "init", "model": model}
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
            return

        raise WorkerError(
            response.get("kind", "error"),
            response.get("message", "unknown worker error"),
            missing=response.get("missing"),
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

    def get(self, plugin_ref: str, manifest: "PluginManifest", cd: Path) -> WorkerHandle:
        """*plugin_ref* is the original CLI-facing reference (e.g.
        ``"../models_inactive/seem"`` for a parked plugin, or just
        ``"seem"`` for a normal one) — passed through to the worker
        subprocess, which needs that exact shape to resolve a parked
        plugin via its own load_plugin() call (see
        pluginloader._parse_inactive_ref). *manifest* is the plugin's
        own manifest — its clean ``id`` keys this pool (a messy
        path-shaped ref can't), and its ``requirements``/
        ``pip_options``/``patches`` are what auto-provisioning actually
        installs (see _provision_plugin_env), replacing the old
        pyproject.toml-extras-name mechanism.
        """
        plugin_id = manifest.id
        pyenv_name = manifest.pyenv
        if plugin_id not in self._workers:
            from .pyenv import env_dir, env_python_path, install_hint
            from .util import get_auto_provision_enabled, get_downloads_allowed

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
                        f"first. Run `testdrive -T {plugin_ref}` (or -TT {plugin_ref}) once to "
                        f"set it up automatically, then re-run this command.",
                    )
                requirements = [req["pip"] for req in manifest.requirements]
                if not get_auto_provision_enabled():
                    # --no-auto-provision: set up by hand instead, same
                    # instructions the framework's own environment needs.
                    env_directory = env_dir(cd, pyenv_name)
                    hint = install_hint(cd, pyenv_name, requirements, manifest.pip_options)
                    raise WorkerError(
                        "env_not_configured",
                        f"plugin '{plugin_id}' needs its '{pyenv_name}' environment set up, "
                        f"but --no-auto-provision was given. Set it up by hand:\n"
                        f'    python3 -m venv "{env_directory}"\n'
                        f'    "{python_path}" -m pip install -e ."\n'
                        f"    {hint}",
                    )
                _provision_plugin_env(
                    plugin_id, pyenv_name, env_dir(cd, pyenv_name), python_path,
                    requirements, manifest.pip_options, manifest.patches,
                )
                if not python_path.exists():  # pragma: no cover — defensive
                    raise WorkerError(
                        "error",
                        f"set up '{pyenv_name}' for plugin '{plugin_id}' but the expected "
                        f"interpreter still isn't there ({python_path})",
                    )
            self._workers[plugin_id] = WorkerHandle(plugin_ref, python_path)
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
