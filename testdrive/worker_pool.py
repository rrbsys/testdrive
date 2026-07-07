"""Parent-side management of worker subprocesses for plugins with a
non-"framework" ``pyenv`` (see ``PluginManifest.pyenv``, ``pyenv.py``,
and ``worker_main.py``'s module docstring for the wire protocol).

One worker subprocess per plugin id, spawned lazily on first use and
kept alive for the rest of this testdrive invocation — so a loop-mode
run over many images pays a plugin's initialize() cost once, not once
per image. Call ``shutdown_all_workers()`` when the CLI invocation is
done (``cli.main()`` does this in a ``finally`` block).
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
            from .pyenv import env_python_path

            python_path = env_python_path(cd, pyenv_name)
            if not python_path.exists():
                env_dir = python_path.parent.parent
                raise WorkerError(
                    "env_not_configured",
                    f"plugin '{plugin_id}' needs the '{pyenv_name}' environment "
                    f"(expected interpreter: {python_path}), which doesn't exist yet.\n"
                    f"Create it with:\n"
                    f'    python3.12 -m venv "{env_dir}"\n'
                    f'    "{python_path}" -m pip install -e . -e ".[{plugin_id}]"',
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
