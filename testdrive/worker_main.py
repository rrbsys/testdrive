"""Worker subprocess entry point for plugins with a non-"framework"
``pyenv`` (see ``PluginManifest.pyenv`` and ``pyenv.py``).

Run as ``python -m testdrive.worker_main <plugin_id>`` using *that
plugin's own* dedicated virtual environment's interpreter — never the
framework's. Reads line-delimited JSON requests from stdin, writes one
line-delimited JSON response per request to stdout. Kept alive across
multiple requests within a single testdrive invocation (see
``worker_pool.WorkerPool``) specifically so a plugin's initialize() —
which can be extremely expensive, see Molmo — only has to run once per
CLI invocation, not once per image in a loop-mode run.

Wire protocol (each message is exactly one line of JSON; the real
stdout file descriptor is reserved for these lines only — see
``_protect_stdout()`` for how that's enforced even against native
(non-Python) writes to fd 1, not just Python-level ``print()``):

    parent -> worker
        {"cmd": "init", "model": str | null}
        {"cmd": "detect", "image_path": str, "prompt": str,
         "threshold": float, "model": str | null}
        {"cmd": "task", "image_path": str, "task_prompt": str,
         "model": str | null}
        {"cmd": "shutdown"}

    worker -> parent
        # response to "init" (also implicitly run before the first
        # "detect"/"task", if not already done — so a plain detect-only
        # caller never needs to send "init" separately)
        {"ok": true}
        {"ok": false, "kind": "missing_dependency", "missing": [str, ...]}
        {"ok": false, "kind": "cache_not_populated", "message": str}
        {"ok": false, "kind": "error", "message": str}

        # response to "detect" (one line per request)
        {"ok": true, "detections": [{"label": str, "score": float,
                                      "bbox": [int, int, int, int]}, ...]}
        {"ok": false, "kind": "missing_dependency", "missing": [str, ...]}
        {"ok": false, "kind": "cache_not_populated", "message": str}
        {"ok": false, "kind": "error", "message": str}

        # response to "task" (text-output tasks; see PluginManifest.tasks)
        {"ok": true, "text_result": str}
        {"ok": false, "kind": "missing_dependency", "missing": [str, ...]}
        {"ok": false, "kind": "cache_not_populated", "message": str}
        {"ok": false, "kind": "error", "message": str}

Plugin/library logging goes to stderr (Python logging's default),
which the parent lets pass through directly rather than capturing, so
worker-side messages are still visible to the person running
testdrive. Anything a plugin's own dependencies write to stdout
instead — Python ``print()`` calls buried in vendored research code,
or even native (non-Python) writes like the startup banners some MPI
implementations print straight to fd 1 in C, bypassing ``sys.stdout``
entirely — also ends up visible on stderr rather than corrupting the
JSON response line the parent is waiting for; see
``_protect_stdout()``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any

log = logging.getLogger("testdrive.worker_main")

_protocol_out: Any = None  # set by _protect_stdout() before any request handling


def _protect_stdout() -> None:
    """Reserve the real stdout file descriptor for the JSON protocol.

    Must run before importing anything plugin-related (pluginloader,
    the plugin module itself, and whatever it imports — torch,
    mpi4py, transformers, ...), since any of those can write to
    stdout during import or later inside initialize()/detect(). A
    plain ``sys.stdout = sys.stderr`` reassignment only redirects
    Python-level writes; some native libraries (mpi4py/OpenMPI's
    startup banners are a known example) write straight to fd 1 in C,
    which bypasses ``sys.stdout`` entirely and would still land on
    the same pipe the parent is reading the JSON response from,
    corrupting that line. Duplicating the real fd 1 aside first, then
    pointing fd 1 itself at stderr, protects against both cases: every
    future write to "stdout" — Python or native — actually lands on
    stderr (still visible, doesn't break the protocol), while
    ``_respond()`` keeps a private, untouched handle to the original
    fd 1 to send the actual protocol lines through.
    """
    global _protocol_out
    real_stdout_fd = os.dup(1)
    os.dup2(2, 1)  # fd 1 (stdout) now aliases fd 2 (stderr) for everyone
    _protocol_out = os.fdopen(real_stdout_fd, "w", buffering=1)
    sys.stdout = sys.stderr  # keep the Python-level object consistent too


def _respond(payload: dict[str, Any]) -> None:
    _protocol_out.write(json.dumps(payload) + "\n")
    _protocol_out.flush()


def _apply_parent_settings() -> None:
    """Mirror the parent process's util.py settings.

    These are plain module-level state in util.py, which is per-process
    — a subprocess starts with its own fresh copy (defaults), it
    doesn't inherit the parent's in-memory values just because it's a
    child process. The parent passes them down as env vars at spawn
    time (see worker_pool.WorkerHandle.__init__) specifically so this
    worker respects the same cache-discipline / --max-parallel-files
    settings the parent CLI invocation was run with.
    """
    from .util import set_downloads_allowed, set_max_parallel_files

    set_downloads_allowed(os.environ.get("TESTDRIVE_WORKER_DOWNLOADS_ALLOWED") == "1")

    raw = os.environ.get("TESTDRIVE_WORKER_MAX_PARALLEL_FILES", "")
    if raw:
        set_max_parallel_files(int(raw))


def _serve_fatal_error(message: str) -> None:
    """Reply to every request with the same fatal error until shutdown.

    Used when the plugin itself couldn't even be loaded — there's no
    plugin object to retry with, but the worker still needs to respond
    to whatever requests the parent sends rather than hanging silently.
    """
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            request = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if request.get("cmd") == "shutdown":
            return
        _respond({"ok": False, "kind": "error", "message": message})


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m testdrive.worker_main <plugin_id>", file=sys.stderr)
        return 1

    # Must happen before any plugin-related import below — those can
    # pull in arbitrary third-party code (torch, mpi4py, transformers,
    # a plugin's own vendored repo, ...) that might write to stdout
    # during import alone, before we ever get to request handling.
    _protect_stdout()

    plugin_id = argv[1]
    _apply_parent_settings()

    from .imageio import load_image
    from .pluginloader import PluginLoadError, load_plugin
    from .util import CacheNotPopulatedError

    try:
        loaded = load_plugin(plugin_id)
    except PluginLoadError as exc:
        _serve_fatal_error(f"could not load plugin '{plugin_id}': {exc}")
        return 1

    plugin = loaded.instantiate()
    initialized = False
    init_error: dict[str, Any] | None = None

    def ensure_initialized(model: str | None) -> dict[str, Any] | None:
        """Run the model-override + dependency-check + initialize()
        sequence exactly once per worker lifetime; every later call
        (from either "init" or "detect" requests) is a cheap no-op that
        just returns whatever happened the first time. Returns an error
        dict on failure, None on success.
        """
        nonlocal initialized, init_error
        if initialized or init_error is not None:
            return init_error

        try:
            if model:
                plugin.set_model_override(model)

            installed, missing = plugin.is_installed()
            if not installed:
                init_error = {"ok": False, "kind": "missing_dependency", "missing": missing}
            else:
                plugin.initialize()
                initialized = True
        except CacheNotPopulatedError as exc:
            init_error = {"ok": False, "kind": "cache_not_populated", "message": str(exc)}
        except Exception as exc:  # noqa: BLE001
            # Full traceback to stderr — visible directly (see
            # _protect_stdout(); this can't corrupt the JSON protocol
            # pipe) — since the short message alone often isn't enough
            # to tell where inside vendored plugin code things broke
            # (e.g. a hardcoded .cuda() call three layers deep in a
            # submodule constructor, not anywhere near the line that
            # actually raised).
            traceback.print_exc(file=sys.stderr)
            init_error = {"ok": False, "kind": "error", "message": f"initialize() failed: {exc}"}

        return init_error

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            request = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            _respond({"ok": False, "kind": "error", "message": f"bad request JSON: {exc}"})
            continue

        cmd = request.get("cmd")
        if cmd == "shutdown":
            break

        if cmd == "init":
            error = ensure_initialized(request.get("model"))
            _respond(error if error is not None else {"ok": True})
            continue

        if cmd == "task":
            error = ensure_initialized(request.get("model"))
            if error is not None:
                _respond(error)
                continue
            try:
                image = load_image(Path(request["image_path"]))
                text_result = plugin.run_task(image, request["task_prompt"])
                _respond({"ok": True, "text_result": text_result})
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                _respond({"ok": False, "kind": "error", "message": f"run_task() raised: {exc}"})
            continue

        if cmd != "detect":
            _respond({"ok": False, "kind": "error", "message": f"unknown command: {cmd!r}"})
            continue

        error = ensure_initialized(request.get("model"))
        if error is not None:
            _respond(error)
            continue

        try:
            image = load_image(Path(request["image_path"]))
            detections = plugin.detect(image, request["prompt"], threshold=request["threshold"])
            _respond(
                {
                    "ok": True,
                    "detections": [
                        {"label": d.label, "score": d.score, "bbox": list(d.bbox)}
                        for d in detections
                    ],
                }
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            _respond({"ok": False, "kind": "error", "message": f"detect() raised: {exc}"})

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
