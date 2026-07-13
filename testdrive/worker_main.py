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

Wire protocol (each message is exactly one line of JSON; nothing else
is ever written to stdout, so the pipe can't get corrupted by stray
prints):

    parent -> worker
        {"cmd": "init", "model": str | null}
        {"cmd": "detect", "image_path": str, "prompt": str,
         "threshold": float, "model": str | null}
        {"cmd": "shutdown"}

    worker -> parent
        # response to "init" (also implicitly run before the first
        # "detect", if not already done — so a plain detect-only caller
        # never needs to send "init" separately)
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

The worker never writes anything to stdout except these response
lines — plugin/library logging goes to stderr (Python logging's
default), which the parent lets pass through directly rather than
capturing, so worker-side messages are still visible to the person
running testdrive.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("testdrive.worker_main")


def _respond(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


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
            _respond({
                "ok": True,
                "detections": [
                    {"label": d.label, "score": d.score, "bbox": list(d.bbox)}
                    for d in detections
                ],
            })
        except Exception as exc:  # noqa: BLE001
            _respond({"ok": False, "kind": "error", "message": f"detect() raised: {exc}"})

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
