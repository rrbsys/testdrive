"""Model cache directory resolution.

All HF/transformers downloads are routed through a single cache
directory so they are predictable and easy to share or delete.

Resolution order
----------------
1. ``TESTDRIVE_CACHE`` environment variable (absolute or relative path).
2. ``cache/`` inside the project root — determined as the parent of
   this file's package directory, i.e. ``<repo>/cache/``.

The resolved path is created on first access if it does not exist.

Usage in plugin ``initialize()``
---------------------------------
::

    from ..cache import cache_dir

    processor = AutoProcessor.from_pretrained(repo, cache_dir=cache_dir())
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(
                    repo, cache_dir=cache_dir())
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger("testdrive.cache")

# Project root = parent of the ``testdrive/`` package directory
_PACKAGE_DIR = Path(__file__).parent  # …/testdrive/
_PROJECT_ROOT = _PACKAGE_DIR.parent  # …/

_DEFAULT_CACHE = _PROJECT_ROOT / "cache"
_ENV_VAR = "TESTDRIVE_CACHE"


def cache_dir() -> Path:
    """Return the resolved, guaranteed-to-exist cache directory.

    Reads ``TESTDRIVE_CACHE`` on every call so that the env var can be
    changed at runtime (useful in tests).
    """
    raw = os.environ.get(_ENV_VAR, "")
    if raw.strip():
        path = Path(raw.strip()).expanduser().resolve()
    else:
        path = _DEFAULT_CACHE.resolve()

    path.mkdir(parents=True, exist_ok=True)
    log.debug("cache dir: %s  (source: %s)", path, _ENV_VAR if raw.strip() else "default")
    return path


def cache_info() -> dict[str, str]:
    """Return a dict describing the active cache configuration (for ``-M`` / ``-I``)."""
    raw = os.environ.get(_ENV_VAR, "")
    source = f"${_ENV_VAR}" if raw.strip() else "default"
    return {
        "path": str(cache_dir()),
        "source": source,
        "env_var": _ENV_VAR,
    }
