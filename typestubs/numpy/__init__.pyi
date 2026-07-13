"""Deliberately minimal placeholder stub for numpy — not a real stub.

Why this exists: recent numpy releases bundle their own inline
``__init__.pyi`` (PEP 561), which uses the ``type`` statement (PEP
695, Python 3.12+ only). mypy parses stub files using the grammar
implied by the configured ``python_version`` (pinned to ``"3.10"`` in
pyproject.toml — this project's actual supported floor, per
``requires-python``), so it hits a hard syntax error trying to parse
numpy's own stub, before semantic analysis (and any per-module
``[[tool.mypy.overrides]]`` setting, e.g. ``follow_imports = "skip"``)
ever gets a chance to apply. That override alone did not avoid the
crash in practice.

This file sidesteps the problem at its root: it's placed under
``mypy_path`` (see ``[tool.mypy]`` in pyproject.toml), which mypy
searches *before* installed packages' own stubs — so mypy resolves
``import numpy`` to this file instead of ever reading the real,
currently-unparseable one. Every attribute is typed ``Any``, which is
the same level of rigor already applied to torch/transformers/etc.
(see the ``ignore_missing_imports`` override above) — this project has
exactly one numpy call site (``models/samgd.py``), so the loss of
checking here is minimal and matches the existing untyped-dependency
posture.

Delete this file (and the ``mypy_path`` entry) once either: numpy
ships stubs parseable under this project's floor ``python_version``
again, or this project's floor is raised to 3.12+.

Note: this file alone only covers a bare ``import numpy`` — numpy's
own *submodules* (e.g. ``numpy._typing``, pulled in transitively by
other packages) are handled separately, via the
``follow_imports = "skip"`` override for ``"numpy.*"`` in
pyproject.toml. That combination matters: some of those submodules hit
the same PEP 695 problem in regular ``.py`` source (not just stub
files), where ``follow_imports = "skip"`` actually works (it only
applies to ``.py`` files, never ``.pyi`` stubs — which is exactly why
it alone couldn't stop the crash on this package's own ``__init__.pyi``
before this file existed).
"""

from typing import Any

def __getattr__(name: str) -> Any: ...
