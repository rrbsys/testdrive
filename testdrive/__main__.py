import sys
from .cli import entrypoint


def _needs_safe_reexec() -> bool:
    """True if this process wasn't started with -P / PYTHONSAFEPATH and
    the "testdrive" package has ended up as a corrupted, empty PEP 420
    namespace package instead of the real one.

    `python -m testdrive` (and plain `python testdrive/__main__.py`)
    prepend the current working directory to sys.path. If cwd has - or
    sits right next to - a directory literally named "testdrive" (common:
    testdrive's own default cache/home directory is named exactly that,
    so this reliably bites anyone who runs `-m testdrive` from their own
    home directory), the stdlib PathFinder claims "testdrive" as an empty
    namespace package from that shadowing directory before this
    package's own (editable-install) finder is ever consulted - even
    though the finder is entirely correct and the real package is right
    there. The result is a `sys.modules["testdrive"]` with no
    `__version__`/`__file__` (repr shows "unknown location"), while
    individual submodules like this one can still load fine via a
    same-package fallback in the editable finder - which is what makes
    the resulting ImportError ("cannot import name '__version__' from
    'testdrive' (unknown location)") so confusing to hit blind.

    `-P` (PYTHONSAFEPATH, Python 3.11+) disables that cwd-prepending
    behavior entirely. Re-exec with it once, rather than trying to detect
    and repair the already-corrupted module object after the fact.
    """
    if getattr(sys.flags, "safe_path", False):
        return False  # already safe - not (or already past) the bug
    if sys.version_info < (3, 11):
        return False  # -P isn't available; nothing we can do here
    pkg = sys.modules.get("testdrive")
    return pkg is None or not hasattr(pkg, "__version__")


if _needs_safe_reexec():
    import os

    os.execv(sys.executable, [sys.executable, "-P", "-m", "testdrive", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(entrypoint())
