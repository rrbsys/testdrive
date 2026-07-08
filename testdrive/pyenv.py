"""Enforces that testdrive always runs from its own dedicated virtual
environment, ``cache/pyenv/framework`` (or wherever ``PLUGIN["pyenv"]``
points for a given plugin — see ``PluginManifest.pyenv``).

Why this exists: plugging more and more model plugins into the same
Python installation is a guaranteed road to dependency hell (the
"tensorflow disaster" we already hit once with transformers). Rather
than let that accumulate silently, testdrive insists on a single,
known-good, dedicated environment for the framework itself. Plugins
that need their own incompatible environment get one for real, via a
worker subprocess — see ``worker_pool.py`` / ``worker_main.py`` and
``PluginManifest.pyenv``'s docstring.

If testdrive is started from anywhere else, it tries to transparently
relaunch itself from ``cache/pyenv/framework`` if that environment
already exists — or, if it doesn't exist yet, stops and tells the user
exactly how to create it, rather than silently running under some
arbitrary interpreter that may or may not have the right dependencies
(or *any* dependencies at all).

This check only applies to real CLI invocations (the installed
``testdrive`` command, and ``python -m testdrive``) via
``cli.entrypoint()`` — it deliberately does not run for direct calls to
``cli.main()``, which is how the test suite drives the CLI. Set
``TESTDRIVE_SKIP_PYENV_CHECK=1`` to bypass it entirely (used by CI's
smoke test, which intentionally runs the core framework from whatever
plain environment the runner set up, not a dedicated one).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("testdrive.pyenv")

_SKIP_ENV_VAR = "TESTDRIVE_SKIP_PYENV_CHECK"
_RELAUNCH_MARKER = "TESTDRIVE_PYENV_RELAUNCHED"

#: The environment name every plugin uses unless it declares its own
#: (see PluginManifest.pyenv). Matches the default there.
FRAMEWORK_ENV_NAME = "framework"


def env_dir(cd: Path, name: str = FRAMEWORK_ENV_NAME) -> Path:
    """Return ``cache/pyenv/<platform>/<name>`` (not guaranteed to exist).

    Keyed by ``sys.platform`` because a venv is never portable across
    operating systems (different interpreter binary, different
    site-packages layout, compiled extensions built for the wrong OS,
    …) — sharing one ``cache/`` directory between, say, a Windows
    machine and a macOS one (a real scenario here: this project's cache
    has been used from both native Windows and Wine on macOS) would
    otherwise mean one platform's venv silently gets treated as if it
    were usable on the other.
    """
    return cd / "pyenv" / sys.platform / name


def recommended_python() -> str:
    """The python command this platform's setup instructions recommend.

    3.11 on macOS, 3.12 elsewhere — as of this writing, some of this
    project's heavier dependencies lag behind on macOS/3.12 wheel
    availability, so 3.11 is the safer default there specifically.
    """
    return "python3.11" if sys.platform == "darwin" else "python3.12"


def env_python_path(cd: Path, name: str = FRAMEWORK_ENV_NAME) -> Path:
    """Return the expected python executable inside ``cache/pyenv/<name>``.

    Doesn't check existence — callers that need to know whether the
    environment is actually set up should check ``.exists()`` on the
    result themselves.
    """
    d = env_dir(cd, name)
    if os.name == "nt":
        return d / "Scripts" / "python.exe"
    return d / "bin" / "python"


def is_in_env(cd: Path, name: str = FRAMEWORK_ENV_NAME) -> bool:
    """True if the *currently running* interpreter is that environment.

    Compares ``sys.prefix`` (the active venv's root directory) against
    ``cache/pyenv/<name>`` — not ``sys.executable``. A venv's
    ``bin/python`` is very often a symlink back to the system
    interpreter (routine on macOS in particular), so resolving symlinks
    on the executable path alone can make plain system Python and "the
    venv" look identical. ``sys.prefix`` reflects which environment is
    actually active regardless of that.
    """
    try:
        current = Path(sys.prefix).resolve()
        expected = env_dir(cd, name).resolve()
    except OSError:
        return False
    return current == expected


def _relaunch_instructions(cd: Path) -> str:
    py = env_python_path(cd)
    env = env_dir(cd)
    version = recommended_python()

    if os.name == "nt":
        venv_cmd = f'py -{version.removeprefix("python")} -m venv "{env}"'
    else:
        venv_cmd = f'{version} -m venv "{env}"'

    return (
        "testdrive isn't running from its dedicated environment.\n\n"
        f"Expected interpreter:\n    {py}\n\n"
        "That environment doesn't exist yet. Set it up once (3 steps —\n"
        "skip step 1 if you've already got `testdrive` running some other\n"
        "way, e.g. you're seeing this message at all):\n\n"
        "    1) pip install -e .\n"
        "       (into whatever Python you're using right now — this is what\n"
        "       makes the `testdrive` command exist at all; without it,\n"
        "       there's nothing to relaunch from cache/pyenv/... in the first\n"
        "       place, on a fresh checkout or a fresh shell)\n\n"
        f"    2) {venv_cmd}\n"
        "       (the dedicated environment itself)\n\n"
        f'    3) "{py.parent / ("pip.exe" if os.name == "nt" else "pip")}" install -e .\n'
        "       (testdrive again, this time INTO that dedicated environment —\n"
        "       step 1 alone does not do this, they're two separate installs)\n\n"
        "Then just re-run the same testdrive command — it will automatically\n"
        "relaunch itself from that environment from now on."
    )


def ensure_framework_env(cd: Path) -> None:
    """Guard entry point: make sure we're running from ``cache/pyenv/framework``.

    - Already there → returns immediately, no-op.
    - Not there, but the environment exists → relaunches via ``os.execve``
      (replaces the current process; does not return on success).
    - Not there, and the environment doesn't exist → prints setup
      instructions to stderr and exits with ``ExitCode.PYENV_NOT_CONFIGURED``.

    Bypassed entirely if ``TESTDRIVE_SKIP_PYENV_CHECK`` is set (any
    non-empty value) — for CI and advanced/development use.

    Relaunching is tried **at most once** per process tree (tracked via
    an injected marker env var, not a loop counter — ``execve`` replaces
    the process, so there's no in-memory state to loop on). Without this,
    a target that exists as a file but isn't actually a valid venv (e.g.
    missing ``pyvenv.cfg``, so Python never adopts it as ``sys.prefix``)
    would relaunch into itself, still not satisfy ``is_in_env()``, and
    relaunch again — forever. One bounded retry turns that into a clean,
    immediate, diagnosable error instead of a silent infinite loop.
    """
    if os.environ.get(_SKIP_ENV_VAR):
        log.debug("%s set — skipping framework-env check", _SKIP_ENV_VAR)
        return

    if is_in_env(cd):
        return

    from .util import ExitCode

    if os.environ.get(_RELAUNCH_MARKER):
        print(
            f"[testdrive] relaunched via {env_python_path(cd)} but still not running "
            "from the expected environment afterward. That path may not be a valid "
            "virtual environment (e.g. missing pyvenv.cfg) — recreate it with:\n\n"
            f'    {recommended_python()} -m venv "{env_dir(cd)}"\n',
            file=sys.stderr,
        )
        sys.exit(ExitCode.PYENV_NOT_CONFIGURED)

    target = env_python_path(cd)
    if target.exists():
        print(f"[testdrive] relaunching via {target} ...", file=sys.stderr)
        try:
            relaunch_env = os.environ.copy()
            relaunch_env[_RELAUNCH_MARKER] = "1"
            os.execve(str(target), [str(target), "-m", "testdrive"] + sys.argv[1:], relaunch_env)
            return  # pragma: no cover — os.execve never returns on success
        except OSError as exc:
            print(f"[testdrive] could not relaunch via {target}: {exc}\n", file=sys.stderr)
            # fall through to the setup instructions below

    print(_relaunch_instructions(cd), file=sys.stderr)
    sys.exit(ExitCode.PYENV_NOT_CONFIGURED)
