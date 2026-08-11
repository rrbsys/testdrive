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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

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


def env_pip_path(cd: Path, name: str = FRAMEWORK_ENV_NAME) -> Path:
    """Return the expected pip executable inside ``cache/pyenv/<name>``
    (same existence caveat as env_python_path).
    """
    d = env_dir(cd, name)
    if os.name == "nt":
        return d / "Scripts" / "pip.exe"
    return d / "bin" / "pip"


def install_hint(cd: Path, name: str, packages: list[str], pip_options: str = "") -> str:
    """A single, copy-pasteable command to install ``packages`` into the
    named environment — the full OS-correct path to that environment's
    own pip, not just a bare ``pip install ...`` (which would silently
    run against whatever pip happens to be on PATH, not necessarily the
    right environment at all). *pip_options* (see
    ``PluginManifest.pip_options``) is inserted before the package
    list, exactly as ``run_pip_install`` below would use it — e.g.
    ``--no-build-isolation`` for a plugin whose build needs it.
    """
    pip = env_pip_path(cd, name)
    opts = f"{pip_options} " if pip_options else ""
    pkgs = " ".join(f'"{p}"' for p in packages)
    return f'"{pip}" install {opts}{pkgs}'


def run_pip_install(
    python_path: Path,
    requirements: list[str],
    pip_options: str = "",
    patches: list[dict[str, str]] | None = None,
    announce: "Callable[[str], None] | None" = None,
) -> None:
    """Install *requirements* (already-formatted pip arguments, e.g.
    ``"torch==2.0.0"`` or ``"name @ git+https://..."``) into the
    environment at *python_path* — this is the one place that actually
    runs the pip command a plugin's own manifest describes (see
    ``PluginManifest.requirements``/``pip_options``/``patches``), used
    both for a plugin with its own dedicated pyenv (see
    ``worker_pool._provision_plugin_env``) and for a "framework"-pyenv
    plugin being auto-provisioned directly into the shared framework
    environment (see ``selftest.py``).

    *patches* (see ``PluginManifest.patches``) are applied — via
    ``apply_source_patches`` below — after installing every non-VCS
    entry in *requirements* but before any ``"... @ git+..."`` entry:
    the ordering a plugin like seem needs (patch an already-installed
    dependency's vendored header before building something else from
    source against it), and the reason this doesn't just run a single
    ``pip install`` covering everything at once even when *patches* is
    empty and there's nothing to split for.

    Raises ``subprocess.CalledProcessError`` (pip itself failed) or
    ``RuntimeError`` (a patch's target file is missing, or its "find"
    text doesn't appear exactly once — see ``apply_source_patches``).
    Callers translate either into their own error type with more
    context (``WorkerError``, etc.) — this function only runs the
    commands.
    """
    import shlex
    import subprocess

    _announce = announce or (lambda _msg: None)
    patches = patches or []

    extra_args = shlex.split(pip_options) if pip_options else []
    vcs_reqs = [r for r in requirements if " @ git+" in r or r.startswith("git+")]
    normal_reqs = [r for r in requirements if r not in vcs_reqs]

    def _pip_install(pkgs: list[str]) -> None:
        if not pkgs:
            return
        cmd = [str(python_path), "-m", "pip", "install", *extra_args, *pkgs]
        _announce(f"installing via {' '.join(cmd)}  (this may take a while)")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    _pip_install(normal_reqs)
    if patches:
        apply_source_patches(python_path, patches, announce=_announce)
    _pip_install(vcs_reqs)


def apply_source_patches(
    python_path: Path,
    patches: list[dict[str, str]],
    announce: "Callable[[str], None] | None" = None,
) -> None:
    """Apply ``PluginManifest.patches``-shaped patches to the environment
    at *python_path*.

    Resolves *that* environment's own site-packages directory by
    running a one-liner inside it (not this process's own
    ``sysconfig`` — a plugin's dedicated pyenv is very often a
    different Python version/layout entirely than the framework's),
    then for each patch: if "replace" is already present in the target
    file, treats it as already applied and moves on — provisioning
    stays safe to re-run against an already-patched environment.
    Otherwise requires "find" to appear in the target file exactly
    once (same safety property this project's own development process
    already relies on for editing *this* source tree) before
    substituting it for "replace".

    Raises ``RuntimeError`` if a target file doesn't exist, or "find"
    doesn't appear exactly once and "replace" isn't already present
    either — most likely an unexpected dependency version this patch
    was never written against.
    """
    import subprocess

    _announce = announce or (lambda _msg: None)
    if not patches:
        return

    site_packages = Path(
        subprocess.run(
            [str(python_path), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    for patch in patches:
        target = site_packages / patch["target"]
        find, replace = patch["find"], patch["replace"]
        _announce(f"patching {target} ...")
        if not target.exists():
            raise RuntimeError(f"patch target does not exist: {target}")
        text = target.read_text()
        if replace in text and find not in text:
            continue  # already applied — safe to re-run provisioning
        count = text.count(find)
        if count != 1:
            raise RuntimeError(
                f"patch target {target}: expected the patch's 'find' text to "
                f"appear exactly once, found {count} instead — the installed "
                f"dependency version likely doesn't match what this patch was "
                f"written against"
            )
        target.write_text(text.replace(find, replace))


def check_modules_installed(
    python_path: Path, requirements: list[dict[str, str]]
) -> tuple[bool, list[str]]:
    """The out-of-process equivalent of ``DetectorPlugin.is_installed()``.

    Checks, via a quick subprocess using *python_path*'s own
    interpreter, which of *requirements* (``PluginManifest.requirements``
    -shaped: each a dict with ``"pip"``/``"module"`` keys) can actually
    be imported there.

    Needed for any plugin whose dependencies live in a dedicated pyenv
    (see ``PluginManifest.pyenv``) other than the one this process
    happens to be running in — an in-process ``importlib.import_module``
    check there would just be probing the wrong interpreter's
    site-packages entirely, and would report a fully-provisioned plugin
    as "missing" (see cli.py's ``-I`` for the caller; this was the same
    class of bug ``-T``/``-TT`` already had to fix for exactly this
    reason — see ``selftest.run_selftest``'s ``uses_worker`` branch).

    Deliberately narrower than the real worker protocol
    (``worker_pool.py``/``worker_main.py``): only imports modules, never
    starts a long-lived worker, never triggers auto-provisioning or a
    download. Safe to call freely for a status check even with
    downloads disabled.

    Returns ``(False, [f"error probing environment: ..."])`` — not an
    exception — if the probe itself can't run (interpreter missing,
    times out, crashes): from a status-reporting caller's point of
    view that's just another reason to call this plugin "not usable
    right now", same shape as an actually-missing package.
    """
    import json as _json
    import subprocess

    if not requirements:
        return True, []

    modules = [req["module"] for req in requirements]
    probe = (
        "import importlib, json\n"
        f"modules = {modules!r}\n"
        "missing = []\n"
        "for m in modules:\n"
        "    try:\n"
        "        importlib.import_module(m)\n"
        "    except ImportError:\n"
        "        missing.append(m)\n"
        "print(json.dumps(missing))\n"
    )
    try:
        completed = subprocess.run(
            [str(python_path), "-c", probe],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, [f"error probing environment: {exc}"]

    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        return False, [f"error probing environment: {detail}"]

    try:
        missing_modules = set(_json.loads(completed.stdout))
    except (_json.JSONDecodeError, ValueError):
        return False, [f"error probing environment: unexpected output {completed.stdout!r}"]

    missing_pip = [req["pip"] for req in requirements if req["module"] in missing_modules]
    return (len(missing_pip) == 0, missing_pip)


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
    - Not there, but the environment exists → relaunches into it and does
      not return: on POSIX via ``os.execve`` (a true in-place exec — same
      PID, so anything waiting on us waits on the *real* run); on Windows
      via a synchronous child process, after which we ``sys.exit()`` with
      its exact return code (see the platform note below for why this
      split exists).
    - Not there, and the environment doesn't exist → prints setup
      instructions to stderr and exits with ``ExitCode.PYENV_NOT_CONFIGURED``.

    Bypassed entirely if ``TESTDRIVE_SKIP_PYENV_CHECK`` is set (any
    non-empty value) — for CI and advanced/development use.

    Relaunching is tried **at most once** per process tree (tracked via
    an injected marker env var, not a loop counter — the relaunch, on
    either platform, does not return to this frame on success, so there's
    no in-memory state to loop on). Without this, a target that exists as
    a file but isn't actually a valid venv (e.g. missing ``pyvenv.cfg``,
    so Python never adopts it as ``sys.prefix``) would relaunch into
    itself, still not satisfy ``is_in_env()``, and relaunch again —
    forever. One bounded retry turns that into a clean, immediate,
    diagnosable error instead of a silent infinite loop.

    Platform note — why Windows can't just use ``os.execve`` too:
    On POSIX, ``execve`` really does replace the current process image in
    place; the PID never changes and nothing above us (a shell, a wrapper
    script, a CI runner) ever regains control until the *relaunched* run
    finishes. Windows has no such syscall. CPython emulates ``os.execve``
    there via the C runtime's "overlay" spawn mode, which is fire-and-forget
    from the parent's point of view: it starts the new interpreter as a
    *separate* process and then immediately terminates the current one,
    without waiting for the new one to do anything. Any process tree above
    us that's watching *this* PID (cmd.exe, a console-script launcher .exe,
    a CI step) sees testdrive "finish" the instant the relaunch is kicked
    off — while the real run keeps going in the background, racing
    whatever the caller does next (read output files, report a step as
    complete, tear down a temp dir the worker still needs) and interleaving
    its console output with anything typed afterward. That's silent data-
    race territory, not just a cosmetic glitch, so on Windows we spawn the
    relaunch as a real child process and block on it with ``subprocess.run``
    instead, then propagate its exit code ourselves.
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
        relaunch_env = os.environ.copy()
        relaunch_env[_RELAUNCH_MARKER] = "1"
        argv = [str(target), "-m", "testdrive"] + sys.argv[1:]

        if sys.platform == "win32":
            # See the platform note in this function's docstring: on
            # Windows, os.execve does not block whatever launched *us*, so
            # we spawn synchronously ourselves and wait for the real exit
            # code instead of trusting a fire-and-forget "exec".
            import subprocess

            try:
                completed = subprocess.run(argv, env=relaunch_env)
            except OSError as exc:
                print(f"[testdrive] could not relaunch via {target}: {exc}\n", file=sys.stderr)
                # fall through to the setup instructions below
            else:
                sys.exit(completed.returncode)
        else:
            try:
                os.execve(str(target), argv, relaunch_env)
                return  # pragma: no cover — os.execve never returns on success
            except OSError as exc:
                print(f"[testdrive] could not relaunch via {target}: {exc}\n", file=sys.stderr)
                # fall through to the setup instructions below

    print(_relaunch_instructions(cd), file=sys.stderr)
    sys.exit(ExitCode.PYENV_NOT_CONFIGURED)
