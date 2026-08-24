#!/usr/bin/env python3
"""
autoconfig.py

Bootstraps a testdrive installation the first time it is run, or inspects
an existing one on subsequent runs.

Behavior:
  1. If <testdrive-home> does not exist:
     - create it, download and unpack <testdrive-remoteurl>
     - create a framework virtualenv (using --python-path) and editable-
       install it (`pip install -e .[dev]`, one combined call). This is
       required by testdrive's own design: a plugin with its own
       dedicated pyenv (e.g. yolo11's "newenv") gets provisioned by
       testdrive running `pip install -e <project_root>` into it, where
       project_root is derived from testdrive's own __file__ at
       runtime - which only points at a real pyproject.toml if
       testdrive itself was installed editable (see pyenv.py's
       relaunch instructions and worker_pool._project_root()).
     - for each plugin: run `<framework-python> -m testdrive -TT <plugin>`
     - for each plugin: print `testdrive -I <plugin>` install details, or
       the failure output from the last `-TT` run if it failed
  2. If <testdrive-home> exists:
     - reuse <tdhome>/cache/pyenv/<sys.platform>/framework/{bin,Scripts}/python
       as the interpreter
     - for each plugin: print `testdrive -I <plugin>` install details
     - with --force-provisioning: any plugin -I reports as 'installed: no'
       gets provisioned via `testdrive -T <plugin>` first, then re-checked

Both `-TT` and `-I` are invoked with TESTDRIVE_CACHE pinned to
<tdhome>/cache. testdrive resolves its own cache/pyenv layout from
TESTDRIVE_CACHE if set, and otherwise falls back to a path relative to
wherever the testdrive package itself is installed (i.e. inside the venv's
site-packages) - which does *not* match <tdhome>/cache and, worse, makes
testdrive's own "am I running from my dedicated env" self-check compare
sys.executable against a wrong, nested expected path (reported as
"testdrive isn't running from its dedicated environment" even when it
actually is). Pinning TESTDRIVE_CACHE keeps testdrive's notion of home in
sync with the layout this script creates.

Usage:
    python autoconfig.py [options]

Options (defaults shown):
    --testdrive-remoteurl  https://github.com/rrbsys/testdrive/archive/refs/heads/main.zip
    --testdrive-home       <home>/testdrive
    --plugins              "yunet,yolo11"
    --python-path          "python"
    --force-provisioning   (off) provision existing-home plugins reported as not installed
"""

import argparse
import io
import json
import logging
import os
import platform
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("autoconfig")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    home = Path.home()
    parser = argparse.ArgumentParser(description="Bootstrap or inspect a testdrive installation.")
    parser.add_argument(
        "--testdrive-remoteurl",
        default="https://github.com/rrbsys/testdrive/archive/refs/heads/main.zip",
        # tag:    default="https://github.com/rrbsys/testdrive/archive/refs/tags/v0.1.4.zip",
        # branch: default="https://github.com/rrbsys/testdrive/archive/refs/heads/some/branch.zip",
        # commit: default="https://github.com/rrbsys/testdrive/archive/abcdef0.zip",
        help="URL of the testdrive source archive to download on first run.",
    )
    parser.add_argument(
        "--testdrive-home",
        default=str(home / "testdrive"),
        help="Directory that holds the testdrive install (source + envs + cache).",
    )
    parser.add_argument(
        "--plugins",
        default="yunet,yolo11",
        help="Comma separated list of plugins to try out / report on.",
    )
    parser.add_argument(
        "--python-path",
        default="python",
        help="Python interpreter used to create the framework environment.",
    )
    parser.add_argument(
        "--force-provisioning",
        action="store_true",
        help=(
            "On an existing testdrive home, run `testdrive -T <plugin>` to "
            "provision any plugin that `-I` currently reports as "
            "'installed: no', then re-check and report its updated status. "
            "By default, inspecting an existing home is read-only. No "
            "effect on a fresh install - bootstrap already provisions "
            "every plugin unconditionally."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return parser.parse_args(argv)


def split_plugins(raw):
    return [p.strip() for p in raw.split(",") if p.strip()]


# --------------------------------------------------------------------------
# Download / unpack
# --------------------------------------------------------------------------
def download_and_unpack(url: str, dest_dir: Path) -> Path:
    """Download `url` (a zip archive) and unpack it under dest_dir.

    Returns the path to the (single) top-level directory the archive
    extracted into.
    """
    log.info("Downloading testdrive source from %s", url)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as resp:
        data = resp.read()

    log.info("Unpacking archive into %s", dest_dir)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.strip()]
        top_levels = {n.split("/", 1)[0] for n in names}
        zf.extractall(dest_dir)

    if len(top_levels) != 1:
        raise RuntimeError(
            f"Expected exactly one top-level directory in archive, found: {top_levels}"
        )

    return dest_dir / next(iter(top_levels))


# --------------------------------------------------------------------------
# Framework environment
# --------------------------------------------------------------------------
def cache_dir(td_home: Path) -> Path:
    """<tdhome>/cache - passed to testdrive via TESTDRIVE_CACHE so its own
    cache_dir()/pyenv layout lines up with what this script creates."""
    return td_home / "cache"


def framework_env_dir(td_home: Path) -> Path:
    return cache_dir(td_home) / "pyenv" / sys.platform / "framework"


def framework_python(td_home: Path) -> Path:
    env_dir = framework_env_dir(td_home)
    if platform.system() == "Windows":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def testdrive_env(td_home: Path) -> dict:
    """Environment for any subprocess that runs `-m testdrive ...` - pins
    TESTDRIVE_CACHE so testdrive resolves cache/pyenv paths under
    <tdhome>/cache instead of relative to its own install location."""
    env = os.environ.copy()
    env["TESTDRIVE_CACHE"] = str(cache_dir(td_home))
    return env


def create_framework_env(td_home: Path, python_path: str) -> Path:
    env_dir = framework_env_dir(td_home)
    log.info("Creating framework env at %s (using %s)", env_dir, python_path)
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([python_path, "-m", "venv", str(env_dir)], check=True)
    return framework_python(td_home)


def pip_install(env_python: Path, *args: str, cwd: Path = None):
    log.info("Installing %s%s", " ".join(args), f" (cwd={cwd})" if cwd else "")
    subprocess.run(
        [str(env_python), "-m", "pip", "install", *args],
        check=True,
        cwd=str(cwd) if cwd else None,
    )


# --------------------------------------------------------------------------
# Plugin operations (both shell out to the framework env's own testdrive,
# not to pip, so framework-pyenv plugins *and* plugins with their own
# dedicated pyenv - see PluginManifest.pyenv - are handled the same way
# testdrive itself handles them)
def _safe_cwd(env_python: Path) -> Path:
    """A working directory for `-m testdrive` subprocess calls that is
    guaranteed not to contain a subdirectory literally named "testdrive".

    `python -m X` prepends the process's cwd to sys.path (unless run with
    `-P` / PYTHONSAFEPATH, which needs Python 3.11+ - testdrive only
    requires >=3.10, so it can't be relied on alone). If cwd has a
    sibling/child directory named "testdrive" - which is exactly what
    happens by default, since --testdrive-home defaults to
    "<home>/testdrive" and it's easy to end up running this script *from*
    <home> - the stdlib PathFinder claims "testdrive" as an empty PEP 420
    namespace package from that shadowing directory before the editable
    install's own finder (appended later in sys.meta_path) is ever
    consulted. That produces exactly this failure, regardless of platform:
        ImportError: cannot import name '__version__' from 'testdrive'
        (unknown location)
    even though the real, correctly editable-installed package is right
    there and would resolve fine from any other cwd. The venv's own
    interpreter directory (`.../bin` or `...\\Scripts`) is a safe, always-
    available choice - it does not, and cannot sensibly, contain a
    "testdrive" subdirectory.
    """
    return env_python.parent


def _supports_dash_p(env_python: Path) -> bool:
    """Whether `env_python` is Python 3.11+ and therefore supports `-P`
    (PYTHONSAFEPATH), which disables prepending cwd/script-dir to
    sys.path. Checked once per env rather than assumed, since testdrive
    itself only requires >=3.10."""
    try:
        result = subprocess.run(
            [str(env_python), "-c", "import sys; print(sys.version_info >= (3, 11))"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and result.stdout.strip() == "True"


def _testdrive_argv(env_python: Path, *args: str) -> list:
    """Build the argv for an `-m testdrive ...` subprocess call, adding
    `-P` when supported.

    Belt-and-suspenders alongside `_safe_cwd()`/passing an explicit
    `cwd=`: cwd/script-dir insertion into sys.path is what lets a
    directory literally named "testdrive" (e.g. <testdrive_home> itself,
    or anything sitting next to it) get claimed by the stdlib PathFinder
    as an empty PEP 420 namespace package ahead of the real editable
    install's own finder - producing
        ImportError: cannot import name '__version__' from 'testdrive'
        (unknown location)
    even though the real, correctly editable-installed package is right
    there. Pinning `cwd` fixes this whenever the subprocess layer honors
    it correctly; `-P` fixes it at the interpreter level regardless of
    that, which matters when running under environments (e.g. Wine) where
    subprocess cwd handling may not exactly match a native, interactive
    invocation.
    """
    argv = [str(env_python)]
    if _supports_dash_p(env_python):
        argv.append("-P")
    argv += ["-m", "testdrive", *args]
    return argv


# --------------------------------------------------------------------------
def run_testdrive_t(env_python: Path, td_home: Path, plugin: str):
    """Run `testdrive -T <plugin>` (self-test) using the framework env.

    Unlike `-TT` (example-test - see run_testdrive_tt()), `-T` is what
    actually auto-provisions a plugin's dependencies: for a "framework"
    pyenv plugin it installs missing deps straight into the running
    framework env (see selftest.py's run_selftest()); for a plugin with
    its own dedicated pyenv it triggers that env's worker-based
    auto-provisioning (see worker_pool._provision_plugin_env()) - the
    same mechanism `-TT` also triggers for worker-routed plugins, which
    is why a dedicated-pyenv plugin like yolo11 auto-provisions correctly
    under a bare `-TT`, while a framework-pyenv plugin like yunet does
    not: `-TT`'s dependency check for framework-pyenv plugins is a plain
    check-and-fail with no provisioning step (see cli.py's
    _run_detect_one) - only `-T` provisions that case. Called before
    `-TT` in bootstrap() so both plugin kinds end up provisioned the same
    way before the real (example-image-based) test runs.

    Returns (success: bool, output: str).
    """
    log.info("Running testdrive -T %s (provisioning)", plugin)
    try:
        result = subprocess.run(
            _testdrive_argv(env_python, "-T", plugin),
            check=False,
            capture_output=True,
            text=True,
            env=testdrive_env(td_home),
            cwd=str(_safe_cwd(env_python)),
        )
    except FileNotFoundError as exc:
        return False, str(exc)

    success = result.returncode == 0
    output = (result.stdout or "") + (result.stderr or "")
    return success, output.strip()


def run_testdrive_tt(env_python: Path, td_home: Path, plugin: str):
    """Run `testdrive -TT <plugin>` using the framework env.

    Returns (success: bool, output: str).
    """
    log.info("Running testdrive -TT %s", plugin)
    try:
        result = subprocess.run(
            _testdrive_argv(env_python, "-TT", plugin),
            check=False,
            capture_output=True,
            text=True,
            env=testdrive_env(td_home),
            cwd=str(_safe_cwd(env_python)),
        )
    except FileNotFoundError as exc:
        return False, str(exc)

    success = result.returncode == 0
    output = (result.stdout or "") + (result.stderr or "")
    return success, output.strip()


def _format_install_status(r: dict) -> str:
    lines = [f"pyenv     : {r.get('pyenv')}  ({r.get('pyenv_path')})"]
    lines.append(f"installed : {'yes' if r.get('installed') else 'no'}")
    if r.get("backend"):
        lines.append(f"backend   : {r['backend']}")
    missing = r.get("missing") or []
    if missing:
        lines.append("missing   :")
        for m in missing:
            lines.append(f"    {m}")
    return "\n".join(lines)


def _query_install_status(env_python: Path, td_home: Path, plugin: str):
    """Run `testdrive -I <plugin> --json` and return the raw status dict.

    Returns the parsed dict on success, an error string if status
    couldn't be determined (non-JSON output, plugin not found, etc.), or
    None if the interpreter/command isn't available at all.
    """
    try:
        result = subprocess.run(
            _testdrive_argv(env_python, "-I", plugin, "--json"),
            check=False,
            capture_output=True,
            text=True,
            env=testdrive_env(td_home),
            cwd=str(_safe_cwd(env_python)),
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0 and not result.stdout.strip():
        # Genuine failure (plugin not found, env broken, etc.) - surface
        # stderr rather than silently reporting "not installed".
        return (result.stderr or "").strip() or None

    try:
        data = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        return (result.stdout or result.stderr or "").strip() or None

    # `-I <plugin>` returns a single dict; `-I` (all) returns a list.
    if isinstance(data, list):
        data = data[0] if data else {}

    return data


def list_install_details(env_python: Path, td_home: Path, plugin: str):
    """Return `testdrive -I <plugin>` install status as formatted text,
    or None if the interpreter/command isn't available at all."""
    status = _query_install_status(env_python, td_home, plugin)
    if isinstance(status, dict):
        return _format_install_status(status)
    return status


# --------------------------------------------------------------------------
# Main flows
# --------------------------------------------------------------------------
def bootstrap(td_home: Path, remote_url: str, python_path: str, plugins):
    log.info("testdrive home %s does not exist - bootstrapping", td_home)

    td_home.mkdir(parents=True, exist_ok=True)
    source_dir = download_and_unpack(remote_url, td_home / "src")
    env_python = create_framework_env(td_home, python_path)

    # Editable install (`-e`), not regular - this is required by
    # testdrive's own design, not optional: for a plugin with its own
    # dedicated pyenv (e.g. yolo11's "newenv"), testdrive provisions
    # that env by running `pip install -e <project_root>` into it,
    # where project_root is computed as
    # Path(__file__).resolve().parent.parent from inside the *running*
    # testdrive package (see worker_pool._project_root()). With a
    # regular install, __file__ resolves to a site-packages copy with
    # no pyproject.toml next to it, and that provisioning step fails
    # with "does not appear to be a Python project". With an editable
    # install, __file__ resolves back to this real checkout, where
    # pyproject.toml genuinely exists - exactly as pyenv.py's own
    # relaunch instructions describe (step 1: `pip install -e .`).
    #
    # One combined install (base + dev extra together), not two
    # separate `-e .` / `-e .[dev]` calls: two sequential editable
    # installs regenerate the editable finder/.pth twice for no
    # benefit and are a needless source of inconsistency.
    pip_install(env_python, "-e", ".[dev]", cwd=source_dir)

    tt_results = {}
    for plugin in plugins:
        # -T first: provisions the plugin's dependencies (framework-env
        # install, or dedicated-pyenv auto-setup) - see run_testdrive_t().
        # Its own pass/fail isn't the signal we report; -TT is. But
        # skipping it means framework-pyenv plugins never get their
        # deps installed at all, since -TT alone won't do that for them.
        t_success, t_output = run_testdrive_t(env_python, td_home, plugin)
        if not t_success:
            log.warning("testdrive -T %s (provisioning) failed:\n%s", plugin, t_output)
        tt_results[plugin] = run_testdrive_tt(env_python, td_home, plugin)

    report = {}
    for plugin in plugins:
        success, output = tt_results[plugin]
        if success:
            details = list_install_details(env_python, td_home, plugin)
            report[plugin] = details or "(installed, no status available)"
        else:
            report[plugin] = f"FAILED: {output}"

    return report


def inspect_existing(td_home: Path, plugins, force_provisioning: bool = False):
    log.info("testdrive home %s exists - inspecting", td_home)
    env_python = framework_python(td_home)

    if not env_python.exists():
        log.warning("Expected framework python not found at %s", env_python)

    report = {}
    for plugin in plugins:
        status = _query_install_status(env_python, td_home, plugin)

        if force_provisioning and isinstance(status, dict) and not status.get("installed"):
            # Same provisioning step bootstrap() already runs for every
            # plugin up front (see run_testdrive_t()) - here it's only
            # run for plugins that -I already confirmed are actually
            # missing, so a normal `inspect_existing` run stays read-only
            # by default.
            log.info("Provisioning %s (currently reported as not installed)", plugin)
            t_success, t_output = run_testdrive_t(env_python, td_home, plugin)
            if not t_success:
                log.warning("testdrive -T %s (provisioning) failed:\n%s", plugin, t_output)
            status = _query_install_status(env_python, td_home, plugin)

        if isinstance(status, dict):
            report[plugin] = _format_install_status(status)
        else:
            report[plugin] = status or "(not installed / no status available)"

    return report


def write_launcher(td_home: Path) -> Path:
    """Write a small launcher (testdrive.bat / testdrive) at the top of
    <td_home> that pins TESTDRIVE_CACHE and calls the framework env's own
    installed `testdrive` console-script.

    Without this, manually running testdrive (rather than through this
    script, which always sets TESTDRIVE_CACHE for its own subprocess
    calls - see testdrive_env()) hits a real, separate gap: testdrive's
    own default cache location (when TESTDRIVE_CACHE isn't set) is
    <project_root>/cache - relative to wherever its *source checkout*
    lives - which doesn't match this script's layout, where the venv/
    cache deliberately live at <td_home>/cache, a sibling of src/, so
    that re-downloading a newer source version doesn't strand the
    existing venv. The mismatch surfaces as testdrive's own
    "isn't running from its dedicated environment" self-check failing,
    pointing at a nonexistent path nested inside src/. This launcher
    means a person never has to remember `set/export TESTDRIVE_CACHE=...`
    by hand to use testdrive directly.

    Uses the console-script entry point (not `-m testdrive`), which also
    sidesteps the separate cwd/sys.path shadowing issue `-m` can hit when
    run from a directory that has - or sits next to - one literally
    named "testdrive" (see _safe_cwd()/_testdrive_argv() above, and the
    same-shaped fix applied directly in testdrive's own __main__.py).
    """
    env_python = framework_python(td_home)
    testdrive_exe = env_python.parent / (
        "testdrive.exe" if platform.system() == "Windows" else "testdrive"
    )
    cache = cache_dir(td_home)

    if platform.system() == "Windows":
        launcher = td_home / "testdrive.bat"
        launcher.write_text(
            f'@echo off\r\nset "TESTDRIVE_CACHE={cache}"\r\n"{testdrive_exe}" %*\r\n',
            encoding="utf-8",
        )
    else:
        launcher = td_home / "testdrive"
        launcher.write_text(
            f'#!/bin/sh\nexport TESTDRIVE_CACHE="{cache}"\nexec "{testdrive_exe}" "$@"\n',
            encoding="utf-8",
        )
        launcher.chmod(launcher.stat().st_mode | 0o111)

    return launcher


def print_report(report):
    print("\n=== Plugin report ===")
    for plugin, details in report.items():
        print(f"\n--- {plugin} ---")
        print(details)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    td_home = Path(args.testdrive_home).expanduser()
    plugins = split_plugins(args.plugins)

    if not td_home.exists():
        report = bootstrap(td_home, args.testdrive_remoteurl, args.python_path, plugins)
    else:
        report = inspect_existing(td_home, plugins, force_provisioning=args.force_provisioning)

    print_report(report)

    launcher = write_launcher(td_home)
    print(f"\nFor manual use, run testdrive via: {launcher}")

    if any(str(v).startswith("FAILED") for v in report.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
