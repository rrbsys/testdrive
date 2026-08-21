#!/usr/bin/env python3
"""
autoconfig.py

Bootstraps a testdrive installation the first time it is run, or inspects
an existing one on subsequent runs.

Behavior:
  1. If <testdrive-home> does not exist:
     - create it, download and unpack <testdrive-remoteurl>
     - create a framework virtualenv (using --python-path) and
       `pip install .` / `pip install .[dev]` into it (a regular,
       non-editable install - relies on testdrive's pyproject.toml
       shipping examples/ as package-data; see that file's history for
       why an editable install was tried and reverted)
     - for each plugin: run `<framework-python> -m testdrive -TT <plugin>`
     - for each plugin: print `testdrive -I <plugin>` install details, or
       the failure output from the last `-TT` run if it failed
  2. If <testdrive-home> exists:
     - reuse <tdhome>/cache/pyenv/<sys.platform>/framework/{bin,Scripts}/python
       as the interpreter
     - for each plugin: print `testdrive -I <plugin>` install details

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
    --testdrive-remoteurl  https://github.com/rrbsys/testdrive/archive/refs/tags/v0.1.3.zip
    --testdrive-home       <home>/testdrive
    --plugins              "yunet,yolo11"
    --python-path          "python"
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
        default="https://github.com/rrbsys/testdrive/archive/refs/tags/v0.1.3.zip",
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
# --------------------------------------------------------------------------
def run_testdrive_tt(env_python: Path, td_home: Path, plugin: str):
    """Run `testdrive -TT <plugin>` using the framework env.

    Returns (success: bool, output: str).
    """
    log.info("Running testdrive -TT %s", plugin)
    try:
        result = subprocess.run(
            [str(env_python), "-m", "testdrive", "-TT", plugin],
            check=False,
            capture_output=True,
            text=True,
            env=testdrive_env(td_home),
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


def list_install_details(env_python: Path, td_home: Path, plugin: str):
    """Return `testdrive -I <plugin>` install status, or None if the
    interpreter/command isn't available at all."""
    try:
        result = subprocess.run(
            [str(env_python), "-m", "testdrive", "-I", plugin, "--json"],
            check=False,
            capture_output=True,
            text=True,
            env=testdrive_env(td_home),
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

    return _format_install_status(data)


# --------------------------------------------------------------------------
# Main flows
# --------------------------------------------------------------------------
def bootstrap(td_home: Path, remote_url: str, python_path: str, plugins):
    log.info("testdrive home %s does not exist - bootstrapping", td_home)

    td_home.mkdir(parents=True, exist_ok=True)
    source_dir = download_and_unpack(remote_url, td_home / "src")
    env_python = create_framework_env(td_home, python_path)

    # Regular installs: testdrive's pyproject.toml now declares
    # examples/ as package-data (see the fix to that file), so a plain
    # `pip install .` ships it correctly - no need for an editable
    # install to make __file__ resolve back to the source tree. That's
    # deliberate: editable installs turned out to have their own
    # breakage (a PEP 660 redirect finder that, at least on Windows,
    # can leave the package with no resolvable __file__/location,
    # breaking testdrive/cli.py's `from . import __version__`).
    pip_install(env_python, ".", cwd=source_dir)
    pip_install(env_python, ".[dev]", cwd=source_dir)

    tt_results = {}
    for plugin in plugins:
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


def inspect_existing(td_home: Path, plugins):
    log.info("testdrive home %s exists - inspecting", td_home)
    env_python = framework_python(td_home)

    if not env_python.exists():
        log.warning("Expected framework python not found at %s", env_python)

    report = {}
    for plugin in plugins:
        details = list_install_details(env_python, td_home, plugin)
        report[plugin] = details or "(not installed / no status available)"

    return report


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
        report = inspect_existing(td_home, plugins)

    print_report(report)

    if any(str(v).startswith("FAILED") for v in report.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
