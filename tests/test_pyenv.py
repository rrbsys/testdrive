"""Tests for pyenv.py — the framework's own dedicated-environment guard.

Nothing here spins up real subprocesses or virtual environments (that's
covered, end-to-end, by test_worker_pool.py's provisioning test); these
are pure unit tests of the path/relaunch logic, with os.name/sys.platform
and the process-replacing bits (os.execve/sys.exit) mocked out.
"""

from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest.mock as mock
from pathlib import Path


# ---------------------------------------------------------------------------
# env_dir / recommended_python
# ---------------------------------------------------------------------------


def test_env_dir_default_name_is_framework():
    from testdrive.pyenv import FRAMEWORK_ENV_NAME, env_dir

    cd = Path("/some/cache")
    assert env_dir(cd).name == FRAMEWORK_ENV_NAME


def test_env_dir_keyed_by_platform():
    import testdrive.pyenv as pyenv

    cd = Path("/some/cache")
    with mock.patch.object(pyenv.sys, "platform", "linux"):
        linux_dir = pyenv.env_dir(cd)
    with mock.patch.object(pyenv.sys, "platform", "darwin"):
        darwin_dir = pyenv.env_dir(cd)

    assert linux_dir == cd / "pyenv" / "linux" / "framework"
    assert darwin_dir == cd / "pyenv" / "darwin" / "framework"
    assert linux_dir != darwin_dir


def test_env_dir_custom_name():
    from testdrive.pyenv import env_dir

    cd = Path("/some/cache")
    assert env_dir(cd, "newenv").name == "newenv"


def test_recommended_python_is_3_11_on_macos():
    import testdrive.pyenv as pyenv

    with mock.patch.object(pyenv.sys, "platform", "darwin"):
        assert pyenv.recommended_python() == "python3.11"


def test_recommended_python_is_3_12_elsewhere():
    import testdrive.pyenv as pyenv

    with mock.patch.object(pyenv.sys, "platform", "linux"):
        assert pyenv.recommended_python() == "python3.12"
    with mock.patch.object(pyenv.sys, "platform", "win32"):
        assert pyenv.recommended_python() == "python3.12"


# ---------------------------------------------------------------------------
# env_python_path / env_pip_path
# ---------------------------------------------------------------------------


def test_env_python_path_posix():
    import testdrive.pyenv as pyenv

    cd = Path("/some/cache")
    with mock.patch.object(pyenv.os, "name", "posix"):
        p = pyenv.env_python_path(cd)
    assert p == cd / "pyenv" / sys.platform / "framework" / "bin" / "python"


def test_env_python_path_windows():
    import testdrive.pyenv as pyenv

    cd = Path("/some/cache")
    with mock.patch.object(pyenv.os, "name", "nt"):
        p = pyenv.env_python_path(cd)
    assert p == cd / "pyenv" / sys.platform / "framework" / "Scripts" / "python.exe"


def test_env_pip_path_posix():
    import testdrive.pyenv as pyenv

    cd = Path("/some/cache")
    with mock.patch.object(pyenv.os, "name", "posix"):
        p = pyenv.env_pip_path(cd)
    assert p == cd / "pyenv" / sys.platform / "framework" / "bin" / "pip"


def test_env_pip_path_windows():
    import testdrive.pyenv as pyenv

    cd = Path("/some/cache")
    with mock.patch.object(pyenv.os, "name", "nt"):
        p = pyenv.env_pip_path(cd)
    assert p == cd / "pyenv" / sys.platform / "framework" / "Scripts" / "pip.exe"


# ---------------------------------------------------------------------------
# install_hint
# ---------------------------------------------------------------------------


def test_install_hint_uses_environments_own_pip():
    import testdrive.pyenv as pyenv

    cd = Path("/some/cache")
    with mock.patch.object(pyenv.os, "name", "posix"):
        hint = pyenv.install_hint(cd, "newenv", ["torch", "ultralytics"])
        expected_pip = pyenv.env_pip_path(cd, "newenv")

    assert str(expected_pip) in hint
    assert '"torch"' in hint
    assert '"ultralytics"' in hint
    assert hint.startswith(f'"{expected_pip}" install')


def test_install_hint_empty_packages():
    from testdrive.pyenv import install_hint

    cd = Path("/some/cache")
    hint = install_hint(cd, "framework", [])
    assert hint.strip().endswith("install")


# ---------------------------------------------------------------------------
# is_in_env
# ---------------------------------------------------------------------------


def test_is_in_env_true_when_prefix_matches():
    import tempfile

    import testdrive.pyenv as pyenv

    with tempfile.TemporaryDirectory() as td:
        cd = Path(td)
        expected = pyenv.env_dir(cd)
        expected.mkdir(parents=True)
        with mock.patch.object(pyenv.sys, "prefix", str(expected)):
            assert pyenv.is_in_env(cd) is True


def test_is_in_env_false_when_prefix_differs():
    import tempfile

    import testdrive.pyenv as pyenv

    with tempfile.TemporaryDirectory() as td:
        cd = Path(td)
        with mock.patch.object(pyenv.sys, "prefix", "/somewhere/else"):
            assert pyenv.is_in_env(cd) is False


def test_is_in_env_false_on_resolve_oserror():
    import testdrive.pyenv as pyenv

    cd = Path("/some/cache")

    def _raise(*a, **kw):
        raise OSError("boom")

    with mock.patch.object(Path, "resolve", _raise):
        assert pyenv.is_in_env(cd) is False


# ---------------------------------------------------------------------------
# ensure_framework_env
# ---------------------------------------------------------------------------


def _run_ensure(cd: Path):
    """Call ensure_framework_env(cd), capturing stderr and any SystemExit."""
    import testdrive.pyenv as pyenv

    buf = io.StringIO()
    exit_code = None
    with mock.patch.object(sys, "stderr", buf):
        try:
            pyenv.ensure_framework_env(cd)
        except SystemExit as exc:
            exit_code = exc.code
    return exit_code, buf.getvalue()


def test_ensure_framework_env_skips_when_env_var_set():
    import testdrive.pyenv as pyenv

    cd = Path("/some/cache")
    backup = os.environ.get(pyenv._SKIP_ENV_VAR)
    os.environ[pyenv._SKIP_ENV_VAR] = "1"
    try:
        boom = AssertionError("should not be called")
        with mock.patch.object(pyenv, "is_in_env", side_effect=boom):
            # Should return immediately without even checking is_in_env.
            pyenv.ensure_framework_env(cd)
    finally:
        if backup is None:
            del os.environ[pyenv._SKIP_ENV_VAR]
        else:
            os.environ[pyenv._SKIP_ENV_VAR] = backup


def test_ensure_framework_env_noop_when_already_in_env():
    import testdrive.pyenv as pyenv

    cd = Path("/some/cache")
    os.environ.pop(pyenv._SKIP_ENV_VAR, None)
    with mock.patch.object(pyenv, "is_in_env", return_value=True):
        # No exception, no exit — just returns.
        pyenv.ensure_framework_env(cd)


def test_ensure_framework_env_relaunches_when_target_exists():
    import testdrive.pyenv as pyenv

    cd = Path("/some/cache")
    os.environ.pop(pyenv._SKIP_ENV_VAR, None)
    os.environ.pop(pyenv._RELAUNCH_MARKER, None)

    captured = {}

    def fake_execve(path, argv, env):
        captured["path"] = path
        captured["argv"] = argv
        captured["env"] = env
        # Real os.execve never returns on success; our fake just does,
        # matching how ensure_framework_env's own `return` afterward
        # would be reached if it somehow did.

    with mock.patch.object(sys, "argv", ["testdrive"]), \
         mock.patch.object(pyenv, "is_in_env", return_value=False), \
         mock.patch.object(pyenv.Path, "exists", return_value=True), \
         mock.patch.object(pyenv.os, "execve", side_effect=fake_execve):
        exit_code, stderr = _run_ensure(cd)

    assert exit_code is None
    assert captured["env"][pyenv._RELAUNCH_MARKER] == "1"
    assert str(pyenv.env_python_path(cd)) == captured["path"]
    assert captured["argv"][0] == captured["path"]
    assert captured["argv"][1:] == ["-m", "testdrive"]
    assert "relaunching" in stderr


def test_ensure_framework_env_missing_env_prints_setup_instructions():
    import testdrive.pyenv as pyenv

    cd = Path("/some/cache")
    os.environ.pop(pyenv._SKIP_ENV_VAR, None)
    os.environ.pop(pyenv._RELAUNCH_MARKER, None)

    with mock.patch.object(pyenv, "is_in_env", return_value=False), \
         mock.patch.object(pyenv.Path, "exists", return_value=False):
        exit_code, stderr = _run_ensure(cd)

    from testdrive.util import ExitCode

    assert exit_code == ExitCode.PYENV_NOT_CONFIGURED
    assert "pip install -e ." in stderr
    assert str(pyenv.env_python_path(cd)) in stderr
    assert str(pyenv.env_dir(cd)) in stderr


def test_ensure_framework_env_execve_oserror_falls_back_to_instructions():
    import testdrive.pyenv as pyenv

    cd = Path("/some/cache")
    os.environ.pop(pyenv._SKIP_ENV_VAR, None)
    os.environ.pop(pyenv._RELAUNCH_MARKER, None)

    def fake_execve(*a, **kw):
        raise OSError("no such file")

    with mock.patch.object(pyenv, "is_in_env", return_value=False), \
         mock.patch.object(pyenv.Path, "exists", return_value=True), \
         mock.patch.object(pyenv.os, "execve", side_effect=fake_execve):
        exit_code, stderr = _run_ensure(cd)

    from testdrive.util import ExitCode

    assert exit_code == ExitCode.PYENV_NOT_CONFIGURED
    assert "could not relaunch" in stderr
    assert "pip install -e ." in stderr


def test_ensure_framework_env_bounded_retry_when_relaunch_marker_already_set():
    """If we already relaunched once (marker set) and still aren't in the
    expected env, stop immediately with a diagnosable error instead of
    relaunching forever (see ensure_framework_env's docstring).
    """
    import testdrive.pyenv as pyenv

    cd = Path("/some/cache")
    os.environ.pop(pyenv._SKIP_ENV_VAR, None)
    os.environ[pyenv._RELAUNCH_MARKER] = "1"
    try:
        boom = AssertionError("must not relaunch again")
        with mock.patch.object(pyenv, "is_in_env", return_value=False), \
             mock.patch.object(pyenv.os, "execve", side_effect=boom):
            exit_code, stderr = _run_ensure(cd)
    finally:
        del os.environ[pyenv._RELAUNCH_MARKER]

    from testdrive.util import ExitCode

    assert exit_code == ExitCode.PYENV_NOT_CONFIGURED
    assert "still not running" in stderr
    assert "pyvenv.cfg" in stderr


def test_ensure_framework_env_relaunch_argv_forwards_cli_args():
    import testdrive.pyenv as pyenv

    cd = Path("/some/cache")
    os.environ.pop(pyenv._SKIP_ENV_VAR, None)
    os.environ.pop(pyenv._RELAUNCH_MARKER, None)

    captured = {}

    def fake_execve(path, argv, env):
        captured["argv"] = argv

    argv_backup = sys.argv
    sys.argv = ["testdrive", "groundingdino", "photo.jpg", "cat"]
    try:
        with mock.patch.object(pyenv, "is_in_env", return_value=False), \
             mock.patch.object(pyenv.Path, "exists", return_value=True), \
             mock.patch.object(pyenv.os, "execve", side_effect=fake_execve):
            _run_ensure(cd)
    finally:
        sys.argv = argv_backup

    assert captured["argv"][1:3] == ["-m", "testdrive"]
    assert captured["argv"][3:] == ["groundingdino", "photo.jpg", "cat"]
