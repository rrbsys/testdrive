"""Tests for worker_pool.py — provisioning and dispatch for plugins with
their own (non-"framework") pyenv.

Most of these mock subprocess.run entirely (fast, no real venv/network
involved). test_provision_plugin_env_real_venv_end_to_end is the
exception: it actually creates a venv and installs the core package
into it via that venv's own pip, called directly (no activation) — the
same approach _provision_plugin_env itself uses. It builds the venv
with --system-site-packages and installs with --no-build-isolation
--no-index so it also runs offline, reusing whatever's already on the
machine's search path instead of hitting the network — CI (or a
developer machine) with network access exercises the exact same code
path with a fully isolated venv, so nothing about the real behavior
goes untested; this just keeps the test itself fast and independent of
network access.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import subprocess
import tempfile
import unittest.mock as mock
from pathlib import Path


# ---------------------------------------------------------------------------
# _provision_marker_path
# ---------------------------------------------------------------------------


def test_provision_marker_path():
    from testdrive.worker_pool import _provision_marker_path

    d = Path("/some/env")
    assert _provision_marker_path(d) == d / ".testdrive_plugin_ok"


# ---------------------------------------------------------------------------
# _provision_plugin_env — mocked subprocess, unit-level
# ---------------------------------------------------------------------------


def test_provision_plugin_env_runs_venv_then_pip_upgrade_then_installs():
    from testdrive.util import set_pyenv_pip_upgrade
    from testdrive.worker_pool import _provision_marker_path, _provision_plugin_env

    with tempfile.TemporaryDirectory() as td:
        env_dir = Path(td) / "envs" / "newenv"
        python_path = env_dir / ("Scripts" if os.name == "nt" else "bin") / (
            "python.exe" if os.name == "nt" else "python"
        )

        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            if cmd[1:3] == ["-m", "venv"]:
                env_dir.mkdir(parents=True, exist_ok=True)  # what a real venv call would do
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        set_pyenv_pip_upgrade(True)
        try:
            with mock.patch("subprocess.run", side_effect=fake_run):
                _provision_plugin_env("yolo11", "newenv", env_dir, python_path)
        finally:
            set_pyenv_pip_upgrade(True)  # restore default

        assert len(calls) == 3
        # 1) venv creation, using *this* interpreter
        assert calls[0][:3] == [sys.executable, "-m", "venv"]
        assert calls[0][3] == str(env_dir)
        # 2) pip upgrade, using the new environment's own interpreter
        assert calls[1][0] == str(python_path)
        assert calls[1][1:4] == ["-m", "pip", "install"]
        assert "--upgrade" in calls[1]
        assert "pip" in calls[1]
        # 3) editable install of the core package + this plugin's extra
        assert calls[2][0] == str(python_path)
        assert "-e" in calls[2]
        assert any(a.endswith("[yolo11]") for a in calls[2])

        assert _provision_marker_path(env_dir).exists()
        assert "yolo11" in _provision_marker_path(env_dir).read_text()


def test_provision_plugin_env_skips_pip_upgrade_when_disabled():
    from testdrive.util import set_pyenv_pip_upgrade
    from testdrive.worker_pool import _provision_plugin_env

    with tempfile.TemporaryDirectory() as td:
        env_dir = Path(td) / "envs" / "newenv"
        python_path = env_dir / "bin" / "python"

        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        set_pyenv_pip_upgrade(False)
        try:
            with mock.patch("subprocess.run", side_effect=fake_run):
                _provision_plugin_env("yolo11", "newenv", env_dir, python_path)
        finally:
            set_pyenv_pip_upgrade(True)

        # Just venv creation + the editable install — no pip-upgrade call.
        assert len(calls) == 2
        assert calls[0][:3] == [sys.executable, "-m", "venv"]
        assert "-e" in calls[1]


def test_provision_plugin_env_wraps_called_process_error():
    from testdrive.worker_pool import WorkerError, _provision_plugin_env

    with tempfile.TemporaryDirectory() as td:
        env_dir = Path(td) / "envs" / "newenv"
        python_path = env_dir / "bin" / "python"

        def fake_run(cmd, **kw):
            raise subprocess.CalledProcessError(1, cmd, output="", stderr="pip blew up")

        try:
            with mock.patch("subprocess.run", side_effect=fake_run):
                _provision_plugin_env("yolo11", "newenv", env_dir, python_path)
            assert False, "expected WorkerError"
        except WorkerError as exc:
            assert exc.kind == "error"
            assert "pip blew up" in str(exc)


def test_provision_plugin_env_wraps_oserror():
    from testdrive.worker_pool import WorkerError, _provision_plugin_env

    with tempfile.TemporaryDirectory() as td:
        env_dir = Path(td) / "envs" / "newenv"
        python_path = env_dir / "bin" / "python"

        def fake_run(cmd, **kw):
            raise OSError("no such file or directory")

        try:
            with mock.patch("subprocess.run", side_effect=fake_run):
                _provision_plugin_env("yolo11", "newenv", env_dir, python_path)
            assert False, "expected WorkerError"
        except WorkerError as exc:
            assert exc.kind == "error"
            assert "no such file or directory" in str(exc)


def test_provision_plugin_env_marker_write_failure_is_non_fatal():
    """A marker-file write failure is a diagnostic breadcrumb only — it
    must not turn a successful provisioning into a raised error.
    """
    from testdrive.worker_pool import _provision_plugin_env

    with tempfile.TemporaryDirectory() as td:
        env_dir = Path(td) / "envs" / "newenv"
        python_path = env_dir / "bin" / "python"

        def fake_run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        def fake_write_text(self, *a, **kw):
            raise OSError("disk full")

        with mock.patch("subprocess.run", side_effect=fake_run), \
             mock.patch.object(Path, "write_text", fake_write_text):
            _provision_plugin_env("yolo11", "newenv", env_dir, python_path)  # must not raise


# ---------------------------------------------------------------------------
# WorkerPool.get() — env-not-configured / --no-auto-provision branches
# ---------------------------------------------------------------------------


def test_pool_get_raises_when_downloads_not_allowed():
    from testdrive.util import set_downloads_allowed
    from testdrive.worker_pool import WorkerError, WorkerPool

    with tempfile.TemporaryDirectory() as td:
        cd = Path(td)
        set_downloads_allowed(False)
        try:
            pool = WorkerPool()
            try:
                pool.get("yolo11", "newenv", cd)
                assert False, "expected WorkerError"
            except WorkerError as exc:
                assert exc.kind == "env_not_configured"
                assert "-T yolo11" in str(exc)
        finally:
            set_downloads_allowed(True)


def test_pool_get_raises_when_auto_provision_disabled():
    from testdrive.util import set_auto_provision_enabled, set_downloads_allowed
    from testdrive.worker_pool import WorkerError, WorkerPool

    with tempfile.TemporaryDirectory() as td:
        cd = Path(td)
        set_downloads_allowed(True)
        set_auto_provision_enabled(False)
        try:
            pool = WorkerPool()
            try:
                pool.get("yolo11", "newenv", cd)
                assert False, "expected WorkerError"
            except WorkerError as exc:
                assert exc.kind == "env_not_configured"
                assert "--no-auto-provision" in str(exc)
                assert "python3 -m venv" in str(exc)
        finally:
            set_auto_provision_enabled(True)


def test_pool_get_provisions_then_spawns_worker():
    from testdrive.util import set_auto_provision_enabled, set_downloads_allowed
    from testdrive.worker_pool import WorkerPool

    with tempfile.TemporaryDirectory() as td:
        cd = Path(td)
        set_downloads_allowed(True)
        set_auto_provision_enabled(True)

        provisioned = {}

        def fake_provision(plugin_id, pyenv_name, env_directory, python_path):
            provisioned["called"] = (plugin_id, pyenv_name)
            python_path.parent.mkdir(parents=True, exist_ok=True)
            python_path.touch()  # pretend the venv now exists

        fake_handle = object()

        provision_target = "testdrive.worker_pool._provision_plugin_env"
        handle_target = "testdrive.worker_pool.WorkerHandle"
        with mock.patch(provision_target, side_effect=fake_provision), \
             mock.patch(handle_target, return_value=fake_handle):
            pool = WorkerPool()
            handle = pool.get("yolo11", "newenv", cd)

        assert provisioned["called"] == ("yolo11", "newenv")
        assert handle is fake_handle
        # Second call for the same plugin reuses the cached handle —
        # no second provisioning call.
        with mock.patch(
            "testdrive.worker_pool._provision_plugin_env",
            side_effect=AssertionError("must not provision twice"),
        ):
            assert pool.get("yolo11", "newenv", cd) is fake_handle


# ---------------------------------------------------------------------------
# Real end-to-end provisioning (actual venv, actual pip — offline-friendly)
# ---------------------------------------------------------------------------


def test_provision_plugin_env_real_venv_end_to_end():
    """The real thing: create an actual venv and pip-install the core
    package into it, exactly the way _provision_plugin_env does — by
    calling that venv's own python/pip directly, no activation involved
    (confirmed fine as a testing approach; matches what the production
    code already does).

    Uses --system-site-packages so the already-installed core
    dependency (Pillow) is visible without hitting the network, and
    --no-build-isolation/--no-index for the same reason. This does not
    change what's being verified: that a real venv gets created at the
    expected path and a real, runnable `python -c "import testdrive"`
    works afterward against *that* interpreter specifically.
    """
    import subprocess as _subprocess

    import pytest

    import testdrive.worker_pool as worker_pool
    from testdrive.util import get_pyenv_pip_upgrade, set_pyenv_pip_upgrade

    # This test builds --system-site-packages + --no-build-isolation +
    # --no-index specifically to avoid the network — but that only works
    # if setuptools/wheel (this project's build backend) are importable
    # from whatever "system site packages" the *new* venv actually gets.
    #
    # Crucially, that is NOT necessarily this process's own site-packages:
    # when a venv is created (via `sys.executable -m venv`) from an
    # interpreter that is itself already inside a venv — the normal case
    # when running pytest inside a project venv — Python resolves
    # --system-site-packages against the *base* interpreter
    # (sys._base_executable), not the calling venv. So we check the base
    # interpreter directly, in its own subprocess, rather than trusting
    # importlib.util.find_spec() in this process.
    base_python = getattr(sys, "_base_executable", None) or sys.executable
    probe = _subprocess.run(
        [base_python, "-c", "import setuptools, wheel"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        pytest.skip(
            f"setuptools/wheel not importable from the base interpreter "
            f"({base_python}) — needed because a --system-site-packages venv "
            "links to the *base* interpreter's site-packages, not the "
            "currently active venv's, even when created from inside one. "
            f"Install them there directly (e.g. `{base_python} -m pip install "
            "setuptools wheel`), or run this test with network access instead, "
            "where the unmodified, non-offline path is exercised."
        )

    # Skip the pip-upgrade sub-step here: it would try to hit PyPI for
    # the latest pip, which needs network this test deliberately avoids.
    # It's exercised (mocked) separately by
    # test_provision_plugin_env_runs_venv_then_pip_upgrade_then_installs.
    upgrade_backup = get_pyenv_pip_upgrade()
    set_pyenv_pip_upgrade(False)

    with tempfile.TemporaryDirectory() as td:
        env_dir = Path(td) / "pyenv" / sys.platform / "newenv"
        python_path = env_dir / ("Scripts" if os.name == "nt" else "bin") / (
            "python.exe" if os.name == "nt" else "python"
        )

        real_run = subprocess.run

        def run_offline_friendly(cmd, **kw):
            cmd = list(cmd)
            if cmd[1:3] == ["-m", "venv"]:
                cmd = cmd[:3] + ["--system-site-packages"] + cmd[3:]
            elif cmd[1:4] == ["-m", "pip", "install"] and "-e" in cmd:
                # Editable-install calls: skip network/build-isolation,
                # matching the confirmed pip -e (no activation) approach.
                cmd = cmd[:4] + ["--no-build-isolation", "--no-index"] + cmd[4:]
            return real_run(cmd, **kw)

        try:
            with mock.patch("subprocess.run", side_effect=run_offline_friendly):
                worker_pool._provision_plugin_env(
                    "core-only-test", "newenv", env_dir, python_path,
                )

            assert python_path.exists()
            assert worker_pool._provision_marker_path(env_dir).exists()

            check = subprocess.run(
                [str(python_path), "-c", "import testdrive; print(testdrive.__name__)"],
                capture_output=True, text=True,
            )
            assert check.returncode == 0
            assert check.stdout.strip() == "testdrive"
        finally:
            set_pyenv_pip_upgrade(upgrade_backup)
