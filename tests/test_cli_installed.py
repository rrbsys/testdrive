"""Tests for `-I` / cmd_installed's pyenv-aware install-status logic.

Covers the bug this was fixed for (an in-process importlib check is
only correct when this process actually *is* the plugin's own pyenv —
see cli._plugin_install_status's docstring), plus the new
single-plugin and '*' CLI surface.
"""

from __future__ import annotations

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

import testdrive.cli as cli
import testdrive.pyenv as pyenv
from testdrive.pluginloader import PluginLoadError, load_plugin


# ---------------------------------------------------------------------------
# _plugin_install_status
# ---------------------------------------------------------------------------


def test_framework_pyenv_active_checks_in_process():
    """When this process genuinely is the plugin's own ("framework")
    pyenv, the fast in-process instance.is_installed() path is used —
    and, critically, a plugin-specific override of is_installed() gets
    to run (the out-of-process probe below can't call it at all).
    """
    import unittest.mock as mock

    loaded = load_plugin("yunet")  # a real "framework"-pyenv plugin
    cd = Path("/some/cache")

    called = {}

    class FakeInstance:
        def is_installed(self):
            called["ran"] = True
            return (True, [])

    with (
        mock.patch.object(pyenv, "env_python_path", return_value=Path("/fake/python")),
        mock.patch.object(Path, "exists", return_value=True),
        mock.patch.object(pyenv, "is_in_env", return_value=True),
        mock.patch.object(loaded, "instantiate", return_value=FakeInstance()),
    ):
        status = cli._plugin_install_status(loaded, cd)

    assert called.get("ran") is True
    assert status["installed"] is True
    assert status["pyenv"] == "framework"
    assert status["pyenv_active"] is True


def test_framework_pyenv_still_checks_in_process_when_cache_pyenv_dir_is_absent():
    """The exact scenario the CI core-only smoke test exercises
    (TESTDRIVE_SKIP_PYENV_CHECK=1, no cache/pyenv/framework directory
    at all — see ci.yml's core-only-smoke-test job): a "framework"
    plugin still means "runs in this same process" regardless of
    whether the dedicated cache/pyenv/framework directory happens to
    exist. Gating on that (an earlier version of this fix did) wrongly
    reported every framework-pyenv plugin as "not set up" in exactly
    this case.
    """
    import unittest.mock as mock

    loaded = load_plugin("yunet")
    cd = Path("/some/cache")

    called = {}

    class FakeInstance:
        def is_installed(self):
            called["ran"] = True
            return (False, ["some-package"])

    boom = AssertionError("must not probe out-of-process for a framework-pyenv plugin")

    with (
        mock.patch.object(Path, "exists", return_value=False),  # no cache/pyenv/framework at all
        mock.patch.object(pyenv, "check_modules_installed", side_effect=boom),
        mock.patch.object(loaded, "instantiate", return_value=FakeInstance()),
    ):
        status = cli._plugin_install_status(loaded, cd)

    assert called.get("ran") is True
    assert status["installed"] is False
    assert status["missing"] == ["some-package"]
    assert status["pyenv_exists"] is False


def test_dedicated_pyenv_not_set_up_reports_missing_without_importing_anything():
    """yolo11 declares its own 'newenv' pyenv. If that environment
    doesn't exist yet, there's nothing to import-check at all — report
    it as not installed with a clear next step, and don't touch
    is_installed()/check_modules_installed (nothing to probe).
    """
    import unittest.mock as mock

    loaded = load_plugin("yolo11")
    cd = Path("/some/cache")

    boom = AssertionError("must not probe a nonexistent environment")

    with (
        mock.patch.object(Path, "exists", return_value=False),
        mock.patch.object(pyenv, "check_modules_installed", side_effect=boom),
    ):
        status = cli._plugin_install_status(loaded, cd)

    assert status["installed"] is False
    assert status["pyenv"] == "newenv"
    assert status["pyenv_exists"] is False
    assert status["pyenv_active"] is False
    assert any("not set up" in m for m in status["missing"])
    assert any("-T yolo11" in m for m in status["missing"])


def test_dedicated_pyenv_set_up_probes_out_of_process_not_in_process():
    """yolo11's 'newenv' environment exists, but this (framework)
    process is never *that* environment — must probe the real target
    interpreter via check_modules_installed, not call
    instance.is_installed() in-process (which would check the wrong
    site-packages and is exactly the bug being fixed here).
    """
    import unittest.mock as mock

    loaded = load_plugin("yolo11")
    cd = Path("/some/cache")

    probe_calls = []

    def fake_probe(python_path, requirements):
        probe_calls.append((python_path, requirements))
        return (True, [])

    boom = AssertionError("must not check in-process for a dedicated, inactive pyenv")

    with (
        mock.patch.object(Path, "exists", return_value=True),
        mock.patch.object(pyenv, "is_in_env", return_value=False),
        mock.patch.object(pyenv, "check_modules_installed", side_effect=fake_probe),
        mock.patch.object(loaded, "instantiate", side_effect=boom),
    ):
        status = cli._plugin_install_status(loaded, cd)

    assert status["installed"] is True
    assert status["pyenv"] == "newenv"
    assert status["pyenv_exists"] is True
    assert status["pyenv_active"] is False
    assert len(probe_calls) == 1
    assert probe_calls[0][1] == loaded.manifest.requirements


# ---------------------------------------------------------------------------
# cmd_installed: -I / -I PLUGIN / -I "*"
# ---------------------------------------------------------------------------


def _run_installed(capsys, **kwargs):
    exit_code = cli.cmd_installed(as_json=True, **kwargs)
    out = capsys.readouterr().out
    return exit_code, json.loads(out) if out.strip() else None


def test_installed_bare_covers_every_plugin(capsys):
    import unittest.mock as mock

    from testdrive.pluginloader import iter_loadable_plugins

    all_ids = {p.manifest.id for p in iter_loadable_plugins()}

    with mock.patch.object(cli, "_plugin_install_status", return_value={"id": "x"}):
        exit_code, results = _run_installed(capsys)

    assert exit_code == cli.ExitCode.SUCCESS
    assert len(results) == len(all_ids)


def test_installed_star_same_as_bare(capsys):
    import unittest.mock as mock

    with mock.patch.object(cli, "_plugin_install_status", return_value={"id": "x"}):
        _, bare = _run_installed(capsys)
        _, star = _run_installed(capsys, plugin_id="*")

    assert bare == star


def test_installed_single_plugin_filters_to_just_that_one(capsys):
    exit_code, results = _run_installed(capsys, plugin_id="yolo11")

    assert exit_code == cli.ExitCode.SUCCESS
    assert len(results) == 1
    assert results[0]["id"] == "yolo11"
    assert results[0]["pyenv"] == "newenv"


def test_installed_unknown_plugin_is_an_error(capsys):
    exit_code, _ = _run_installed(capsys, plugin_id="does-not-exist-at-all")
    assert exit_code == cli.ExitCode.PLUGIN_NOT_FOUND


def test_installed_single_plugin_matches_direct_load_plugin():
    # Sanity: the id we filter by is the same one load_plugin() itself
    # would resolve, so '-I yolo11' and '-I YOLO11' aren't silently
    # different code paths from how every other -X PLUGIN flag works.
    try:
        load_plugin("yolo11")
    except PluginLoadError:
        raise AssertionError("expected 'yolo11' to be a real, loadable plugin id for this test")


# ---------------------------------------------------------------------------
# argparse: -I nargs behavior
# ---------------------------------------------------------------------------


def test_arg_parse_bare_dash_i_means_all():
    ns = cli.build_parser().parse_args(["-I"])
    assert ns.installed == "*"


def test_arg_parse_dash_i_with_plugin():
    ns = cli.build_parser().parse_args(["-I", "yolo11"])
    assert ns.installed == "yolo11"


def test_arg_parse_dash_i_with_star():
    ns = cli.build_parser().parse_args(["-I", "*"])
    assert ns.installed == "*"


def test_arg_parse_no_dash_i_is_none():
    ns = cli.build_parser().parse_args(["-L"])
    assert ns.installed is None
