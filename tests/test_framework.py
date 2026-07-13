"""Tests for Commits 1-3."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

from testdrive.detection import (
    FRAMEWORK_API_VERSION,
    Detection,
    DetectionResult,
    PluginManifest,
)
from testdrive.imageio import derive_output_paths
from testdrive.pluginloader import (
    PluginLoadError,
    iter_loadable_plugins,
    list_plugins,
    load_plugin,
)


# ---------------------------------------------------------------------------
# detection.py
# ---------------------------------------------------------------------------


def test_framework_api_version_is_one():
    assert FRAMEWORK_API_VERSION == 1


def test_detection_to_dict():
    d = Detection(label="cat", score=0.9, bbox=(1, 2, 3, 4))
    assert d.to_dict()["bbox"] == (1, 2, 3, 4)


def test_manifest_requires_id():
    try:
        PluginManifest.from_dict({"name": "X"})
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_manifest_from_dict_ok():
    m = PluginManifest.from_dict({"id": "x", "name": "X", "version": "0.1.0", "api": 1})
    assert m.id == "x"
    assert m.requirements == []


def test_manifest_requirements_pip_module_format():
    m = PluginManifest.from_dict(
        {
            "id": "x",
            "requirements": [
                {"pip": "Pillow", "module": "PIL"},
                {"pip": "torch", "module": "torch"},
            ],
        }
    )
    assert m.requirements[0]["pip"] == "Pillow"
    assert m.requirements[0]["module"] == "PIL"


def test_detection_result_count_and_summary():
    r = DetectionResult(
        image_path=Path("photo.jpg"),
        image_size=(800, 600),
        plugin_id="testplugin",
        plugin_name="Test Plugin",
        plugin_version="0.1.0",
        prompt="person",
        threshold=0.3,
        detections=[
            Detection(label="person", score=0.91, bbox=(10, 10, 50, 90)),
            Detection(label="person", score=0.78, bbox=(60, 10, 100, 90)),
        ],
        elapsed_ms=123.4,
    )
    assert r.count == 2
    summary = r.summary()
    assert "Matches  : 2" in summary
    assert "person" in summary
    assert "123 ms" in summary


def test_detection_result_to_dict_serialisable():
    import json

    r = DetectionResult(
        image_path=Path("x.jpg"),
        image_size=(100, 100),
        plugin_id="p",
        plugin_name="P",
        plugin_version="0",
        prompt="cat",
        threshold=0.3,
        matches_path=Path("x-matches.png"),
        redacted_path=Path("x-redacted.png"),
    )
    s = json.dumps(r.to_dict())
    assert '"x-matches.png"' in s


# ---------------------------------------------------------------------------
# imageio.py
# ---------------------------------------------------------------------------


def test_derive_output_paths_simple():
    m, r = derive_output_paths(Path("photo.jpg"))
    assert m == Path("photo-matches.png")
    assert r == Path("photo-redacted.png")


def test_derive_output_paths_preserves_dir():
    m, r = derive_output_paths(Path("/data/img/dog.JPEG"))
    assert m == Path("/data/img/dog-matches.png")
    assert r == Path("/data/img/dog-redacted.png")


# ---------------------------------------------------------------------------
# annotate.py  (requires Pillow)
# ---------------------------------------------------------------------------


def _pil_available() -> bool:
    try:
        import PIL  # noqa: F401

        return True
    except ImportError:
        return False


def test_draw_boxes_returns_same_size():
    if not _pil_available():
        print("  SKIP (Pillow not available)")
        return
    from PIL import Image
    from testdrive.annotate import draw_boxes

    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    detections = [Detection(label="cat", score=0.9, bbox=(10, 10, 80, 80))]
    result = draw_boxes(img, detections)
    assert result.size == img.size
    assert result is not img  # must be a copy


def test_draw_boxes_no_detections():
    if not _pil_available():
        print("  SKIP (Pillow not available)")
        return
    from PIL import Image
    from testdrive.annotate import draw_boxes

    img = Image.new("RGB", (100, 100), color=(0, 0, 0))
    result = draw_boxes(img, [])
    assert result.size == img.size


def test_redact_fills_with_black():
    if not _pil_available():
        print("  SKIP (Pillow not available)")
        return
    from PIL import Image
    from testdrive.annotate import redact

    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    bbox = (10, 10, 90, 90)
    detections = [Detection(label="face", score=0.99, bbox=bbox)]
    result = redact(img, detections)

    # Centre of the bbox should be black
    cx = (bbox[0] + bbox[2]) // 2
    cy = (bbox[1] + bbox[3]) // 2
    assert result.getpixel((cx, cy)) == (0, 0, 0), "expected black fill in redacted area"

    # Outside the bbox the original white should remain
    assert result.getpixel((0, 0)) == (255, 255, 255)


# ---------------------------------------------------------------------------
# pluginloader.py
# ---------------------------------------------------------------------------


def test_list_plugins_backward_compatible():
    # Originally hardcoded to a fixed list (["groundingdino", "owlv2"],
    # then later every plugin that existed at the time) — inherently
    # fragile, since it breaks every time a plugin is added, or
    # parked/unparked (e.g. into/out of models_inactive/). What
    # list_plugins() actually promises "backward compatible" behavior
    # for is its *contract*: a sorted, duplicate-free list of plugin id
    # strings matching whatever's currently discoverable — not which
    # specific plugins that happens to be on any given machine.
    result = list_plugins()
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)
    assert result == sorted(result)
    assert len(result) == len(set(result))
    assert result == sorted(p.manifest.id for p in iter_loadable_plugins())


def test_discovers_groundingdino_and_owlv2():
    ids = {p.manifest.id for p in iter_loadable_plugins()}
    assert "groundingdino" in ids
    assert "owlv2" in ids


def test_load_plugin_by_id():
    loaded = load_plugin("groundingdino")
    assert loaded.manifest.name == "Grounding DINO"
    assert loaded.manifest.license == "Apache-2.0"


def test_load_unknown_plugin_raises():
    try:
        load_plugin("does-not-exist")
        assert False, "expected PluginLoadError"
    except PluginLoadError:
        pass


def test_inactive_plugins_absent_from_discovery():
    # Parked plugins live in the sibling models_inactive/ package and
    # must never surface through the normal discovery path — not in
    # iter_loadable_plugins(), and therefore not in list_plugins() or
    # any '*' loop built from it (see cli.py).
    ids = {p.manifest.id for p in iter_loadable_plugins()}
    assert "molmo" not in ids
    assert "molmo7b" not in ids
    assert "seem" not in ids
    assert "molmo" not in list_plugins()


def test_load_plugin_reaches_inactive_plugin_by_explicit_path():
    loaded = load_plugin("../models_inactive/molmo")
    assert loaded.manifest.id == "molmo"
    assert loaded.manifest.name == "MolmoE"


def test_load_plugin_inactive_path_accepts_local_os_separator():
    import os

    # Built with os.path.join rather than hardcoded, so this exercises
    # whatever separator the OS actually running this test uses —
    # that's the "os-dependent" part.
    ref = os.path.join(os.pardir, "models_inactive", "seem")
    loaded = load_plugin(ref)
    assert loaded.manifest.id == "seem"


def test_load_plugin_inactive_path_unknown_module_raises():
    try:
        load_plugin("../models_inactive/does-not-exist")
        assert False, "expected PluginLoadError"
    except PluginLoadError:
        pass


def test_load_plugin_plain_id_does_not_match_inactive_shape():
    # A bare id containing "models_inactive" as text, but not shaped
    # like the special two-level-up reference, is just an (unknown)
    # plugin id — never accidentally treated as an inactive-plugin path.
    try:
        load_plugin("models_inactive/molmo")
        assert False, "expected PluginLoadError"
    except PluginLoadError:
        pass


def test_display_name_resolves_inactive_ref_to_bare_basename():
    from testdrive.pluginloader import display_name

    assert display_name("../models_inactive/samgd") == "samgd"


def test_display_name_passes_through_ordinary_plugin_id():
    from testdrive.pluginloader import display_name

    assert display_name("owlv2") == "owlv2"


def test_cli_find_example_image_resolves_inactive_plugin_path():
    # Regression test: examples/<name> lookup must use the resolved
    # basename, not the raw "../models_inactive/samgd" string embedded
    # verbatim into a path (which would resolve outside examples/
    # entirely instead of finding examples/samgd/...).
    import testdrive.cli as cli

    found = cli._find_example_image("../models_inactive/samgd")
    assert found is not None
    image_path, prompt, expected = found
    assert image_path.name == "image1-prompt-green_triangle-1matches.png"
    assert prompt == "green triangle"
    assert expected == 1


def test_seem_manifest_uses_own_isolated_pyenv():
    # The whole point of parking seem.py with a non-"framework" pyenv:
    # its exotic dependency chain must never be able to affect any
    # other plugin's (or the framework's own) environment.
    loaded = load_plugin("../models_inactive/seem")
    m = loaded.manifest
    assert m.id == "seem"
    assert m.pyenv == "seem"
    assert m.pyenv != "framework"


def test_seem_manifest_declares_focalt_and_focall_variants():
    loaded = load_plugin("../models_inactive/seem")
    m = loaded.manifest
    assert set(m.models) == {"focalt", "focall"}
    assert m.model == "focalt"


def test_seem_not_installed_without_its_exotic_deps():
    # In any environment that doesn't have the "seem" extra's exotic
    # deps installed (the overwhelming common case, including CI) —
    # is_installed() must report it missing rather than crash, via the
    # same default requirements-probing behavior every other plugin
    # relies on (no override needed here — see seem.py itself).
    loaded = load_plugin("../models_inactive/seem")
    plugin = loaded.instantiate()
    installed, missing = plugin.is_installed()
    assert installed is False
    assert len(missing) > 0


def test_seem_absent_from_discovery_like_other_parked_plugins():
    ids = {p.manifest.id for p in iter_loadable_plugins()}
    assert "seem" not in ids


# ---------------------------------------------------------------------------
# owlv2 model file
# ---------------------------------------------------------------------------


def test_owlv2_parse_prompt_single():
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from testdrive.models.owlv2 import _parse_prompt

    assert _parse_prompt("person") == ["person"]


def test_owlv2_parse_prompt_multi():
    from testdrive.models.owlv2 import _parse_prompt

    assert _parse_prompt("cat, dog, bird") == ["cat", "dog", "bird"]


def test_owlv2_parse_prompt_strips_whitespace():
    from testdrive.models.owlv2 import _parse_prompt

    assert _parse_prompt("  cat ,  dog  ") == ["cat", "dog"]


def test_owlv2_parse_prompt_empty_parts_ignored():
    from testdrive.models.owlv2 import _parse_prompt

    assert _parse_prompt("cat,,dog") == ["cat", "dog"]


def test_owlv2_manifest_complete():
    loaded = load_plugin("owlv2")
    m = loaded.manifest
    assert m.hf_repo == "google/owlv2-base-patch16-ensemble"
    assert m.license == "Apache-2.0"
    assert any(r["module"] == "PIL" for r in m.requirements)
    assert any(r["module"] == "torch" for r in m.requirements)
    assert any(r["module"] == "transformers" for r in m.requirements)


# ---------------------------------------------------------------------------
# selftest.py  (mock plugin - no real model weights needed)
# ---------------------------------------------------------------------------


def _register_mock_plugin(
    plugin_id: str = "mockplugin",
    *,
    init_raises: Exception | None = None,
    detect_raises: Exception | None = None,
    bad_detections: bool = False,
    bad_score: bool = False,
):
    """Return a run_selftest() result for a fully in-memory mock plugin."""
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from testdrive.detection import Detection, PluginManifest, FRAMEWORK_API_VERSION
    from testdrive.plugin import DetectorPlugin

    class MockPlugin(DetectorPlugin):
        def initialize(self):
            if init_raises:
                raise init_raises

        def detect(self, image, prompt, threshold=0.3):
            if detect_raises:
                raise detect_raises
            if bad_detections:
                return "not-a-list"
            if bad_score:
                return [Detection(label="x", score=2.5, bbox=(0, 0, 10, 10))]
            return [Detection(label="block", score=0.85, bbox=(10, 20, 80, 90))]

    manifest = PluginManifest(
        id=plugin_id,
        name="Mock Plugin",
        version="0.0.1",
        api=FRAMEWORK_API_VERSION,
        requirements=[],  # no deps → is_installed() always returns True
        sample_prompt="block",
    )

    # Patch pluginloader.load_plugin just for this call
    from testdrive.pluginloader import LoadedPlugin
    import testdrive.selftest as st

    # Monkey-patch the load_plugin inside the selftest module's namespace
    import testdrive.pluginloader as pl

    _orig = pl.load_plugin

    def _fake_load(pid):
        if pid == plugin_id:
            loaded = LoadedPlugin(
                module_name=plugin_id,
                manifest=manifest,
                plugin_class=MockPlugin,
            )
            return loaded
        return _orig(pid)

    pl.load_plugin = _fake_load
    try:
        result = st.run_selftest(plugin_id)
    finally:
        pl.load_plugin = _orig

    return result


def test_selftest_mock_plugin_passes():
    result = _register_mock_plugin()
    assert result.passed, f"expected pass, failures: {result.failures}"
    assert result.detection_count == 1


def test_selftest_mock_plugin_init_failure():
    result = _register_mock_plugin(init_raises=RuntimeError("GPU not found"))
    assert not result.passed
    assert any("initialize" in s for s in result.steps)
    assert any("GPU not found" in f for f in result.failures)


def test_selftest_mock_plugin_detect_raises():
    result = _register_mock_plugin(detect_raises=ValueError("bad input"))
    assert not result.passed
    assert any("detect" in s for s in result.steps)


def test_selftest_mock_plugin_bad_return_type():
    result = _register_mock_plugin(bad_detections=True)
    assert not result.passed
    assert any("list" in f for f in result.failures)


def test_selftest_mock_plugin_bad_score():
    result = _register_mock_plugin(bad_score=True)
    assert not result.passed
    assert any("score" in f for f in result.failures)


def test_selftest_unknown_plugin():
    from testdrive.selftest import run_selftest

    result = run_selftest("does-not-exist")
    assert not result.passed
    assert result.failures


def test_selftest_summary_contains_plugin_id():
    result = _register_mock_plugin()
    assert "mockplugin" in result.summary()


def test_selftest_summary_shows_pass():
    result = _register_mock_plugin()
    assert "PASS" in result.summary()


def test_selftest_summary_shows_fail_on_failure():
    result = _register_mock_plugin(init_raises=RuntimeError("boom"))
    assert "FAIL" in result.summary()


# ---------------------------------------------------------------------------
# CLI -T integration
# ---------------------------------------------------------------------------


def test_cli_selftest_unknown_plugin_exits_nonzero():
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from testdrive.cli import main

    code = main(["-T", "does-not-exist"])
    assert code != 0


def test_cli_selftest_json_output():
    import json as _json
    from io import StringIO
    from unittest.mock import patch
    from testdrive.cli import main

    # Use the same mock-plugin trick, patching pluginloader
    from testdrive.detection import Detection, PluginManifest, FRAMEWORK_API_VERSION
    from testdrive.plugin import DetectorPlugin
    from testdrive.pluginloader import LoadedPlugin
    import testdrive.pluginloader as pl

    class QuickPlugin(DetectorPlugin):
        def initialize(self):
            pass

        def detect(self, image, prompt, threshold=0.3):
            return [Detection(label="x", score=0.9, bbox=(0, 0, 5, 5))]

    manifest = PluginManifest(
        id="quickplugin",
        name="Quick",
        version="0",
        api=FRAMEWORK_API_VERSION,
        requirements=[],
        sample_prompt="x",
    )
    _orig = pl.load_plugin

    def _fake(pid):
        if pid == "quickplugin":
            return LoadedPlugin("quickplugin", manifest, QuickPlugin)
        return _orig(pid)

    pl.load_plugin = _fake

    buf = StringIO()
    try:
        with patch("sys.stdout", buf):
            code = main(["-T", "quickplugin", "--json"])
    finally:
        pl.load_plugin = _orig

    assert code == 0
    data = _json.loads(buf.getvalue())
    assert data["passed"] is True
    assert data["detection_count"] == 1


# ---------------------------------------------------------------------------
# cache.py
# ---------------------------------------------------------------------------


def test_cache_default_path_is_cache_subdir():
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    env_backup = os.environ.pop("TESTDRIVE_CACHE", None)
    try:
        from testdrive.cache import cache_dir, _PROJECT_ROOT

        path = cache_dir()
        assert path == (_PROJECT_ROOT / "cache").resolve()
        assert path.exists()
    finally:
        if env_backup is not None:
            os.environ["TESTDRIVE_CACHE"] = env_backup


def test_cache_env_var_overrides_default():
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        os.environ["TESTDRIVE_CACHE"] = td
        try:
            # Reload to pick up new env var value
            from testdrive.cache import cache_dir
            from pathlib import Path

            assert cache_dir() == Path(td).resolve()
        finally:
            del os.environ["TESTDRIVE_CACHE"]


def test_cache_dir_is_created_if_missing():
    import os
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        target = str(Path(td) / "deep" / "nested" / "cache")
        os.environ["TESTDRIVE_CACHE"] = target
        try:
            from testdrive.cache import cache_dir

            p = cache_dir()
            assert p.exists()
        finally:
            del os.environ["TESTDRIVE_CACHE"]


def test_cache_info_shows_source():
    import os

    os.environ.pop("TESTDRIVE_CACHE", None)
    from testdrive.cache import cache_info

    info = cache_info()
    assert info["source"] == "default"
    assert "cache" in info["path"]
    assert info["env_var"] == "TESTDRIVE_CACHE"


def test_cache_info_shows_env_var_source():
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        os.environ["TESTDRIVE_CACHE"] = td
        try:
            from testdrive.cache import cache_info

            info = cache_info()
            assert info["source"] == "$TESTDRIVE_CACHE"
        finally:
            del os.environ["TESTDRIVE_CACHE"]


# ---------------------------------------------------------------------------
# util.py: mask_to_bbox / ensure_git_repo
# ---------------------------------------------------------------------------


def test_mask_to_bbox_tight_box_around_nonzero_pixels():
    import numpy as np

    from testdrive.util import mask_to_bbox

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True  # rows 2..4, cols 3..6
    assert mask_to_bbox(mask) == (3, 2, 6, 4)


def test_mask_to_bbox_empty_mask_returns_none():
    import numpy as np

    from testdrive.util import mask_to_bbox

    assert mask_to_bbox(np.zeros((10, 10), dtype=bool)) is None


def test_mask_to_bbox_single_pixel():
    import numpy as np

    from testdrive.util import mask_to_bbox

    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 3] = True
    assert mask_to_bbox(mask) == (3, 2, 3, 2)


def test_ensure_git_repo_clones_and_checks_out_ref():
    import subprocess
    import tempfile
    import unittest.mock as mock
    from pathlib import Path

    from testdrive.util import ensure_git_repo

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[:2] == ["git", "clone"]:
            Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with tempfile.TemporaryDirectory() as td:
        cd = Path(td)
        with mock.patch("subprocess.run", side_effect=fake_run):
            result = ensure_git_repo("https://example.com/x.git", "abc123", cd, "seem")

        assert result == cd / "repos" / "seem"
        assert (result / ".testdrive_complete").exists()
        assert calls[0][:2] == ["git", "clone"]
        assert calls[0][2] == "https://example.com/x.git"
        assert calls[1][:4] == ["git", "-C", calls[1][2], "checkout"]
        assert calls[1][4] == "abc123"


def test_ensure_git_repo_second_call_is_a_no_op():
    import subprocess
    import tempfile
    import unittest.mock as mock
    from pathlib import Path

    from testdrive.util import ensure_git_repo

    def fake_run(cmd, **kw):
        if cmd[:2] == ["git", "clone"]:
            Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with tempfile.TemporaryDirectory() as td:
        cd = Path(td)
        with mock.patch("subprocess.run", side_effect=fake_run):
            ensure_git_repo("https://example.com/x.git", "abc123", cd, "seem")

        with mock.patch(
            "subprocess.run", side_effect=AssertionError("must not clone again")
        ):
            result = ensure_git_repo("https://example.com/x.git", "abc123", cd, "seem")
        assert result == cd / "repos" / "seem"


def test_ensure_git_repo_raises_cache_not_populated_when_downloads_disallowed():
    import tempfile
    from pathlib import Path

    from testdrive.util import CacheNotPopulatedError, ensure_git_repo, set_downloads_allowed

    set_downloads_allowed(False)
    try:
        with tempfile.TemporaryDirectory() as td:
            try:
                ensure_git_repo("https://example.com/x.git", "abc123", Path(td), "seem")
                assert False, "expected CacheNotPopulatedError"
            except CacheNotPopulatedError:
                pass
    finally:
        set_downloads_allowed(True)


def test_ensure_git_repo_failed_checkout_leaves_no_partial_state():
    """A bad ref (or any failure after the clone) must not leave behind
    a directory that a later call would mistake for a completed clone.
    """
    import subprocess
    import tempfile
    import unittest.mock as mock
    from pathlib import Path

    from testdrive.util import ensure_git_repo

    def fake_run(cmd, **kw):
        if cmd[:2] == ["git", "clone"]:
            Path(cmd[-1]).mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        # the "git checkout" step fails
        raise subprocess.CalledProcessError(1, cmd, output="", stderr="unknown revision")

    with tempfile.TemporaryDirectory() as td:
        cd = Path(td)
        with mock.patch("subprocess.run", side_effect=fake_run):
            try:
                ensure_git_repo("https://example.com/x.git", "bad-ref", cd, "seem")
                assert False, "expected CalledProcessError"
            except subprocess.CalledProcessError:
                pass

        assert not (cd / "repos" / "seem").exists()
        assert not (cd / "repos" / "seem.part").exists()


def test_owlv2_initialize_uses_local_snapshot_dir():
    """Verify initialize() downloads via ensure_local_repo() and calls
    from_pretrained() with that local directory — not the repo id plus a
    cache_dir= kwarg, which is the older approach this framework moved
    away from specifically to avoid huggingface_hub's symlinked snapshot
    cache (broken on some Windows installs, and under Wine).
    """
    import os
    import sys
    import tempfile
    import types
    import unittest.mock as mock
    from pathlib import Path

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from testdrive.detection import PluginManifest

    captured = {}

    # --- fake huggingface_hub: record the snapshot_download call, create
    # the directory it's asked to populate (as the real one would) ---
    fake_hfh = types.ModuleType("huggingface_hub")

    def fake_snapshot_download(repo_id, local_dir, **kw):
        captured.setdefault("snapshot_download_calls", []).append(repo_id)
        Path(local_dir).mkdir(parents=True, exist_ok=True)

    def fake_list_repo_files(repo_id):
        return ["config.json", "preprocessor_config.json"]

    fake_hfh.snapshot_download = fake_snapshot_download
    fake_hfh.list_repo_files = fake_list_repo_files

    # --- fake transformers: from_pretrained() just records what path
    # it was called with ---
    fake_transformers = types.ModuleType("transformers")

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path, **kw):
            captured["processor_path"] = path
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kw):
            captured["model_path"] = path
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    fake_transformers.AutoProcessor = FakeProcessor
    fake_transformers.AutoImageProcessor = FakeProcessor
    fake_transformers.Owlv2Processor = FakeProcessor
    fake_transformers.Owlv2ForObjectDetection = FakeModel

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    env_backup = os.environ.pop("TESTDRIVE_CACHE", None)
    try:
        with tempfile.TemporaryDirectory() as td:
            os.environ["TESTDRIVE_CACHE"] = td
            with mock.patch.dict(
                "sys.modules",
                {
                    "huggingface_hub": fake_hfh,
                    "transformers": fake_transformers,
                    "torch": fake_torch,
                },
            ):
                from testdrive.models.owlv2 import Plugin, PLUGIN

                plugin = Plugin()
                plugin.manifest = PluginManifest.from_dict(PLUGIN)
                plugin._initialized = False
                plugin.initialize()

            from testdrive.cache import cache_dir
            from testdrive.util import _local_repo_dir

            expected_local_dir = _local_repo_dir(PLUGIN["id"], cache_dir())
    finally:
        os.environ.pop("TESTDRIVE_CACHE", None)
        if env_backup is not None:
            os.environ["TESTDRIVE_CACHE"] = env_backup

    assert captured.get("processor_path") == expected_local_dir
    assert captured.get("model_path") == expected_local_dir
    assert PLUGIN["hf_repo"] in captured.get("snapshot_download_calls", [])


# ---------------------------------------------------------------------------
# groundingdino model file
# ---------------------------------------------------------------------------


def test_gdino_parse_prompt_single():
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from testdrive.models.groundingdino import _parse_prompt

    gdino_text, labels = _parse_prompt("person")
    assert labels == ["person"]
    assert gdino_text == "person ."


def test_gdino_parse_prompt_multi():
    from testdrive.models.groundingdino import _parse_prompt

    gdino_text, labels = _parse_prompt("cat, dog, bird")
    assert labels == ["cat", "dog", "bird"]
    assert gdino_text == "cat . dog . bird ."


def test_gdino_parse_prompt_lowercases():
    from testdrive.models.groundingdino import _parse_prompt

    _, labels = _parse_prompt("Person, CAR")
    assert labels == ["person", "car"]


def test_gdino_parse_prompt_strips_whitespace():
    from testdrive.models.groundingdino import _parse_prompt

    gdino_text, labels = _parse_prompt("  cat ,  dog  ")
    assert labels == ["cat", "dog"]
    assert "cat" in gdino_text and "dog" in gdino_text


def test_gdino_parse_prompt_trailing_dot():
    from testdrive.models.groundingdino import _parse_prompt

    gdino_text, _ = _parse_prompt("person, car")
    assert gdino_text.endswith(" .")


def test_gdino_manifest_complete():
    loaded = load_plugin("groundingdino")
    m = loaded.manifest
    assert m.hf_repo == "IDEA-Research/grounding-dino-base"
    assert m.license == "Apache-2.0"
    assert any(r["module"] == "torch" for r in m.requirements)
    assert any(r["module"] == "transformers" for r in m.requirements)


def test_gdino_selftest_missing_deps():
    """Self-test should report MISSING_DEPENDENCY when is_installed()
    says a plugin's dependencies aren't met.

    This used to call run_selftest("groundingdino") with nothing
    mocked, assuming torch/transformers would be absent — true in a
    bare CI container, false on any machine that's actually been used
    to run testdrive for real. On a fully set-up dev machine, the real
    is_installed() reports everything present, so this test's own
    assertion (guarded by `if not result.passed:`) silently never ran —
    and instead a full self-test proceeded to actually download and run
    Grounding DINO for real as an accidental multi-minute integration
    test disguised as a fast unit test. Mocking is_installed() directly
    makes this deterministic and fast regardless of what's actually
    installed.
    """
    import unittest.mock as mock
    from testdrive.selftest import run_selftest
    from testdrive.models.groundingdino import Plugin

    with mock.patch.object(Plugin, "is_installed", return_value=(False, ["torch", "transformers"])):
        result = run_selftest("groundingdino")

    assert not result.passed
    dep_step = next((s for s in result.steps if "dependencies" in s), None)
    assert dep_step is not None
    assert result.failures and "missing" in result.failures[0]


def test_gdino_selftest_mock_passes():
    """Full self-test pipeline with mocked torch+transformers."""
    from testdrive.detection import Detection, PluginManifest, FRAMEWORK_API_VERSION
    from testdrive.plugin import DetectorPlugin
    from testdrive.pluginloader import LoadedPlugin
    import testdrive.pluginloader as pl

    class FakePlugin(DetectorPlugin):
        def initialize(self):
            pass

        def detect(self, image, prompt, threshold=0.3):
            return [Detection(label="person", score=0.88, bbox=(5, 5, 50, 120))]

    manifest = PluginManifest(
        id="gdino_mock",
        name="GDino Mock",
        version="0",
        api=FRAMEWORK_API_VERSION,
        requirements=[],
        sample_prompt="person",
    )
    _orig = pl.load_plugin

    def _fake(pid):
        if pid == "gdino_mock":
            return LoadedPlugin("gdino_mock", manifest, FakePlugin)
        return _orig(pid)

    pl.load_plugin = _fake
    try:
        from testdrive.selftest import run_selftest

        result = run_selftest("gdino_mock")
    finally:
        pl.load_plugin = _orig

    assert result.passed, f"expected pass, failures: {result.failures}"
    assert result.detection_count == 1


def test_gdino_initialize_uses_local_snapshot_dir():
    """Verify initialize() downloads via ensure_local_repo() and calls
    from_pretrained() with that local directory — not the repo id plus a
    cache_dir= kwarg (see test_owlv2_initialize_uses_local_snapshot_dir
    for why).
    """
    import os
    import tempfile
    import types
    import unittest.mock as mock
    from pathlib import Path

    captured = {}

    fake_hfh = types.ModuleType("huggingface_hub")

    def fake_snapshot_download(repo_id, local_dir, **kw):
        captured.setdefault("snapshot_download_calls", []).append(repo_id)
        Path(local_dir).mkdir(parents=True, exist_ok=True)

    def fake_list_repo_files(repo_id):
        return ["config.json", "preprocessor_config.json"]

    fake_hfh.snapshot_download = fake_snapshot_download
    fake_hfh.list_repo_files = fake_list_repo_files

    fake_transformers = types.ModuleType("transformers")

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path, **kw):
            captured["processor_path"] = path
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kw):
            captured["model_path"] = path
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    fake_transformers.AutoProcessor = FakeProcessor
    fake_transformers.AutoImageProcessor = FakeProcessor
    fake_transformers.AutoModelForZeroShotObjectDetection = FakeModel

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    env_backup = os.environ.pop("TESTDRIVE_CACHE", None)
    try:
        with tempfile.TemporaryDirectory() as td:
            os.environ["TESTDRIVE_CACHE"] = td
            with mock.patch.dict(
                "sys.modules",
                {
                    "huggingface_hub": fake_hfh,
                    "transformers": fake_transformers,
                    "torch": fake_torch,
                },
            ):
                from testdrive.models.groundingdino import Plugin, PLUGIN

                plugin = Plugin()
                plugin.manifest = PluginManifest.from_dict(PLUGIN)
                plugin._initialized = False
                plugin.initialize()

            from testdrive.cache import cache_dir
            from testdrive.util import _local_repo_dir

            expected_local_dir = _local_repo_dir(PLUGIN["id"], cache_dir())
    finally:
        os.environ.pop("TESTDRIVE_CACHE", None)
        if env_backup is not None:
            os.environ["TESTDRIVE_CACHE"] = env_backup

    assert captured.get("processor_path") == expected_local_dir
    assert captured.get("model_path") == expected_local_dir
    assert PLUGIN["hf_repo"] in captured.get("snapshot_download_calls", [])
