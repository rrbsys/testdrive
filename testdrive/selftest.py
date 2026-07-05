"""Plugin self-test runner (``testdrive -T <plugin>``).

The self-test exercises the full plugin pipeline against a small
synthetic image so you can verify a plugin is correctly installed and
produces valid output without needing a real photograph.

What it checks
--------------
1. Dependencies are importable (``is_installed()``).
2. ``initialize()`` completes without error.
3. ``detect()`` returns a ``list[Detection]`` in finite time.
4. Every returned ``Detection`` has valid fields:
   - ``score`` in [0.0, 1.0]
   - ``bbox`` is a 4-tuple of ints with x1 ≤ x2, y1 ≤ y2
   - ``label`` is a non-empty string

A synthetic image is used intentionally: detections are *not* required
(the prompt is unlikely to match noise), so the test is purely about
pipeline correctness, not model quality.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger("testdrive.selftest")

_SYNTHETIC_SIZE = (224, 224)
_SYNTHETIC_COLOR = (100, 149, 237)  # cornflower blue - recognisable, not white/black


@dataclass
class SelfTestResult:
    plugin_id: str
    passed: bool = False
    steps: list[str] = field(default_factory=list)  # human-readable step log
    failures: list[str] = field(default_factory=list)
    detection_count: int = 0
    elapsed_ms: float = 0.0

    def add_step(self, label: str, ok: bool, note: str = "") -> None:
        mark = "OK  " if ok else "FAIL"
        line = f"  {mark}  {label}"
        if note:
            line += f"  ({note})"
        self.steps.append(line)

    def summary(self) -> str:
        lines = [f"Self-test : {self.plugin_id}"]
        lines.extend(self.steps)
        if self.failures:
            lines.append("")
            for f in self.failures:
                lines.append(f"  ✗  {f}")
        lines.append("")
        lines.append("PASS" if self.passed else "FAIL")
        return "\n".join(lines)


def _make_synthetic_image() -> "PILImage.Image":
    """Return a small solid-colour RGB image with geometric shapes.

    Using a non-trivial image (rectangles in different colours) gives
    models something to look at while keeping the test fully offline
    and reproducible.
    """
    from PIL import Image, ImageDraw

    w, h = _SYNTHETIC_SIZE
    img = Image.new("RGB", (w, h), color=_SYNTHETIC_COLOR)
    draw = ImageDraw.Draw(img)

    # A few coloured rectangles - give object detectors something to score
    draw.rectangle([20, 20, 90, 110], fill=(220, 50, 50))  # red block
    draw.rectangle([120, 30, 200, 100], fill=(50, 180, 50))  # green block
    draw.rectangle([50, 130, 180, 200], fill=(255, 200, 0))  # yellow block

    return img


def _validate_detections(
    detections: object,
    failures: list[str],
) -> int:
    """Validate the return value of ``detect()``.

    Returns the number of detections found, or 0 on type errors.
    Appends a description of every validation failure to *failures*.
    """
    from .detection import Detection

    if not isinstance(detections, list):
        failures.append(f"detect() must return list[Detection], got {type(detections).__name__}")
        return 0

    for i, det in enumerate(detections):
        if not isinstance(det, Detection):
            failures.append(f"detection[{i}] is {type(det).__name__}, expected Detection")
            continue

        if not isinstance(det.label, str) or not det.label:
            failures.append(f"detection[{i}].label must be a non-empty str")

        if not (0.0 <= det.score <= 1.0):
            failures.append(f"detection[{i}].score={det.score:.4f} is outside [0, 1]")

        if len(det.bbox) != 4:
            failures.append(f"detection[{i}].bbox must be a 4-tuple, got {det.bbox!r}")
        else:
            x1, y1, x2, y2 = det.bbox
            if x1 > x2 or y1 > y2:
                failures.append(f"detection[{i}].bbox has x1 > x2 or y1 > y2: {det.bbox!r}")

    return len(detections)


def run_selftest(plugin_id: str) -> SelfTestResult:
    """Run the self-test for *plugin_id* and return a :class:`SelfTestResult`.

    This is the main entry point called by the CLI.  It never raises;
    all errors are captured in the result object.
    """
    from .pluginloader import PluginLoadError, load_plugin

    result = SelfTestResult(plugin_id=plugin_id)
    t_total = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Step 1: load plugin
    # ------------------------------------------------------------------ #
    try:
        loaded = load_plugin(plugin_id)
        result.add_step("load plugin", ok=True)
    except PluginLoadError as exc:
        result.add_step("load plugin", ok=False, note=str(exc))
        result.failures.append(str(exc))
        return result

    # ------------------------------------------------------------------ #
    # Step 2: dependency check
    # ------------------------------------------------------------------ #
    plugin = loaded.instantiate()
    installed, missing = plugin.is_installed()
    if not installed:
        note = "missing: " + ", ".join(missing)
        result.add_step("dependencies", ok=False, note=note)
        result.failures.append(f"missing packages: {', '.join(missing)}")
        return result
    result.add_step("dependencies", ok=True)

    # ------------------------------------------------------------------ #
    # Step 3: build synthetic test image
    # ------------------------------------------------------------------ #
    try:
        image = _make_synthetic_image()
        w, h = image.size
        result.add_step("synthetic image", ok=True, note=f"{w}×{h} RGB")
    except Exception as exc:  # noqa: BLE001
        result.add_step("synthetic image", ok=False, note=str(exc))
        result.failures.append(f"could not create test image: {exc}")
        return result

    # ------------------------------------------------------------------ #
    # Step 4: initialize plugin
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    try:
        plugin.initialize()
        init_ms = (time.perf_counter() - t0) * 1000
        result.add_step("initialize", ok=True, note=f"{init_ms:.0f} ms")
    except Exception as exc:  # noqa: BLE001
        init_ms = (time.perf_counter() - t0) * 1000
        result.add_step("initialize", ok=False, note=str(exc))
        result.failures.append(f"initialize() raised: {exc}")
        return result

    # ------------------------------------------------------------------ #
    # Step 5: run detection
    # ------------------------------------------------------------------ #
    prompt = plugin.manifest.sample_prompt or "object"
    log.debug("self-test prompt: %r", prompt)

    t0 = time.perf_counter()
    try:
        detections = plugin.detect(image, prompt, threshold=0.1)
        detect_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:  # noqa: BLE001
        detect_ms = (time.perf_counter() - t0) * 1000
        result.add_step("detect", ok=False, note=str(exc))
        result.failures.append(f"detect() raised: {exc}")
        return result

    # ------------------------------------------------------------------ #
    # Step 6: validate return value
    # ------------------------------------------------------------------ #
    validation_failures: list[str] = []
    count = _validate_detections(detections, validation_failures)

    ok = len(validation_failures) == 0
    note = f"{detect_ms:.0f} ms  →  {count} detection(s)"
    result.add_step("detect", ok=ok, note=note)
    result.add_step(f'prompt: "{prompt}"', ok=True)

    if validation_failures:
        result.failures.extend(validation_failures)
        return result

    result.detection_count = count
    result.elapsed_ms = (time.perf_counter() - t_total) * 1000
    result.passed = True
    return result
