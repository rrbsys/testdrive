"""PaddleOCR text-block detector plugin.

Like ``yunet``, this is fixed-vocabulary rather than open-vocabulary:
there's exactly one thing it ever reports — a "textblock" — so the
first word of ``prompt`` must be ``"textblock"`` (case-insensitive) to
confirm intent, mirroring how ``yunet`` gates on ``"face"``. An optional
second word restricts to a language (``en``/``zh``/...). Prompt ``"ocr"``
is a text task that dumps all recognized lines (see ``PLUGIN["tasks"]``).

Unlike a normal fixed-vocabulary detector though, each detection's
*label* isn't just the class name — the framework's annotate.py
renders whatever we put in ``Detection.label`` alongside the score, so
this plugin follows the repo's ``<class>:<attribute>`` labeling
convention to report each block's *detected language* as a suffix on
the class name (``"textblock:en"``, ``"textblock:zh"``, ...), with
``score`` being PaddleOCR's own recognition confidence for that block.
That's what puts "language detected and confidence" on each green box
without any framework changes — draw_boxes() already renders
``f"{label} {score}"``.

CURRENT LIMITATION — English/Chinese only, no German yet: the plan was
to run two pipelines (``lang="ch"`` for Chinese+English, ``lang=
"german"`` for German) and merge their outputs, since no single
PaddleOCR recognition model currently covers all three languages at
once. In practice, as of this writing every recent ``ocr_version`` for
``lang="german"`` hits a confirmed upstream paddle_inference bug
(newer PP-OCRv5/v6 models are exported in a PIR static-graph format
that predictor construction rejects with "InvalidArgument: Type of
attribute: strides is not right" —
https://github.com/PaddlePaddle/PaddleOCR/issues/15908), while forcing
the older ``ocr_version="PP-OCRv4"`` to dodge that bug turns out to
ship no German model at all (``ValueError: No models are available
for lang='german' and ocr_version='PP-OCRv4'``). So for now this
plugin only runs the "ch" pipeline (Chinese + English), and a German
text block will get *mis*-labeled ``"textblock:en"`` (Latin script with no CJK
characters — see ``_is_cjk``) rather than going undetected. Revisit
once the upstream PIR issue is fixed and a compatible German model
line exists again; the ``_run_pipeline``/NMS-merge structure below is
kept as-is (rather than simplified down to a single call) specifically
so re-adding a second pipeline later is a two-line change, not a
rewrite.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from ..cache import cache_dir
from ..detection import Detection
from ..plugin import DetectorPlugin

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger("testdrive.models.paddleocr")

PLUGIN_API = 1

# paddlex (PaddleOCR's underlying pipeline runner) resolves its model
# cache root as a *module-level* constant the moment paddlex.utils.cache
# is first imported:
#
#     CACHE_DIR = os.environ.get("PADDLE_PDX_CACHE_HOME", DEFAULT_CACHE_DIR)
#
# — not lazily inside a function, so this has to be set before paddlex
# is imported *anywhere* in the process, even once. That ruled out
# setting it inside initialize(): DetectorPlugin.is_installed() (called
# by worker_main.py before initialize()) does a real
# importlib.import_module("paddleocr"), which pulls paddlex in via
# import machinery well before initialize() ever runs. Setting it here
# instead, at our own module's import time, works because pluginloader
# always imports this file (to discover PLUGIN/Plugin) before it can
# call is_installed() or initialize() on it — so this is the earliest
# point in this plugin's lifecycle we get to run any code at all.
#
# Routed through the same cache/ directory (and TESTDRIVE_CACHE
# override) every other plugin uses, under its own "paddleocr" subdir
# so it doesn't collide with the HF-style caches other plugins use.
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir() / "models" / "paddleocr"))
(cache_dir() / "models" / "paddleocr").mkdir(parents=True, exist_ok=True)

# Wine / some Windows setups: oneDNN + PIR crashes with
#   NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support
#   [pir::ArrayAttribute<pir::DoubleAttribute>]
# (see onednn_instruction.cc)
# Must be set before paddle/paddleocr import. paddlex may still force
# FLAGS_enable_pir_api=1 when device_type=cpu; enable_mkldnn=False is the real fix.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"

# The only prompt this plugin ever honors — see the module docstring.
_TEXTBLOCK_PROMPT = "textblock"

# Greedy-NMS overlap threshold for merging multiple pipelines' candidate
# boxes for the same physical text block (see module docstring — only
# one pipeline runs right now, but this stays ready for a second).
_NMS_IOU_THRESHOLD = 0.3

# Common CJK Unicode block ranges (CJK Unified Ideographs + Extension A
# + CJK punctuation) — enough to tell "this line is Chinese" from
# "this line is English" without a separate language-ID dependency,
# since the "ch" pipeline is only ever asked to pick between those two.
_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x3000, 0x303F),  # CJK punctuation
    (0xFF00, 0xFFEF),  # Fullwidth forms
)

PLUGIN = {
    "id": "paddleocr",
    "name": "PaddleOCR",
    "version": "0.1.0",
    "api": PLUGIN_API,
    "description": (
        "OCR-based text block detection using PaddleOCR's PP-OCRv4 pipeline. "
        "Fixed single-class vocabulary — it only detects text blocks, so "
        "'prompt' isn't an open-vocabulary description; it must simply be "
        "'textblock' to confirm intent (any other prompt reports nothing, "
        "with a warning). Each box's label is 'textblock:<lang>' (e.g. "
        "'textblock:en'), following the repo's <class>:<attribute> "
        "labeling convention, with score as OCR confidence. Currently "
        "English/Chinese only — see module docstring re: German."
    ),
    "author": "PaddlePaddle / Baidu",
    "homepage": "https://github.com/PaddlePaddle/PaddleOCR",
    "license": "Apache-2.0",
    "license_url": "https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE",
    "backend": "paddleocr (PP-OCRv4)",
    "task": "fixed-vocabulary (text-block-only) detection + language ID",
    "supports": [
        "text block detection",
        "en/zh recognition in one image",
        "per-box language label + OCR confidence",
    ],
    "requirements": [
        {"pip": "Pillow", "module": "PIL"},
        {"pip": "numpy", "module": "numpy"},
        {"pip": "opencv-python", "module": "cv2"},
        {"pip": "paddlepaddle", "module": "paddle"},
        {"pip": "paddleocr>=3.0", "module": "paddleocr"},
    ],
    "sample_prompt": "textblock",
    "test_threshold": "default",
    # paddlepaddle is its own large tensor framework (not torch, not
    # tensorflow) with its own pinned dependency chain — isolating it
    # avoids the exact "one plugin's upgrade breaks everyone else"
    # problem PluginManifest.pyenv's docstring warns about.
    "pyenv": "paddleocr",
    "classes": [_TEXTBLOCK_PROMPT],
    "language": "en",
    "languages": ["en", "zh", "de-not_implemented_yet"],
    "tasks": {
        "ocr": "ocr",
    },
}


def _is_cjk(text: str) -> bool:
    """True if *text* looks predominantly Chinese rather than English.

    Only needs to distinguish those two, since the "ch" pipeline below
    is never asked to recognize anything else. (A German block will
    also fall through to "not CJK" — mislabeled as English, see module
    docstring.)
    """
    if not text:
        return False
    cjk_count = sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES))
    return cjk_count / len(text) > 0.3


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _parse_prompt(prompt: str) -> tuple[str, str | None]:
    """Split ``prompt`` into (class, optional_language)."""
    parts = prompt.strip().split()
    if not parts:
        return "", None
    cls = parts[0].lower()
    lang = parts[1].lower() if len(parts) > 1 else None
    if len(parts) > 2:
        log.warning(
            "paddleocr ignores extra prompt words after language: %s",
            " ".join(parts[2:]),
        )
    return cls, lang


class Plugin(DetectorPlugin):
    """PaddleOCR fixed single-class (textblock) detector with language ID."""

    _ocr_cjk: Any  # lang="ch": Simplified/Traditional Chinese + English
    _initialized: bool = False

    def initialize(self) -> None:
        if self._initialized:
            return

        from paddleocr import PaddleOCR

        log.info("loading PaddleOCR (ch pipeline)...")
        # Doc-orientation/unwarping/textline-orientation are all aimed
        # at scanned-document quirks (skew, warping, upside-down
        # pages) that don't apply to a synthetic test image with
        # horizontal text — leaving them on only costs extra model
        # downloads and inference time for no benefit here.
        #
        # ocr_version is pinned to "PP-OCRv4" rather than the current
        # default (PP-OCRv5/v6) to dodge a confirmed upstream
        # paddle_inference bug with the newer PIR-based model export
        # format — see the module docstring for the full story and the
        # (currently unresolved) tradeoff this pin implies for German.
        common_kwargs = dict(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            ocr_version="PP-OCRv4",
            enable_mkldnn=False,  # required: oneDNN + PIR crashes under Wine/Win
            device="cpu",
        )
        self._ocr_cjk = PaddleOCR(lang="ch", **common_kwargs)

        self._initialized = True
        log.info("PaddleOCR ready")

    def _run_pipeline(
        self, ocr: Any, img_bgr: Any, *, fixed_label: str | None, threshold: float
    ) -> list[Detection]:
        """Run one PaddleOCR pipeline over the full image.

        *fixed_label* is the language to report for every detection
        from this pipeline (e.g. "de"), or ``None`` to decide per-box
        via :func:`_is_cjk` (the "ch" pipeline, which recognizes both
        Chinese and English). Either way the final ``Detection.label``
        is ``"textblock:<lang>"``, per the repo's ``<class>:<attribute>``
        convention (see module docstring) — not just the bare language.
        """
        out: list[Detection] = []
        for page in ocr.predict(input=img_bgr):
            texts = page["rec_texts"]
            scores = page["rec_scores"]
            boxes = page["rec_boxes"]
            for text, score, box in zip(texts, scores, boxes):
                score = float(score)
                if score < threshold:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box)
                lang = fixed_label if fixed_label else ("zh" if _is_cjk(text) else "en")
                label = f"{_TEXTBLOCK_PROMPT}:{lang}"
                out.append(Detection(label=label, score=score, bbox=(x1, y1, x2, y2)))
        return out


    def _collect_texts(self, img_bgr: Any) -> list[str]:
        """Return every recognized text line from the image (no threshold)."""
        texts_out: list[str] = []
        for page in self._ocr_cjk.predict(input=img_bgr):
            for text in page["rec_texts"]:
                t = str(text).strip()
                if t:
                    texts_out.append(t)
        return texts_out

    def run_task(self, image: "PILImage.Image", task_prompt: str) -> str:
        """Dump every OCR line found in the image (prompt ``ocr``)."""
        if not self._initialized:
            self.initialize()

        import cv2
        import numpy as np

        img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        texts = self._collect_texts(img_bgr)
        log.debug("run_task(%s): %d text line(s)", task_prompt, len(texts))
        return "\n".join(texts)

    def detect(
        self, image: "PILImage.Image", prompt: str, threshold: float = 0.5
    ) -> list[Detection]:
        """Detect text blocks and label each with its language + OCR confidence.

        ``prompt`` must be ``"textblock"`` (case-insensitive) — see the
        module docstring. Anything else never matches, with a warning
        logged rather than raising, matching ``yunet``/``yolo11``'s
        handling of an unknown class/prompt so a loop-mode run over
        several plugins doesn't blow up on a prompt that just isn't
        this plugin's.
        """
        if not self._initialized:
            self.initialize()

        cls, force_lang = _parse_prompt(prompt)
        if cls != _TEXTBLOCK_PROMPT:
            log.warning(
                "paddleocr only detects '%s'; prompt '%s' will never match",
                _TEXTBLOCK_PROMPT,
                prompt,
            )
            return []

        languages: list[str] = (
            list(self.manifest.languages)
            if self.manifest.languages
            else ["en", "zh", "de-not_implemented_yet"]
        )

        if force_lang is not None:
            if force_lang not in languages:
                log.warning(
                    "paddleocr does not support language '%s' (known: %s); "
                    "falling back to auto-detect",
                    force_lang,
                    ", ".join(languages),
                )
                force_lang = None
            elif force_lang == "de-not_implemented_yet":
                log.warning(
                    "German (de) is not implemented yet for paddleocr "
                    "(upstream paddle_inference PIR bug — see module docstring); "
                    "falling back to auto-detect (en/zh)"
                )
                force_lang = None

        import cv2
        import numpy as np

        img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

        # Only the "ch" (Chinese+English) pipeline runs for now — see
        # module docstring re: German. Kept as a list + NMS pass rather
        # than a single flat call so a second pipeline can be added
        # back with a one-line append once German is unblocked upstream.
        candidates = self._run_pipeline(
            self._ocr_cjk, img_bgr, fixed_label=None, threshold=threshold
        )

        if force_lang is not None:
            candidates = [d for d in candidates if d.label.endswith(f":{force_lang}")]

        candidates.sort(key=lambda d: d.score, reverse=True)
        kept: list[Detection] = []
        for det in candidates:
            if any(_iou(det.bbox, k.bbox) > _NMS_IOU_THRESHOLD for k in kept):
                continue
            kept.append(det)

        log.debug(
            "detect: %d text block(s) above threshold %.2f (lang filter=%s)",
            len(kept),
            threshold,
            force_lang or "auto",
        )
        return kept
