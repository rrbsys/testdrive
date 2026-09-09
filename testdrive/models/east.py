"""EAST scene-text detector plugin (OpenCV DNN).

Prompts: ``"wordblock"`` (one Detection per individual text region)
and ``"lineblock"`` (text regions grouped into lines). Deliberately
named to match the repo's ``<class>block`` = bbox-only convention
(see paddleocr.py's ``"textblock"``) rather than paddleocr's
identically-shaped ``"ocrword"``/``"ocrline"`` prompts, because EAST is
a *detector only*: it was trained purely to find where text is (as
axis-ish-aligned quadrilaterals), never to recognize *what* the text
says. The ``"ocr*"`` prefix is reserved for prompts that actually
recognize text content (paddleocr's), so using it here — despite the
superficial word/line similarity — would misleadingly imply this
plugin does too. Concretely:

* Neither prompt ever sets ``Detection.text`` — there's no recognized
  string to put there, so no "word"/"line" key ever appears in
  ``<plugin>_redactions.json`` for this plugin, and no
  ``<plugin>_text.txt`` is ever produced (that whole mechanism keys
  off ``Detection.text`` being set — see cli.py's directory-loop path).
* Labels are the bare class name (``"wordblock"``, ``"lineblock"``),
  with no ``:<lang>`` suffix — EAST has no language identification
  either, only geometry.

EAST (github.com/argman/EAST) is itself a *word*-level detector — that
maps directly onto ``"wordblock"``. There's no notion of a "line" in
the model's output at all; ``"lineblock"`` is purely a geometric
post-processing step this plugin adds on top (see
``_group_into_lines``), clustering nearby word-level boxes by vertical
position — not a second model capability.

Runs via OpenCV's DNN module (``cv2.dnn.readNet``) on the classic
``frozen_east_text_detection.pb`` checkpoint — no torch/transformers,
CPU-friendly, same spirit as yunet.py. EAST requires the input size to
be a multiple of 32; this plugin always resizes to a fixed
``_INPUT_WIDTH``x``_INPUT_HEIGHT`` working resolution and maps
detections back to the original image size afterwards (see
``detect()``), rather than trying to preserve the original resolution
exactly — a deliberate speed/simplicity tradeoff, same as YuNet's fixed
320x320 default before ``setInputSize()`` is called per-image (except
EAST's input size is fixed for the whole run, not adjusted per image).

The decode step below (turning EAST's raw score/geometry feature maps
into boxes) also ignores the model's per-cell rotation angle output and
always produces an axis-aligned box — the standard simplification for
this model when you only need "where is the text", not "what's its
exact rotated quadrilateral". This works well for the common
near-horizontal-text case; noticeably rotated text will get a looser
axis-aligned box than its true rotated extent.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from ..cache import cache_dir
from ..detection import Detection
from ..plugin import DetectorPlugin
from ..util import download_file

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger("testdrive.models.east")

PLUGIN_API = 1

# The two prompts this plugin ever honors — see the module docstring.
_WORD_PROMPT = "wordblock"
_LINE_PROMPT = "lineblock"

# EAST requires width/height that are both multiples of 32. Kept small
# (and thus fast) by default, same "good enough, not exhaustive"
# tradeoff as YuNet's 320x320 default input size — bump these if you
# need to pick up small text in high-resolution images.
_INPUT_WIDTH = 320
_INPUT_HEIGHT = 320

# IoU threshold used for both the raw-detection NMS pass (deduping
# overlapping candidate boxes around the same text region) and the
# line-grouping vertical-proximity check below.
_NMS_IOU_THRESHOLD = 0.4

# oyyd/frozen_east_text_detection.pb re-hosts argman/EAST's official
# release checkpoint as a plain (non-LFS) blob directly in a small git
# repo, so raw.githubusercontent.com serves the real ~92MB binary, not
# an LFS pointer stub — same "avoid the git-lfs-via-raw-URL trap"
# rationale documented in yunet.py's _MODEL_URL comment, just solved
# here by a repo that never used LFS in the first place rather than by
# using a dedicated HF mirror.
_MODEL_URL = (
    "https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/"
    "master/frozen_east_text_detection.pb"
)
_MODEL_FILENAME = "frozen_east_text_detection.pb"

# The two output layers EAST's frozen graph exposes: per-cell
# text/no-text confidence, and per-cell box geometry (4 edge distances
# + 1 rotation angle) — see _decode_predictions.
_LAYER_SCORES = "feature_fusion/Conv_7/Sigmoid"
_LAYER_GEOMETRY = "feature_fusion/concat_3"

PLUGIN = {
    "id": "east",
    "name": "EAST",
    "version": "0.1.0",
    "api": PLUGIN_API,
    "description": (
        "Efficient and Accurate Scene Text detector (EAST) via OpenCV's DNN "
        "module. Fixed vocabulary, geometry-only — it finds text regions but "
        "never recognizes their content, so 'prompt' isn't an open-vocabulary "
        "description; it must simply be 'wordblock' (one box per detected text "
        "region — EAST's native word-level output) or 'lineblock' (the same "
        "regions geometrically grouped into lines by this plugin, not a "
        "separate model capability) to confirm intent (any other prompt "
        "reports nothing, with a warning). Named '*block' rather than "
        "paddleocr's superficially-similar 'ocrword'/'ocrline' on purpose — "
        "per the repo convention, 'ocr*' means detected text is recognized "
        "and attached to the detection, '*block' means bbox-only — and this "
        "plugin never recognizes text, so no Detection.text and no ':<lang>' "
        "label suffix either (EAST has no language ID, only geometry); score "
        "is just EAST's own text/no-text confidence."
    ),
    "author": "Xinyu Zhou et al. (EAST) / OpenCV DNN port",
    "homepage": "https://github.com/argman/EAST",
    "license": "MIT",
    "license_url": "https://github.com/argman/EAST/blob/master/LICENSE",
    "backend": "opencv-dnn (EAST)",
    "task": "fixed-vocabulary (text-region-only) detection, word or line grouping",
    "supports": [
        "word-level text region detection ('wordblock')",
        "line-level text region grouping ('lineblock', geometric post-processing)",
        "fast CPU inference",
        "no torch/transformers dependency",
    ],
    "requirements": [
        {"pip": "Pillow", "module": "PIL"},
        {"pip": "numpy", "module": "numpy"},
        {"pip": "opencv-python", "module": "cv2"},
    ],
    "sample_prompt": "wordblock",
    "test_threshold": "default",
    "pyenv": "framework",
    "classes": [_WORD_PROMPT, _LINE_PROMPT],
}


def _decode_predictions(
    scores: Any, geometry: Any, min_confidence: float
) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    """Turn EAST's raw ``scores``/``geometry`` feature maps into
    axis-aligned candidate boxes (in the resized-input's coordinate
    space — the caller maps back to the original image) + confidences,
    filtering out anything below *min_confidence*.

    Each cell of the (downsampled by 4x) feature map holds a text/
    no-text confidence and, in ``geometry``, 5 numbers: the distance
    from that cell to each of its box's 4 edges (top/right/bottom/left)
    plus a rotation angle — the standard EAST "RBOX" geometry encoding.
    See the module docstring re: the rotation angle being ignored here.
    """
    num_rows, num_cols = scores.shape[2:4]
    boxes: list[tuple[float, float, float, float]] = []
    confidences: list[float] = []

    for y in range(num_rows):
        scores_row = scores[0, 0, y]
        dist_top = geometry[0, 0, y]
        dist_right = geometry[0, 1, y]
        dist_bottom = geometry[0, 2, y]
        dist_left = geometry[0, 3, y]
        angles_row = geometry[0, 4, y]

        for x in range(num_cols):
            score = float(scores_row[x])
            if score < min_confidence:
                continue

            # The feature map is 4x downsampled from the input.
            offset_x, offset_y = x * 4.0, y * 4.0
            angle = float(angles_row[x])
            cos_a, sin_a = math.cos(angle), math.sin(angle)

            h = float(dist_top[x]) + float(dist_bottom[x])
            w = float(dist_right[x]) + float(dist_left[x])

            end_x = offset_x + cos_a * float(dist_right[x]) + sin_a * float(dist_bottom[x])
            end_y = offset_y - sin_a * float(dist_right[x]) + cos_a * float(dist_bottom[x])
            start_x = end_x - w
            start_y = end_y - h

            boxes.append((start_x, start_y, end_x, end_y))
            confidences.append(score)

    return boxes, confidences


def _group_into_lines(
    items: list[tuple[tuple[int, int, int, int], float]],
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Greedily cluster word-level ``(bbox, score)`` detections into
    line-level ``(bbox, score)`` detections by vertical (y-center)
    proximity — EAST has no notion of a "line" itself (see module
    docstring), so this is purely geometric post-processing: each
    output line's box is the enclosing rectangle of the words assigned
    to it, and its score is the mean of their scores.

    Not layout-aware (no attempt at column/paragraph detection, or at
    ordering left-to-right within a line beyond what the input order
    already gives it) — good enough to turn a scattering of word boxes
    into readable "rows" of text, not a full document-layout analyzer.
    Output is sorted top-to-bottom for a stable, reading-order sequence
    of detection.
    """
    if not items:
        return []

    # Sort by vertical center first so nearby words tend to be
    # considered together and running per-line averages stay stable.
    ordered = sorted(items, key=lambda t: (t[0][1] + t[0][3]) / 2)

    lines: list[dict[str, Any]] = []
    for bbox, score in ordered:
        cy = (bbox[1] + bbox[3]) / 2
        h = bbox[3] - bbox[1]
        placed = False
        for line in lines:
            if abs(line["cy"] - cy) <= _NMS_IOU_THRESHOLD * max(h, line["h"], 1):
                line["boxes"].append(bbox)
                line["scores"].append(score)
                line["cy"] = sum((b[1] + b[3]) / 2 for b in line["boxes"]) / len(line["boxes"])
                line["h"] = sum(b[3] - b[1] for b in line["boxes"]) / len(line["boxes"])
                placed = True
                break
        if not placed:
            lines.append({"cy": cy, "h": h, "boxes": [bbox], "scores": [score]})

    out: list[tuple[tuple[int, int, int, int], float]] = []
    for line in lines:
        xs1 = [b[0] for b in line["boxes"]]
        ys1 = [b[1] for b in line["boxes"]]
        xs2 = [b[2] for b in line["boxes"]]
        ys2 = [b[3] for b in line["boxes"]]
        merged_bbox = (min(xs1), min(ys1), max(xs2), max(ys2))
        merged_score = sum(line["scores"]) / len(line["scores"])
        out.append((merged_bbox, merged_score))

    out.sort(key=lambda t: t[0][1])
    return out


class Plugin(DetectorPlugin):
    """EAST fixed-vocabulary (word/line text-region) detector."""

    _net: Any
    _initialized: bool = False

    def initialize(self) -> None:
        if self._initialized:
            return

        import cv2

        cd = cache_dir()
        checkpoint = download_file(
            _MODEL_URL,
            cd / "checkpoints" / "east",
            filename=_MODEL_FILENAME,
        )

        log.info("loading EAST...")
        self._net = cv2.dnn.readNet(str(checkpoint))
        self._initialized = True
        log.info("EAST ready")

    def detect(
        self, image: "PILImage.Image", prompt: str, threshold: float = 0.5
    ) -> list[Detection]:
        """Detect text regions and label each with EAST's own confidence.

        ``prompt`` must be ``"wordblock"`` or ``"lineblock"`` (case-
        insensitive) — see the module docstring. Anything else never
        matches, with a warning logged rather than raising, matching
        paddleocr/yunet/yolo11's handling of an unknown class/prompt so
        a loop-mode run over several plugins doesn't blow up on a
        prompt that just isn't this plugin's. Unlike paddleocr's
        identically-named prompts, no language word is accepted here —
        EAST has no language identification, only geometry — and one
        present in the prompt is ignored with a warning rather than
        causing a mismatch.
        """
        if not self._initialized:
            self.initialize()

        parts = prompt.strip().lower().split()
        cls = parts[0] if parts else ""
        if len(parts) > 1:
            log.warning(
                "east has no language filtering; ignoring extra prompt word(s): %s",
                " ".join(parts[1:]),
            )

        if cls not in (_WORD_PROMPT, _LINE_PROMPT):
            log.warning(
                "east only detects '%s' or '%s'; prompt '%s' will never match",
                _WORD_PROMPT,
                _LINE_PROMPT,
                prompt,
            )
            return []

        import cv2
        import numpy as np

        img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        orig_h, orig_w = img_bgr.shape[:2]
        ratio_w = orig_w / float(_INPUT_WIDTH)
        ratio_h = orig_h / float(_INPUT_HEIGHT)

        resized = cv2.resize(img_bgr, (_INPUT_WIDTH, _INPUT_HEIGHT))
        blob = cv2.dnn.blobFromImage(
            resized,
            1.0,
            (_INPUT_WIDTH, _INPUT_HEIGHT),
            (123.68, 116.78, 103.94),
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        scores, geometry = self._net.forward([_LAYER_SCORES, _LAYER_GEOMETRY])

        raw_boxes, confidences = _decode_predictions(scores, geometry, threshold)
        if not raw_boxes:
            log.debug("detect: no text regions above threshold %.2f", threshold)
            return []

        # NMSBoxes wants (x, y, w, h) rects, not (x1, y1, x2, y2).
        rects = [
            (int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1)))
            for x1, y1, x2, y2 in raw_boxes
        ]
        keep = cv2.dnn.NMSBoxes(rects, confidences, threshold, _NMS_IOU_THRESHOLD)
        keep = [int(i) for i in np.array(keep).flatten()] if len(keep) else []

        words: list[tuple[tuple[int, int, int, int], float]] = []
        for i in keep:
            x1, y1, x2, y2 = raw_boxes[i]
            bbox = (
                max(0, int(round(x1 * ratio_w))),
                max(0, int(round(y1 * ratio_h))),
                min(orig_w, int(round(x2 * ratio_w))),
                min(orig_h, int(round(y2 * ratio_h))),
            )
            words.append((bbox, confidences[i]))

        if cls == _WORD_PROMPT:
            detections = [
                Detection(label=_WORD_PROMPT, score=score, bbox=bbox) for bbox, score in words
            ]
            log.debug(
                "detect: %d word-level text region(s) above threshold %.2f",
                len(detections),
                threshold,
            )
            return detections

        lines = _group_into_lines(words)
        detections = [
            Detection(label=_LINE_PROMPT, score=score, bbox=bbox) for bbox, score in lines
        ]
        log.debug(
            "detect: %d line(s) (from %d word-level region(s)) above threshold %.2f",
            len(detections),
            len(words),
            threshold,
        )
        return detections
