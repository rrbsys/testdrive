"""YuNet face detector plugin (OpenCV).

Like YOLO11, YuNet is a *fixed*-vocabulary detector, but even more
narrowly so: it only ever detects one thing — human faces. There's no
free-text, open-vocabulary understanding at inference time, and unlike
YOLO11 there isn't even a choice of *which* known class to report.
``prompt`` here is purely a confirmation of intent: it must be
``"face"`` (case-insensitive, surrounding whitespace ignored) or
nothing is reported at all, with a warning logged explaining why.

YuNet is tiny (~230KB) and runs comfortably in real time on CPU, via
OpenCV's DNN module (``cv2.FaceDetectorYN``) rather than
torch/transformers — no GPU, and no heavyweight ML framework, needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..cache import cache_dir
from ..detection import Detection
from ..plugin import DetectorPlugin
from ..util import download_file

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger("testdrive.models.yunet")

PLUGIN_API = 1

# The only prompt YuNet ever honors — see the module docstring.
_FACE_PROMPT = "face"

# The onnx model file lives in OpenCV's own model zoo repo, but that
# repo tracks it via git-lfs — raw.githubusercontent.com only ever
# serves the LFS *pointer* text for a file like this, never the actual
# binary, regardless of who fetches it. The dedicated HF mirror of
# just this model resolves the real binary through its normal
# /resolve/ endpoint instead (same pattern already used for YOLO11's
# checkpoints in yolo11.py), so that's used here as a plain URL via
# download_file() rather than a full ensure_local_repo() snapshot,
# since only this one file is needed.
_MODEL_URL = (
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
    "face_detection_yunet_2023mar.onnx"
)
_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"

PLUGIN = {
    "id": "yunet",
    "name": "YuNet",
    "version": "0.1.0",
    "api": PLUGIN_API,
    "description": (
        "Light-weight, fast CNN-based face detector from OpenCV Zoo. Fixed "
        "single-class vocabulary — it only detects faces, so 'prompt' isn't "
        "an open-vocabulary description; it must simply be 'face' to confirm "
        "intent (any other prompt reports nothing, with a warning)."
    ),
    "author": "Shiqi Yu / OpenCV",
    "homepage": "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
    "license": "MIT",
    "license_url": "https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE",
    "backend": "opencv-dnn",
    "hf_repo": "opencv/face_detection_yunet",
    "task": "fixed-vocabulary (face-only) object detection",
    "supports": ["face detection", "fast CPU inference", "no torch/transformers dependency"],
    "requirements": [
        {"pip": "Pillow", "module": "PIL"},
        {"pip": "numpy", "module": "numpy"},
        {"pip": "opencv-python", "module": "cv2"},
    ],
    "sample_prompt": "face",
    "test_threshold": "default",
    "pyenv": "framework",
    "classes": [_FACE_PROMPT],
}


class Plugin(DetectorPlugin):
    """YuNet fixed single-class (face) detector."""

    _detector: Any
    _initialized: bool = False

    def initialize(self) -> None:
        if self._initialized:
            return

        import cv2

        cd = cache_dir()
        checkpoint = download_file(
            _MODEL_URL, cd / "checkpoints" / "yunet", filename=_MODEL_FILENAME,
        )

        log.info("loading YuNet...")
        # input_size is a placeholder here — detect() calls
        # setInputSize() with the actual image dimensions before every
        # call, since it varies per image and OpenCV's DNN backend
        # infers on the exact shape given (see opencv_zoo issue #44).
        self._detector = cv2.FaceDetectorYN.create(
            model=str(checkpoint),
            config="",
            input_size=(320, 320),
            score_threshold=0.3,
            nms_threshold=0.3,
            top_k=5000,
        )

        self._initialized = True
        log.info("YuNet ready")

    def detect(self, image: "PILImage.Image", prompt: str, threshold: float = 0.3) -> list[Detection]:
        """Detect faces.

        ``prompt`` must be ``"face"`` (case-insensitive) — see the
        module docstring. Anything else never matches, with a warning
        logged, exactly mirroring how YOLO11 handles an unknown class
        name rather than raising, since a loop-mode run over several
        plugins (``'*'``) shouldn't blow up just because YuNet's one
        class doesn't apply to that iteration's prompt.
        """
        if not self._initialized:
            self.initialize()

        if prompt.strip().lower() != _FACE_PROMPT:
            log.warning(
                "yunet only detects '%s'; prompt '%s' will never match",
                _FACE_PROMPT, prompt,
            )
            return []

        import cv2
        import numpy as np

        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = img_bgr.shape[:2]

        self._detector.setInputSize((width, height))
        self._detector.setScoreThreshold(float(threshold))
        _, faces = self._detector.detect(img_bgr)

        detections: list[Detection] = []
        if faces is not None:
            for face in faces:
                # Each row: x, y, w, h, then 5 landmark (x, y) pairs,
                # then the confidence score — see FaceDetectorYN docs.
                x, y, w, h = face[:4]
                score = float(face[-1])
                x1 = max(0, int(round(x)))
                y1 = max(0, int(round(y)))
                x2 = min(width, int(round(x + w)))
                y2 = min(height, int(round(y + h)))
                detections.append(Detection(
                    label=_FACE_PROMPT,
                    score=score,
                    bbox=(x1, y1, x2, y2),
                ))

        log.debug("detect: %d detection(s) above threshold %.2f", len(detections), threshold)
        return detections
