"""YOLO11 detector plugin (Ultralytics).

Unlike every other plugin in this framework, YOLO11 is a *fixed*-
vocabulary detector — it always predicts from the 80 COCO classes it
was trained on, regardless of what text you give it. There's no
open-vocabulary text understanding at inference time. ``prompt`` here
instead selects which of those 80 known classes to report
(comma-separated, matched case-insensitively); anything typed that
isn't one of them simply never matches, with a warning logged.

In exchange for giving up open-vocabulary flexibility, YOLO11 is
dramatically faster and lighter than every other plugin here — even
the largest variant (yolo11x) is well under a gigabyte, and the
smallest (yolo11n) runs in real time on CPU. See ``PLUGIN["models"]``
for the full size lineup and ``--model`` on the CLI to pick one other
than the default.
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

log = logging.getLogger("testdrive.models.yolo11")

PLUGIN_API = 1

# Exactly what `YOLO("yolo11m.pt").names.values()` returns — all five
# sizes share this identical 80-class COCO label set. Listed statically
# here so `-M yolo11` can show it without downloading or importing
# anything.
_COCO_80_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
assert len(_COCO_80_CLASSES) == 80

PLUGIN = {
    "id": "yolo11",
    "name": "YOLO11",
    "version": "0.1.0",
    "api": PLUGIN_API,
    "description": (
        "Fast, fixed-vocabulary (COCO-80) real-time object detector from "
        "Ultralytics. Unlike this framework's other plugins, it does not "
        "understand free-text prompts \u2014 'prompt' selects which of its 80 "
        "trained classes to report, not an open-vocabulary description."
    ),
    "author": "Ultralytics",
    "homepage": "https://docs.ultralytics.com/models/yolo11/",
    "license": "AGPL-3.0",
    "license_url": "https://www.gnu.org/licenses/agpl-3.0.html",
    "backend": "ultralytics",
    "hf_repo": "Ultralytics/YOLO11",
    "task": "fixed-vocabulary object detection",
    "supports": ["80 fixed COCO classes", "multiple model sizes (--model)", "fast CPU inference"],
    "requirements": [
        {"pip": "Pillow", "module": "PIL"},
        {"pip": "torch>=2.2,<2.5", "module": "torch"},
        {"pip": "ultralytics>=8.3", "module": "ultralytics"},
    ],
    "sample_prompt": "person",
    "test_threshold": "default",
    "pyenv": "newenv",
    "models": ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"],
    "model": "yolo11m",
    "classes": _COCO_80_CLASSES,
}


class Plugin(DetectorPlugin):
    """YOLO11 fixed-vocabulary object detector."""

    _model: Any
    _device: str
    _initialized: bool = False

    def initialize(self) -> None:
        if self._initialized:
            return

        import torch
        from ultralytics import YOLO

        model_name = self.manifest.model or "yolo11m"
        if model_name not in self.manifest.models:
            raise ValueError(
                f"unknown YOLO11 model '{model_name}'; choices: {', '.join(self.manifest.models)}"
            )

        cd = cache_dir()
        # One file per size, not a full repo snapshot (ensure_local_repo
        # would pull down all five .pt files via snapshot_download just
        # to use one of them) — download_file() fetches exactly the one
        # checkpoint requested, still through the same cache-discipline
        # gate (set_downloads_allowed) as every other plugin.
        checkpoint_url = (
            f"https://huggingface.co/{self.manifest.hf_repo}/resolve/main/{model_name}.pt"
        )
        checkpoint = download_file(
            checkpoint_url,
            cd / "checkpoints" / "yolo11",
            filename=f"{model_name}.pt",
        )

        log.info("loading YOLO11 (%s)...", model_name)
        self._model = YOLO(str(checkpoint))

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

        self._initialized = True
        log.info("YOLO11 (%s) ready", model_name)

    def detect(
        self, image: "PILImage.Image", prompt: str, threshold: float = 0.3
    ) -> list[Detection]:
        """Detect objects among YOLO11's fixed 80 COCO classes.

        ``prompt`` is a comma-separated allow-list of class names to
        report (case-insensitive) — not an open-vocabulary description,
        see the module docstring. An empty allow-list (e.g. an empty
        prompt) reports everything YOLO11 finds, unfiltered. Anything
        YOLO11 detects outside a non-empty allow-list is filtered out
        here, not "not found by the model" — the model itself always
        looks for all 80 classes regardless of prompt.
        """
        if not self._initialized:
            self.initialize()

        wanted = {p.strip().lower() for p in prompt.split(",") if p.strip()}
        known = {c.lower() for c in _COCO_80_CLASSES}
        unknown = wanted - known
        if unknown:
            log.warning(
                "prompt term(s) %s are not among YOLO11's 80 known classes and will never "
                "match; see `testdrive -M yolo11` for the full list",
                sorted(unknown),
            )

        results = self._model.predict(image, conf=threshold, device=self._device, verbose=False)

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                label = result.names[int(box.cls[0])]
                if wanted and label.lower() not in wanted:
                    continue
                score = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        label=label,
                        score=score,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                    )
                )

        log.debug("detect: %d detection(s) above threshold %.2f", len(detections), threshold)
        return detections
