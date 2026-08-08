"""Grounding DINO detector plugin (transformers backend).

Previously used the official ``groundingdino-py`` package, which pulls in
a large, fragile dependency chain (including a protobuf requirement that
frequently conflicts with other installed packages) and needs manually
downloaded config/checkpoint files that this plugin never actually
shipped a path for. Grounding DINO has since been natively supported in
``transformers`` (``GroundingDinoForObjectDetection`` / ``AutoProcessor``),
so this plugin now uses that instead — no extra dependency beyond
``transformers`` itself, and it benefits from the shared ``load_processor``/
``load_model`` cache handling like every other plugin.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..cache import cache_dir
from ..detection import Detection
from ..plugin import DetectorPlugin
from ..util import load_processor, load_model

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger("testdrive.models.groundingdino")

PLUGIN_API = 1


def _parse_prompt(prompt: str) -> tuple[str, list[str]]:
    """Split a comma-separated prompt into Grounding DINO's expected text
    format plus the individual (lowercased) labels.

    Grounding DINO's vocabulary/tokenizer were trained on lowercase,
    period-separated captions built with a space on both sides of each
    period (``"cat . dog . bird ."``, the convention used by the
    original Grounding DINO demo scripts) — both the composed text and
    the returned labels are lowercased accordingly.

    ``"person"``        -> ``("person .", ["person"])``
    ``"Person, CAR"``    -> ``("person . car .", ["person", "car"])``
    """
    labels = [p.strip().lower() for p in prompt.split(",") if p.strip()]
    if not labels:
        labels = [prompt.strip().lower()]
    text = " . ".join(labels) + " ."
    return text, labels


PLUGIN = {
    "id": "groundingdino",
    "name": "Grounding DINO",
    "version": "0.2.0",
    "api": PLUGIN_API,
    "description": "Open-set object detector.",
    "author": "IDEA Research",
    "homepage": "https://huggingface.co/IDEA-Research/grounding-dino-base",
    "license": "Apache-2.0",
    "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
    "backend": "transformers",
    "hf_repo": "IDEA-Research/grounding-dino-base",
    "task": "zero-shot object detection",
    "supports": ["text prompts", "confidence threshold"],
    "requirements": [
        {"pip": "Pillow", "module": "PIL"},
        {"pip": "torch>=2.2,<2.5", "module": "torch"},
        {"pip": "transformers==4.50.3", "module": "transformers"},
    ],
    "sample_prompt": "red star",
    "test_threshold": "default",
    "pyenv": "framework",
}


class Plugin(DetectorPlugin):
    """Grounding DINO zero-shot object detector."""

    _processor: Any
    _model: Any
    _device: str
    _initialized: bool = False

    def initialize(self) -> None:
        if self._initialized:
            return

        import torch
        from transformers import AutoModelForZeroShotObjectDetection

        repo = self.manifest.hf_repo
        cd = cache_dir()

        log.info("loading Grounding DINO processor...")
        self._processor = load_processor(repo, cd, self.manifest.id)

        log.info("loading Grounding DINO model...")
        self._model = load_model(repo, cd, self.manifest.id, AutoModelForZeroShotObjectDetection)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()

        self._initialized = True
        log.info("Grounding DINO ready")

    def detect(
        self, image: "PILImage.Image", prompt: str, threshold: float = 0.3
    ) -> list[Detection]:
        """Run Grounding DINO inference and return bounding-box detections.

        Parameters
        ----------
        image:
            RGB ``PIL.Image`` loaded by the framework.
        prompt:
            Free-text description of what to look for. Use commas to
            specify multiple labels: ``"cat, dog, bird"``. Grounding
            DINO expects each label period-terminated internally; that
            formatting is handled here.
        threshold:
            Minimum box confidence score (0-1) to include a detection.
        """
        import torch

        if not self._initialized:
            self.initialize()

        # Grounding DINO's text format is period-separated phrases,
        # e.g. "cat . dog . bird ." — build that from a comma-separated
        # (or single) prompt.
        text, labels = _parse_prompt(prompt)
        log.debug("text prompt: %r  threshold=%.2f", text, threshold)

        inputs = self._processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = [image.size[::-1]]  # (height, width)
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=threshold,
            text_threshold=0.25,
            target_sizes=target_sizes,
        )

        detections: list[Detection] = []
        result = results[0]  # single image → first (only) element

        boxes = result["boxes"].cpu().tolist()
        scores = result["scores"].cpu().tolist()
        # Newer transformers return "text_labels" (strings) and deprecate
        # "labels" as text (moving it to integer class ids); older ones
        # only have "labels" as strings. Prefer text_labels when present —
        # checked via 'in' rather than result.get("text_labels",
        # result.get("labels")), since that always evaluates the "labels"
        # default eagerly and triggers the deprecation warning every call
        # even when text_labels is available.
        raw_labels = result["text_labels"] if "text_labels" in result else result["labels"]

        for box, score, label in zip(boxes, scores, raw_labels):
            label_text = label if isinstance(label, str) else str(label)
            if not label_text.strip():
                # Grounding DINO occasionally decodes an empty phrase for a
                # box whose matched token span didn't align to a full word
                # (more common at low text_threshold, as in the self-test).
                # That's not a usable label, so skip it rather than emit an
                # invalid empty-label detection.
                continue
            x1, y1, x2, y2 = box
            detections.append(
                Detection(
                    label=label_text,
                    score=float(score),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        log.debug("detect: %d detection(s) above threshold %.2f", len(detections), threshold)
        return detections
