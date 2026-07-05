"""OWL-ViT detector plugin."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..cache import cache_dir
from ..detection import Detection
from ..plugin import DetectorPlugin
from ..util import load_processor, load_model

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger("testdrive.models.owlvit")

PLUGIN_API = 1

PLUGIN = {
    "id": "owlvit",
    "name": "OWL-ViT",
    "version": "0.1.0",
    "api": PLUGIN_API,
    "description": "Open-vocabulary object detection with OWL-ViT.",
    "author": "Google",
    "homepage": "https://huggingface.co/google/owlvit-base-patch32",
    "license": "Apache-2.0",
    "backend": "transformers",
    "hf_repo": "google/owlvit-base-patch32",
    "task": "zero-shot object detection",
    "supports": ["text prompts", "confidence threshold", "multiple labels"],
    "requirements": [
        {"pip": "Pillow", "module": "PIL"},
        {"pip": "torch", "module": "torch"},
        {"pip": "transformers", "module": "transformers"},
    ],
    "sample_prompt": "orange square",
    "test_threshold": "0.05",
}


def _parse_prompt(prompt: str) -> list[str]:
    """Split a comma-separated prompt string into a list of text queries.

    ``"cat, dog"`` → ``["cat", "dog"]``
    ``"person"``   → ``["person"]``
    """
    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    return parts if parts else [prompt.strip()]


class Plugin(DetectorPlugin):
    """OWL-ViT zero-shot object detector."""

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

        self._processor = load_processor(repo, cd)
        self._model = load_model(repo, cd, AutoModelForZeroShotObjectDetection)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()

        self._initialized = True
        log.info("OWL-ViT ready")

    def detect(
        self, image: "PILImage.Image", prompt: str, threshold: float = 0.3
    ) -> list[Detection]:
        """Run OWL-ViT inference and return bounding-box detections.

        Parameters
        ----------
        image:
            RGB ``PIL.Image`` loaded by the framework.
        prompt:
            Free-text description of what to look for. Use commas to
            specify multiple labels: ``"cat, dog, bird"``.
        threshold:
            Minimum confidence score (0-1) to include a detection.
        """
        import torch

        if not self._initialized:
            self.initialize()

        text_queries = _parse_prompt(prompt)
        log.debug("text queries: %s  threshold=%.2f", text_queries, threshold)

        # Processor expects a list-of-lists: [[q1, q2, ...]] for one image
        inputs = self._processor(text=[text_queries], images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # post_process_object_detection is deprecated in favor of this
        # method (same OwlViTProcessor base class OWLv2 uses), which also
        # returns text_labels directly instead of requiring a manual
        # index -> text_queries[idx] lookup.
        target_sizes = torch.tensor([image.size[::-1]], device=self._device)
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=threshold,
            target_sizes=target_sizes,
            text_labels=[text_queries],
        )

        detections: list[Detection] = []
        result = results[0]  # single image → first (only) element

        boxes = result["boxes"].cpu().tolist()
        scores = result["scores"].cpu().tolist()
        text_labels = result["text_labels"]

        if not boxes and log.isEnabledFor(logging.DEBUG):
            # Nothing cleared the threshold. Re-run post-processing with
            # threshold=0 just to peek at the top raw scores, so -vv can
            # tell "the model is genuinely unsure about this image" apart
            # from "something upstream is broken" — a 0.30 default cutoff
            # can quietly hide a 0.28 near-miss otherwise.
            debug_result = self._processor.post_process_grounded_object_detection(
                outputs=outputs,
                threshold=0.0,
                target_sizes=target_sizes,
                text_labels=[text_queries],
            )[0]
            top = sorted(
                zip(debug_result["scores"].tolist(), debug_result["text_labels"]),
                reverse=True,
            )[:5]
            log.debug(
                "no detections cleared threshold=%.2f; top raw scores (score, label): %s",
                threshold,
                top,
            )

        for box, score, label_text in zip(boxes, scores, text_labels):
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
