"""OWLv2 detector plugin.

Full inference implementation using Hugging Face ``transformers``.

Model : google/owlv2-base-patch16-ensemble
Task  : zero-shot object detection (text-prompted bounding boxes)

Prompt syntax
-------------
A single string, optionally comma-separated for multiple labels:

    "cat"
    "cat, dog"
    "person, bicycle, car"

Each part becomes a separate text query. The model scores all queries
against the image in a single forward pass.

Cache
-----
Model files are stored under the directory resolved by
``testdrive.cache.cache_dir()`` — set ``TESTDRIVE_CACHE`` to override.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..cache import cache_dir
from ..detection import Detection
from ..plugin import DetectorPlugin
from ..util import load_processor, load_model

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

log = logging.getLogger("testdrive.models.owlv2")

PLUGIN_API = 1

PLUGIN = {
    "id": "owlv2",
    "name": "OWLv2",
    "version": "0.1.0",
    "api": PLUGIN_API,
    "description": "Scaling open-vocabulary object detection with self-training.",
    "author": "Google Research",
    "homepage": "https://huggingface.co/docs/transformers/model_doc/owlv2",
    "license": "Apache-2.0",
    "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
    "backend": "transformers",
    "hf_repo": "google/owlv2-base-patch16-ensemble",
    "task": "zero-shot object detection",
    "supports": ["text prompts", "confidence threshold", "multiple labels (comma-separated)"],
    "requirements": [
        {"pip": "Pillow", "module": "PIL"},
        {"pip": "torch>=2.2,<2.5", "module": "torch"},
        {"pip": "transformers==4.50.3", "module": "transformers"},
        {"pip": "scipy", "module": "scipy"},
    ],
    "sample_prompt": "purple pentagon",
    "test_threshold": "default",
    "pyenv": "framework",
}


def _parse_prompt(prompt: str) -> list[str]:
    """Split a comma-separated prompt string into a list of text queries.

    ``"cat, dog"`` → ``["cat", "dog"]``
    ``"person"``   → ``["person"]``
    """
    parts = [p.strip() for p in prompt.split(",") if p.strip()]
    return parts if parts else [prompt.strip()]


# _load_image_processor removed — direct Owlv2Processor.from_pretrained is reliable per model card.


class Plugin(DetectorPlugin):
    """OWLv2 zero-shot object detector.

    Lazily loads processor and model on the first call to
    :meth:`initialize`. After that, :meth:`detect` can be called
    repeatedly without re-loading weights.
    """

    _processor: "Owlv2Processor"
    _model: "Owlv2ForObjectDetection"
    _device: str
    _initialized: bool = False

    def initialize(self) -> None:
        if self._initialized:
            return

        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        repo = self.manifest.hf_repo
        cd = cache_dir()
        log.info("cache dir : %s", cd)

        log.info("loading OWLv2 processor...")
        self._processor = load_processor(repo, cd, self.manifest.id, processor_class=Owlv2Processor)

        log.info("loading OWLv2 model...")
        self._model = load_model(repo, cd, self.manifest.id, Owlv2ForObjectDetection)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("using device: %s", self._device)

        self._model = self._model.to(self._device)
        self._model.eval()

        self._initialized = True
        log.info("OWLv2 ready")

    def initialize_DISABLED(self) -> None:
        """Download (or load from cache) processor and model weights.

        Constructs the processor manually from its two sub-components
        (image processor + ``CLIPTokenizerFast``) to sidestep the
        auto-detection path in ``Owlv2Processor.from_pretrained()``, which
        fails on transformers versions where ``Owlv2ImageProcessor`` is not
        registered in the auto-mapping.

        The image processor is loaded with a version-aware fallback chain:
          1. ``Owlv2ImageProcessor``  – correct name, not always registered
          2. ``OwlViTImageProcessor`` – predecessor; identical pipeline,
             reliably registered across all current transformers versions
          3. ``AutoImageProcessor``   – generic fallback; reads the config
             file and picks whatever class is available
        """
        if self._initialized:
            return

        import torch
        from transformers import (
            Owlv2ForObjectDetection,
            Owlv2Processor,
        )

        repo = self.manifest.hf_repo
        cd = cache_dir()
        log.info("cache dir : %s", cd)

        log.info("loading OWLv2 processor from '%s' ...", repo)
        self._processor = load_processor(repo, cd, self.manifest.id, processor_class=Owlv2Processor)

        log.info("loading OWLv2 model from '%s' ...", repo)
        self._model = load_model(repo, cd, self.manifest.id, Owlv2ForObjectDetection)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("using device: %s", self._device)

        self._model = self._model.to(self._device)
        self._model.eval()

        self._initialized = True
        log.info("OWLv2 ready")

    def detect(
        self,
        image: "PILImage.Image",
        prompt: str,
        threshold: float = 0.3,
    ) -> list[Detection]:
        """Run OWLv2 inference and return bounding-box detections.

        Parameters
        ----------
        image:
            RGB ``PIL.Image`` loaded by the framework.
        prompt:
            Free-text description of what to look for. Use commas to
            specify multiple labels: ``"cat, dog, bird"``.
        threshold:
            Minimum confidence score (0–1) to include a detection.
        """
        import torch

        if not self._initialized:
            self.initialize()

        text_queries = _parse_prompt(prompt)
        log.debug("text queries: %s  threshold=%.2f", text_queries, threshold)

        # Processor expects a list-of-lists: [[q1, q2, ...]] for one image
        inputs = self._processor(
            text=[text_queries],
            images=image,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process: convert raw logits → boxes in pixel coordinates.
        # post_process_object_detection is deprecated in favor of this
        # method, which also returns text_labels directly instead of
        # requiring a manual index -> text_queries[idx] lookup.
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
