"""Florence-2 detector plugin."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..cache import cache_dir
from ..detection import Detection
from ..plugin import DetectorPlugin
from ..util import load_processor, load_model

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger("testdrive.models.florence2")

PLUGIN_API = 1

PLUGIN = {
    "id": "florence2",
    "name": "Florence-2",
    "version": "0.1.0",
    "api": PLUGIN_API,
    "description": "Powerful vision foundation model from Microsoft (object detection, captioning, etc.).",
    "author": "Microsoft",
    "homepage": "https://huggingface.co/microsoft/Florence-2-base",
    "license": "MIT",
    "backend": "transformers",
    "hf_repo": "microsoft/Florence-2-base",
    "task": "multi-task vision",
    "supports": ["text prompts", "object detection", "captioning"],
    "requirements": [
        {"pip": "Pillow", "module": "PIL"},
        {"pip": "torch", "module": "torch"},
        {"pip": "transformers", "module": "transformers"},
        {"pip": "einops", "module": "einops"},
        {"pip": "timm", "module": "timm"},
        # flash-attn is optional (for faster inference)
        # {"pip": "flash-attn", "module": "flash_attn", "optional": True},
    ],
    "sample_prompt": "blue triangle",
    "test_threshold": "default",
    "pyenv": "framework",
}


class Plugin(DetectorPlugin):
    """Florence-2 multi-task model."""

    _processor: Any
    _model: Any
    _device: str
    _initialized: bool = False

    def initialize(self) -> None:
        if self._initialized:
            return

        import torch
        from transformers import AutoModelForCausalLM

        repo = self.manifest.hf_repo
        cd = cache_dir()

        log.info("loading Florence-2 processor...")
        self._processor = load_processor(repo, cd, self.manifest.id, trust_remote_code=True)

        log.info("loading Florence-2 model...")
        self._model = load_model(repo, cd, self.manifest.id, AutoModelForCausalLM, trust_remote_code=True)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._model.eval()

        self._initialized = True
        log.info("Florence-2 ready")

    def detect(
        self, image: "PILImage.Image", prompt: str, threshold: float = 0.3
    ) -> list[Detection]:
        """Detect ``prompt`` using Florence-2's phrase-grounding task.

        Florence-2's plain ``<OD>`` task detects a *fixed* built-in
        vocabulary of object classes and — despite looking like it takes
        free text — errors if anything follows the task token ("Task
        token <OD> should be the only token in the text."). For
        prompt-driven detection, Florence-2 instead has a dedicated task,
        ``<CAPTION_TO_PHRASE_GROUNDING>``, which takes the phrase *after*
        the task token and returns boxes grounded to that phrase.

        Florence-2 doesn't emit a confidence score for grounded phrases
        (it's a deterministic beam-search decode, not a scored proposal
        list), so ``threshold`` has nothing to filter against and every
        returned box gets ``score=1.0``.
        """
        import torch

        if not self._initialized:
            self.initialize()

        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        text_input = task_prompt + prompt

        inputs = self._processor(text=text_input, images=image, return_tensors="pt").to(
            self._device
        )

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3,
            )

        generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed = self._processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )

        result = parsed.get(task_prompt, {})
        bboxes = result.get("bboxes", [])
        labels = result.get("labels", [])

        detections: list[Detection] = []
        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            label_text = label.strip() if isinstance(label, str) and label.strip() else prompt
            detections.append(
                Detection(
                    label=label_text,
                    score=1.0,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        log.debug("detect: %d detection(s)", len(detections))
        return detections
