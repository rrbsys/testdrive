"""SAM + Grounding DINO combo plugin (segmentation + detection)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..cache import cache_dir
from ..detection import Detection
from ..plugin import DetectorPlugin
from ..util import load_processor, load_model, download_file

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger("testdrive.models.samgd")

PLUGIN_API = 1

# SAM checkpoints aren't distributed through the HF hub — they're plain
# files on Meta's public CDN. vit_h (the default used below) is the
# largest/most accurate variant; vit_l and vit_b are smaller/faster.
SAM_CHECKPOINT_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

PLUGIN = {
    "id": "samgd",
    "name": "SAM + Grounding DINO",
    "version": "0.1.0",
    "api": PLUGIN_API,
    "description": "Grounding DINO for detection + SAM for segmentation.",
    "author": "Meta + IDEA Research",
    "homepage": "https://github.com/IDEA-Research/GroundingDINO + https://segment-anything.com",
    "license": "Apache-2.0 / BSD",
    "backend": "transformers + segment-anything",
    "hf_repo": "IDEA-Research/grounding-dino-base",
    "task": "detection + segmentation",
    "supports": ["text prompts", "boxes", "masks"],
    "requirements": [
        {"pip": "Pillow", "module": "PIL"},
        {"pip": "torch", "module": "torch"},
        {"pip": "numpy", "module": "numpy"},
        {"pip": "transformers", "module": "transformers"},
        {"pip": "segment-anything", "module": "segment_anything"},
    ],
    "sample_prompt": "green triangle",
    "test_threshold": "default",
}


class Plugin(DetectorPlugin):
    """SAM + Grounding DINO combo."""

    _gd_model: Any = None
    _sam_model: Any = None
    _sam_predictor: Any = None
    _processor: Any = None
    _device: str
    _initialized: bool = False

    def initialize(self) -> None:
        if self._initialized:
            return

        import torch
        from transformers import AutoModelForZeroShotObjectDetection
        from segment_anything import sam_model_registry, SamPredictor

        # Load Grounding DINO
        repo = self.manifest.hf_repo
        cd = cache_dir()
        self._processor = load_processor(repo, cd)
        self._gd_model = load_model(repo, cd, AutoModelForZeroShotObjectDetection)

        # Load SAM (ViT-H by default)
        sam_variant = "vit_h"
        log.info("loading SAM checkpoint (%s)...", sam_variant)
        checkpoint = download_file(SAM_CHECKPOINT_URLS[sam_variant], cd / "checkpoints")

        import warnings

        with warnings.catch_warnings():
            # segment_anything's own checkpoint loader calls torch.load()
            # without weights_only=True and triggers this every time; it's
            # third-party code we don't control, and the checkpoint comes
            # from Meta's official CDN over HTTPS, so it's expected noise
            # rather than something actionable here.
            warnings.filterwarnings("ignore", category=FutureWarning, message=r".*weights_only.*")
            self._sam_model = sam_model_registry[sam_variant](checkpoint=str(checkpoint))

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._gd_model = self._gd_model.to(self._device)
        self._sam_model = self._sam_model.to(self._device)
        self._sam_predictor = SamPredictor(self._sam_model)

        self._initialized = True
        log.info("SAM + Grounding DINO ready")

    def detect(
        self, image: "PILImage.Image", prompt: str, threshold: float = 0.3
    ) -> list[Detection]:
        """Detect ``prompt`` with Grounding DINO, then refine each box into
        a tight, mask-derived bounding box with SAM.

        The output schema (:class:`Detection`) is bbox-only — there's no
        mask field to return — so SAM's contribution here is a tighter,
        more accurate box than Grounding DINO's raw prediction, derived
        from the actual segmented region rather than the model's coarse
        box regression.
        """
        import numpy as np
        import torch

        if not self._initialized:
            self.initialize()

        # --- 1. Grounding DINO: candidate boxes for the prompt ---
        labels = [p.strip() for p in prompt.split(",") if p.strip()]
        text = ". ".join(labels) + "."
        log.debug("text prompt: %r  threshold=%.2f", text, threshold)

        inputs = self._processor(images=image, text=text, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            gd_outputs = self._gd_model(**inputs)

        target_sizes = [image.size[::-1]]  # (height, width)
        gd_result = self._processor.post_process_grounded_object_detection(
            gd_outputs,
            inputs["input_ids"],
            threshold=threshold,
            text_threshold=0.25,
            target_sizes=target_sizes,
        )[0]

        gd_boxes = gd_result["boxes"].cpu().numpy()
        gd_scores = gd_result["scores"].cpu().tolist()
        raw_labels = gd_result["text_labels"] if "text_labels" in gd_result else gd_result["labels"]

        if len(gd_boxes) == 0:
            return []

        # --- 2. SAM: refine each box into a tight, mask-derived bbox ---
        image_np = np.array(image.convert("RGB"))
        self._sam_predictor.set_image(image_np)

        input_boxes = torch.tensor(gd_boxes, dtype=torch.float32, device=self._device)
        transformed_boxes = self._sam_predictor.transform.apply_boxes_torch(
            input_boxes, image_np.shape[:2]
        )

        with torch.no_grad():
            masks, _iou_preds, _low_res = self._sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

        detections: list[Detection] = []
        for mask, score, label, gd_box in zip(masks, gd_scores, raw_labels, gd_boxes):
            label_text = label if isinstance(label, str) else str(label)
            if not label_text.strip():
                # Same empty-phrase edge case as the groundingdino plugin —
                # skip rather than emit an invalid empty-label detection.
                continue

            mask_np = mask[0].cpu().numpy()  # (H, W) bool
            ys, xs = np.where(mask_np)
            if len(xs) == 0 or len(ys) == 0:
                # SAM produced an empty mask for this box (rare); fall
                # back to Grounding DINO's raw box rather than dropping
                # the detection entirely.
                x1, y1, x2, y2 = gd_box
            else:
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

            detections.append(
                Detection(
                    label=label_text,
                    score=float(score),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        log.debug("detect: %d detection(s) above threshold %.2f", len(detections), threshold)
        return detections
