"""Molmo detector plugin — PARKED, not currently discovered.

This lives in ``models/_inactive/`` (see ``pluginloader.
iter_loadable_plugins`` for why that's invisible to the framework), so
it is intentionally *not* loaded by the framework right now. Move it
back to ``models/`` (and fix its relative imports from ``...`` back to
``..`` — see git history / molmo7b.py's note for why that matters) if
you want to reactivate it.

Points at MolmoE-1B-0924 (mixture-of-experts, ~1B *active* parameters
per token) rather than the larger dense Molmo-7B-D (parked separately
as ``molmo7b.py``) — chosen on the assumption that fewer active
parameters would mean a lighter, faster plugin. That assumption was
wrong in the way that matters most for a CLI tool: MoE models store
*every* expert on disk regardless of how many activate per token, so
the download is essentially the same size either way — this "slim"
1B-active model still downloads ~29GB of fp32 weights.

Concretely: ``-TT molmo`` did eventually **pass** (confirmed on
real hardware, Wine/macOS) — but took roughly 11 minutes for one
synthetic-image detection, after an initialize() step that dwarfs that.
That's not a hard failure, but it's also not something a "try before
you install anything big" CLI tool should default to running. Parked
rather than deleted so the option stays available for anyone with the
patience, a GPU, or a genuine need for Molmo's pointing/counting
capabilities specifically.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...cache import cache_dir
from ...detection import Detection
from ...plugin import DetectorPlugin
from ...util import load_processor, load_model

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger("testdrive.models.molmo")

PLUGIN_API = 1

PLUGIN = {
    "id": "molmo",
    "name": "MolmoE",
    "version": "0.2.0",
    "api": PLUGIN_API,
    "description": (
        "Multimodal open-vocabulary detection from AllenAI (mixture-of-experts, "
        "~1B active params — see models/_inactive/molmo7b.py for the larger dense "
        "7B variant if you have the hardware for it)."
    ),
    "author": "AllenAI",
    "homepage": "https://huggingface.co/allenai/MolmoE-1B-0924",
    "license": "Apache-2.0",
    "backend": "transformers",
    "hf_repo": "allenai/MolmoE-1B-0924",
    "task": "multi-modal detection",
    "supports": ["text prompts", "points (approximated as boxes)", "detailed captions"],
    "requirements": [
        {"pip": "Pillow", "module": "PIL"},
        {"pip": "torch", "module": "torch"},
        {"pip": "transformers", "module": "transformers"},
    ],
    "sample_prompt": "largest circle",
    "test_threshold": "default",
    "pyenv": "framework",
}


def _parse_molmo_points(text: str) -> list[tuple[float, float, str]]:
    """Parse Molmo's pointing output into ``(x_pct, y_pct, label)`` triples.

    Molmo doesn't emit boxes — it's trained to point, replying to a
    "Point to X" prompt with markup like::

        <point x="45.3" y="63.2" alt="the cat">the cat</point>

    or, for multiple instances of the same thing::

        <points x1="12.1" y1="30.4" x2="70.2" y2="41.0" alt="dogs">dogs</points>

    Coordinates are percentages (0-100) of image width/height, not pixel
    coordinates.
    """
    import re

    results: list[tuple[float, float, str]] = []

    for m in re.finditer(r'<point\s+x="([\d.]+)"\s+y="([\d.]+)"[^>]*alt="([^"]*)"[^>]*>', text):
        x, y, alt = m.groups()
        results.append((float(x), float(y), alt))

    for m in re.finditer(r'<points\s+([^>]*?)alt="([^"]*)"[^>]*>', text):
        attrs, alt = m.groups()
        xs = {int(idx): float(val) for idx, val in re.findall(r'x(\d+)="([\d.]+)"', attrs)}
        ys = {int(idx): float(val) for idx, val in re.findall(r'y(\d+)="([\d.]+)"', attrs)}
        for idx in sorted(xs):
            if idx in ys:
                results.append((xs[idx], ys[idx], alt))

    return results


class Plugin(DetectorPlugin):
    """Molmo multimodal model."""

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

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Loading in bfloat16 rather than the default fp32 roughly halves
        # memory/compute vs. fp32, at the standard small precision-loss
        # cost — same numeric range as fp32 (unlike float16, no overflow
        # risk), a reasonable trade for a detection tool. Matters less
        # here than it did for the dense 7B variant (this MoE model only
        # activates ~1B params per token to begin with), but there's no
        # reason not to. GPUs load in their native dtype instead.
        dtype = torch.bfloat16 if self._device == "cpu" else "auto"

        self._processor = load_processor(repo, cd, self.manifest.id, trust_remote_code=True)
        self._model = load_model(
            repo,
            cd,
            self.manifest.id,
            AutoModelForCausalLM,
            trust_remote_code=True,
            torch_dtype=dtype,
        )

        self._model = self._model.to(self._device)
        self._model.eval()

        self._initialized = True
        log.info("Molmo ready")

    def detect(
        self, image: "PILImage.Image", prompt: str, threshold: float = 0.3
    ) -> list[Detection]:
        """Detect ``prompt`` via Molmo's pointing capability.

        Molmo is trained to *point* at things, not draw boxes — a
        "Point to X" prompt gets back ``<point x=".." y=".." alt="X">``
        (or ``<points ...>`` for several instances). Since
        :class:`Detection` needs a bbox, each point gets turned into a
        small box centered on it (a fixed percentage of the image's
        shorter side) rather than a box that actually traces the
        object's extent — this is an approximation, not a real
        detection box.

        Like Florence-2's phrase grounding, Molmo doesn't emit a
        confidence score for points, so ``threshold`` has nothing to
        filter against and every result gets ``score=1.0``.
        """
        import torch
        from transformers import GenerationConfig

        if not self._initialized:
            self.initialize()

        text_prompt = f"Point to {prompt}"
        inputs = self._processor.process(images=[image], text=text_prompt)

        # The model may be loaded in a non-default dtype (bfloat16 on
        # CPU, see initialize()), but processor.process() always
        # produces float32 tensors — so floating-point inputs (pixel
        # values etc.) need casting to match, or the model's matmuls
        # fail with a dtype mismatch. Integer tensors (input_ids, token
        # indices, ...) must NOT be cast — only floating-point ones.
        model_dtype = next(self._model.parameters()).dtype
        inputs = {
            k: (
                v.to(self._device, dtype=model_dtype)
                if v.is_floating_point()
                else v.to(self._device)
            ).unsqueeze(0)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            output = self._model.generate_from_batch(
                inputs,
                # Pointing responses are short — a handful of
                # <point .../> or <points .../> tags at most — so there's
                # no reason to budget for 300 tokens.
                GenerationConfig(max_new_tokens=100, stop_strings="<|endoftext|>"),
                tokenizer=self._processor.tokenizer,
            )

        generated_tokens = output[0, inputs["input_ids"].size(1) :]
        generated_text = self._processor.tokenizer.decode(
            generated_tokens, skip_special_tokens=True
        )
        log.debug("Molmo generated: %r", generated_text)

        points = _parse_molmo_points(generated_text)

        width, height = image.size
        radius_px = max(6, int(0.04 * min(width, height)))

        detections: list[Detection] = []
        for x_pct, y_pct, label in points:
            cx = x_pct / 100.0 * width
            cy = y_pct / 100.0 * height
            x1 = max(0, int(cx - radius_px))
            y1 = max(0, int(cy - radius_px))
            x2 = min(width, int(cx + radius_px))
            y2 = min(height, int(cy + radius_px))
            label_text = label.strip() if label and label.strip() else prompt
            detections.append(Detection(label=label_text, score=1.0, bbox=(x1, y1, x2, y2)))

        log.debug("detect: %d point(s) -> %d detection(s)", len(points), len(detections))
        return detections
