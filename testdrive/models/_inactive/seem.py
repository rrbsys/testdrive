"""SEEM (Segment Everything Everywhere All At Once) plugin.

SEEM was never ported into ``transformers``. The only Hugging Face repo
for it (``xdecoder/SEEM``) contains raw ``.pt`` checkpoint files with no
``config.json``/``preprocessor_config.json``/modeling code — there is
nothing for ``AutoProcessor``/``AutoModel*`` to load, even with
``trust_remote_code=True``. Using it for real would mean vendoring the
original ``UX-Decoder/Segment-Everything-Everywhere-All-At-Once`` GitHub
code and its own custom loader, which is out of scope here (and exactly
the kind of fragile custom-package dependency the ``groundingdino``
plugin was moved away from).

This plugin is therefore marked as not-installed with a clear reason
via ``is_installed()``, so ``-T``/``-I`` and normal detection runs fail
fast with an honest explanation instead of a confusing error surfaced
from deep inside ``transformers``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ...detection import Detection
from ...plugin import DetectorPlugin

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger("testdrive.models.seem")

PLUGIN_API = 1

_UNIMPLEMENTED_REASON = (
    "seem (unimplemented: no transformers-compatible SEEM model exists — "
    "see plugin homepage for details)"
)

PLUGIN = {
    "id": "seem",
    "name": "SEEM",
    "version": "0.1.0",
    "api": PLUGIN_API,
    "description": "Segment Everything Everywhere All At Once. NOT IMPLEMENTED: "
    "no transformers-compatible model repo exists for SEEM.",
    "author": "UX-Decoder",
    "homepage": "https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once",
    "license": "Apache-2.0",
    "backend": "transformers",
    "hf_repo": "",
    "task": "panoptic segmentation",
    "supports": ["text prompts", "masks"],
    "requirements": [
        {"pip": "Pillow", "module": "PIL"},
        {"pip": "torch", "module": "torch"},
        {"pip": "transformers", "module": "transformers"},
    ],
    "sample_prompt": "yellow circle",
    "test_threshold": "default",
}


class Plugin(DetectorPlugin):
    """SEEM universal segmentation — currently unimplemented, see module docstring."""

    _model: Any
    _processor: Any
    _device: str
    _initialized: bool = False

    def is_installed(self) -> tuple[bool, list[str]]:
        # Always reports "not installed": there's no real dependency to
        # install here, just no viable model backend (see module
        # docstring). This makes -T/-I and normal detect runs fail fast
        # with a clear message instead of a deep, confusing HF error.
        return (False, [_UNIMPLEMENTED_REASON])

    def initialize(self) -> None:
        raise NotImplementedError(_UNIMPLEMENTED_REASON)

    def detect(
        self, image: "PILImage.Image", prompt: str, threshold: float = 0.3
    ) -> list[Detection]:
        raise NotImplementedError(_UNIMPLEMENTED_REASON)
