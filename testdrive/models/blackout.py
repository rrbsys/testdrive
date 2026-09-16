"""Blackout plugin — always detects the entire image as a single match.

This is a trivial "full-image redaction" detector.  It returns exactly one
Detection whose bbox covers the whole image, regardless of the prompt.
The framework's normal redaction path then paints the entire image with
the configured background colour and overlays the configured text
(centred and auto-scaled to 90 % of the image area).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..detection import Detection
from ..plugin import DetectorPlugin

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger("testdrive.models.blackout")

PLUGIN_API = 1

PLUGIN = {
    "id": "blackout",
    "name": "Blackout",
    "version": "0.1.0",
    "api": PLUGIN_API,
    "description": (
        "Trivial full-image detector.  Always returns a single match that "
        "covers the entire image, so the redacted output is a solid-colour "
        "canvas with the configured overlay text."
    ),
    "author": "Testdrive",
    "homepage": "",
    "license": "MIT",
    "license_url": "",
    "backend": "none",
    "hf_repo": "",
    "task": "full-image redaction",
    "supports": ["full-image blackout", "no model required"],
    "requirements": [
        {"pip": "Pillow", "module": "PIL"},
    ],
    "sample_prompt": "blackout",
    "test_threshold": "default",
    "pyenv": "framework",
    # Redaction style (consumed by the framework's redact path when present)
    "redact_text": "The quick brown fox jumps over the lazy dog",
    "redact_bgcolor": "lightgrey",
}


class Plugin(DetectorPlugin):
    """Always reports one detection covering the whole image."""

    _initialized: bool = False

    def initialize(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        log.info("Blackout ready (no model)")

    def detect(
        self, image: "PILImage.Image", prompt: str, threshold: float = 0.3
    ) -> list[Detection]:
        """Return a single Detection whose bbox is the entire image."""
        if not self._initialized:
            self.initialize()

        width, height = image.size
        # One match only: the whole image
        return [
            Detection(
                label="blackout",
                score=1.0,
                bbox=(0, 0, width, height),
            )
        ]
