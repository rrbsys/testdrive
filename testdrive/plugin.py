"""Abstract base class that all detector plugins must implement.

A plugin module exposes:

* ``PLUGIN_API`` (int) - the plugin API version it targets.
* ``PLUGIN`` (dict)    - manifest fields (see ``detection.PluginManifest``).
* ``Plugin`` (class)   - a subclass of :class:`DetectorPlugin`.

No plugin parses CLI args, draws boxes, or writes files - that's all
owned by the framework.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any

from .detection import Detection, PluginManifest


class DetectorPlugin(ABC):
    """Base class for all testdrive detector plugins."""

    #: Set by the loader to a :class:`PluginManifest` instance.
    manifest: PluginManifest

    @abstractmethod
    def initialize(self) -> None:
        """Load model weights / set up backend resources (called once, lazily)."""
        raise NotImplementedError

    @abstractmethod
    def detect(self, image: Any, prompt: str, threshold: float = 0.3) -> list[Detection]:
        """Run inference and return a list of :class:`Detection`."""
        raise NotImplementedError

    def is_installed(self) -> tuple[bool, list[str]]:
        """Return ``(installed, missing_pip_packages)``.

        Each entry in ``manifest.requirements`` is a dict with keys:
          ``"pip"``    - the pip install name (e.g. ``"Pillow"``)
          ``"module"`` - the Python import name to probe (e.g. ``"PIL"``)

        Returns the pip names of any packages that cannot be imported.
        Plugins may override this for more precise checks.
        """
        missing: list[str] = []
        for req in self.manifest.requirements:
            module_name = req["module"]
            try:
                importlib.import_module(module_name)
            except ImportError:
                missing.append(req["pip"])
        return (len(missing) == 0, missing)
