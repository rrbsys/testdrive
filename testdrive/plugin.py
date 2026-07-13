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
from dataclasses import replace
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

    def set_model_override(self, model: str) -> None:
        """Override which model variant to load (the CLI's --model).

        Only meaningful for plugins that declare more than one choice in
        ``manifest.models`` (e.g. YOLO11's n/s/m/l/x sizes) — plugins
        with a single fixed model reject any override. The default
        implementation validates ``model`` against ``manifest.models``
        and swaps ``manifest.model``; plugins just need to read
        ``self.manifest.model`` in ``initialize()`` to pick the right
        file/repo.

        Replaces ``self.manifest`` with a new instance (dataclasses.
        replace) rather than mutating it in place — ``LoadedPlugin.
        instantiate()`` hands every instance the *same* PluginManifest
        object by reference, so an in-place mutation would leak into
        every other instance of this plugin for the rest of the process.
        """
        if not self.manifest.models:
            raise ValueError(f"plugin '{self.manifest.id}' does not support --model (no models declared)")
        if model not in self.manifest.models:
            raise ValueError(
                f"invalid model '{model}' for plugin '{self.manifest.id}'; "
                f"choices: {', '.join(self.manifest.models)}"
            )
        self.manifest = replace(self.manifest, model=model)
