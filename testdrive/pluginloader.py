"""Plugin discovery and loading.

Plugins are discovered purely by scanning ``testdrive/models/*.py`` -
there is no registry to edit. Each module is imported, its manifest is
validated, and its declared ``PLUGIN_API`` is checked against
``FRAMEWORK_API_VERSION`` before it is considered usable.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass

from . import models as models_pkg
from .detection import FRAMEWORK_API_VERSION, PluginManifest
from .plugin import DetectorPlugin

log = logging.getLogger("testdrive.pluginloader")


class PluginLoadError(Exception):
    """Raised when a plugin module exists but cannot be loaded."""


@dataclass
class LoadedPlugin:
    """A successfully discovered plugin, not yet instantiated/initialized."""

    module_name: str
    manifest: PluginManifest
    plugin_class: type[DetectorPlugin]

    def instantiate(self) -> DetectorPlugin:
        instance = self.plugin_class()
        instance.manifest = self.manifest
        return instance


def list_plugins() -> list[str]:
    """Return the sorted list of plugin ids found in ``models/``.

    Kept for backward compatibility with Commit 1's simple filename
    scan, but now reports validated manifest ids rather than raw
    filenames.
    """
    return sorted(p.manifest.id for p in iter_loadable_plugins())


def iter_loadable_plugins() -> list[LoadedPlugin]:
    """Import every module in ``testdrive.models`` and return the valid plugins.

    Modules that fail to import or fail manifest validation are
    skipped with a logged warning instead of crashing discovery for
    the whole framework.
    """
    plugins: list[LoadedPlugin] = []
    for module_info in pkgutil.iter_modules(models_pkg.__path__):
        module_name = module_info.name
        if module_name.startswith("_"):
            continue
        try:
            plugins.append(_load_module(module_name))
        except PluginLoadError as exc:
            log.warning("skipping plugin '%s': %s", module_name, exc)
    return plugins


def load_plugin(plugin_id: str) -> LoadedPlugin:
    """Load a single plugin by its manifest id.

    Raises
    ------
    PluginLoadError
        If no module produces a plugin with the given id, or the
        plugin fails validation.
    """
    for module_info in pkgutil.iter_modules(models_pkg.__path__):
        module_name = module_info.name
        if module_name.startswith("_"):
            continue
        try:
            loaded = _load_module(module_name)
        except PluginLoadError:
            continue
        if loaded.manifest.id == plugin_id:
            return loaded
    raise PluginLoadError(f"no plugin found with id '{plugin_id}'")


def _load_module(module_name: str) -> LoadedPlugin:
    full_name = f"{models_pkg.__name__}.{module_name}"
    try:
        module = importlib.import_module(full_name)
    except Exception as exc:  # noqa: BLE001 - any import error is a load error
        raise PluginLoadError(f"import failed: {exc}") from exc

    plugin_api = getattr(module, "PLUGIN_API", None)
    if plugin_api is None:
        raise PluginLoadError("missing module-level PLUGIN_API")

    if plugin_api != FRAMEWORK_API_VERSION:
        raise PluginLoadError(
            f"requires plugin API {plugin_api}, framework supports {FRAMEWORK_API_VERSION}"
        )

    raw_manifest = getattr(module, "PLUGIN", None)
    if raw_manifest is None:
        raise PluginLoadError("missing module-level PLUGIN manifest dict")

    try:
        manifest = PluginManifest.from_dict(raw_manifest)
    except ValueError as exc:
        raise PluginLoadError(f"invalid manifest: {exc}") from exc

    plugin_class = getattr(module, "Plugin", None)
    if plugin_class is None:
        raise PluginLoadError("missing module-level Plugin class")
    if not (isinstance(plugin_class, type) and issubclass(plugin_class, DetectorPlugin)):
        raise PluginLoadError("Plugin must be a subclass of DetectorPlugin")

    return LoadedPlugin(module_name=module_name, manifest=manifest, plugin_class=plugin_class)
