"""Plugin discovery and loading.

Plugins are discovered purely by scanning ``testdrive/models/*.py`` -
there is no registry to edit. Each module is imported, its manifest is
validated, and its declared ``PLUGIN_API`` is checked against
``FRAMEWORK_API_VERSION`` before it is considered usable.

Parked plugins living in the sibling ``testdrive/models_inactive/``
package are deliberately invisible to that scan (see
``iter_loadable_plugins``) — they never appear in ``-L``, ``-M '*'``,
``-T '*'``, ``-TT '*'``, or anywhere else that means "every plugin".
They're still reachable one at a time, though, via an explicit
``../models_inactive/<name>``-shaped reference — see ``load_plugin``.
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from . import models as models_pkg
from . import models_inactive as models_inactive_pkg
from .detection import FRAMEWORK_API_VERSION, PluginManifest
from .plugin import DetectorPlugin

log = logging.getLogger("testdrive.pluginloader")

# The directory name (not a full path) that marks an explicit
# reference to a parked plugin — see _parse_inactive_ref.
_INACTIVE_PACKAGE_NAME = "models_inactive"


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
    """Load a single plugin, normally by its manifest id.

    *plugin_id* is usually a manifest id (e.g. ``"owlv2"``), resolved
    by scanning ``models/`` exactly like ``iter_loadable_plugins()``.

    It may also be an explicit reference to a *parked* plugin sitting
    in ``models_inactive/`` instead — shaped like
    ``../models_inactive/molmo`` (or ``..\\models_inactive\\molmo`` on
    Windows; either separator works, since this is parsed with
    :mod:`pathlib` rather than matched as a literal string) — naming
    that module's filename, not necessarily its manifest id. This is
    the only way to reach a parked plugin: they're never scanned by
    ``iter_loadable_plugins``, so they never show up in ``-L``,
    ``-M '*'``, ``-T '*'``, ``-TT '*'``, or a bare ``'*'`` detect run —
    reaching one always means spelling out this exact reference, on
    purpose, from any of ``-M``/``-T``/``-TT``/a normal detect run.

    Raises
    ------
    PluginLoadError
        If no module produces a plugin with the given id (or, for an
        inactive-plugin reference, if that specific module doesn't
        exist or fails validation).
    """
    inactive_name = _parse_inactive_ref(plugin_id)
    if inactive_name is not None:
        return _load_module(inactive_name, package=models_inactive_pkg)

    for module_info in pkgutil.iter_modules(models_pkg.__path__):
        module_name = module_info.name
        if module_name.startswith("_"):
            continue
        try:
            loaded = _load_module(module_name)
        except PluginLoadError as exc:
            # Previously silent (unlike iter_loadable_plugins(), which
            # already logs this) - a module failing to import here used
            # to surface only as a generic "no plugin found with id
            # '<id>'" once every module had been tried and none matched,
            # with no trace of *why* any of them failed.
            log.warning("skipping plugin '%s': %s", module_name, exc)
            continue
        if loaded.manifest.id == plugin_id:
            return loaded
    raise PluginLoadError(f"no plugin found with id '{plugin_id}'")


def _parse_inactive_ref(plugin_id: str) -> str | None:
    """Recognise an explicit ``../models_inactive/<name>``-shaped
    reference to a parked plugin (see ``load_plugin``), returning the
    module basename if *plugin_id* has that shape, else ``None``.

    Parsed with :class:`pathlib.Path` rather than matched as a literal
    string so either slash style works, matching however the local OS
    itself accepts paths (backslash-or-forward-slash on Windows,
    forward-slash on everything else).
    """
    parts = Path(plugin_id).parts
    if len(parts) == 3 and parts[0] == os.pardir and parts[1] == _INACTIVE_PACKAGE_NAME:
        return parts[2]
    return None


def display_name(plugin_id: str) -> str:
    """The bare name to use for filesystem lookups (e.g. an examples/
    subdirectory) and user-facing display — the trailing basename for
    an explicit ``../models_inactive/<name>`` reference (see
    ``load_plugin``), or *plugin_id* unchanged for an ordinary manifest
    id. Callers that build a path out of a plugin identifier (rather
    than just passing it straight to ``load_plugin``) should use this
    instead of the raw value, or an inactive-plugin reference like
    ``../models_inactive/samgd`` ends up literally embedded in the
    path (e.g. ``examples/../models_inactive/samgd/...``, which
    resolves outside ``examples/`` entirely rather than to it).
    """
    return _parse_inactive_ref(plugin_id) or plugin_id


def _load_module(module_name: str, package: ModuleType = models_pkg) -> LoadedPlugin:
    full_name = f"{package.__name__}.{module_name}"
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
