"""Core data structures: Detection, PluginManifest, DetectionResult."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

#: Plugin API version implemented by this version of the framework.
FRAMEWORK_API_VERSION = 1


@dataclass
class Detection:
    """A single bounding-box detection returned by a plugin.

    ``bbox`` is (x1, y1, x2, y2) in absolute pixel coordinates.
    """

    label: str
    score: float
    bbox: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionResult:
    """The full result of one ``testdrive <plugin> <image> <prompt>`` run.

    Every plugin returns exactly the same result type - the framework
    is responsible for building this object from raw detections.
    """

    # --- input metadata ---
    image_path: Path
    image_size: tuple[int, int]  # (width, height)
    plugin_id: str
    plugin_name: str
    plugin_version: str
    prompt: str
    threshold: float

    #: Which model variant was actually used, for plugins that declare
    #: more than one (see PluginManifest.models / --model). Empty for
    #: every plugin with just one fixed model.
    model: str = ""

    # --- outputs ---
    detections: list[Detection] = field(default_factory=list)
    elapsed_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    # --- derived output paths (set by the framework after saving) ---
    matches_path: Path | None = None
    redacted_path: Path | None = None

    @property
    def count(self) -> int:
        return len(self.detections)

    def summary(self) -> str:
        """Return the human-readable CLI summary block."""
        lines = [
            f"Plugin   : {self.plugin_name} ({self.plugin_id})",
            f"Image    : {self.image_path.name}  ({self.image_size[0]}x{self.image_size[1]})",
            f'Prompt   : "{self.prompt}"',
            f"Threshold: {self.threshold:.2f}",
        ]
        if self.model:
            lines.append(f"Model    : {self.model}")
        lines += [
            f"Matches  : {self.count}",
            f"Time     : {self.elapsed_ms:.0f} ms",
        ]
        if self.matches_path:
            lines.append(f"Saved    : {self.matches_path}")
        if self.redacted_path:
            lines.append(f"           {self.redacted_path}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"Warning  : {w}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["image_path"] = str(self.image_path)
        d["matches_path"] = str(self.matches_path) if self.matches_path else None
        d["redacted_path"] = str(self.redacted_path) if self.redacted_path else None
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class PluginManifest:
    """Static metadata every plugin must declare via its ``PLUGIN`` dict."""

    id: str
    name: str = ""
    version: str = ""
    api: int = FRAMEWORK_API_VERSION

    description: str = ""
    author: str = ""
    homepage: str = ""

    license: str = ""
    license_url: str = ""

    backend: str = ""
    hf_repo: str = ""
    task: str = ""

    supports: list[str] = field(default_factory=list)

    #: Each entry: {"pip": <pip install argument>, "module": <import
    #: name to probe for is_installed()>}. "pip" now carries the real,
    #: exact version pin this plugin needs (e.g. "torch==2.0.0", or a
    #: "name @ git+https://..." VCS reference) — not just a bare
    #: package name — since it doubles as this plugin's actual
    #: auto-provisioning install list (see pyenv.run_pip_install,
    #: worker_pool._provision_plugin_env, and selftest.py's
    #: framework-pyenv auto-provision branch), not just something
    #: shown in a "missing dependency" hint message.
    requirements: list[dict[str, str]] = field(default_factory=list)

    #: Extra flags appended to every pip install command run for this
    #: plugin's requirements (auto-provisioning or the manual
    #: --no-auto-provision hint alike) — e.g. "--no-build-isolation"
    #: for a plugin whose build needs the current environment rather
    #: than pip's normal isolated build env (see seem.py, whose old
    #: GPU-only dependency chain needs this). Empty string (the
    #: default) adds nothing.
    pip_options: str = ""

    #: Source patches to apply mid-provisioning, after installing every
    #: non-VCS entry in ``requirements`` but before any "name @ git+..."
    #: entry (which is exactly the ordering a plugin like seem needs:
    #: patch an already-installed dependency's vendored header *before*
    #: building something from source against it). Each entry:
    #: {"target": <path relative to the environment's site-packages,
    #: e.g. "torch/include/c10/util/foo.h">, "find": <exact text to
    #: replace, must appear exactly once>, "replace": <replacement
    #: text>}. Idempotent — if "find" is already gone and "replace" is
    #: already present, re-provisioning is a safe no-op rather than an
    #: error. See pyenv.apply_source_patches.
    patches: list[dict[str, str]] = field(default_factory=list)

    sample_prompt: str = ""

    #: Threshold -TT should use for this plugin's example test, as a
    #: string. Either "default" (use the CLI's normal detect() default,
    #: _DEFAULT_THRESHOLD) or a numeric string like "0.05" for plugins
    #: that are known to need a different value on synthetic example
    #: images specifically (e.g. a much lower threshold on out-of-
    #: distribution flat-shaded shapes than a real photo would need).
    #: An explicit --threshold on the CLI always overrides this.
    test_threshold: str = "default"

    #: For plugins with multiple selectable model variants (e.g. YOLO11's
    #: n/s/m/l/x sizes): the full list of valid choices, and which one is
    #: the default. Empty ``models`` means the plugin has exactly one
    #: fixed model and --model doesn't apply to it (see
    #: DetectorPlugin.set_model_override).
    models: list[str] = field(default_factory=list)
    model: str = ""

    #: For fixed-vocabulary plugins (e.g. YOLO11's 80 COCO classes):
    #: every class the model can report. Empty for open-vocabulary
    #: plugins, which accept arbitrary prompt text instead.
    classes: list[str] = field(default_factory=list)

    #: Which named virtual environment this plugin runs in. Every
    #: plugin shares the "framework" environment (cache/pyenv/framework)
    #: by default — deliberately, since installing more and more
    #: plugins' dependencies into one Python installation is a
    #: guaranteed route to dependency hell (we've already hit this once:
    #: a transformers upgrade pulling in a whole TensorFlow chain that
    #: broke every other plugin). A plugin declaring its own pyenv value
    #: (e.g. "modelx") signals it needs isolation from that shared set;
    #: several plugins sharing a non-default name (e.g. "legacy-torch1")
    #: signals they're mutually compatible with each other but not with
    #: "framework".
    #:
    #: NOTE: a plugin declaring a non-"framework" pyenv actually runs in
    #: a worker subprocess using that environment's interpreter (see
    #: worker_pool.py / worker_main.py) — not just a label. The worker
    #: is spawned lazily on first use and kept alive for the rest of the
    #: invocation, so a loop-mode run over many images pays that
    #: plugin's initialize() cost once, not once per image.
    pyenv: str = "framework"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PluginManifest":
        if "id" not in data:
            raise ValueError("plugin manifest is missing required field: 'id'")
        known_fields = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
