"""SEEM (Segment Everything Everywhere All At Once) plugin.

**Status: scaffolded, not yet real.** The plumbing this file needs —
its own isolated ``seem`` pyenv (see ``pyenv:`` below and
``PluginManifest.pyenv``), vendoring SEEM's own modeling code via
``ensure_git_repo`` (see ``util.py``), and downloading its raw ``.pt``
checkpoint via ``download_file`` — is real and follows this
framework's normal conventions throughout. What's *not* verified yet
is the actual model-construction and inference calls inside
``initialize()``/``detect()`` below: they're a best-effort
reconstruction of the pattern UX-Decoder's demo scripts share across
X-Decoder/SEEM/Semantic-SAM (all from the same authors), not something
checked against this exact repo/ref's real source. Every place that's
true is marked ``TODO(seem):`` — clone the repo yourself (see
``SEEM_REPO_URL``/``SEEM_REPO_REF``) and fix up import paths / argument
names / config file names against what's actually there before
expecting this to run.

Why this plugin can attempt real vendored-research-code integration at
all, unlike the framework's other plugins: SEEM was never ported into
``transformers`` — the only Hugging Face repo for it (``xdecoder/SEEM``)
ships raw ``.pt`` checkpoint files with no ``config.json``/processor/
modeling code, nothing for ``AutoModel*`` to load. The upstream
``UX-Decoder/Segment-Everything-Everywhere-All-At-Once`` GitHub repo
also isn't pip-installable as-is (no ``setup.py``/``pyproject.toml`` —
its own install docs are a manual ``git clone`` + ``PYTHONPATH``
workflow), and it wants its own old/exotic dependency chain (a
``detectron2`` fork, an old pinned torch). Every other plugin in this
framework deliberately avoids exactly this shape of dependency — but
this plugin's ``"pyenv": "seem"`` (rather than the default
``"framework"``) means all of that lives in its own auto-provisioned
virtual environment (see ``worker_pool.py``), never touching the
framework's own environment or any other plugin's. That isolation is
what makes attempting this safe now, where it wasn't before.

This also stays parked in ``models_inactive/`` rather than moving to
``models/`` — see ``pluginloader.py``'s module docstring and the
README's parked-plugins section. It's reachable one at a time via
``../models_inactive/seem`` from ``-M``/``-T``/``-TT``/a normal detect
run, but never discovered by ``-L``/``-I``/a bare ``'*'`` — a
``detectron2`` build (compiler, possibly CUDA toolkit) failing partway
through shouldn't be something a plain ``testdrive '*' photo.jpg cat``
can trigger for someone who only wanted, say, YOLO11.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..cache import cache_dir
from ..detection import Detection
from ..plugin import DetectorPlugin
from ..util import download_file, ensure_git_repo

if TYPE_CHECKING:
    from PIL import Image as PILImage

log = logging.getLogger("testdrive.models_inactive.seem")

PLUGIN_API = 1

# Pinned to a tag, not a branch — ensure_git_repo() caches whatever it
# clones forever (see its own docstring), so a branch name would
# silently keep serving whatever commit happened to be HEAD the first
# time this ran, on every machine, indefinitely. Resolve this to the
# exact commit SHA the tag currently points at before relying on this
# for anything beyond local experimentation — tags are far less likely
# to move than branches, but nothing stops it.
SEEM_REPO_URL = "https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once.git"
SEEM_REPO_REF = "v1.0"

# Raw checkpoint files on the HF hub (see module docstring — there's no
# config.json/processor alongside these, just the pytorch state dict).
SEEM_CHECKPOINT_BASE_URL = "https://huggingface.co/xdecoder/SEEM/resolve/main"
SEEM_CHECKPOINT_FILES = {
    "focalt": "seem_focalt_v2.pt",  # smaller/faster backbone
    "focall": "seem_focall_v1.pt",  # larger, presumably-more-accurate backbone
}
_DEFAULT_VARIANT = "focalt"

PLUGIN = {
    "id": "seem",
    "name": "SEEM",
    "version": "0.1.0",
    "api": PLUGIN_API,
    "description": "Segment Everything Everywhere All At Once. Scaffolded but not yet "
    "real — see module docstring for exactly what's verified vs. a "
    "best-effort placeholder. Auto-provisions its own 'seem' pyenv on first use.",
    "author": "UX-Decoder",
    "homepage": "https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once",
    "license": "Apache-2.0",
    "backend": "vendored (see module docstring)",
    "hf_repo": "xdecoder/SEEM",  # informational only — raw checkpoints, not from_pretrained()able
    "task": "panoptic segmentation",
    "supports": ["text prompts", "boxes"],  # masks reduced to boxes — see util.mask_to_bbox
    "requirements": [
        {"pip": "Pillow", "module": "PIL"},
        {"pip": "torch", "module": "torch"},
        {"pip": "opencv-python", "module": "cv2"},
        {"pip": "detectron2 @ git+https://github.com/MaureenZOU/detectron2-xyz.git",
         "module": "detectron2"},
    ],
    "sample_prompt": "yellow circle",
    "test_threshold": "default",
    "models": ["focalt", "focall"],
    "model": _DEFAULT_VARIANT,
    "pyenv": "seem",
}


class Plugin(DetectorPlugin):
    """SEEM universal segmentation — see module docstring for current status."""

    _model: Any = None
    _device: str = "cpu"
    _repo_dir: Any = None
    _initialized: bool = False

    def initialize(self) -> None:
        if self._initialized:
            return

        import sys

        import torch

        cd = cache_dir()
        variant = self.manifest.model or _DEFAULT_VARIANT

        # 1. Vendor SEEM's own modeling code (not a real pip package —
        #    see module docstring) straight into the cache, same
        #    discipline as every other download this framework makes —
        #    respects set_downloads_allowed() via CacheNotPopulatedError,
        #    same as ensure_local_repo()/download_file() do.
        self._repo_dir = ensure_git_repo(SEEM_REPO_URL, SEEM_REPO_REF, cd, self.manifest.id)
        demo_code_dir = self._repo_dir / "demo_code"
        if str(demo_code_dir) not in sys.path:
            sys.path.insert(0, str(demo_code_dir))

        # 2. Download the selected checkpoint variant.
        checkpoint_file = SEEM_CHECKPOINT_FILES[variant]
        checkpoint = download_file(
            f"{SEEM_CHECKPOINT_BASE_URL}/{checkpoint_file}",
            cd / "checkpoints" / "seem",
        )

        # 3. Build the model.
        #
        # TODO(seem): everything from here down is the unverified part
        # (see module docstring) — this follows the BaseModel/
        # build_model/load_opt_from_config_files pattern shared across
        # UX-Decoder's demo scripts, but hasn't been checked against
        # this exact repo/ref. Inspect self._repo_dir (specifically
        # demo_code/demo/seem/app.py or demo_code/tasks/) and correct
        # the imports/config path/call shape below to match.
        from modeling import build_model  # type: ignore[import-not-found]
        from modeling.BaseModel import BaseModel  # type: ignore[import-not-found]
        from utils.arguments import load_opt_from_config_files  # type: ignore[import-not-found]
        from utils.distributed import init_distributed  # type: ignore[import-not-found]

        config_file = demo_code_dir / "configs" / "seem" / f"seem_{variant}_lang.yaml"
        opt = load_opt_from_config_files([str(config_file)])
        opt = init_distributed(opt)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = (
            BaseModel(opt, build_model(opt))
            .from_pretrained(str(checkpoint))
            .eval()
            .to(self._device)
        )

        self._initialized = True
        log.info("SEEM (%s) ready", variant)

    def detect(
        self, image: "PILImage.Image", prompt: str, threshold: float = 0.3
    ) -> list[Detection]:
        if not self._initialized:
            self.initialize()

        # TODO(seem): the actual inference call. Once this is real, the
        # shape should look roughly like:
        #
        #   from util import mask_to_bbox  # not "from ..util" here —
        #       this method already runs inside the "seem" worker
        #       subprocess (see worker_main.py), which has this
        #       framework's own package on sys.path too
        #   results = self._model.forward(image, text_query=prompt)  # or similar
        #   detections = []
        #   for mask, score, label in results:  # whatever the real shape is
        #       if score < threshold:
        #           continue
        #       box = mask_to_bbox(mask)
        #       if box is not None:
        #           detections.append(Detection(label=label, score=float(score), bbox=box))
        #   return detections
        #
        # (Detection is bbox-only — see samgd.py's module docstring —
        # so masks reduce to boxes here the same way SAM's do there,
        # via the shared util.mask_to_bbox helper.)
        raise NotImplementedError(
            "seem: repo/checkpoint plumbing and model construction are wired up "
            "(see initialize()), but the actual inference call isn't implemented "
            "yet — see the TODO(seem) comments in this method"
        )
