"""SEEM (Segment Everything Everywhere All At Once) plugin.

**Status: mostly real now, one open uncertainty.** The plumbing —
its own isolated ``seem`` pyenv (see ``pyenv:`` below and
``PluginManifest.pyenv``), vendoring SEEM's own modeling code via
``ensure_git_repo`` (see ``util.py``), and downloading its raw ``.pt``
checkpoint via ``download_file`` — is real and follows this
framework's normal conventions throughout. ``initialize()``'s model
construction and ``detect()``'s inference call are now confirmed
against the real ``v1.0``-tagged repo source (``demo/seem/app.py`` and
``demo/seem/tasks/interactive.py``'s ``'Text'`` branch), not a guess —
see the comments at each step for exactly what that confirmed against.
The one still-open item is marked ``TODO(seem):`` in ``detect()``: the
raw scale of ``vl_similarity()``'s output (from
``modeling/language/loss.py``, not yet inspected) isn't confirmed, so
the score returned is a sigmoid-clamped guess rather than a checked
value — worth confirming against that module before trusting
``threshold`` filtering here in anything but an approximate way.

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
    "focalt": "seem_focalt_v0.pt",  # smaller/faster backbone
    "focall": "seem_focall_v0.pt",  # larger, presumably-more-accurate backbone
}
_DEFAULT_VARIANT = "focalt"

PLUGIN = {
    "id": "seem",
    "name": "SEEM",
    "version": "0.1.0",
    "api": PLUGIN_API,
    "description": "Segment Everything Everywhere All At Once. Model construction and "
    "inference are confirmed against the real repo source — see module "
    "docstring for the one still-open scoring-scale uncertainty. "
    "Auto-provisions its own 'seem' pyenv on first use.",
    "author": "UX-Decoder",
    "homepage": "https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once",
    "license": "Apache-2.0",
    "backend": "vendored (see module docstring)",
    "hf_repo": "xdecoder/SEEM",  # informational only — raw checkpoints, not from_pretrained()able
    "task": "panoptic segmentation",
    "supports": ["text prompts", "boxes"],  # masks reduced to boxes — see util.mask_to_bbox
    "requirements": [
        {"pip": "torch==2.0.0", "module": "torch"},
        {"pip": "setuptools<81", "module": "setuptools"},
        {"pip": "nltk", "module": "nltk"},
        {"pip": "einops", "module": "einops"},
        {"pip": "mpi4py", "module": "mpi4py"},
        {"pip": "numpy==1.26.4", "module": "numpy"},
        {"pip": "timm==0.4.12", "module": "timm"},
        {"pip": "transformers==4.34.0", "module": "transformers"},
        {"pip": "kornia==0.7.0", "module": "kornia"},
        {"pip": "opencv-python==4.8.1.78", "module": "cv2"},
        {"pip": "Pillow==9.4.0", "module": "PIL"},
        {"pip": "detectron2 @ git+https://github.com/MaureenZOU/detectron2-xyz.git",
         "module": "detectron2"},
    ],
    # detectron2-xyz's C++ extension (built above with --no-build-isolation,
    # since its setup.py imports torch directly at build time — see
    # worker_pool.py/pyenv.run_pip_install) doesn't compile against
    # torch==2.0.0's bundled headers under a modern Apple Clang (21.x):
    # pybind11's operator""_a literal-operator declaration inside
    # strong_type.h — vendored into torch's own installed headers — hits
    # a hard error under recent Clang's stricter C++ conformance
    # checking. Confirmed by hand: guarding out that one template
    # specialization (unused by anything detectron2-xyz actually needs)
    # with #if 0/#endif is enough to let it compile clean. See
    # PluginManifest.patches / pyenv.apply_source_patches for how/when
    # this gets applied — after every non-VCS requirement above is
    # installed (so torch's headers exist to patch) but before the
    # detectron2-xyz requirement (so the patch is in place before
    # anything tries to compile against it).
    "patches": [
        {
            "target": "torch/include/c10/util/strong_type.h",
            "find": (
                "template <typename T, typename Tag, typename ... M>\n"
                "struct is_arithmetic<::strong::type<T, Tag, M...>>\n"
                "  : is_base_of<::strong::arithmetic::modifier<::strong::type<T, Tag, M...>>,\n"
                "               ::strong::type<T, Tag, M...>>\n"
                "{\n"
                "};\n"
            ),
            "replace": (
                "#if 0\n"
                "template <typename T, typename Tag, typename ... M>\n"
                "struct is_arithmetic<::strong::type<T, Tag, M...>>\n"
                "  : is_base_of<::strong::arithmetic::modifier<::strong::type<T, Tag, M...>>,\n"
                "               ::strong::type<T, Tag, M...>>\n"
                "{\n"
                "};\n"
                "#endif\n"
            ),
        },
    ],
    # detectron2-xyz's setup.py imports torch directly at build time (to
    # detect CUDA/pick compile flags) without declaring it under
    # [build-system] requires, so pip's normal isolated build env — which
    # only ever contains the build backend, nothing this project installs
    # — can't see it: --no-build-isolation makes the build run against
    # this environment instead, where torch (pinned above) already is.
    "pip_options": "--no-build-isolation",
    "sample_prompt": "yellow circle",
    "test_threshold": "default",
    "models": ["focalt", "focall"],
    "model": _DEFAULT_VARIANT,
    "pyenv": "seem",
}


class Plugin(DetectorPlugin):
    """SEEM universal segmentation — see module docstring for current status."""

    _model: Any = None
    _device: Any = "cpu"
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
        if str(self._repo_dir) not in sys.path:
            sys.path.insert(0, str(self._repo_dir))

        # 2. Download the selected checkpoint variant.
        checkpoint_file = SEEM_CHECKPOINT_FILES[variant]
        checkpoint = download_file(
            f"{SEEM_CHECKPOINT_BASE_URL}/{checkpoint_file}",
            cd / "checkpoints" / "seem",
        )

        # 3. Build the model — this shape (config -> load_opt_from_config_files
        # -> BaseModel(opt, build_model(opt)).from_pretrained(...)) is
        # confirmed against demo/seem/app.py in the real v1.0-tagged repo.
        # It loads "configs/seem/{variant}_unicl_lang_demo.yaml" (not
        # "_v1.yaml" — that config doesn't pair with the "_v0.pt"
        # checkpoints app.py itself downloads) — see
        # SEEM_CHECKPOINT_FILES above, which matches that pairing.
        from modeling import build_model  # type: ignore[import-not-found]
        from modeling.BaseModel import BaseModel  # type: ignore[import-not-found]
        from utils.arguments import load_opt_from_config_files  # type: ignore[import-not-found]

        # app.py calls utils.distributed.init_distributed(opt) here, but
        # that function unconditionally shells out to `hostname -I`
        # in apply_distributed() (see utils/distributed.py) whenever
        # opt['rank'] == 0 — true for single-process CPU use too, not
        # just real multi-node runs — and that flag is GNU/Linux-only
        # (BSD/macOS `hostname` has no -I). The actual
        # torch.distributed.init_process_group() call inside
        # apply_distributed() is properly gated behind
        # opt['world_size'] > 1, so for single-process use the whole
        # call computes a master address/port that's then thrown away
        # — dead weight wrapped around a platform-specific crash. This
        # replicates just init_distributed()'s own "no MPI" branch
        # (world_size=1, rank=0) directly instead of calling it, since
        # OMPI_COMM_WORLD_SIZE is never set for this plugin's use case.
        opt = load_opt_from_config_files(
            [str(self._repo_dir / "configs" / "seem" / f"{variant}_unicl_lang_demo.yaml")]
        )
        opt["CUDA"] = torch.cuda.is_available()
        opt["world_size"] = 1
        opt["local_size"] = 1
        opt["rank"] = 0
        opt["local_rank"] = 0
        opt["device"] = torch.device("cuda", 0) if opt["CUDA"] else torch.device("cpu")

        self._device = opt["device"]

        if not opt["CUDA"]:
            # modeling/architectures/seem_model_demo.py's from_config()
            # (reached via build_model() below) has one line that
            # unconditionally calls torch.cuda.current_device() for a
            # dilation-kernel tensor's device= kwarg, regardless of
            # opt['CUDA'] — a real bug in the vendored v1.0 source for
            # CPU-only use, confirmed via full traceback:
            # AssertionError: Torch not compiled with CUDA enabled,
            # raised from torch.cuda._lazy_init(). Patching the
            # vendored file on disk would be more surgical but far
            # more fragile (path/line depend on the exact checkout,
            # silently stops applying if SEEM_REPO_REF ever moves).
            # Monkeypatching current_device() to hand back
            # torch.device("cpu") instead is equivalent here — that
            # return value is only ever forwarded straight into
            # torch.ones(..., device=X), which accepts a torch.device
            # object just as well as a CUDA index — and scoped to only
            # the case where CUDA genuinely isn't available, so a real
            # GPU machine's behavior is completely unaffected: nothing
            # in a CPU-only run can legitimately depend on getting a
            # real CUDA index back from this call anyway, since that
            # would already require CUDA to be available.
            torch.cuda.current_device = lambda: torch.device("cpu")

            # Same story, different call shape: other points in the
            # vendored code — e.g. modeling/language/vlpencoder.py's
            # get_text_token_embeddings(), reached from detect() on
            # every call, not just here in initialize() — call
            # .cuda() directly on tensors/modules, again with no
            # CPU-only guard (confirmed via full traceback:
            # AssertionError: Torch not compiled with CUDA enabled,
            # from `tokens = {key: value.cuda() ...}`). Rather than
            # patch each call site as it's individually discovered,
            # monkeypatch Tensor.cuda()/Module.cuda() themselves to
            # no-ops (return self unchanged) — the standard technique
            # for running a GPU-only-authored codebase CPU-only.
            # Correct, not just non-crashing: everything already lives
            # on CPU by default here (nothing upstream moves anything
            # to a CUDA device when opt['CUDA'] is False), so ".cuda()"
            # becoming a no-op matches reality everywhere at once,
            # rather than requiring every call site to be found one
            # crash at a time.
            torch.Tensor.cuda = lambda self, *args, **kwargs: self
            torch.nn.Module.cuda = lambda self, *args, **kwargs: self

        self._model = (
            BaseModel(opt, build_model(opt))
            .from_pretrained(str(checkpoint))
            .eval()
            .to(self._device)
        )

        # forward_prediction_heads() (modeling/interface/seem_demo.py,
        # reached from evaluate_demo() during detect() below)
        # unconditionally calls lang_encoder.compute_similarity(),
        # regardless of which task is active — confirmed via full
        # traceback: AttributeError: 'LanguageEncoder' object has no
        # attribute 'default_text_embeddings'. That's not something
        # specific to the Panoptic task as this comment used to assume
        # (interactive.py's 'Text' branch never looks at this
        # particular result, but the forward pass computes it anyway
        # as a side effect) — it needs get_text_embeddings() to have
        # registered a "default"-named vocabulary first, exactly like
        # app.py does right after loading the model.
        from utils.constants import COCO_PANOPTIC_CLASSES  # type: ignore[import-not-found]

        with torch.no_grad():
            self._model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                COCO_PANOPTIC_CLASSES + ["background"], is_eval=True
            )

        self._initialized = True
        log.info("SEEM (%s) ready", variant)

    def detect(
        self, image: "PILImage.Image", prompt: str, threshold: float = 0.3
    ) -> list[Detection]:
        if not self._initialized:
            self.initialize()

        import numpy as np
        import torch
        import torch.nn.functional as F
        from torchvision import transforms

        from ..util import mask_to_bbox

        # This is demo/seem/tasks/interactive.py's interactive_infer_image(),
        # trimmed to just its 'Text' branch (grounding by a free-text
        # prompt — the only task this plugin's Detection-based interface
        # needs) and made device-agnostic (the original hardcodes .cuda()
        # throughout, since the real demo assumes a GPU box). Panoptic/
        # Stroke/Example/Audio/Video paths are dropped entirely — not
        # reachable from a plain (image, prompt, threshold) call. See
        # modeling.language.loss.vl_similarity for what "grounding_class"
        # /the resulting out_prob actually represent — not yet inspected,
        # so the score below is out_prob's raw value, clamped through a
        # sigmoid to land in Detection's expected roughly-[0,1] range;
        # confirm that's actually what out_prob's scale calls for.
        from modeling.language.loss import vl_similarity  # type: ignore[import-not-found]

        resize = transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC)
        image_ori = resize(image.convert("RGB"))
        width, height = image_ori.size
        image_np = np.asarray(image_ori)
        image_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).to(self._device)

        data = {"image": image_tensor, "height": height, "width": width, "text": [prompt]}

        self._model.model.task_switch["spatial"] = False
        self._model.model.task_switch["visual"] = False
        self._model.model.task_switch["grounding"] = True
        self._model.model.task_switch["audio"] = False

        with torch.no_grad():
            results, image_size, extra = self._model.model.evaluate_demo([data])

            pred_masks = results["pred_masks"][0]
            v_emb = results["pred_captions"][0]
            t_emb = extra["grounding_class"]

            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-7)
            v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-7)

            temperature = self._model.model.sem_seg_head.predictor.lang_encoder.logit_scale
            out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)

            matched_id = out_prob.max(0)[1]
            pred_masks_pos = pred_masks[matched_id, :, :]
            score = torch.sigmoid(out_prob.max(0)[0]).item()

            mask = (
                F.interpolate(pred_masks_pos[None,], image_size[-2:], mode="bilinear")[
                    0, :, :height, :width
                ]
                > 0.0
            ).float().cpu().numpy()

        if score < threshold:
            return []
        box = mask_to_bbox(mask[0])
        if box is None:
            return []
        # box is in image_ori's (post-Resize(512)) pixel coordinates,
        # not the original image's — scale back before returning, or
        # the box lands proportionally too large/offset when drawn
        # onto the original-sized canvas (confirmed: this is why the
        # very first successful detection's -matches image showed no
        # visible box at all — it wasn't missing, just off-frame).
        orig_width, orig_height = image.size
        scale_x = orig_width / width
        scale_y = orig_height / height
        x1, y1, x2, y2 = box
        box = (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
        )
        return [Detection(label=prompt, score=score, bbox=box)]
