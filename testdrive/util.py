"""Shared utilities: exit codes and logging setup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import ssl

log = logging.getLogger("testdrive.util")

# Set via --max-parallel-files on the CLI (see set_max_parallel_files below).
_max_parallel_files: int | None = None

# Set by the CLI around each command (see set_downloads_allowed below).
# Defaults to True so library/test usage isn't restricted unless a
# caller opts in.
_downloads_allowed: bool = True


def set_max_parallel_files(n: int | None) -> None:
    """Cap how many files a single repo snapshot downloads in parallel.

    ``huggingface_hub.snapshot_download`` downloads up to ``max_workers``
    (default 8) files concurrently. On a slow or asymmetric link, several
    multi-GB ``*.safetensors`` shards downloading at once just divides
    the same bandwidth ~8 ways rather than actually going any faster —
    and each individual transfer can end up crawling (or timing out)
    as a result. Capping concurrency, even down to 1, keeps each
    transfer's effective throughput usable.
    """
    global _max_parallel_files
    _max_parallel_files = n


class CacheNotPopulatedError(RuntimeError):
    """Raised when a model/checkpoint isn't cached locally yet and
    downloads are currently disallowed (see ``set_downloads_allowed``).
    """

    def __init__(self, what: str, path: Path):
        self.what = what
        self.path = path
        super().__init__(f"'{what}' is not cached yet (would download to {path})")


def set_downloads_allowed(allowed: bool) -> None:
    """Control whether a model/checkpoint may actually be downloaded.

    The CLI sets this to ``False`` around a plain detect run and
    ``True`` around ``-T``/``-TT`` (see cli.py). A plain ``testdrive
    <plugin> <image> <prompt>`` run hitting an uncached model otherwise
    turns into a surprise multi-gigabyte, multi-minute blocking download
    in the middle of what looked like a normal detection command —
    which, stacked on top of a naturally slow model's already-long init
    time, is easy to mistake for a hang (this is exactly what happened
    working through Molmo). ``-T``/``-TT`` are explicitly "populate the
    cache" commands, so they re-enable downloads; a plain detect run
    against an already-cached model is completely unaffected either way,
    since nothing here downloads anything once cached.
    """
    global _downloads_allowed
    _downloads_allowed = allowed


def get_downloads_allowed() -> bool:
    """Current value set by set_downloads_allowed() (default True).

    Used by worker_pool.py to propagate this process's setting to a
    worker subprocess, which doesn't inherit Python module state across
    the process boundary the way it would if everything ran in one
    process.
    """
    return _downloads_allowed


def get_max_parallel_files() -> int | None:
    """Current value set by set_max_parallel_files() (default None).

    Same propagation purpose as get_downloads_allowed().
    """
    return _max_parallel_files


# Set via --no-auto-provision on the CLI (see set_auto_provision_enabled
# below). Defaults to True so library/test usage isn't restricted unless
# a caller opts out.
_auto_provision_enabled: bool = True


def set_auto_provision_enabled(enabled: bool) -> None:
    """Control whether a plugin's own (non-"framework") pyenv may be
    automatically created and pip-installed into on first use.

    Defaults to True. Pass False (``--no-auto-provision`` on the CLI)
    to require the person to set that environment up by hand instead —
    e.g. because they want control over exactly what gets installed, or
    because auto-provisioning failed/misbehaved and they're debugging it
    manually. Only relevant to plugins that declare a pyenv other than
    "framework"; has no effect otherwise. Also gated behind
    get_downloads_allowed() regardless of this setting — provisioning a
    new environment is the same class of "don't do this by surprise
    during a plain detect run" action as downloading model weights.
    """
    global _auto_provision_enabled
    _auto_provision_enabled = enabled


def get_auto_provision_enabled() -> bool:
    """Current value set by set_auto_provision_enabled() (default True)."""
    return _auto_provision_enabled


# Set via --pyenv-pip-upgrade/--no-pyenv-pip-upgrade on the CLI (see
# set_pyenv_pip_upgrade below). Defaults to True.
_pyenv_pip_upgrade: bool = True


def set_pyenv_pip_upgrade(enabled: bool) -> None:
    """Control whether auto-provisioning a plugin's environment (see
    worker_pool._provision_plugin_env) upgrades pip in that environment
    before installing anything else into it.

    Defaults to True: a plugin's dependency list can be substantial
    (torch, transformers, ...), and an old bundled pip is a common
    source of install failures (missing wheel support, outdated
    resolver, etc.) that are confusing to diagnose from a "package X
    failed to install" error alone. Pass False
    (``--no-pyenv-pip-upgrade``) to skip this and use whatever pip
    ``venv`` itself bundled.
    """
    global _pyenv_pip_upgrade
    _pyenv_pip_upgrade = enabled


def get_pyenv_pip_upgrade() -> bool:
    """Current value set by set_pyenv_pip_upgrade() (default True)."""
    return _pyenv_pip_upgrade


class ExitCode:
    """Process exit codes, documented so testdrive scripts/CI nicely."""

    SUCCESS = 0
    CLI_ERROR = 1
    PLUGIN_NOT_FOUND = 2
    MISSING_DEPENDENCY = 3
    INFERENCE_FAILED = 4
    IMAGE_UNREADABLE = 5
    OUTPUT_WRITE_FAILED = 6
    LOOP_PARTIAL_FAILURE = 7  # '*' loop mode: at least one plugin in the loop failed
    PYENV_NOT_CONFIGURED = 8  # not running from cache/pyenv/framework, and it doesn't exist yet


def setup_logging(verbosity: int) -> None:
    """0 -> WARNING, 1 (-v) -> INFO, 2+ (-vv) -> DEBUG."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(level=level, format="%(levelname)-7s %(message)s")
    logging.getLogger("testdrive").setLevel(level)


def _local_repo_dir(plugin_id: str, cd: Path) -> Path:
    """Plain, non-symlinked directory a repo snapshot is downloaded into
    — one directory per *plugin*, not per HF repo. A plugin's repo can
    change (e.g. switching to a different upstream checkpoint) without
    leaving an orphaned, differently-named cache directory behind, and
    ``cache/models/<plugin id>/`` is a much more legible layout to
    browse by hand than a mangled ``org__repo-name`` directory.
    """
    return cd / "models" / plugin_id


def ensure_local_repo(repo: str, cd: Path, plugin_id: str) -> Path:
    """Download a full HF repo snapshot into a plain local directory.

    ``huggingface_hub``'s default cache stores files as content-addressed
    blobs and exposes them through symlinked "snapshot" trees. That
    symlink layer is unreliable on some Windows installs and consistently
    broken under Wine: ``Path.is_symlink()`` reports ``True`` while
    ``Path.exists()`` reports ``False``, and reads fail with
    ``OSError: [Errno 22] Invalid argument`` — which then surfaces deep
    inside ``transformers`` as confusing "file not found" / "unrecognized
    processing class" errors, even though the files downloaded fine.

    Downloading into a real directory via ``local_dir=`` sidesteps the
    symlink layer entirely: files land on disk as ordinary files. This
    works identically on Linux, macOS, native Windows, and Wine, and is
    independent of the installed ``huggingface_hub``/``transformers``
    version.

    Lands in ``cache/models/<plugin_id>/`` (see ``_local_repo_dir``) —
    keyed by plugin, not by the HF repo string.

    A ``.testdrive_complete`` marker file is written once every file has
    been downloaded, so repeated calls (e.g. once for the processor, once
    for the model) don't re-download anything.

    Repos that ship both ``*.safetensors`` and a legacy format
    (``pytorch_model.bin`` / ``*.ckpt`` / ``*.h5`` / ``*.msgpack``) for the
    *same* weights only need the safetensors copy — ``from_pretrained()``
    prefers it automatically — so the legacy duplicate is skipped to avoid
    downloading the same weights twice.
    """
    import warnings

    from huggingface_hub import snapshot_download, list_repo_files

    local_dir = _local_repo_dir(plugin_id, cd)
    marker = local_dir / ".testdrive_complete"
    if marker.exists():
        return local_dir

    if not _downloads_allowed:
        raise CacheNotPopulatedError(repo, local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)

    ignore_patterns = None
    try:
        files = list_repo_files(repo)
        if any(f.endswith(".safetensors") for f in files):
            ignore_patterns = ["*.bin", "*.ckpt", "*.h5", "*.msgpack"]
    except Exception:
        pass  # fall back to downloading everything if listing fails

    dl_kwargs: dict[str, Any] = {
        "repo_id": repo,
        "local_dir": local_dir,
        "ignore_patterns": ignore_patterns,
    }
    if _max_parallel_files is not None:
        dl_kwargs["max_workers"] = _max_parallel_files

    with warnings.catch_warnings():
        # Newer huggingface_hub deprecated local_dir_use_symlinks (it's a
        # no-op there: local_dir= already always writes real files), but
        # still accepts it rather than raising. Silence just that notice —
        # it's expected, not something the user needs to see every run.
        warnings.filterwarnings("ignore", message=r".*local_dir_use_symlinks.*")
        try:
            # Older huggingface_hub (<0.23) may still symlink into its own
            # cache even with local_dir= unless told not to.
            snapshot_download(local_dir_use_symlinks=False, **dl_kwargs)
        except TypeError:
            # Even older huggingface_hub that doesn't accept the kwarg at all.
            snapshot_download(**dl_kwargs)

    marker.touch()
    return local_dir


def load_processor(
    repo: str,
    cd: Path,
    plugin_id: str,
    processor_class: type[Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Robust processor loader with multiple fallbacks.

    Downloads the full repo into a plain local directory first (see
    ``ensure_local_repo``) so loading never depends on the HF hub's
    symlinked snapshot cache, then tries ``processor_class`` (if given),
    falling back to ``AutoProcessor`` and ``AutoImageProcessor``.

    Extra ``kwargs`` (e.g. ``trust_remote_code=True``) are forwarded to
    every ``from_pretrained`` call attempted below.
    """
    from collections.abc import Callable

    from transformers import AutoProcessor, AutoImageProcessor

    local = ensure_local_repo(repo, cd, plugin_id)

    attempts: list[tuple[str, Callable[[], Any]]] = []
    if processor_class:
        attempts.append(
            (processor_class.__name__, lambda: processor_class.from_pretrained(local, **kwargs))
        )
    attempts.extend(
        [
            ("AutoProcessor", lambda: AutoProcessor.from_pretrained(local, **kwargs)),
            ("AutoImageProcessor", lambda: AutoImageProcessor.from_pretrained(local, **kwargs)),
        ]
    )

    errors: list[str] = []
    for name, loader in attempts:
        try:
            return loader()
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        f"Processor load failed for {repo} (local snapshot: {local}):\n" + "\n".join(errors)
    )


def load_model(repo: str, cd: Path, plugin_id: str, model_class: type[Any], **kwargs: Any) -> Any:
    """Load ``model_class`` from the same plain local snapshot directory
    used by :func:`load_processor`, avoiding the HF hub's symlinked
    snapshot cache (see ``ensure_local_repo``). If :func:`load_processor`
    already downloaded this repo, this reuses that download.
    """
    local = ensure_local_repo(repo, cd, plugin_id)
    return model_class.from_pretrained(local, **kwargs)


def _ssl_context() -> "ssl.SSLContext":
    """Build an SSL context that works on hosts with an incomplete system
    CA store (notably older Windows, e.g. 8.1 under corporate roots).

    Prefers ``certifi``'s Mozilla CA bundle when installed; falls back to
    the platform default context otherwise. Callers that hit
    ``CERTIFICATE_VERIFY_FAILED`` without certifi should
    ``pip install certifi`` into the relevant pyenv.
    """

    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_file(url: str, dest_dir: Path, filename: str | None = None) -> Path:
    """Download a plain file (not from the HF hub) into ``dest_dir``.

    Used for checkpoints distributed as raw URLs rather than through a
    Hugging Face repo (e.g. Meta's SAM checkpoints), so they aren't left
    depending on the current working directory or on being downloaded by
    hand beforehand. Skips the download if the file already exists.

    Downloads to a ``.part`` file first and only renames it into place on
    success, so an interrupted download can't leave a corrupt file that
    looks "already downloaded" on the next run.

    Uses :func:`_ssl_context` so TLS verification works on hosts whose
    system CA store is incomplete (Windows 8.1 is a common case) when
    ``certifi`` is installed.
    """
    import shutil
    import ssl
    import urllib.error
    import urllib.request

    dest_dir.mkdir(parents=True, exist_ok=True)
    name = filename or url.rsplit("/", 1)[-1]
    dest = dest_dir / name

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    if not _downloads_allowed:
        raise CacheNotPopulatedError(url, dest)

    log.info("downloading %s -> %s", url, dest)
    tmp = dest.with_name(dest.name + ".part")
    try:
        ctx = _ssl_context()
        with urllib.request.urlopen(url, context=ctx) as resp:
            with open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
        tmp.replace(dest)
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        # Surface a actionable hint when the failure is the common
        # "broken system CA store" case on older Windows.
        err = str(exc.reason) if getattr(exc, "reason", None) else str(exc)
        if "CERTIFICATE_VERIFY_FAILED" in err or isinstance(
            getattr(exc, "reason", None), ssl.SSLError
        ):
            raise urllib.error.URLError(
                f"{exc.reason}; install certifi into this environment "
                f"(pip install certifi) so downloads can use Mozilla's CA "
                f"bundle instead of the system store"
            ) from exc
        raise
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def ensure_git_repo(url: str, ref: str, cd: Path, plugin_id: str) -> Path:
    """Clone a pinned commit/tag of a git repo into ``cache/repos/<plugin_id>/``.

    For vendoring a research codebase that isn't a real pip package (no
    ``setup.py``/``pyproject.toml`` for pip to build) but is still just
    a git repo — e.g. SEEM's own modeling code, as opposed to its
    pip-installable ``detectron2`` fork dependency, which belongs in
    ``pyproject.toml`` instead. Callers typically ``sys.path.insert()``
    the returned directory (or a subdirectory of it) before importing
    from it.

    ``ref`` should be a commit SHA or tag, not a branch — this caches
    forever once cloned (see the marker file below), so a branch name
    would silently keep serving whatever commit happened to be HEAD the
    first time it was cloned, on every machine, indefinitely.

    A ``.testdrive_complete`` marker file is written once the checkout
    succeeds, so repeated calls are a no-op — same convention as
    :func:`ensure_local_repo`. Clones into a ``.part``-suffixed
    directory first and only renames it into place on success (mirroring
    :func:`download_file`'s own atomicity trick), so an interrupted or
    failed clone/checkout can't leave a corrupt directory that looks
    "already cloned" on the next run.

    Raises ``CacheNotPopulatedError`` if not yet cloned and downloads are
    currently disallowed (see ``set_downloads_allowed``), and lets
    ``subprocess.CalledProcessError`` propagate as-is on a real git
    failure (bad URL, bad ref, no network, etc.) — same "don't wrap what
    we don't need to" approach as ``ensure_local_repo`` takes with
    ``huggingface_hub`` errors.
    """
    import shutil
    import subprocess

    target = cd / "repos" / plugin_id
    marker = target / ".testdrive_complete"
    if marker.exists():
        return target

    if not _downloads_allowed:
        raise CacheNotPopulatedError(f"{url}@{ref}", target)

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)

    log.info("cloning %s@%s -> %s", url, ref, target)
    try:
        subprocess.run(["git", "clone", url, str(tmp)], check=True, capture_output=True, text=True)
        subprocess.run(
            ["git", "-C", str(tmp), "checkout", ref], check=True, capture_output=True, text=True
        )
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    shutil.rmtree(target, ignore_errors=True)
    tmp.replace(target)
    marker.touch()
    return target


def mask_to_bbox(mask: Any) -> tuple[int, int, int, int] | None:
    """Tight ``(x1, y1, x2, y2)`` box around a boolean/0-1 mask array's
    nonzero pixels, or ``None`` if the mask is empty.

    :class:`~testdrive.detection.Detection` is bbox-only — there's no
    mask field to return (see ``samgd.py``'s module docstring) — so any
    plugin whose native output is a segmentation mask (SAM, SEEM, ...)
    needs to reduce each mask to a box before returning a ``Detection``.
    Shared here since more than one plugin does exactly this reduction,
    the same way, rather than duplicating the ``np.where`` dance in each.
    """
    import numpy as np

    mask_np = np.asarray(mask)
    ys, xs = np.where(mask_np)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
