"""Shared utilities: exit codes and logging setup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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


def _local_repo_dir(repo: str, cd: Path) -> Path:
    """Plain, non-symlinked directory a repo snapshot is downloaded into."""
    safe = repo.replace("/", "__")
    return cd / "models" / safe


def ensure_local_repo(repo: str, cd: Path) -> Path:
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

    local_dir = _local_repo_dir(repo, cd)
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


def load_processor(repo: str, cd: Path, processor_class: type[Any] | None = None, **kwargs: Any) -> Any:
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

    local = ensure_local_repo(repo, cd)

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


def load_model(repo: str, cd: Path, model_class: type[Any], **kwargs: Any) -> Any:
    """Load ``model_class`` from the same plain local snapshot directory
    used by :func:`load_processor`, avoiding the HF hub's symlinked
    snapshot cache (see ``ensure_local_repo``). If :func:`load_processor`
    already downloaded this repo, this reuses that download.
    """
    local = ensure_local_repo(repo, cd)
    return model_class.from_pretrained(local, **kwargs)


def download_file(url: str, dest_dir: Path, filename: str | None = None) -> Path:
    """Download a plain file (not from the HF hub) into ``dest_dir``.

    Used for checkpoints distributed as raw URLs rather than through a
    Hugging Face repo (e.g. Meta's SAM checkpoints), so they aren't left
    depending on the current working directory or on being downloaded by
    hand beforehand. Skips the download if the file already exists.

    Downloads to a ``.part`` file first and only renames it into place on
    success, so an interrupted download can't leave a corrupt file that
    looks "already downloaded" on the next run.
    """
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
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest
