"""Image I/O helpers.

The framework is the only part of the project that ever reads or
writes image files. Plugins receive a ``PIL.Image.Image`` object and
return ``Detection`` instances - they never touch the filesystem.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("testdrive.imageio")

# Lazily imported so that ``testdrive -L / -M / -I`` work on machines
# that don't have Pillow installed.
try:
    from PIL import Image

    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False


def _require_pil() -> None:
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required for image I/O: pip install Pillow")


def load_image(path: Path) -> "Image.Image":
    """Open *path* and return an RGB ``PIL.Image``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    OSError
        If Pillow cannot decode the file (unrecognised format, corrupt
        data, …).
    ImportError
        If Pillow is not installed.
    """
    _require_pil()
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")

    log.debug("loading image: %s", path)
    image: Image.Image = Image.open(path)

    if image.mode != "RGB":
        log.debug("converting %s → RGB (was %s)", path.name, image.mode)
        image = image.convert("RGB")

    return image


def save_image(image: "Image.Image", path: Path) -> None:
    """Save *image* as a PNG to *path*, creating parent directories as needed.

    Raises
    ------
    OSError
        If the file cannot be written.
    """
    _require_pil()
    path.parent.mkdir(parents=True, exist_ok=True)
    log.debug("saving image: %s", path)
    image.save(path, format="PNG", optimize=False)


def derive_output_paths(
    image_path: Path,
    plugin_suffix: str | None = None,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Return ``(matches_path, redacted_path)`` derived from *image_path*.

    If ``plugin_suffix`` is given (used by loop mode, ``'*'`` as the
    plugin id), it's appended to the basename before the ``-matches``/
    ``-redacted`` suffix, so results from different plugins run over the
    same image don't collide.

    If ``output_dir`` is given (``--output-dir`` on the CLI), output
    files land there instead of next to *image_path* — handy when
    running over many images/plugins so results don't scatter across
    whatever directories the input images happened to live in.
    ``output_dir`` is created if it doesn't exist yet.

    Examples
    --------
    >>> derive_output_paths(Path("photo.jpg"))
    (PosixPath('photo-matches.png'), PosixPath('photo-redacted.png'))

    >>> derive_output_paths(Path("/data/img/dog.JPEG"))
    (PosixPath('/data/img/dog-matches.png'), PosixPath('/data/img/dog-redacted.png'))

    >>> derive_output_paths(Path("photo.jpg"), plugin_suffix="owlv2")
    (PosixPath('photo-owlv2-matches.png'), PosixPath('photo-owlv2-redacted.png'))

    >>> derive_output_paths(Path("in/photo.jpg"), output_dir=Path("out"))
    (PosixPath('out/photo-matches.png'), PosixPath('out/photo-redacted.png'))
    """
    stem = image_path.stem
    if plugin_suffix:
        stem = f"{stem}-{plugin_suffix}"
    parent = output_dir if output_dir is not None else image_path.parent
    return (
        parent / f"{stem}-matches.png",
        parent / f"{stem}-redacted.png",
    )
