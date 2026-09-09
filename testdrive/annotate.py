"""Image annotation engine.

The framework owns all drawing logic. Plugins never draw anything;
they return ``Detection`` lists and the framework decides how to
visualise them.

Two modes:
  ``draw_boxes``  – green bounding boxes with label + score overlay.
  ``redact``      – solid black rectangles covering each detection.
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image as PILImage

from .detection import Detection

log = logging.getLogger("testdrive.annotate")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BOX_COLOR = "#00FF00"  # bright green
_BOX_WIDTH = 3  # border thickness in pixels
_REDACT_COLOR = "#000000"  # solid black
_LABEL_BG = "#00CC00"  # slightly darker green for label background
_LABEL_FG = "#000000"  # black text on green background
_LABEL_PAD = 4  # padding (px) around label text


def _get_draw_module() -> tuple[ModuleType, ModuleType]:
    """Return (PIL.Image, PIL.ImageDraw) or raise ImportError.

    These are modules, not classes — despite the common `Image.open(...)`/
    `ImageDraw.Draw(...)` usage looking class-like, `Image` and `ImageDraw`
    are themselves plain modules exposing factory functions/classes.
    """
    try:
        from PIL import Image, ImageDraw

        return Image, ImageDraw
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow is required for annotation: pip install Pillow") from exc


def _get_font(size: int = 14) -> Any:
    """Return a PIL font, falling back gracefully to the built-in default."""

    try:
        from PIL import ImageFont

        try:
            # Try loading a bundled TrueType font (available on most systems)
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except (OSError, IOError):
            pass
        try:
            return ImageFont.truetype("arial.ttf", size)
        except (OSError, IOError):
            pass
        # PIL built-in bitmap font - always available, small but readable
        return ImageFont.load_default()
    except Exception:  # noqa: BLE001 - never crash the annotation engine
        return None


def draw_boxes(
    image: "PILImage.Image",
    detections: list[Detection],
    *,
    box_color: str = _BOX_COLOR,
    box_width: int = _BOX_WIDTH,
    label_bg: str = _LABEL_BG,
    label_fg: str = _LABEL_FG,
    font_size: int = 14,
    show_score: bool = True,
) -> "PILImage.Image":
    """Return a copy of *image* with green bounding boxes and labels drawn.

    Parameters
    ----------
    image:
        Source image (not modified in place).
    detections:
        List of detections to annotate.
    box_color:
        Hex colour for the box outline.
    box_width:
        Stroke width in pixels.
    label_bg:
        Background colour for the label banner.
    label_fg:
        Text colour for the label banner.
    font_size:
        Approximate font size in points; actual size depends on the
        available font (PIL's built-in default ignores this).
    show_score:
        Whether to include the confidence score in the label.
    """
    _, ImageDraw = _get_draw_module()
    font = _get_font(font_size)

    result = image.copy()
    draw = ImageDraw.Draw(result, "RGBA")

    if not detections:
        log.debug("draw_boxes: no detections to draw")
        return result

    for det in detections:
        x1, y1, x2, y2 = det.bbox

        # --- bounding box ---
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_width)

        # --- label text ---
        label = det.label
        if show_score:
            label = f"{label} {det.score:.2f}"

        # Measure text so we can size the background banner
        if font is not None:
            try:
                # Pillow ≥ 10 changed the API
                bbox_text = draw.textbbox((0, 0), label, font=font)
                tw = bbox_text[2] - bbox_text[0]
                th = bbox_text[3] - bbox_text[1]
            except AttributeError:
                tw, th = draw.textsize(label, font=font)
        else:
            tw, th = len(label) * 7, 12

        pad = _LABEL_PAD
        bx1 = x1
        by1 = max(0, y1 - th - pad * 2)
        bx2 = x1 + tw + pad * 2
        by2 = y1

        draw.rectangle([bx1, by1, bx2, by2], fill=label_bg)
        draw.text((bx1 + pad, by1 + pad), label, fill=label_fg, font=font)

    log.debug("draw_boxes: annotated %d detection(s)", len(detections))
    return result


#: The only builtins a manifest's ``replace_eval`` expression can reach —
#: everything it should plausibly need to slice/measure/rebuild a piece
#: of text (e.g. "text[:1] + '*' * (len(text) - 1)"), and nothing that
#: reads files, imports modules, or otherwise escapes this one string.
_REPLACE_EVAL_BUILTINS = {"len": len, "str": str, "min": min, "max": max}


def _eval_replacement(text: str, expr: str) -> str | None:
    """Evaluate *expr* (a manifest ``replace_eval``) with ``text`` bound
    to the detected text, in a namespace restricted to
    ``_REPLACE_EVAL_BUILTINS``.

    Returns the masked replacement, or ``None`` if *expr* itself
    evaluates to ``None`` — the caller's signal to skip the
    white-box-with-text rendering entirely and fall back to a plain
    solid-rectangle redaction, e.g. for an expression like
    ``"None if text.isdigit() else text[:1] + '*' * (len(text) - 1)"``
    that deliberately opts a particular piece of text out of masking.

    Falls back to a full-mask of the same length (never ``None``) on
    any evaluation error, so a bad/unexpected expression degrades
    safely rather than crashing the whole redaction pass.
    """
    try:
        result = eval(  # noqa: S307
            expr, {"__builtins__": _REPLACE_EVAL_BUILTINS}, {"text": text}
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("redact: replace_eval %r failed on %r: %s", expr, text, exc)
        return "*" * len(text)
    return None if result is None else str(result)


_REDACT_TEXT_BG = "#FFFFFF"  # white box behind a replace_eval'd item
_REDACT_TEXT_FG = "#000000"  # black text drawn on top of it


def redact(
    image: "PILImage.Image",
    detections: list[Detection],
    *,
    fill_color: str = _REDACT_COLOR,
    replace_minchar: int = 0,
    replace_eval: str = "",
    font_size: int = 14,
) -> "PILImage.Image":
    """Return a copy of *image* with each detection redacted.

    Parameters
    ----------
    image:
        Source image (not modified in place).
    detections:
        List of detections to redact.
    fill_color:
        Fill colour for the plain solid-rectangle redaction (used
        whenever a detection has no ``text``, *replace_eval* is empty
        — i.e. the plugin isn't configured for word-/line-level
        masking at all — or *replace_eval* evaluates to ``None`` for
        that particular detection).
    replace_minchar, replace_eval:
        For detections that carry ``Detection.text`` (per-word/
        per-line OCR detections — see ``PluginManifest.replace_minchar``/
        ``replace_eval``), only used when *replace_eval* is non-empty:

        * text of length <= *replace_minchar* is considered too short
          to matter and is left completely untouched — no box is
          drawn over it at all, so the original pixels show through.
        * longer text is masked by evaluating *replace_eval* (with
          ``text`` bound to the detected text). If the expression
          evaluates to a real value, it's drawn as black text over a
          white box (not the plain black rectangle used elsewhere), so
          the redaction is legible rather than just an opaque block.
          If it evaluates to ``None``, this detection instead falls
          back to the plain solid-rectangle redaction.
    """
    _, ImageDraw = _get_draw_module()

    result = image.copy()
    draw = ImageDraw.Draw(result)

    if not detections:
        log.debug("redact: no detections to redact")
        return result

    font = _get_font(font_size) if replace_eval else None

    for det in detections:
        x1, y1, x2, y2 = det.bbox

        if det.text and replace_eval:
            text = det.text
            if len(text) <= replace_minchar:
                # Short enough to leave as-is: draw nothing, original
                # pixels stay visible.
                continue
            masked = _eval_replacement(text, replace_eval)
            if masked is not None:
                draw.rectangle([x1, y1, x2, y2], fill=_REDACT_TEXT_BG)
                if font is not None:
                    draw.text((x1 + 2, y1 + 1), masked, fill=_REDACT_TEXT_FG, font=font)
                continue
            # replace_eval deliberately opted this one out (-> None):
            # fall through to the plain solid-rectangle redaction below.

        draw.rectangle([x1, y1, x2, y2], fill=fill_color)

    log.debug("redact: redacted %d detection(s)", len(detections))
    return result
