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
    """Return a PIL TrueType font, falling back gracefully to the built-in default.

    Tries well-known absolute paths first so large point sizes work even when
    the process CWD is not a font directory.
    """
    try:
        from PIL import ImageFont
    except ImportError:
        return None

    candidates = [
        # Linux (Debian/Ubuntu, Fedora, Arch, etc.)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        # macOS (system + user Library)
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSText.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Helvetica.ttc",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/Arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        # Bare names (CWD or fontconfig path)
        "DejaVuSans.ttf",
        "arial.ttf",
        "Arial.ttf",
        "Helvetica.ttc",
    ]
    for path in candidates:
        try:
            # .ttc collections need an explicit face index on some Pillow builds
            if path.lower().endswith(".ttc"):
                return ImageFont.truetype(path, size, index=0)
            return ImageFont.truetype(path, size)
        except (OSError, IOError, ValueError):
            continue
    try:
        # PIL built-in bitmap font – always available but tiny / non-scalable
        return ImageFont.load_default()
    except Exception:  # noqa: BLE001
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


def _measure_text(draw: Any, text: str, font: Any) -> tuple[int, int]:
    """Return (width, height) of *text* rendered with *font*."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    except AttributeError:
        tw, th = draw.textsize(text, font=font)
        return int(tw), int(th)


def _wrap_text(draw: Any, text: str, font: Any, max_w: int) -> list[str]:
    """Greedy word-wrap *text* so each line fits within *max_w* pixels."""
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = current + " " + word
        tw, _ = _measure_text(draw, trial, font)
        if tw <= max_w:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _block_size(draw: Any, lines: list[str], font: Any, line_gap: float = 1.15) -> tuple[int, int]:
    """Return (width, height) of a multi-line block."""
    if not lines:
        return 0, 0
    widths = []
    heights = []
    for line in lines:
        tw, th = _measure_text(draw, line, font)
        widths.append(tw)
        heights.append(th)
    block_w = max(widths) if widths else 0
    # Approximate inter-line spacing from the tallest single line
    line_h = max(heights) if heights else 0
    block_h = int(line_h * line_gap * (len(lines) - 1) + line_h) if lines else 0
    return block_w, block_h


def _fit_centered_text(
    draw: Any,
    text: str,
    box: tuple[int, int, int, int],
    *,
    target_frac: float = 0.9,
    fill: str = "#000000",
) -> None:
    """Draw *text* centred inside *box*, auto-scaled to *target_frac* of the area.

    Longer strings are word-wrapped.  Binary-searches a TrueType font size
    so the resulting multi-line block occupies roughly ``target_frac`` of
    the box width *and* height (tighter constraint wins).
    """
    x1, y1, x2, y2 = box
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    max_w = max(1, int(box_w * target_frac))
    max_h = max(1, int(box_h * target_frac))

    lo, hi = 4, max(box_h * 2, 32)
    best_font = None
    best_lines: list[str] = [text]
    best_bw = best_bh = 0

    for _ in range(20):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        font = _get_font(mid)
        if font is None:
            break

        lines = _wrap_text(draw, text, font, max_w)
        bw, bh = _block_size(draw, lines, font)

        if bw <= max_w and bh <= max_h:
            best_font = font
            best_lines = lines
            best_bw, best_bh = bw, bh
            lo = mid + 1
        else:
            hi = mid - 1

    if best_font is None:
        best_font = _get_font(max(12, min(box_h // 10, 48)))
        if best_font is None:
            return
        best_lines = _wrap_text(draw, text, best_font, max_w)
        best_bw, best_bh = _block_size(draw, best_lines, best_font)

    # Vertical start so the whole block is centred
    ty = y1 + (box_h - best_bh) // 2
    # Per-line height for stepping
    _, single_h = _measure_text(draw, best_lines[0] if best_lines else "X", best_font)
    line_step = int(single_h * 1.15)

    for i, line in enumerate(best_lines):
        tw, _ = _measure_text(draw, line, best_font)
        tx = x1 + (box_w - tw) // 2
        draw.text((tx, ty + i * line_step), line, fill=fill, font=best_font)


def redact(
    image: "PILImage.Image",
    detections: list[Detection],
    *,
    fill_color: str = _REDACT_COLOR,
    replace_minchar: int = 0,
    replace_eval: str = "",
    font_size: int = 14,
    redact_text: str = "",
    redact_bgcolor: str = "",
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
        that particular detection).  Overridden by *redact_bgcolor*
        when the latter is non-empty.
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
    redact_text, redact_bgcolor:
        Optional plugin-level overlay.  When *redact_text* is non-empty
        the rectangle is filled with *redact_bgcolor* (or *fill_color*
        if the bgcolor is empty) and the text is drawn centred and
        auto-scaled to ~90 % of the detection area.
    """
    _, ImageDraw = _get_draw_module()

    result = image.copy()
    draw = ImageDraw.Draw(result)

    if not detections:
        log.debug("redact: no detections to redact")
        return result

    font = _get_font(font_size) if replace_eval else None
    effective_fill = redact_bgcolor or fill_color

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

        draw.rectangle([x1, y1, x2, y2], fill=effective_fill)

        if redact_text:
            _fit_centered_text(
                draw,
                redact_text,
                (x1, y1, x2, y2),
                target_frac=0.9,
                fill="#000000",
            )

    log.debug("redact: redacted %d detection(s)", len(detections))
    return result
