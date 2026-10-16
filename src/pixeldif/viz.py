"""Drawing helpers: colour maps, labelled tiles, grids and simple plots.

Sheets are meant to be read by a model as a single still image, so every tile
carries its own label (what it shows, the lag in frames and milliseconds,
and a headline number).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def font(size: int = 16) -> ImageFont.ImageFont:
    for path in _FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def to_float(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float32) / 255.0


def to_uint8(image: np.ndarray) -> np.ndarray:
    return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def luminance(image: np.ndarray) -> np.ndarray:
    """Rec. 709 weights on (gamma-encoded) RGB in [0, 1]."""
    return image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722


def colormap(values: np.ndarray, vmax: float, name: str = "inferno") -> np.ndarray:
    """Scalar field in [0, vmax] -> RGB float image."""
    maps = {"inferno": cv2.COLORMAP_INFERNO, "turbo": cv2.COLORMAP_TURBO, "magma": cv2.COLORMAP_MAGMA}
    scaled = np.clip(values / max(vmax, 1e-6), 0.0, 1.0)
    bgr = cv2.applyColorMap((scaled * 255).astype(np.uint8), maps[name])
    return bgr[..., ::-1].astype(np.float32) / 255.0


def overlay_heat(frame: np.ndarray, values: np.ndarray, vmax: float, dim: float = 0.35) -> np.ndarray:
    """Heat map blended over a darkened, desaturated copy of the frame."""
    grey = luminance(frame)[..., None].repeat(3, axis=2) * dim
    heat = colormap(values, vmax)
    alpha = np.clip(values / max(vmax, 1e-6), 0.0, 1.0)[..., None] ** 0.6
    return grey * (1.0 - alpha) + heat * alpha


def tile(image: np.ndarray, title: str, subtitle: str = "", width: int = 400) -> Image.Image:
    """A float or uint8 image scaled to `width` with a caption bar above it."""
    array = to_uint8(image) if image.dtype != np.uint8 else image
    h, w = array.shape[:2]
    height = max(1, int(round(h * width / w)))
    picture = Image.fromarray(array).resize((width, height), Image.Resampling.LANCZOS)
    bar = 44 if subtitle else 26
    canvas = Image.new("RGB", (width, height + bar), (18, 20, 24))
    canvas.paste(picture, (0, bar))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), title, fill=(235, 235, 235), font=font(15))
    if subtitle:
        draw.text((6, 23), subtitle, fill=(170, 180, 190), font=font(13))
    return canvas


def grid(tiles: list[list[Image.Image]], heading: str = "", gap: int = 4) -> Image.Image:
    """Rows of tiles (rows may differ in length) under an optional heading."""
    widths = [sum(t.width for t in row) + gap * (len(row) - 1) for row in tiles]
    heights = [max(t.height for t in row) for row in tiles]
    top = 34 if heading else 0
    sheet = Image.new("RGB", (max(widths), top + sum(heights) + gap * (len(tiles) - 1)), (8, 9, 11))
    draw = ImageDraw.Draw(sheet)
    if heading:
        draw.text((8, 7), heading, fill=(255, 255, 255), font=font(18))
    y = top
    for row, height in zip(tiles, heights):
        x = 0
        for t in row:
            sheet.paste(t, (x, y))
            x += t.width + gap
        y += height + gap
    return sheet


def draw_box(image: np.ndarray, box: tuple[int, int, int, int], color=(0.2, 1.0, 0.4), thickness: int = 2) -> np.ndarray:
    out = np.ascontiguousarray(to_uint8(image) if image.dtype != np.uint8 else image.copy())
    x, y, w, h = box
    cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), tuple(int(c * 255) for c in color), thickness)
    return out


def save(image: Image.Image | np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(to_uint8(image) if image.dtype != np.uint8 else image)
    image.save(path)
    return path
