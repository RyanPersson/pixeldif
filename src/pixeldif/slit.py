"""Slit-scan (time-slice) images: one line of pixels from every frame, stacked.

Time runs down the image. Something still draws a vertical stripe; something
moving steadily draws a slanted line; waving grass and billowing steam draw
wavy or braided texture; a pop or a texture swap draws a horizontal seam where
every pixel along the slit changes at once. A model can read all of that from
one still image.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import viz


def parse_line(text: str, width: int, height: int) -> tuple[float, float, float, float]:
    """'row:0.5', 'col:0.3' (fractions) or 'x0,y0,x1,y1' (fractions or pixels)."""
    text = text.strip()
    if text.startswith("row:"):
        y = float(text[4:]) * (height - 1 if float(text[4:]) <= 1 else 1)
        return 0.0, y, width - 1.0, y
    if text.startswith("col:"):
        x = float(text[4:]) * (width - 1 if float(text[4:]) <= 1 else 1)
        return x, 0.0, x, height - 1.0
    values = [float(v) for v in text.split(",")]
    if len(values) != 4:
        raise ValueError("line is row:Y, col:X or x0,y0,x1,y1")
    if all(v <= 1.0 for v in values):
        values = [values[0] * (width - 1), values[1] * (height - 1), values[2] * (width - 1), values[3] * (height - 1)]
    return tuple(values)


def sample(frames: np.ndarray, line: tuple[float, float, float, float]) -> np.ndarray:
    """(n, length, 3) uint8: the line sampled in every frame."""
    x0, y0, x1, y1 = line
    length = max(2, int(round(np.hypot(x1 - x0, y1 - y0))) + 1)
    xs = np.linspace(x0, x1, length, dtype=np.float32)[None, :]
    ys = np.linspace(y0, y1, length, dtype=np.float32)[None, :]
    rows = [cv2.remap(f, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[0] for f in frames]
    return np.stack(rows)


def change_image(slit: np.ndarray) -> np.ndarray:
    """|row t+1 - row t|: steady motion is faint, pops are bright seams."""
    f = viz.to_float(slit)
    d = np.abs(np.diff(f, axis=0)).max(axis=2)
    return np.concatenate([np.zeros_like(d[:1]), d], axis=0)


def seams(slit: np.ndarray, factor: float = 4.0) -> list[dict]:
    """Rows where the whole-line change jumps well above its local median."""
    energy = change_image(slit).mean(axis=1)
    out = []
    for t in range(1, len(energy)):
        lo, hi = max(1, t - 6), min(len(energy), t + 7)
        local = np.median(np.r_[energy[lo:t], energy[t + 1:hi]]) if hi - lo > 1 else 0.0
        if energy[t] > factor * max(local, 2e-3) and energy[t] > 0.01:
            out.append({"row": t, "change": round(float(energy[t]), 4), "local_median": round(float(local), 4)})
    return out
