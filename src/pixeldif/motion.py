"""Motion at several time scales, for a fixed camera (Posy's motion extraction).

For a base frame t and lags L, the difference f(t + L) - f(t) cancels the
static parts of the picture and keeps what changed. Short lags show fast
motion (flicker, turbulence); long lags show slow motion (drift, billowing).

`decorrelation` turns that into numbers. For each lag and spatial scale it
measures how much of the picture's structure at that scale has been replaced
after the lag:

    D(L, s) = rms(band_s(Y[t+L] - Y[t])) / (sqrt(2) * rms(band_s(Y[t])))

`band_s` is a difference-of-Gaussians band-pass around s pixels and Y is
luminance. D is about 0 when the structure is unchanged and about 1 when it
is fully replaced (two unrelated images with the same statistics). D across
(L, s) is a signature of the motion's texture: steam that billows has small
D at short lags rising slowly; shimmering or noisy effects have large D even
at one frame; a static effect stays near 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import viz
from .source import Source

SCALES = (1, 2, 4, 8, 16, 32)


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


def band(image: np.ndarray, sigma: float) -> np.ndarray:
    fine = image if sigma < 0.75 else _blur(image, sigma)
    return fine - _blur(image, 2.0 * sigma)


def crop(array: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    if roi is None:
        return array
    x, y, w, h = roi
    return array[..., y:y + h, x:x + w, :] if array.ndim == 4 else array[y:y + h, x:x + w, ...]


@dataclass
class Signature:
    """Decorrelation and change statistics for one clip region."""

    fps: float
    lags: list[int]
    scales: tuple[int, ...]
    decorrelation: np.ndarray  # (lags, scales)
    change_rms: np.ndarray  # (lags, scales), absolute band rms of the difference
    mean_abs: list[float]  # mean |difference| per lag (all channels)
    changed_fraction: list[float]  # fraction of pixels with |difference| > 4 %
    samples: int
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "fps": self.fps,
            "lags_frames": self.lags,
            "lags_ms": [round(l / self.fps * 1000, 1) for l in self.lags],
            "scales_px": list(self.scales),
            "decorrelation": np.round(self.decorrelation, 4).tolist(),
            "change_rms": np.round(self.change_rms, 5).tolist(),
            "mean_abs_change": [round(v, 5) for v in self.mean_abs],
            "changed_fraction": [round(v, 4) for v in self.changed_fraction],
            "samples": self.samples,
            **self.extra,
        }


def signature(frames: np.ndarray, fps: float, lags: list[int], starts: list[int],
              roi=None, scales=SCALES) -> Signature:
    """Average statistics over the base frames `starts` (indices into frames)."""
    lum = [viz.luminance(viz.to_float(crop(f, roi))) for f in frames]
    decor = np.zeros((len(lags), len(scales)))
    change = np.zeros_like(decor)
    mean_abs = np.zeros(len(lags))
    changed = np.zeros(len(lags))
    count = np.zeros(len(lags))
    base_bands = {}
    for i, lag in enumerate(lags):
        for s in starts:
            if s + lag >= len(frames):
                continue
            d = lum[s + lag] - lum[s]
            rgb = np.abs(viz.to_float(crop(frames[s + lag], roi)) - viz.to_float(crop(frames[s], roi)))
            mean_abs[i] += rgb.mean()
            changed[i] += (rgb.max(axis=2) > 0.04).mean()
            for j, sigma in enumerate(scales):
                key = (s, j)
                if key not in base_bands:
                    base_bands[key] = float(np.sqrt(np.mean(band(lum[s], sigma) ** 2)))
                rms = float(np.sqrt(np.mean(band(d, sigma) ** 2)))
                change[i, j] += rms
                decor[i, j] += rms / (np.sqrt(2.0) * max(base_bands[key], 1e-4))
            count[i] += 1
    count = np.maximum(count, 1)
    return Signature(
        fps=fps, lags=lags, scales=tuple(scales),
        decorrelation=decor / count[:, None], change_rms=change / count[:, None],
        mean_abs=(mean_abs / count).tolist(), changed_fraction=(changed / count).tolist(),
        samples=int(count.min()),
    )


def timescale_map(frames: np.ndarray, lags: list[int], starts: list[int], roi=None,
                  floor: float = 0.02, blur: float = 3.0):
    """Per pixel, the lag at which local change reaches half its largest value.

    Returns (lag map in frames, peak local change). Flicker and fast motion
    saturate within the first lags; slow drift keeps growing to the longest."""
    energy = []
    for lag in lags:
        acc = None
        n = 0
        for s in starts:
            if s + lag >= len(frames):
                continue
            mag = change_magnitude(crop(frames[s], roi), crop(frames[s + lag], roi))
            mag = _blur(mag, blur)
            acc = mag if acc is None else acc + mag
            n += 1
        energy.append(acc / max(n, 1))
    energy = np.stack(energy)  # (lags, h, w)
    peak = energy.max(axis=0)
    reached = energy >= 0.5 * peak[None]
    first = reached.argmax(axis=0)
    log_lags = np.log2(np.array(lags, np.float32))
    # Interpolate within the bracketing lags for a smooth map.
    prev = np.maximum(first - 1, 0)
    e0 = np.take_along_axis(energy, prev[None], 0)[0]
    e1 = np.take_along_axis(energy, first[None], 0)[0]
    frac = np.clip((0.5 * peak - e0) / np.maximum(e1 - e0, 1e-6), 0, 1)
    frac = np.where(first == 0, 1.0, frac)
    lag_log = log_lags[prev] + frac * (log_lags[first] - log_lags[prev])
    return np.exp2(lag_log), peak


def timescale_tile(frame: np.ndarray, lag_map: np.ndarray, peak: np.ndarray, lags: list[int], fps: float,
                   width: int, floor: float = 0.02):
    lo, hi = np.log2(lags[0]), np.log2(lags[-1])
    t = (np.log2(lag_map) - lo) / max(hi - lo, 1e-6)
    # Skip turbo's dark ends: fast change must read as bright blue, not black.
    color = viz.colormap(0.12 + 0.83 * np.clip(t, 0, 1), 1.0, "turbo")
    alpha = np.clip((peak - floor) / (3 * floor), 0, 1)[..., None]
    grey = viz.luminance(viz.to_float(frame))[..., None].repeat(3, axis=2) * 0.3
    image = grey * (1 - alpha) + color * alpha
    return viz.tile(image, "time scale of change",
                    f"blue {lags[0] / fps * 1000:.0f} ms (fast) to red {lags[-1] / fps * 1000:.0f} ms (slow); dark = still",
                    width)


def posy(a: np.ndarray, b: np.ndarray, gain: float = 2.0) -> np.ndarray:
    """Posy's overlay: an inverted copy over a time-shifted copy, 50 % each.

    0.5 * (1 - a) + 0.5 * b = 0.5 + 0.5 * (b - a): static areas cancel to
    mid grey, changes keep their colour (amplified by `gain`)."""
    return np.clip(0.5 + 0.5 * gain * (viz.to_float(b) - viz.to_float(a)), 0.0, 1.0)


def change_magnitude(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(viz.to_float(b) - viz.to_float(a)).max(axis=2)


def lag_label(lag: int, fps: float) -> str:
    return f"lag {lag} f = {lag / fps * 1000:.0f} ms"


def motion_rows(frames: np.ndarray, base: int, lags: list[int], fps: float, roi=None,
                gain: float = 2.0, vmax: float | None = None, width: int = 360, name: str = "",
                starts: list[int] | None = None):
    """Two rows of tiles (Posy overlays, heat maps) and the shared heat scale.
    The heat row starts with the time-scale map."""
    a = crop(frames[base], roi)
    mags = [change_magnitude(a, crop(frames[min(base + l, len(frames) - 1)], roi)) for l in lags]
    if vmax is None:
        vmax = max(0.05, max(float(np.percentile(m, 99.5)) for m in mags))
    prefix = f"{name}: " if name else ""
    top = [viz.tile(viz.to_float(a), f"{prefix}frame", "", width)]
    lag_map, peak = timescale_map(frames, lags, starts or [base], roi)
    bottom = [timescale_tile(a, lag_map, peak, lags, fps, width)]
    for lag, mag in zip(lags, mags):
        b = crop(frames[min(base + lag, len(frames) - 1)], roi)
        top.append(viz.tile(posy(a, b, gain), f"{prefix}{lag_label(lag, fps)}",
                            f"overlay x{gain:g}, mean change {mag.mean() * 100:.1f}%", width))
        bottom.append(viz.tile(viz.overlay_heat(viz.to_float(a), mag, vmax), f"{prefix}{lag_label(lag, fps)}",
                               f"heat 0 to {vmax:.2f}; changed >4%: {(mag > 0.04).mean() * 100:.0f}% of pixels", width))
    return top, bottom, vmax


def matrix_tile(values: np.ndarray, lags: list[int], fps: float, scales, title: str,
                vmax: float = 1.0, diverging: bool = False, fmt: str = "{:.2f}") -> "viz.Image.Image":
    """A labelled heat-map table: rows are lags, columns are spatial scales."""
    from PIL import Image, ImageDraw
    cell_w, cell_h, left, top = 64, 26, 118, 58
    w = left + cell_w * len(scales) + 8
    h = top + cell_h * len(lags) + 8
    image = Image.new("RGB", (w, h), (18, 20, 24))
    draw = ImageDraw.Draw(image)
    draw.text((6, 4), title, fill=(235, 235, 235), font=viz.font(15))
    draw.text((left, 30), "spatial scale (px)", fill=(170, 180, 190), font=viz.font(12))
    for j, s in enumerate(scales):
        draw.text((left + j * cell_w + 20, top - 16), str(s), fill=(200, 200, 200), font=viz.font(12))
    for i, lag in enumerate(lags):
        draw.text((6, top + i * cell_h + 6), lag_label(lag, fps).replace("lag ", ""), fill=(200, 200, 200), font=viz.font(12))
        for j in range(len(scales)):
            v = float(values[i, j])
            if diverging:
                t = np.clip(v / vmax, -1, 1)
                color = (int(60 + 180 * max(t, 0)), int(60 + 40 * (1 - abs(t))), int(60 + 180 * max(-t, 0)))
            else:
                c = viz.colormap(np.array([[v]]), vmax, "turbo")[0, 0]
                color = tuple(int(x * 255 * 0.8) for x in c)
            x0, y0 = left + j * cell_w, top + i * cell_h
            draw.rectangle([x0, y0, x0 + cell_w - 2, y0 + cell_h - 2], fill=color)
            draw.text((x0 + 8, y0 + 5), fmt.format(v), fill=(255, 255, 255), font=viz.font(12))
    return image


def load_window(source: Source, start: int, lags: list[int], window: float, samples: int, width: int | None):
    """Frames from `start` covering the sampled base frames plus the longest lag."""
    span = max(1, int(round(window * source.fps)))
    count = span + max(lags) + 1
    frames = source.read(start, count, width)
    last = max(0, len(frames) - 1 - max(lags))
    starts = sorted({int(round(v)) for v in np.linspace(0, min(span, last), max(1, samples))})
    return frames, starts
