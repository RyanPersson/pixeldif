"""How fast things spread and stream, in pixels (or metres) per second.

`extent`: a disturbance growing from a source point (an explosion, a launch
cloud, a splash) against a frame taken before it started. Each frame is
compared with that base frame; the changed region connected to the source is
the disturbance, and its reach left, right and up from the source over time
gives the speed of its fronts. Static or slowly drifting surroundings (fog,
water) change little against the base, so they stay out of the region.

`flow`: the optical-flow field inside a region over time: how fast fog
streams along the ground, how fast smoke churns. It reports the median and
90th-percentile speed of the moving pixels and their mean direction.

With `scale` (metres per pixel at the subject's distance) both report metres
and metres per second, so a game capture and a reference video of different
framing compare in physical units.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import viz


@dataclass
class Reach:
    frame: int
    area_px: int
    left: float
    right: float
    up: float
    box: tuple[int, int, int, int]


def change_mask(frame: np.ndarray, base: np.ndarray, threshold: float, blur: float = 2.0,
                exclude: np.ndarray | None = None) -> np.ndarray:
    a = cv2.GaussianBlur(viz.to_float(frame), (0, 0), blur)
    b = cv2.GaussianBlur(viz.to_float(base), (0, 0), blur)
    # Cameras re-expose as a bright disturbance grows. Match the frame's
    # overall level to the base per channel, measured away from the
    # disturbance (`exclude`), so the exposure change isn't counted.
    keep = np.ones(a.shape[:2], bool) if exclude is None else exclude == 0
    if keep.sum() > 100:
        for c in range(3):
            ratio = np.median((a[..., c][keep] + 0.02) / (b[..., c][keep] + 0.02))
            a[..., c] = (a[..., c] + 0.02) / ratio - 0.02
    mask = (np.abs(a - b).max(axis=2) > threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def reach(frames: np.ndarray, base: np.ndarray, origin: tuple[float, float], threshold: float,
          roi: tuple[int, int, int, int] | None = None, blur: float = 2.0,
          join: int = 40) -> tuple[list[Reach], list[np.ndarray]]:
    """Per frame, the changed region connected to the origin and its reach.
    Blobs closer than `join` pixels count as connected, so a gap that opens
    at the source (a launch cloud clearing around the mount) doesn't lose
    the disturbance."""
    out, masks = [], []
    ox, oy = origin
    previous = None
    for i, frame in enumerate(frames):
        exclude = None
        if previous is not None and previous.any():
            exclude = cv2.dilate(previous, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
        mask = change_mask(frame, base, threshold, blur, exclude)
        if roi is not None:
            x, y, w, h = roi
            keep = np.zeros_like(mask)
            keep[y:y + h, x:x + w] = 1
            mask &= keep
        linked = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * join + 1, 2 * join + 1))) if join > 0 else mask
        count, labels, stats, _ = cv2.connectedComponentsWithStats(linked, connectivity=8)
        component = np.zeros_like(mask)
        if count > 1:
            # The component under the origin, else the one nearest to it.
            label = labels[int(round(oy)), int(round(ox))] if 0 <= oy < mask.shape[0] and 0 <= ox < mask.shape[1] else 0
            if label == 0:
                ys, xs = np.nonzero(labels)
                if len(xs):
                    k = np.argmin((xs - ox) ** 2 + (ys - oy) ** 2)
                    label = labels[ys[k], xs[k]]
            component = ((labels == label) & (mask > 0)).astype(np.uint8) if label else component
        ys, xs = np.nonzero(component)
        if len(xs):
            box = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            out.append(Reach(i, int(len(xs)), float(max(ox - xs.min(), 0)), float(max(xs.max() - ox, 0)),
                             float(max(oy - ys.min(), 0)), box))
        else:
            out.append(Reach(i, 0, 0.0, 0.0, 0.0, (0, 0, 0, 0)))
        masks.append(component)
        previous = component
    return out, masks


def front_speed(values: list[float], fps: float, window: float = 0.5) -> list[float]:
    """Speed of a reach series (units per second): a least-squares slope over
    a sliding window, which tolerates the jagged edge of a billowing cloud."""
    v = np.asarray(values, np.float64)
    half = max(1, int(round(window * fps / 2)))
    t = np.arange(len(v)) / fps
    out = []
    for i in range(len(v)):
        lo, hi = max(0, i - half), min(len(v), i + half + 1)
        if hi - lo < 3:
            out.append(0.0)
            continue
        out.append(float(np.polyfit(t[lo:hi], v[lo:hi], 1)[0]))
    return out


@dataclass
class FlowStats:
    frame: int
    moving_fraction: float
    median_speed: float
    p90_speed: float
    direction_deg: float  # mean direction of moving pixels, 0 = right, 90 = up (image)
    coherence: float  # |mean unit vector|: 1 = all moving the same way, 0 = churning


def flow_stats(frames: np.ndarray, roi=None, preset: str = "medium", min_speed: float = 0.15,
               every: int = 1) -> tuple[list[FlowStats], list[tuple[int, np.ndarray]]]:
    """Flow between frames t and t+1 (every `every` frames) inside `roi`."""
    presets = {"fast": cv2.DISOPTICAL_FLOW_PRESET_FAST, "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
               "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST}
    dis = cv2.DISOpticalFlow_create(presets[preset])
    stats, fields = [], []
    for i in range(0, len(frames) - 1, every):
        a = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        flow = dis.calc(a, b, None)
        if roi is not None:
            x, y, w, h = roi
            flow = flow[y:y + h, x:x + w]
        speed = np.linalg.norm(flow, axis=2)
        moving = speed > min_speed
        if moving.any():
            v = flow[moving]
            s = speed[moving]
            unit = v / s[:, None]
            mean = unit.mean(axis=0)
            direction = float(np.degrees(np.arctan2(-mean[1], mean[0])))
            stats.append(FlowStats(i, float(moving.mean()), float(np.median(s)), float(np.percentile(s, 90)),
                                   direction, float(np.linalg.norm(mean))))
        else:
            stats.append(FlowStats(i, 0.0, 0.0, 0.0, 0.0, 0.0))
        fields.append((i, flow))
    return stats, fields


def flow_image(flow: np.ndarray, vmax: float) -> np.ndarray:
    """Hue = direction, brightness = speed (white = still)."""
    speed = np.linalg.norm(flow, axis=2)
    angle = np.arctan2(-flow[..., 1], flow[..., 0])
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = np.clip(speed / max(vmax, 1e-6) * 255, 0, 255).astype(np.uint8)
    hsv[..., 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def contour_overlay(frame: np.ndarray, mask: np.ndarray, origin: tuple[float, float]) -> np.ndarray:
    image = viz.to_uint8(viz.to_float(frame)).copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (60, 255, 120), 2)
    cv2.drawMarker(image, (int(origin[0]), int(origin[1])), (255, 60, 60), cv2.MARKER_CROSS, 18, 2)
    return image


def plot_series(series: dict[str, list[float]], times: list[float], path, title: str, ylabel: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=110)
    for name, values in series.items():
        ax.plot(times, values, label=name, linewidth=1.6)
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
