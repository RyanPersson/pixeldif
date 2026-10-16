"""Discontinuities in a moving shot: changes that camera motion cannot explain.

A camera moving smoothly through a scene moves every visible pixel along a
smooth path, so frame t + 1 is predictable from frame t: estimate the optical
flow, warp t forward, and the prediction matches. What does not match is

* a *pop*: residual that spikes for one frame pair and is quiet before and
  after it (scatter or an LOD popping in, a texture swap, a shadow cascade
  switch, a light turning on);
* *shimmer*: residual that persists across frames without settling
  (temporal aliasing, z-fighting, noisy effects, flickering lights).

Areas entering the frame at its edges are masked: the prediction cannot see
them. Areas uncovered behind moving objects are not masked, because a pop
also breaks the flow's forward/backward consistency. They recur in every
frame while the object moves, so the before/after test removes them; the
neighbour tolerance grows with the fastest motion in the frame.

Per frame pair:
    r_t      residual of predicting frame t+1 from frame t (max over RGB)
    pop_t    r_t - max(dilate(r_{t-1}), dilate(r_{t+1}))  (neighbours dilated
             to tolerate slight misregistration between frame pairs)
    shimmer  per-pixel temporal median of r over the window
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import viz


@dataclass
class Event:
    frame: int  # index of the frame where the change appears (t + 1)
    box: tuple[int, int, int, int]
    area_fraction: float
    peak: float
    mean: float
    score: float  # area_fraction * mean, used for ranking

    def as_dict(self, fps: float, offset: int) -> dict:
        frame = self.frame + offset
        return {"frame": frame, "time_s": round(frame / fps, 3), "box_xywh": list(self.box),
                "area_fraction": round(self.area_fraction, 5), "peak": round(self.peak, 3),
                "mean": round(self.mean, 3), "score": round(self.score, 6)}


def gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


class Predictor:
    """DIS optical flow with forward/backward consistency."""

    def __init__(self, preset: str = "medium"):
        presets = {"fast": cv2.DISOPTICAL_FLOW_PRESET_FAST, "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
                   "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST}
        self.dis = cv2.DISOpticalFlow_create(presets[preset])

    def predict(self, a: np.ndarray, b: np.ndarray):
        """Frame a warped onto b, the in-frame mask, the flow-consistency
        mask and the flow b -> a."""
        ga, gb = gray(a), gray(b)
        back = self.dis.calc(gb, ga, None)  # for each pixel of b, where it was in a
        fwd = self.dis.calc(ga, gb, None)
        h, w = ga.shape
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        mx, my = xs + back[..., 0], ys + back[..., 1]
        warped = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        inside = (mx >= 1) & (mx <= w - 2) & (my >= 1) & (my <= h - 2)
        # A pixel's round trip b -> a -> b should come back to itself.
        fwd_at = cv2.remap(fwd, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        loop = np.hypot(back[..., 0] + fwd_at[..., 0], back[..., 1] + fwd_at[..., 1])
        speed = np.hypot(back[..., 0], back[..., 1])
        consistent = loop < 0.5 + 0.05 * speed
        # Shrink the in-frame area a little: flow is least reliable at edges.
        inside = cv2.erode(inside.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        return warped, inside, consistent, back


def residuals(frames: np.ndarray, preset: str = "medium", blur: float = 1.2):
    """Residual maps r[t] for pairs (t, t+1), in-frame masks, flow-consistency
    masks, and the median and 95th-percentile flow speed per pair."""
    predictor = Predictor(preset)
    res, masks, consistent, speeds, fast = [], [], [], [], []
    for t in range(len(frames) - 1):
        warped, inside, ok, back = predictor.predict(frames[t], frames[t + 1])
        r = np.abs(viz.to_float(frames[t + 1]) - viz.to_float(warped)).max(axis=2)
        if blur > 0:
            r = cv2.GaussianBlur(r, (0, 0), blur)
        r[~inside] = 0.0
        res.append(r)
        masks.append(inside)
        consistent.append(ok)
        speed = np.hypot(back[..., 0], back[..., 1])
        speeds.append(float(np.median(speed)))
        fast.append(float(np.percentile(speed, 95)))
    return np.stack(res), np.stack(masks), np.stack(consistent), speeds, fast


def pop_scores(res: np.ndarray, masks: np.ndarray, fast: list[float] | None = None,
               tolerance: int = 7) -> np.ndarray:
    """r_t minus the larger (dilated) neighbouring residual; edges use one side.
    The dilation covers the fastest motion, so a moving object's trailing
    edge in a neighbouring pair still overlaps its edge in this one."""
    n = len(res)
    dil = []
    for t in range(n):
        size = tolerance + 2 * int(min(40.0, fast[t] if fast else 0.0))
        dil.append(cv2.dilate(res[t], np.ones((size, size), np.uint8)))
    scores = np.zeros_like(res)
    for t in range(n):
        neighbours = [dil[k] for k in (t - 1, t + 1) if 0 <= k < n]
        if not neighbours:
            continue
        scores[t] = np.clip(res[t] - np.maximum.reduce(neighbours), 0.0, None)
        scores[t][~masks[t]] = 0.0
    return scores


def find_events(scores: np.ndarray, threshold: float, min_area: float, merge: int = 9) -> list[Event]:
    events = []
    h, w = scores.shape[1:]
    kernel = np.ones((merge, merge), np.uint8)
    for t, score in enumerate(scores):
        hot = (score > threshold).astype(np.uint8)
        if not hot.any():
            continue
        joined = cv2.dilate(hot, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
        for k in range(1, n):
            region = (labels == k) & (hot > 0)
            area = region.mean()
            if area < min_area:
                continue
            x, y, bw, bh = (int(v) for v in stats[k, :4])
            values = score[region]
            events.append(Event(t + 1, (x, y, bw, bh), float(area), float(values.max()),
                                float(values.mean()), float(area * values.mean())))
    return sorted(events, key=lambda e: -e.score)


def capture_checks(frames: np.ndarray, speeds: list[float]) -> dict:
    """Signs of a broken capture or a jumpy camera: duplicated frames, and
    frames whose overall motion spikes far above its running median."""
    diffs = [float(np.abs(frames[t + 1].astype(np.int16) - frames[t].astype(np.int16)).mean())
             for t in range(len(frames) - 1)]
    # A repeated frame only points at the capture when the picture is moving
    # on both sides of it (a still or slow camera repeats frames honestly).
    duplicates = [t + 1 for t, d in enumerate(diffs)
                  if d < 0.3 and 0 < t < len(diffs) - 1 and diffs[t - 1] > 1.0 and diffs[t + 1] > 1.0]
    jumps = []
    for t, v in enumerate(speeds):
        window = speeds[max(0, t - 15):t] + speeds[t + 1:t + 16]
        typical = float(np.median(window)) if window else 0.0
        if v > 4.0 * max(typical, 0.5) and v > 4.0:
            jumps.append({"frame": t + 1, "median_flow_px": round(v, 2), "typical_px": round(typical, 2)})
    return {"duplicate_frames": duplicates, "camera_jumps": jumps}


def event_panel(frames: np.ndarray, warped: np.ndarray, score: np.ndarray, event: Event,
                vmax: float, fps: float, offset: int, width: int = 320, note: str = "") -> list:
    """Before, after, the motion-compensated prediction and the highlight, cropped
    around the event, plus the full frame with its box."""
    t = event.frame
    x, y, w, h = event.box
    pad = max(16, int(0.35 * max(w, h)))
    H, W = score.shape
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)

    def cut(image):
        return image[y0:y1, x0:x1]

    label = f"frame {t + offset} ({(t + offset) / fps:.2f} s)"
    after = viz.to_float(frames[t])
    return [
        viz.tile(viz.draw_box(frames[t], event.box), label + (f" · {note}" if note else ""),
                 f"area {event.area_fraction * 100:.2f}%, peak {event.peak:.2f}", width),
        viz.tile(cut(viz.to_float(frames[t - 1])), "before (t-1)", "", width),
        viz.tile(cut(viz.to_float(warped)), "predicted from t-1", "flow-warped", width),
        viz.tile(cut(after), "after (t)", "", width),
        viz.tile(cut(viz.overlay_heat(after, score, vmax)), "unexplained change", f"pop score 0 to {vmax:.2f}", width),
    ]


def timeline(pop_energy: list[float], shimmer_energy: list[float], speeds: list[float],
             events: list[Event], fps: float, offset: int, path, x: list[float] | None = None,
             x_label: str = "time (s)", log_x: bool = False, invert_x: bool = False, title: str = ""):
    """Pop energy, unexplained change and flow speed per frame pair, against
    time or against a recorded per-frame value (e.g. altitude)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if x is None:
        xs = [(i + 1 + offset) / fps for i in range(len(pop_energy))]
    else:
        xs = x
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.6), sharex=True)
    fig.patch.set_facecolor("#121418")
    for ax in axes:
        ax.set_facecolor("#1a1d22")
        ax.tick_params(colors="#c8c8c8")
        for spine in ax.spines.values():
            spine.set_color("#555")
        if log_x:
            ax.set_xscale("log")
    axes[0].plot(xs, pop_energy, color="#ff7a3c", lw=1.2)
    axes[0].set_ylabel("pop energy", color="#e0e0e0")
    top = max(pop_energy) if pop_energy and max(pop_energy) > 0 else 1.0
    for rank, e in enumerate(events[:10]):
        k = e.frame - 1
        if 0 <= k < len(xs):
            axes[0].axvline(xs[k], color="#ffd04a", lw=0.8, alpha=0.6)
            axes[0].text(xs[k], top * (0.92 - 0.08 * (rank % 3)), f"#{rank + 1}", color="#ffd04a", fontsize=8)
    axes[1].plot(xs, shimmer_energy, color="#7ac8ff", lw=1.2)
    axes[1].set_ylabel("unexplained\nchange", color="#e0e0e0")
    axes[2].plot(xs, speeds, color="#9be08a", lw=1.2)
    axes[2].set_ylabel("median flow\n(px/frame)", color="#e0e0e0")
    axes[2].set_xlabel(x_label, color="#e0e0e0")
    if invert_x:
        axes[2].invert_xaxis()
    fig.suptitle(title or "pixeldif pops: transient changes motion cannot explain", color="white")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
