"""Swimming: texture that moves with the viewer instead of with the world.

`pops` warps each frame by the optical flow it measures, so a texture that
slides along with the camera is just more motion to it: explained, never
flagged. `swim` warps by the motion the world allows instead. With the camera
pose of every frame and the distance each pixel shows (a second recording of
the same camera path in the renderer's depth view), every world-fixed point
has exactly one place to be in the next frame:

    world flow   w   where the point a pixel shows lands in the next frame
    viewer flow  c   where it would land if it travelled with the camera (only
                     the camera's turn moves it, like something infinitely far)
    measured     m   optical flow (DIS, forward/backward consistent)

    anchoring    a = (m - w).(c - w) / |c - w|^2
                 0: fixed to the world, 1: moves with the viewer, 0.5: slides
                 along at half the camera's speed. Slip across (c - w), such
                 as a scrolling texture, is kept separately.

Only pixels where c and w differ by at least `min_separation` pixels can tell
the two apart (near things while the camera translates); flat areas (nothing
to track) and points hidden in the next frame are left out.

    leftover     |next(x + w) - this(x)|: what the world warp fails to predict.
                 Swimming texture, but also texture that morphs as the camera
                 moves (resampled at positions tied to the camera), which
                 optical flow would "explain" too, and view-dependent shading
                 (highlights and reflections really do move with the viewer).

    non-rigid    r = m - w wherever the flow is trusted (no separation needed):
                 motion the camera's move through a rigid world doesn't
                 explain, in pixels. Its size catches steady slides (swimming,
                 sway) in any direction, including far away where w and c
                 nearly agree; its frame-to-frame change at the same pixel,
                 the *shimmer*, catches heat-haze-like distortion that crawls
                 with the viewpoint and averages out of any median. A steady
                 offset (a slightly wrong depth) adds to the size, not to the
                 shimmer.

The world-stabilized view warps later frames back into a reference frame by
the world motion: world-fixed content stands still in it, swimming crawls.

Conventions. Poses: `position_m` (or `position`) [x, y, z] in metres in a
frame where the scene is static, `rotation_xyzw` camera-to-frame quaternion,
camera looking along -Z with +Y up (OpenGL/Bevy), `fov_y_deg` vertical field
of view, square pixels. Depth frames: distance along the pixel's ray as a
24-bit code in RGB (R high byte), distance = 10^(code / (2^24 - 1) * 14 - 2)
metres (1 cm to 10^12 m); code 0 means nothing there (sky).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import cv2
import numpy as np

from . import viz

DEPTH_LOG_MIN, DEPTH_LOG_SPAN, DEPTH_CODES = -2.0, 14.0, 16777215.0


def decode_depth(rgb: np.ndarray) -> np.ndarray:
    """Depth-view frame (uint8 RGB) -> metres along the ray, inf for nothing."""
    code = (rgb[..., 0].astype(np.int64) << 16) | (rgb[..., 1].astype(np.int64) << 8) | rgb[..., 2]
    metres = 10.0 ** (code / DEPTH_CODES * DEPTH_LOG_SPAN + DEPTH_LOG_MIN)
    return np.where(code > 0, metres, np.inf)


def encode_depth(metres: np.ndarray) -> np.ndarray:
    """Metres (inf for nothing) -> depth-view RGB, the inverse of decode_depth."""
    finite = np.isfinite(metres) & (metres > 0)
    v = (np.log10(np.where(finite, metres, 1.0)) - DEPTH_LOG_MIN) / DEPTH_LOG_SPAN
    code = np.where(finite, np.clip(np.round(v * DEPTH_CODES), 1, DEPTH_CODES), 0).astype(np.int64)
    return np.stack([(code >> 16) & 255, (code >> 8) & 255, code & 255], axis=-1).astype(np.uint8)


def quaternion_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_quaternion(m: np.ndarray) -> list[float]:
    """3x3 rotation -> [x, y, z, w]."""
    t = np.trace(m)
    if t > 0:
        k = 0.5 / np.sqrt(t + 1)
        q = [(m[2, 1] - m[1, 2]) * k, (m[0, 2] - m[2, 0]) * k, (m[1, 0] - m[0, 1]) * k, 0.25 / k]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k_ = (i + 1) % 3, (i + 2) % 3
        r = np.sqrt(1 + m[i, i] - m[j, j] - m[k_, k_])
        q = [0.0, 0.0, 0.0, (m[k_, j] - m[j, k_]) / (2 * r)]
        q[i] = r / 2
        q[j] = (m[j, i] + m[i, j]) / (2 * r)
        q[k_] = (m[k_, i] + m[i, k_]) / (2 * r)
    return [float(v) for v in q]


@dataclass
class Pose:
    position: np.ndarray  # metres, float64
    rotation: np.ndarray  # camera-to-frame 3x3
    fov_y: float  # radians

    @staticmethod
    def from_probe(value, key: str | None = None) -> "Pose | None":
        if not isinstance(value, dict):
            return None
        if key:
            value = value.get(key)
        elif "position_m" not in value and "position" not in value:
            value = value.get("camera") or value.get("pose")
        if not isinstance(value, dict):
            return None
        position = value.get("position_m", value.get("position"))
        rotation = value.get("rotation_xyzw", value.get("rotation"))
        fov = value.get("fov_y_deg")
        if position is None or rotation is None or fov is None:
            return None
        return Pose(np.asarray(position, np.float64), quaternion_matrix(rotation), np.radians(float(fov)))

    def difference(self, other: "Pose") -> tuple[float, float]:
        """Position (m) and rotation (degrees) between two poses."""
        relative = self.rotation.T @ other.rotation
        angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
        return float(np.linalg.norm(self.position - other.position)), float(angle)


def camera_rays(width: int, height: int, fov_y: float) -> np.ndarray:
    """Unit ray directions in camera space through each pixel centre (h, w, 3)."""
    ty = np.tan(fov_y / 2)
    tx = ty * width / height
    x = ((np.arange(width) + 0.5) / width * 2 - 1) * tx
    y = (1 - (np.arange(height) + 0.5) / height * 2) * ty
    xx, yy = np.meshgrid(x, y)
    rays = np.stack([xx, yy, -np.ones_like(xx)], axis=-1)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def project(points: np.ndarray, width: int, height: int, fov_y: float):
    """Camera-space points -> pixel coordinates (u, v) and distance; points
    behind the camera get NaN."""
    ty = np.tan(fov_y / 2)
    tx = ty * width / height
    forward = -points[..., 2]
    ahead = forward > 1e-9
    safe = np.where(ahead, forward, 1.0)
    u = (points[..., 0] / safe / tx + 1) / 2 * width - 0.5
    v = (1 - points[..., 1] / safe / ty) / 2 * height - 0.5
    u[~ahead] = np.nan
    v[~ahead] = np.nan
    return u, v, np.linalg.norm(points, axis=-1)


def expected_flows(depth: np.ndarray, a: Pose, b: Pose):
    """World flow, viewer flow (h, w, 2) from frame a to b, and the distance
    at which b should see each world point (for the occlusion test)."""
    h, w = depth.shape
    rays = camera_rays(w, h, a.fov_y) @ a.rotation.T  # frame axes
    finite = np.isfinite(depth)
    offset = np.where(finite, depth, 0.0)[..., None] * rays
    shift = a.position - b.position
    to_b = b.rotation  # camera_b = R_b^T (frame vector)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    # A point with no depth (sky) is infinitely far: only the turn moves it.
    world_points = np.where(finite[..., None], offset + shift, rays) @ to_b
    viewer_points = np.where(finite[..., None], offset, rays) @ to_b
    uw, vw, dist = project(world_points, w, h, b.fov_y)
    uc, vc, _ = project(viewer_points, w, h, b.fov_y)
    world = np.stack([uw - xs, vw - ys], axis=-1)
    viewer = np.stack([uc - xs, vc - ys], axis=-1)
    dist = np.where(finite, dist, np.inf)
    return world.astype(np.float32), viewer.astype(np.float32), dist


def smooth_depth(depth: np.ndarray) -> np.ndarray:
    """5x5 median of log distance: volumetric depth (ray-marched clouds) is
    noisy at the step size; sky stays infinite."""
    finite = np.isfinite(depth)
    if not finite.any():
        return depth
    logd = np.log(np.where(finite, depth, np.nanmax(np.where(finite, depth, np.nan)))).astype(np.float32)
    out = np.exp(cv2.medianBlur(logd, 5).astype(np.float64))
    return np.where(finite, out, np.inf)


def resize_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    if depth.shape[1] == width and depth.shape[0] == height:
        return depth
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)


def sample(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx = np.nan_to_num(xs + flow[..., 0], nan=-10.0).astype(np.float32)
    my = np.nan_to_num(ys + flow[..., 1], nan=-10.0).astype(np.float32)
    return cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


@dataclass
class PairResult:
    anchoring: np.ndarray  # NaN where it can't be measured
    slip_across: np.ndarray  # px/frame across (c - w), NaN where not measured
    leftover_world: np.ndarray  # residual after the world warp
    leftover_flow: np.ndarray  # residual after the optical-flow warp (what pops sees)
    separation: np.ndarray  # |c - w| px
    measured: np.ndarray
    world: np.ndarray
    valid: np.ndarray
    inside: np.ndarray
    nonrigid: np.ndarray  # measured - world flow (h, w, 2) px, NaN where the flow isn't trusted


class Analyser:
    def __init__(self, preset: str = "medium", min_separation: float = 0.5, min_texture: float = 2e-5,
                 depth_tolerance: float = 0.25, blur: float = 1.2):
        presets = {"fast": cv2.DISOPTICAL_FLOW_PRESET_FAST, "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
                   "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST}
        self.dis = cv2.DISOpticalFlow_create(presets[preset])
        self.min_separation = min_separation
        self.min_texture = min_texture
        self.depth_tolerance = depth_tolerance
        self.blur = blur

    def pair(self, a: np.ndarray, b: np.ndarray, depth_a: np.ndarray, depth_b: np.ndarray,
             pose_a: Pose, pose_b: Pose) -> PairResult:
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        fwd = self.dis.calc(ga, gb, None)
        back = self.dis.calc(gb, ga, None)
        h, w = ga.shape
        back_at = sample(back, fwd)
        loop = np.hypot(fwd[..., 0] + back_at[..., 0], fwd[..., 1] + back_at[..., 1])
        consistent = loop < 0.5 + 0.05 * np.hypot(fwd[..., 0], fwd[..., 1])
        world, viewer, dist_b = expected_flows(depth_a, pose_a, pose_b)
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        tx, ty = xs + world[..., 0], ys + world[..., 1]
        inside = (tx >= 2) & (tx <= w - 3) & (ty >= 2) & (ty <= h - 3)
        inside &= (xs >= 2) & (xs <= w - 3) & (ys >= 2) & (ys <= h - 3)
        # Visible in b: b's depth where the point lands matches its distance.
        seen = sample(depth_b.astype(np.float32), world)
        with np.errstate(invalid="ignore", divide="ignore"):
            visible = np.where(np.isfinite(dist_b), np.abs(np.log(seen / dist_b)) < self.depth_tolerance,
                               ~np.isfinite(seen))
        texture = cv2.cornerMinEigenVal(ga.astype(np.float32) / 255.0, 7)
        e = viewer - world
        separation = np.hypot(e[..., 0], e[..., 1])
        trusted = inside & visible & consistent & (texture > self.min_texture)
        valid = trusted & (separation >= self.min_separation)
        d = fwd - world
        nonrigid = np.where(trusted[..., None], d, np.nan).astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            anchoring = (d[..., 0] * e[..., 0] + d[..., 1] * e[..., 1]) / separation ** 2
            across = (d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]) / separation
        anchoring = np.where(valid, anchoring, np.nan).astype(np.float32)
        across = np.where(valid, across, np.nan).astype(np.float32)
        fa, fb = viz.to_float(a), viz.to_float(b)
        left_world = np.abs(sample(fb, world) - fa).max(axis=2)
        left_flow = np.abs(sample(fb, fwd) - fa).max(axis=2)
        for r in (left_world, left_flow):
            r[~(inside & visible) | ~np.isfinite(r)] = 0.0
        if self.blur > 0:
            left_world = cv2.GaussianBlur(left_world, (0, 0), self.blur)
            left_flow = cv2.GaussianBlur(left_flow, (0, 0), self.blur)
        return PairResult(anchoring, across, left_world, left_flow, separation, fwd, world, valid, inside & visible,
                          nonrigid)


def nonrigid_size(r: PairResult) -> np.ndarray:
    return np.hypot(r.nonrigid[..., 0], r.nonrigid[..., 1])


def nonrigid_maps(results: list[PairResult], min_share: float = 0.2):
    """Clip maps of the non-rigid flow r = m - w (px): its median size, and the
    shimmer, the root-mean-square change of r between successive pairs at each
    pixel. Also the per-pair median shimmer (NaN for the first pair)."""
    sizes = np.stack([nonrigid_size(r) for r in results])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        size = np.nanmedian(sizes, axis=0)
        size[np.isfinite(sizes).sum(axis=0) < max(2, int(min_share * len(results)))] = np.nan
        shimmer = np.full(size.shape, np.nan, np.float32)
        per_pair = [float("nan")]
        if len(results) > 1:
            total = np.zeros(size.shape, np.float64)
            count = np.zeros(size.shape, np.int32)
            for a, b in zip(results, results[1:]):
                change = np.hypot(b.nonrigid[..., 0] - a.nonrigid[..., 0], b.nonrigid[..., 1] - a.nonrigid[..., 1])
                known = np.isfinite(change)
                total[known] += change[known] ** 2
                count += known
                per_pair.append(float(np.median(change[known])) if known.any() else float("nan"))
            shimmer = np.sqrt(total / np.maximum(count, 1)).astype(np.float32)
            shimmer[count < max(1, int(min_share * (len(results) - 1)))] = np.nan
    return size.astype(np.float32), shimmer, per_pair


def stabilize(frame: np.ndarray, depth_ref: np.ndarray, ref: Pose, pose: Pose):
    """`frame` (seen from `pose`) resampled into the reference view: each
    reference pixel shows what `frame` has at the world point it saw. Also
    the mask of reference pixels that land inside `frame`."""
    world, _, _ = expected_flows(depth_ref, ref, pose)
    out = sample(viz.to_float(frame), world)
    inside = np.isfinite(out).all(axis=2)
    return np.nan_to_num(out, nan=0.0), inside


def anchoring_colors(values: np.ndarray) -> np.ndarray:
    """0 blue (fixed to the world), 0.5 yellow, 1 red (moves with the viewer),
    beyond 1.5 magenta (faster than the viewer); below 0 cyan."""
    stops = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
    colors = np.array([[0.2, 0.95, 1.0], [0.15, 0.4, 1.0], [1.0, 0.85, 0.15], [1.0, 0.2, 0.1], [1.0, 0.2, 0.9]])
    v = np.clip(np.nan_to_num(values, nan=0.0), stops[0], stops[-1])
    return np.stack([np.interp(v, stops, colors[:, k]) for k in range(3)], axis=-1).astype(np.float32)


def anchoring_overlay(frame: np.ndarray, anchoring: np.ndarray, opacity: float = 0.75) -> np.ndarray:
    grey = viz.luminance(viz.to_float(frame))[..., None].repeat(3, axis=2) * 0.45
    known = np.isfinite(anchoring)[..., None]
    return np.where(known, grey * (1 - opacity) + anchoring_colors(anchoring) * opacity, grey)


def arrows(frame: np.ndarray, measured: np.ndarray, world: np.ndarray, valid: np.ndarray, step: int = 24) -> np.ndarray:
    """Measured flow (yellow) and world flow (cyan) on a grid of valid pixels,
    scaled so the typical arrow is about a grid step long."""
    out = viz.to_uint8(viz.luminance(viz.to_float(frame))[..., None].repeat(3, axis=2) * 0.5)
    out = np.ascontiguousarray(out)
    h, w = valid.shape
    speeds = np.hypot(world[..., 0], world[..., 1])[valid]
    scale = 0.6 * step / max(float(np.median(speeds)) if speeds.size else 1.0, 0.05)
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if not valid[y, x]:
                continue
            for flow, color in ((world, (80, 220, 255)), (measured, (255, 220, 60))):
                dx, dy = flow[y, x] * scale
                cv2.arrowedLine(out, (x, y), (int(round(x + dx)), int(round(y + dy))), color, 1, cv2.LINE_AA, tipLength=0.3)
    return out


def crop_box(anchoring: np.ndarray, size: tuple[int, int], threshold: float = 0.5) -> tuple[int, int, int, int]:
    """The window of `size` (w, h) with the most pixels above `threshold`
    (by default the most viewer-anchored ones)."""
    w, h = size
    H, W = anchoring.shape
    w, h = min(w, W), min(h, H)
    hot = (np.nan_to_num(anchoring, nan=0.0) > threshold).astype(np.float32)
    score = cv2.boxFilter(hot, -1, (w, h), normalize=False, borderType=cv2.BORDER_CONSTANT)
    score[:, : w // 2] = score[:, W - (w - w // 2):] = -1
    score[: h // 2] = score[H - (h - h // 2):] = -1
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(x - w // 2), int(y - h // 2), w, h


def regions(anchoring: np.ndarray, threshold: float, min_area: float, merge: int = 9,
            key: str = "median_anchoring") -> list[dict]:
    """Connected areas whose median anchoring (or other map, named `key` in
    the result) is above `threshold`."""
    hot = (np.nan_to_num(anchoring, nan=-1.0) > threshold).astype(np.uint8)
    joined = cv2.morphologyEx(hot, cv2.MORPH_CLOSE, np.ones((merge, merge), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    out = []
    for k in range(1, n):
        region = (labels == k) & np.isfinite(anchoring)
        area = float((labels == k).mean())
        if area < min_area or not region.any():
            continue
        x, y, bw, bh = (int(v) for v in stats[k, :4])
        out.append({"box_xywh": [x, y, bw, bh], "area_fraction": round(area, 5),
                    key: round(float(np.median(anchoring[region])), 3)})
    return sorted(out, key=lambda r: -r["area_fraction"] * r[key])


def timeline(series: dict, x: list[float], x_label: str, path, log_x: bool = False, invert_x: bool = False,
             title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.6), sharex=True)
    fig.patch.set_facecolor("#121418")
    for ax in axes:
        ax.set_facecolor("#1a1d22")
        ax.tick_params(colors="#c8c8c8")
        for spine in ax.spines.values():
            spine.set_color("#555")
        if log_x:
            ax.set_xscale("log")
    axes[0].plot(x, series["median_anchoring"], color="#ff7a3c", lw=1.2, label="median anchoring")
    axes[0].plot(x, series["viewer_share"], color="#ffd04a", lw=1.0, label="share above 0.5")
    axes[0].axhline(0, color="#4a78ff", lw=0.8)
    axes[0].axhline(1, color="#ff3a2a", lw=0.8)
    axes[0].set_ylabel("anchoring\n0 world, 1 viewer", color="#e0e0e0")
    axes[0].legend(facecolor="#1a1d22", labelcolor="#e0e0e0", fontsize=8, loc="upper left")
    axes[1].plot(x, series["leftover_world"], color="#7ac8ff", lw=1.2, label="after world warp")
    axes[1].plot(x, series["leftover_flow"], color="#9be08a", lw=1.0, label="after flow warp (pops)")
    axes[1].set_ylabel("leftover", color="#e0e0e0")
    axes[1].legend(facecolor="#1a1d22", labelcolor="#e0e0e0", fontsize=8, loc="upper left")
    if "shimmer_px" in series:
        axes[2].plot(x, series["nonrigid_px"], color="#ff8fd0", lw=1.2, label="median non-rigid flow |m - w|")
        axes[2].plot(x, series["shimmer_px"], color="#ffe08a", lw=1.0, label="median shimmer (change since the last pair)")
        axes[2].set_ylabel("non-rigid\nflow (px)", color="#e0e0e0")
        axes[2].legend(facecolor="#1a1d22", labelcolor="#e0e0e0", fontsize=8, loc="upper left")
    else:
        axes[2].plot(x, series["measured_fraction"], color="#c8a0ff", lw=1.2)
        axes[2].set_ylabel("measurable\nshare of frame", color="#e0e0e0")
    axes[2].set_xlabel(x_label, color="#e0e0e0")
    if invert_x:
        axes[2].invert_xaxis()
    fig.suptitle(title or "pixeldif swim: does the picture move with the world or with the viewer?", color="white")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
