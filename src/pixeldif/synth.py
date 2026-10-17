"""Synthetic clips with known answers, for tests and demonstrations.

`zoom_pop`: a camera zooming and panning smoothly over a textured landscape
(every pixel moves, nothing should be flagged), with an object that pops into
existence and a patch of ground whose texture is swapped (an LOD change).

`steam`: a fixed camera on a static scene with a slowly billowing plume and a
fast-flickering light, for the motion views.

`swim_flight`: a camera flying over textured ground (turning slowly) with its
poses and depth frames; one patch of ground texture travels with the camera
and one at half its speed, for `swim`.
"""
from __future__ import annotations

import cv2
import numpy as np


def noise_texture(size: int, seed: int, octaves=(256, 64, 16, 4)) -> np.ndarray:
    """Coloured multi-octave value noise, float32 RGB in [0, 1]."""
    rng = np.random.default_rng(seed)
    field = np.zeros((size, size, 3), np.float32)
    weight = 1.0
    for cells in octaves:
        n = max(2, size // cells)
        coarse = rng.random((n, n, 3)).astype(np.float32)
        field += weight * cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC)
        weight *= 0.55
    field -= field.min()
    field /= field.max()
    tint = np.array([0.45, 0.62, 0.35], np.float32)  # greenish landscape
    return np.clip(0.25 + 0.6 * field * tint / tint.max(), 0, 1)


def _render(texture: np.ndarray, zoom: float, cx: float, cy: float, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    m = np.array([[zoom, 0, w / 2 - zoom * cx], [0, zoom, h / 2 - zoom * cy]], np.float32)
    out = cv2.warpAffine(texture, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def zoom_pop(n: int = 90, size=(640, 360), pop_frame: int = 40, swap_frame: int = 65, seed: int = 3):
    """Frames and ground truth: [{'frame', 'kind', 'box_world'}...]."""
    tex = noise_texture(2048, seed)
    alt = noise_texture(2048, seed + 101, octaves=(32, 8, 2))
    world_pop = (1060.0, 1000.0, 22.0)  # object centre and radius in texture pixels
    swap_box = (930, 1030, 90, 60)  # x, y, w, h in texture pixels
    with_object = tex.copy()
    cv2.circle(with_object, (int(world_pop[0]), int(world_pop[1])), int(world_pop[2]), (0.95, 0.85, 0.3), -1, cv2.LINE_AA)
    swapped = with_object.copy()
    x, y, w, h = swap_box
    swapped[y:y + h, x:x + w] = 0.3 + 0.6 * alt[y:y + h, x:x + w] * np.array([0.9, 0.7, 0.5], np.float32)
    frames = []
    for f in range(n):
        s = f / max(1, n - 1)
        zoom = 0.9 * (1.0 + 1.6 * s * s * (3 - 2 * s))  # smooth ease in/out zoom
        cx = 1024 + 40 * np.sin(1.3 * s)
        cy = 1024 + 25 * s
        texture = swapped if f >= swap_frame else with_object if f >= pop_frame else tex
        frames.append(_render(texture, zoom, cx, cy, size))
    truth = [{"frame": pop_frame, "kind": "pop", "world": world_pop},
             {"frame": swap_frame, "kind": "texture swap", "world": swap_box}]
    return np.stack(frames), truth


def steam(n: int = 120, size=(640, 360), fps: float = 60.0, seed: int = 5):
    """A static scene with a billowing plume (slow) and a flickering lamp (fast)."""
    w, h = size
    rng = np.random.default_rng(seed)
    background = noise_texture(1024, seed, octaves=(128, 32, 8))[:h, :w] * 0.8
    blobs = rng.random((40, 4))  # x, y, radius, phase
    frames = []
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for f in range(n):
        t = f / fps
        image = background.copy()
        plume = np.zeros((h, w), np.float32)
        for bx, by, br, bp in blobs:
            # Puffs rise and spread slowly, wobbling sideways.
            px = w * (0.35 + 0.3 * bx) + 12 * np.sin(1.7 * t + 6 * bp)
            py = h * (0.85 - 0.6 * ((by + 0.12 * t) % 1.0))
            r = 18 + 40 * br * (1 + 0.6 * ((by + 0.12 * t) % 1.0))
            plume += np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * r * r))
        alpha = np.clip(plume * 0.6, 0, 0.9)[..., None]
        image = image * (1 - alpha) + 0.92 * alpha
        # A lamp flickering at about 12 Hz.
        lamp = 0.5 + 0.5 * np.sin(2 * np.pi * 12 * t + 3 * np.sin(2 * np.pi * 5 * t))
        cv2.circle(image, (int(0.85 * w), int(0.25 * h)), 14, (1.0, 0.8 * lamp + 0.2, 0.3 * lamp), -1, cv2.LINE_AA)
        frames.append((np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8))
    return np.stack(frames)


def camera_matrix(heading_deg: float, pitch_deg: float) -> np.ndarray:
    """Camera-to-world rotation for a camera looking along -Z with +Y up,
    turned by heading (from -Z toward +X) and pitch (negative down)."""
    h, p = np.radians(heading_deg), np.radians(pitch_deg)
    forward = np.array([np.sin(h) * np.cos(p), np.sin(p), -np.cos(h) * np.cos(p)])
    right = np.array([np.cos(h), 0.0, np.sin(h)])
    up = np.cross(right, forward)
    return np.stack([right, up, -forward], axis=1)


SWIM_BOXES = {"viewer": ((0.06, 0.56, 0.3, 0.36), 1.0), "half": ((0.64, 0.56, 0.3, 0.36), 0.5)}


def swim_flight(n: int = 40, size=(320, 180), fov_deg: float = 60.0, height: float = 40.0, seed: int = 11,
                supersample: int = 2):
    """Frames, depth frames (metres along the ray, inf for sky), probe dicts
    with the camera pose, and the truth: {name: (box as frame fractions,
    expected anchoring)}; the rest of the ground is fixed to the world."""
    from .swim import camera_rays, matrix_quaternion
    w, h = size
    W, H = w * supersample, h * supersample
    texel = 0.1  # metres per texture pixel
    texture = noise_texture(1024, seed, octaves=(128, 32, 12))
    texture = cv2.cvtColor(texture, cv2.COLOR_RGB2HSV)
    texture[..., 1] = np.clip(texture[..., 1] * 1.8, 0, 1)
    texture = cv2.cvtColor(texture, cv2.COLOR_HSV2RGB)
    ys, xs = np.mgrid[0:H, 0:W]
    masks = {}
    for name, ((bx, by, bw, bh), _) in SWIM_BOXES.items():
        masks[name] = (xs >= bx * W) & (xs < (bx + bw) * W) & (ys >= by * H) & (ys < (by + bh) * H)
    velocity = np.array([0.8, 0.0, -1.2])
    frames, depths, probes = [], [], []
    for f in range(n):
        position = np.array([0.0, height, 0.0]) + velocity * f
        rotation = camera_matrix(0.15 * f, -35.0)
        rays = camera_rays(W, H, np.radians(fov_deg)) @ rotation.T
        down = rays[..., 1] < -0.02
        distance = np.where(down, height / np.where(down, -rays[..., 1], 1.0), np.inf)
        point = position + np.where(down, distance, 0.0)[..., None] * rays
        image = np.empty((H, W, 3), np.float32)
        for name, travel in (("world", 0.0), ("viewer", 1.0), ("half", 0.5)):
            mask = masks.get(name, ~(masks["viewer"] | masks["half"]))
            ground = point - travel * position
            mx = (ground[..., 0] / texel).astype(np.float32)
            my = (ground[..., 2] / texel).astype(np.float32)
            sampled = cv2.remap(texture, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            image[mask] = sampled[mask]
        sky = np.array([0.55, 0.7, 0.9], np.float32) * (0.8 + 0.2 * np.clip(rays[..., 1:2], 0, 1))
        image = np.where(down[..., None], image, sky)
        frames.append((np.clip(cv2.resize(image, size, interpolation=cv2.INTER_AREA), 0, 1) * 255 + 0.5).astype(np.uint8))
        small = cv2.resize(np.where(down, distance, 1e9), size, interpolation=cv2.INTER_NEAREST)
        depths.append(np.where(small < 1e8, small, np.inf))
        probes.append({"position_m": position.tolist(), "rotation_xyzw": matrix_quaternion(rotation),
                       "fov_y_deg": fov_deg})
    return np.stack(frames), depths, probes, SWIM_BOXES
