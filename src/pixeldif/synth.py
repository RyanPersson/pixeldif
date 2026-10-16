"""Synthetic clips with known answers, for tests and demonstrations.

`zoom_pop`: a camera zooming and panning smoothly over a textured landscape
(every pixel moves, nothing should be flagged), with an object that pops into
existence and a patch of ground whose texture is swapped (an LOD change).

`steam`: a fixed camera on a static scene with a slowly billowing plume and a
fast-flickering light, for the motion views.
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
