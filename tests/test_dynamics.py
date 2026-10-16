"""Known-answer tests for extent (front speeds) and flow (streaming speed)."""
import cv2
import numpy as np

from pixeldif import dynamics, synth


def background(w=320, h=200, seed=1):
    return (synth.noise_texture(512, seed)[:h, :w] * 255).astype(np.uint8)


def test_extent_measures_a_growing_cloud_front():
    # A bright blob growing from (160, 150): 2 px/frame sideways, 3 px/frame up.
    base = background()
    frames = []
    for i in range(40):
        f = base.copy()
        cv2.ellipse(f, (160, 150), (6 + 2 * i, 4 + 3 * i), 0, 180, 360, (250, 245, 240), -1)
        frames.append(f)
    reaches, _ = dynamics.reach(np.stack(frames), base, (160, 149), threshold=0.12)
    right = dynamics.front_speed([r.right for r in reaches], fps=30.0)
    up = dynamics.front_speed([r.up for r in reaches], fps=30.0)
    assert abs(np.median(right[5:-5]) - 60.0) < 4.0, np.median(right[5:-5])  # 2 px/frame at 30 fps
    assert abs(np.median(up[5:-5]) - 90.0) < 6.0, np.median(up[5:-5])
    # Static surroundings never join the region.
    assert reaches[0].area_px < 1000


def test_flow_measures_streaming_speed_and_direction():
    texture = (synth.noise_texture(512, 4, octaves=(64, 16, 4)) * 255).astype(np.uint8)
    frames = np.stack([np.roll(texture, 2 * i, axis=1)[:200, :320] for i in range(12)])  # 2 px/frame to the right
    stats, _ = dynamics.flow_stats(frames, roi=(40, 40, 240, 120))
    assert abs(np.median([s.median_speed for s in stats]) - 2.0) < 0.25
    assert abs(np.median([s.direction_deg for s in stats])) < 10
    assert np.median([s.coherence for s in stats]) > 0.9
