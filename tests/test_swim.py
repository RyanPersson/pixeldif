"""Known-answer tests for swim: texture fixed to the world, travelling with
the camera, and travelling at half its speed."""
import json

import numpy as np
import pytest
from PIL import Image

from pixeldif import swim, synth
from pixeldif.cli import main


def test_depth_codes_round_trip():
    metres = np.array([[0.01, 1.0, 1234.5], [7.1e7, 1e12, np.inf]])
    back = swim.decode_depth(swim.encode_depth(metres))
    assert np.isinf(back[1, 2])
    finite = np.isfinite(metres)
    assert np.allclose(back[finite], metres[finite], rtol=3e-6)


def test_quaternions_round_trip():
    m = synth.camera_matrix(37.0, -21.0)
    assert np.allclose(swim.quaternion_matrix(swim.matrix_quaternion(m)), m, atol=1e-9)


@pytest.fixture(scope="module")
def flight():
    return synth.swim_flight(n=24)


def _region(values, box, shape, margin=0.04):
    h, w = shape
    bx, by, bw, bh = box
    x0, x1 = int((bx + margin) * w), int((bx + bw - margin) * w)
    y0, y1 = int((by + margin) * h), int((by + bh - margin) * h)
    v = values[y0:y1, x0:x1]
    return v[np.isfinite(v)]


def test_anchoring_reads_world_viewer_and_half_speed_texture(flight):
    frames, depths, probes, boxes = flight
    poses = [swim.Pose.from_probe(p) for p in probes]
    analyser = swim.Analyser()
    stack = np.stack([analyser.pair(frames[t], frames[t + 1], depths[t], depths[t + 1], poses[t], poses[t + 1]).anchoring
                      for t in range(len(frames) - 1)])
    shape = frames.shape[1:3]
    for name, (box, expected) in boxes.items():
        values = _region(stack.transpose(1, 2, 0), box, shape).ravel()
        assert values.size > 500, name
        assert abs(np.median(values) - expected) < 0.12, (name, np.median(values))
    # The world-fixed ground between and above the two patches.
    world = _region(stack.transpose(1, 2, 0), (0.4, 0.55, 0.2, 0.4), shape).ravel()
    world = np.concatenate([world, _region(stack.transpose(1, 2, 0), (0.05, 0.3, 0.9, 0.2), shape).ravel()])
    assert world.size > 500
    assert abs(np.median(world)) < 0.06, np.median(world)


def test_cli_finds_the_travelling_patches(flight, tmp_path):
    frames, depths, probes, boxes = flight
    for folder, images in (("color", frames), ("depth", [swim.encode_depth(d) for d in depths])):
        (tmp_path / folder).mkdir()
        for i, image in enumerate(images):
            Image.fromarray(image).save(tmp_path / folder / f"{i:05d}.png")
        (tmp_path / folder / "pixeldif.json").write_text(json.dumps({"fps": 30}))
        (tmp_path / folder / "frames.jsonl").write_text(
            "".join(json.dumps({"frame": i, "value": p}) + "\n" for i, p in enumerate(probes)))
    out = tmp_path / "out"
    main(["swim", str(tmp_path / "color"), "--depth", str(tmp_path / "depth"), "--out", str(out), "--rows", "2"])
    report = json.loads((out / "report.json").read_text())
    assert report["depth_pairing"]["unmatched_frames"] == []
    h, w = frames.shape[1:3]
    centre = boxes["viewer"][0]
    cx, cy = (centre[0] + centre[2] / 2) * w, (centre[1] + centre[3] / 2) * h
    hit = [r for r in report["fine"]["regions"]
           if r["box_xywh"][0] <= cx <= r["box_xywh"][0] + r["box_xywh"][2]
           and r["box_xywh"][1] <= cy <= r["box_xywh"][1] + r["box_xywh"][3]]
    assert hit and hit[0]["median_anchoring"] > 0.8, report["fine"]["regions"]
    assert (out / "swim.png").exists()


SHIMMER_BOX = (0.38, 0.32, 0.26, 0.2)


@pytest.fixture(scope="module")
def hazy_flight():
    return synth.swim_flight(n=16, shimmer_box=SHIMMER_BOX)


def test_shimmer_finds_heat_haze_that_anchoring_misses(hazy_flight):
    frames, depths, probes, boxes = hazy_flight
    poses = [swim.Pose.from_probe(p) for p in probes]
    analyser = swim.Analyser()
    results = [analyser.pair(frames[t], frames[t + 1], depths[t], depths[t + 1], poses[t], poses[t + 1])
               for t in range(len(frames) - 1)]
    size, shimmer, per_pair = swim.nonrigid_maps(results)
    shape = frames.shape[1:3]
    anchoring = np.stack([r.anchoring for r in results]).transpose(1, 2, 0)
    # The haze crawls over world-fixed ground: anchoring reads 0, shimmer doesn't.
    assert abs(np.median(_region(anchoring, SHIMMER_BOX, shape))) < 0.1
    assert np.median(_region(shimmer, SHIMMER_BOX, shape)) > 0.5
    assert np.median(_region(shimmer, (0.4, 0.62, 0.2, 0.3), shape)) < 0.2
    # Steady slides have a large non-rigid flow but hardly any shimmer.
    for name in ("viewer", "half"):
        box = boxes[name][0]
        assert np.median(_region(size, box, shape)) > 1.0, name
        assert np.median(_region(shimmer, box, shape)) < 0.2, name
    assert np.isnan(per_pair[0]) and len(per_pair) == len(results)


def test_cli_reports_the_shimmering_patch(hazy_flight, tmp_path):
    frames, depths, probes, boxes = hazy_flight
    for folder, images in (("color", frames), ("depth", [swim.encode_depth(d) for d in depths])):
        (tmp_path / folder).mkdir()
        for i, image in enumerate(images):
            Image.fromarray(image).save(tmp_path / folder / f"{i:05d}.png")
        (tmp_path / folder / "pixeldif.json").write_text(json.dumps({"fps": 30}))
        (tmp_path / folder / "frames.jsonl").write_text(
            "".join(json.dumps({"frame": i, "value": p}) + "\n" for i, p in enumerate(probes)))
    out = tmp_path / "out"
    main(["swim", str(tmp_path / "color"), "--depth", str(tmp_path / "depth"), "--out", str(out), "--rows", "2"])
    report = json.loads((out / "report.json").read_text())
    h, w = frames.shape[1:3]
    cx, cy = (SHIMMER_BOX[0] + SHIMMER_BOX[2] / 2) * w, (SHIMMER_BOX[1] + SHIMMER_BOX[3] / 2) * h
    hit = [r for r in report["nonrigid"]["regions"]
           if r["box_xywh"][0] <= cx <= r["box_xywh"][0] + r["box_xywh"][2]
           and r["box_xywh"][1] <= cy <= r["box_xywh"][1] + r["box_xywh"][3]]
    assert hit and hit[0]["median_shimmer_px"] > 0.5, report["nonrigid"]["regions"]
    assert report["nonrigid"]["median_shimmer_px"] < 0.3
