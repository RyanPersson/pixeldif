"""Known-answer tests on synthetic clips."""
import numpy as np
import pytest

from pixeldif import motion, pops, slit, synth
from pixeldif.source import parse_lag, parse_roi


def test_lags_parse_frames_milliseconds_and_seconds():
    assert parse_lag("4", 60) == 4
    assert parse_lag("4f", 30) == 4
    assert parse_lag("50ms", 60) == 3
    assert parse_lag("0.5s", 30) == 15
    assert parse_lag("1ms", 30) == 1  # never below one frame
    with pytest.raises(ValueError):
        parse_lag("soon", 60)


def test_roi_accepts_pixels_and_fractions():
    assert parse_roi("0.5,0,0.5,1", 640, 360) == (320, 0, 320, 360)
    assert parse_roi("10,20,30,40", 640, 360) == (10, 20, 30, 40)


@pytest.fixture(scope="module")
def zoom():
    return synth.zoom_pop(n=80, pop_frame=36, swap_frame=58)


def test_pops_finds_the_pop_and_the_swap_but_not_the_zoom(zoom):
    frames, truth = zoom
    res, masks, _, speeds, fast = pops.residuals(frames)
    scores = pops.pop_scores(res, masks, fast)
    events = pops.find_events(scores, threshold=0.08, min_area=0.0004)
    found = sorted({e.frame for e in events})
    assert found == [36, 58], found
    # The camera really moves: several pixels per frame by the end.
    assert max(speeds) > 2.0


def test_smooth_motion_alone_has_no_events():
    frames, _ = synth.zoom_pop(n=50, pop_frame=999, swap_frame=999)
    res, masks, _, _, fast = pops.residuals(frames)
    events = pops.find_events(pops.pop_scores(res, masks, fast), threshold=0.08, min_area=0.0004)
    assert events == []


def test_decorrelation_grows_with_lag_for_drift_and_is_zero_for_a_still():
    rng = np.random.default_rng(1)
    texture = (synth.noise_texture(512, 7) * 255).astype(np.uint8)
    drift = np.stack([np.roll(texture, k, axis=1)[:200, :300] for k in range(0, 40)])
    still = np.stack([texture[:200, :300]] * 40)
    sig = motion.signature(drift, 60.0, [1, 4, 16], [0, 4, 8], scales=(2, 8))
    flat = motion.signature(still, 60.0, [1, 4, 16], [0, 4, 8], scales=(2, 8))
    d = sig.decorrelation
    assert d[0, 0] < d[1, 0] < d[2, 0]
    assert np.all(flat.decorrelation < 1e-3)


def test_slit_seams_mark_the_pop_frame(zoom):
    frames, truth = zoom
    h, w = frames.shape[1:3]
    # A column through the popped object (it sits right of the centre).
    rows = slit.sample(frames, (0.0, h * 0.5, w - 1.0, h * 0.5))
    seams = [s["row"] for s in slit.seams(rows)]
    assert 36 in seams or 58 in seams, seams
