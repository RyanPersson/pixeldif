"""The virtual clock gives every recorded frame exactly one step of page time."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PAGE = """<!doctype html><canvas id=c width=64 height=16></canvas><script>
const c = document.getElementById('c').getContext('2d');
let start = null;
function frame(t) {
  start = start ?? performance.now();
  const ticks = Math.round((performance.now() - start) / (1000 / 60));
  c.fillStyle = `rgb(${ticks % 256}, ${(ticks >> 8) % 256}, 0)`;
  c.fillRect(0, 0, 64, 16);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>"""


def test_recorded_frames_are_exactly_one_step_apart(tmp_path):
    pytest.importorskip("playwright")
    from pixeldif.record import record
    page = tmp_path / "clock.html"
    page.write_text(PAGE)
    try:
        meta = record(page.as_uri(), tmp_path / "frames", frames=20, fps=60, width=200, height=100,
                      warmup=5, selector="#c", timeout=60)
    except Exception as error:  # no browser available
        pytest.skip(f"browser unavailable: {error}")
    assert meta["frames"] == 20
    values = [np.asarray(Image.open(p).convert("RGB"))[8, 32] for p in sorted((tmp_path / "frames").glob("*.png"))]
    ticks = [int(v[0]) + 256 * int(v[1]) for v in values]
    assert np.all(np.diff(ticks) == 1), ticks


GL_PAGE = """<!doctype html><canvas id=c width=64 height=16></canvas><script>
// Draws from a MessageChannel loop, not requestAnimationFrame, the way winit
// drives a Bevy app on the web.
const gl = document.getElementById('c').getContext('webgl2');
const start = performance.now();
const channel = new MessageChannel();
channel.port1.onmessage = () => {
  const ticks = Math.round((performance.now() - start) / (1000 / 60));
  gl.clearColor((ticks % 256) / 255, ((ticks >> 8) % 256) / 255, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  setTimeout(() => channel.port2.postMessage(0), 4);
};
channel.port2.postMessage(0);
requestAnimationFrame(function keepAlive() { requestAnimationFrame(keepAlive); });
</script>"""


def test_webgl_drawn_outside_animation_frames_is_captured_fresh(tmp_path):
    pytest.importorskip("playwright")
    from pixeldif.record import record
    page = tmp_path / "gl.html"
    page.write_text(GL_PAGE)
    try:
        meta = record(page.as_uri(), tmp_path / "frames", frames=30, fps=60, width=200, height=100,
                      warmup=5, selector="#c", timeout=60)
    except Exception as error:
        pytest.skip(f"browser unavailable: {error}")
    assert meta["frames"] == 30
    values = [np.asarray(Image.open(p).convert("RGB"))[8, 32] for p in sorted((tmp_path / "frames").glob("*.png"))]
    ticks = [int(v[0]) + 256 * int(v[1]) for v in values]
    assert np.all(np.diff(ticks) == 1), ticks
