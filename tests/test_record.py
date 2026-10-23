"""The virtual clock gives every recorded frame exactly one step of page time."""
import json
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


KEY_PAGE = """<!doctype html><canvas id=c tabindex=0 width=64 height=16></canvas><script>
const canvas = document.getElementById('c');
const c = canvas.getContext('2d');
const held = new Set();
canvas.addEventListener('keydown', e => held.add(e.code));
canvas.addEventListener('keyup', e => held.delete(e.code));
function frame() {
  c.fillStyle = held.has('KeyW') ? '#fff' : '#000';
  c.fillRect(0, 0, 64, 16);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>"""


def test_held_keys_reach_the_canvas_for_the_whole_recording(tmp_path):
    pytest.importorskip("playwright")
    from pixeldif.record import record
    page = tmp_path / "keys.html"
    page.write_text(KEY_PAGE)
    try:
        meta = record(page.as_uri(), tmp_path / "frames", frames=10, fps=60, width=200, height=100,
                      warmup=3, selector="#c", timeout=60, hold=["KeyW"])
    except Exception as error:
        pytest.skip(f"browser unavailable: {error}")
    assert meta["hold"] == ["KeyW"]
    values = [np.asarray(Image.open(p).convert("L"))[8, 32] for p in sorted((tmp_path / "frames").glob("*.png"))]
    assert all(v == 255 for v in values), values


def test_only_pages_on_this_machine_count_as_local():
    from pixeldif.record import is_local
    for url in ("file:///tmp/game.html", "http://localhost:8020/play.html", "http://127.0.0.1:8020/",
                "https://127.0.0.2/x", "http://[::1]:8000/"):
        assert is_local(url), url
    for url in ("https://example.com/", "http://192.168.1.10/", "http://localhost.example.com/",
                "http://10.0.0.1/", "javascript:alert(1)", "about:blank"):
        assert not is_local(url), url


HOSTILE_PAGE = """<!doctype html><canvas id=c width=64 height=16></canvas><script>
// The page can reach the recorder's frame binding itself: out-of-range
// frames and anything but PNG data must be refused.
window.attempts = null;
Promise.allSettled([
  window.__pixeldifFrame(-1, 'data:image/png;base64,AAAA', null),
  window.__pixeldifFrame(1000000, 'data:image/png;base64,AAAA', null),
  window.__pixeldifFrame('../escape', 'data:image/png;base64,AAAA', null),
  window.__pixeldifFrame(1.5, 'data:image/png;base64,AAAA', null),
  window.__pixeldifFrame(0, 'data:text/html;base64,PGgxPg==', null),
]).then(results => { window.attempts = results.map(r => r.status); });
const c = document.getElementById('c').getContext('2d');
function frame() { c.fillStyle = '#0f0'; c.fillRect(0, 0, 64, 16); requestAnimationFrame(frame); }
requestAnimationFrame(frame);
</script>"""


def test_the_page_cannot_write_through_the_frame_binding(tmp_path):
    pytest.importorskip("playwright")
    from pixeldif.record import record
    page = tmp_path / "hostile.html"
    page.write_text(HOSTILE_PAGE)
    out = tmp_path / "frames"
    try:
        meta = record(page.as_uri(), out, frames=3, fps=60, width=200, height=100, warmup=3, selector="#c",
                      timeout=60, probe="window.attempts")
    except Exception as error:
        pytest.skip(f"browser unavailable: {error}")
    assert meta["frames"] == 3
    assert sorted(p.name for p in out.glob("*.png")) == ["00000.png", "00001.png", "00002.png"]
    assert not list(tmp_path.glob("escape*"))
    last = (out / "frames.jsonl").read_text().splitlines()[-1]
    assert json.loads(last)["value"] == ["rejected"] * 5
    for path in out.glob("*.png"):
        assert tuple(np.asarray(Image.open(path).convert("RGB"))[8, 32]) == (0, 255, 0)
