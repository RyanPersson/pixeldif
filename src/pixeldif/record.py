"""Record a web game's canvas frame by frame at an exact frame rate.

Real-time recording drops and jitters frames, and capturing each frame slows
the game down. Here the page runs on a virtual clock instead:
`performance.now()` and `Date.now()` advance by exactly 1/fps per animation
frame, however long that frame takes to render and save. The game sees a
steady 60 fps (or whatever --fps says), and every captured frame is exactly
one step apart in game time.

For WebGL the drawing buffer is preserved and draw calls are watched: a frame
is captured, and the clock advanced, only after the engine has drawn the
frame for the current time. Engines that render outside requestAnimationFrame
(winit on the web) would otherwise be captured from stale, double-buffered
frames.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

CLOCK = """
(() => {
  const step = 1000 / %(fps)f;
  let now = performance.now();
  const epoch = Date.now() - now;
  const realRAF = window.requestAnimationFrame.bind(window);
  let queue = new Map();
  let nextId = 1;
  performance.now = () => now;
  Date.now = () => Math.round(epoch + now);
  const state = window.__pixeldif = { capturing: false, index: 0, frames: 0, selector: %(selector)s,
    pending: 0, error: null, ticks: 0, waits: 0, gl: false, drawn: false, probe: null };

  // Engines may render outside requestAnimationFrame (winit on the web
  // updates from its own task queue). Keep the drawing buffer, notice every
  // WebGL draw, and only capture and advance the clock once the engine has
  // drawn the frame for the current time; otherwise a capture can read a
  // stale, double-buffered earlier frame.
  const getContext = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function (type, attributes) {
    const webgl = type === 'webgl' || type === 'webgl2' || type === 'experimental-webgl';
    if (webgl) attributes = Object.assign({}, attributes, { preserveDrawingBuffer: true });
    const context = getContext.call(this, type, attributes);
    if (webgl && context && !context.__pixeldif) {
      context.__pixeldif = true;
      state.gl = true;
      for (const name of ['drawArrays', 'drawElements', 'drawArraysInstanced', 'drawElementsInstanced',
                          'drawRangeElements', 'clear', 'blitFramebuffer']) {
        const original = context[name];
        if (typeof original === 'function') {
          context[name] = function (...args) { state.drawn = true; return original.apply(this, args); };
        }
      }
    }
    return context;
  };

  let scheduled = false;
  function tick() {
    scheduled = false;
    state.ticks += 1;
    const ready = !state.gl || state.drawn || state.waits >= 30;
    if (ready) {
      if (state.capturing && state.index < state.frames) {
        const canvas = document.querySelector(state.selector);
        if (canvas) {
          const url = canvas.toDataURL('image/png');
          let probe = null;
          if (state.probe) { try { probe = JSON.stringify(state.probe()); } catch (e) { probe = JSON.stringify(String(e)); } }
          state.pending += 1;
          window.__pixeldifFrame(state.index, url, probe).then(() => { state.pending -= 1; });
          state.index += 1;
          if (state.index >= state.frames) state.capturing = false;
        }
      }
      now += step;
      state.drawn = false;
      state.waits = 0;
    } else {
      state.waits += 1;
    }
    // Callbacks requested during this frame run next frame, as in a browser.
    const batch = queue;
    queue = new Map();
    for (const cb of batch.values()) {
      try { cb(now); } catch (e) { state.error = String(e); }
    }
    schedule();
  }
  function schedule() {
    if (!scheduled) { scheduled = true; realRAF(tick); }
  }
  window.requestAnimationFrame = cb => { const id = nextId++; queue.set(id, cb); schedule(); return id; };
  window.cancelAnimationFrame = id => { queue.delete(id); };
  schedule();
})();
"""

GPU_ARGS = ["--no-sandbox", "--disable-dev-shm-usage", "--ignore-gpu-blocklist", "--use-angle=vulkan",
            "--enable-features=Vulkan", "--use-vulkan=native", "--disable-vulkan-surface"]


def record(url: str, out: Path, frames: int, fps: float = 60.0, width: int = 1280, height: int = 720,
           ready: str | None = None, setup: list[str] | None = None, warmup: int = 0,
           until: str | None = None, selector: str = "canvas", timeout: float = 600.0,
           headed: bool = False, probe: str | None = None, hold: list[str] | None = None) -> dict:
    """Load `url`, wait for the `ready` JavaScript expression, run each `setup`
    snippet, let `warmup` frames pass (or wait for the `until` expression),
    then save `frames` canvas frames as PNGs in `out`. With `probe`, a
    JavaScript expression evaluated after each captured frame, its values go
    to `frames.jsonl` (one JSON object per frame). `hold` keys (Playwright
    names such as KeyW or ShiftLeft) are pressed on the focused canvas after
    setup, before the warmup, and released after the last frame."""
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    written = {}

    probes = {}

    def save_frame(index: int, data_url: str, probe: str | None = None):
        data = base64.b64decode(data_url.split(",", 1)[1])
        path = out / f"{index:05d}.png"
        path.write_bytes(data)
        written[index] = path
        if probe is not None:
            probes[index] = json.loads(probe)

    started = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed, args=GPU_ARGS)
        page = browser.new_page(viewport={"width": width, "height": height})
        log = []
        page.on("console", lambda m: log.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: log.append(f"[pageerror] {e}"))
        page.expose_function("__pixeldifFrame", save_frame)
        page.add_init_script(CLOCK % {"fps": fps, "selector": json.dumps(selector)})
        page.goto(url, wait_until="load", timeout=timeout * 1000)
        if ready:
            page.wait_for_function(ready, timeout=timeout * 1000)
        for snippet in setup or []:
            page.evaluate(snippet)
        if hold:
            page.focus(selector)
            for key in hold:
                page.keyboard.down(key)
        if until:
            page.wait_for_function(until, timeout=timeout * 1000, polling=100)
        elif warmup:
            page.evaluate("n => new Promise(r => { let k = 0; const f = () => (++k >= n ? r() : requestAnimationFrame(f)); requestAnimationFrame(f); })", warmup)
        if probe:
            page.evaluate("e => { window.__pixeldif.probe = new Function('return (' + e + ')'); }", probe)
        page.evaluate("n => { const s = window.__pixeldif; s.index = 0; s.frames = n; s.capturing = true; }", frames)
        page.wait_for_function("() => !window.__pixeldif.capturing && window.__pixeldif.pending === 0",
                               timeout=timeout * 1000, polling=200)
        error = page.evaluate("() => window.__pixeldif.error")
        for key in hold or []:
            page.keyboard.up(key)
        browser.close()
    meta = {"fps": fps, "frames": len(written), "url": url, "size": [width, height], "setup": setup or [],
            "ready": ready, "until": until, "warmup_frames": warmup, "hold": hold or [], "seconds": round(time.time() - started, 1),
            "page_error": error}
    if probes:
        meta["probe"] = probe
        with (out / "frames.jsonl").open("w") as stream:
            for index in sorted(probes):
                stream.write(json.dumps({"frame": index, "value": probes[index]}) + "\n")
    (out / "pixeldif.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "console.log").write_text("\n".join(log) + "\n")
    return meta
