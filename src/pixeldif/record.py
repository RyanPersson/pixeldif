"""Record a web game's canvas frame by frame at an exact frame rate.

Real-time recording drops and jitters frames, and capturing each frame slows
the game down. Here the page runs on a virtual clock instead:
`performance.now()` and `Date.now()` advance by exactly 1/fps per animation
frame, however long that frame takes to render and save. The game sees a
steady 60 fps (or whatever --fps says), and every captured frame is exactly
one step apart in game time.

The canvas is read inside the same animation-frame task, right after the
game's callbacks have drawn it (WebGL clears the drawing buffer once the frame
is presented).
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
  window.__pixeldif = { capturing: false, index: 0, frames: 0, selector: %(selector)s, pending: 0, error: null, ticks: 0 };
  let scheduled = false;
  function tick() {
    scheduled = false;
    now += step;
    window.__pixeldif.ticks += 1;
    // Callbacks requested during this frame run next frame, as in a browser.
    const batch = queue;
    queue = new Map();
    for (const cb of batch.values()) {
      try { cb(now); } catch (e) { window.__pixeldif.error = String(e); }
    }
    const state = window.__pixeldif;
    if (state.capturing && state.index < state.frames) {
      const canvas = document.querySelector(state.selector);
      if (canvas) {
        const url = canvas.toDataURL('image/png');
        state.pending += 1;
        window.__pixeldifFrame(state.index, url).then(() => { state.pending -= 1; });
        state.index += 1;
        if (state.index >= state.frames) state.capturing = false;
      }
    }
    if (queue.size) schedule();
  }
  function schedule() {
    if (!scheduled) { scheduled = true; realRAF(tick); }
  }
  window.requestAnimationFrame = cb => { const id = nextId++; queue.set(id, cb); schedule(); return id; };
  window.cancelAnimationFrame = id => { queue.delete(id); };
})();
"""

GPU_ARGS = ["--no-sandbox", "--disable-dev-shm-usage", "--ignore-gpu-blocklist", "--use-angle=vulkan",
            "--enable-features=Vulkan", "--use-vulkan=native", "--disable-vulkan-surface"]


def record(url: str, out: Path, frames: int, fps: float = 60.0, width: int = 1280, height: int = 720,
           ready: str | None = None, setup: list[str] | None = None, warmup: int = 0,
           until: str | None = None, selector: str = "canvas", timeout: float = 600.0,
           headed: bool = False) -> dict:
    """Load `url`, wait for the `ready` JavaScript expression, run each `setup`
    snippet, let `warmup` frames pass (or wait for the `until` expression),
    then save `frames` canvas frames as PNGs in `out`."""
    from playwright.sync_api import sync_playwright

    out.mkdir(parents=True, exist_ok=True)
    written = {}

    def save_frame(index: int, data_url: str):
        data = base64.b64decode(data_url.split(",", 1)[1])
        path = out / f"{index:05d}.png"
        path.write_bytes(data)
        written[index] = path

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
        if until:
            page.wait_for_function(until, timeout=timeout * 1000, polling=100)
        elif warmup:
            page.evaluate("n => new Promise(r => { let k = 0; const f = () => (++k >= n ? r() : requestAnimationFrame(f)); requestAnimationFrame(f); })", warmup)
        page.evaluate("n => { const s = window.__pixeldif; s.index = 0; s.frames = n; s.capturing = true; }", frames)
        page.wait_for_function("() => !window.__pixeldif.capturing && window.__pixeldif.pending === 0",
                               timeout=timeout * 1000, polling=200)
        error = page.evaluate("() => window.__pixeldif.error")
        browser.close()
    meta = {"fps": fps, "frames": len(written), "url": url, "size": [width, height], "setup": setup or [],
            "ready": ready, "until": until, "warmup_frames": warmup, "seconds": round(time.time() - started, 1),
            "page_error": error}
    (out / "pixeldif.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "console.log").write_text("\n".join(log) + "\n")
    return meta
