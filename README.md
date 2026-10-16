# pixeldif

See how pictures change over short time scales, as still images and numbers.

AI agents look at games and videos through screenshots, so they can't see what
only shows up over time: how steam billows, how grass waves, a plume
flickering, or scatter popping into view as the camera flies in. `pixeldif`
turns those into sheets an agent can read in one look, plus a JSON report.
It works on video files and on folders of frames (game captures).

The motion views follow the time-offset difference from Posy's
[*Motion Extraction*](https://www.youtube.com/watch?v=NSS6yAMZF78): overlay a
video on an inverted, time-shifted copy of itself and the static parts cancel
to grey, leaving only what changed during the offset.

## Commands

```sh
uv run pixeldif info clip.mp4
uv run pixeldif motion clip.mp4 --at 12.5 --roi 0.3,0.2,0.4,0.6
uv run pixeldif compare reference.mp4 game-frames/ --at-a 41.2 --at-b 3.0
uv run pixeldif pops flythrough.mp4 --start 10 --end 25
uv run pixeldif slit clip.mp4 --line row:0.6 --line col:0.5
uv run pixeldif synth zoom-pop demo-frames/   # a test clip with known answers
uv run pixeldif record http://localhost:8020/game.html frames/ --frames 180 \
    --ready "window.gameReady" --setup "startDemo()"
```

Every command writes a folder (default `pixeldif-out/<command>-<name>/`) with
a PNG sheet and `report.json`, and prints a short summary.

### `motion`: a fixed camera, several time scales

From one moment, the sheet shows the frame and the change after lags of
1, 2, 4, 8, 16 and 32 frames (`--lags` also takes `50ms` or `0.5s`):

- **Overlay row:** Posy's overlay, `0.5 + 0.5·gain·(later − now)`. Static areas
  are grey; moving areas keep their colour.
- **Heat row:** the size of the change on a darkened copy of the frame, with
  one shared scale so the lags compare. The row opens with a **time-scale map**:
  blue where the picture changes within a frame or two (flicker, turbulence),
  red where it only changes over the longest lag (drift), dark where it is still.
- **Decorrelation table** `D(lag, scale)`: how much of the picture's
  structure at each spatial scale has been replaced after each lag. 0 means
  unchanged and 1 means fully replaced. It is averaged over `--samples` base
  frames in `--window` seconds. The table's shape is a fingerprint of the
  motion: billowing steam has small D at short lags that grows slowly;
  shimmer or noise has large D after a single frame.

Use `--roi x,y,w,h` (pixels or fractions) to measure one feature, such as the
exhaust plume or a patch of grass, rather than the whole frame.

### `compare`: game against reference

Runs `motion` on two sources at the same lags in time units (default
multiples of 1/30 s, so 30 and 60 fps sources line up). Both use one heat
scale. The sheet adds `log2(B/A)` of the two decorrelation tables, and the
summary lists where they differ most, for example "at 133 ms and 8 px, B
changes 2.6x less (steadier)".

### `pops`: a moving camera, discontinuities only

For each pair of frames, `pops` estimates optical flow, warps the earlier
frame forward and measures what the warp fails to predict. Smooth camera
motion is predicted, so it cancels. A **pop** is a residual that spikes for
one frame pair and is quiet before and after: scatter or an LOD popping in, a
texture swap, a shadow-cascade switch. The sheet shows the strongest events
(full frame with a box, then before, prediction, after and the unexplained
change, cropped). Two maps follow: **persistent** unexplained change (shimmer,
temporal aliasing, z-fighting), and **where pops happen**, the number of
frames with a pop at each pixel. The second catches many small pops, such as
scatter or shadows, that never form one large event. `timeline.png` plots pop
energy, unexplained change and camera speed over time.

The report also runs **capture checks**: frames repeated while the picture
is moving, and camera jumps (frames whose overall motion spikes far above its
running median). Both mean the capture or the camera path is broken, so fix
that before reading the events. If neighbouring frames fall into two
interleaved families, compare frame *i* with frames *i+1* and *i+2*.
Recordings made with `record --probe` carry per-frame values (for example
altitude) that label each event.

Areas entering at the frame edges are masked. Areas uncovered behind moving
objects recur every frame while the motion continues, so the one-frame test
removes them. Tune with `--threshold` (default 0.08 of full scale) and
`--min-area`.

### `slit`: time slices

Samples a line of pixels from every frame and stacks the lines with time
running down. Still things make vertical stripes and steady motion makes
slants. Waving, billowing and flicker each have their own texture. A pop or
swap is a horizontal seam, and seams are also listed in the report.

### `record`: capture a web game at an exact frame rate

Loads the page in headless Chromium on the GPU, on a virtual clock:
`performance.now()` and `Date.now()` advance by exactly 1/fps per animation
frame, however long a frame takes to render and save. The game sees a steady
frame rate and every PNG is exactly one step apart in game time; real-time
screen recording drops and jitters frames. `--ready` waits for a JavaScript
expression, `--setup` runs snippets (repeatable), and `--until` or
`--warmup` sets when recording starts. The folder gets a `pixeldif.json` with
the frame rate, so the other commands read it directly. `--probe` evaluates a
JavaScript expression after every frame and saves it to `frames.jsonl`.
`--hold KeyW` (repeatable, Playwright key names) holds keys from setup to
the last frame, to fly a game camera along a steady path.

For WebGL the recorder keeps the drawing buffer and watches draw calls. It
captures, and advances the clock, only once the engine has drawn the frame
for the current time. Engines that render outside `requestAnimationFrame`
(Bevy/winit on the web) would otherwise be captured from stale,
double-buffered frames. Hide the game's HUD
in a setup step when you only care about the scene.

## Setup

```sh
uv sync                         # Python 3.12, numpy, OpenCV, Pillow, matplotlib, Playwright
uv run playwright install chromium   # for record
uv run pytest    # known-answer tests on synthetic clips
```

`ffmpeg` and `ffprobe` must be on `PATH` to read video files. Folders of
frames are read directly; put `{"fps": 60}` in `pixeldif.json` inside the
folder (otherwise 60 is assumed, or pass `--fps`).
