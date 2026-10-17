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
uv run pixeldif extent launch.mp4 --origin 935,690 --base 0.5 --scale 0.6
uv run pixeldif flow fog.mp4 --roi 600,700,700,170 --scale 0.6
uv run pixeldif swim flight/ --depth flight-depth/ --video
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

### `extent`: how fast a disturbance spreads

Compares each frame with a base frame from before the disturbance started
(an explosion, a launch cloud, a splash), keeps the changed region
connected to `--origin` (blobs within `--join` pixels count as one), and
reports its reach left, right and up over time, plus front speeds from a
sliding fit. Each frame's overall brightness is matched to the base away
from the disturbance first, because cameras re-expose as a bright cloud
grows. With `--scale` (metres per source pixel at the subject, e.g. from a
building of known height) it reports metres and metres per second, so a
game capture and a reference video of different framing compare directly.

### `flow`: streaming and churning speed

Optical flow inside `--roi` between consecutive frames: the median and
90th-percentile speed of the moving pixels and their direction
*coherence* (1 = everything streams one way, like fog creeping along the
ground; near 0 = churning, like a billowing cloud). Also with `--scale`.
For a fixed camera only.

### `swim`: texture that moves with the viewer instead of the world

`pops` can't see swimming: it warps each frame by the motion it measures,
so a texture sliding along with the camera is just more motion to it.
`swim` warps by the motion the world allows instead. It needs the camera
pose of every frame (`record --probe`) and the distance each pixel shows:
a second recording of the same camera path in the renderer's depth view.
From those, every world-fixed point has one place to be in the next frame
(the *world flow*); a pattern glued to the camera would only turn with it
(the *viewer flow*). Per pixel, the measured optical flow is placed between
the two:

- **anchoring** 0 means the picture moves with the world, 1 with the viewer,
  0.5 sliding along at half the camera's speed. Below 0 means faster than the
  world at the given depth, which is normal for volumes (see below). Pixels
  where the two flows differ by less than `--min-separation` (0.5 px), with
  nothing to track, or hidden in the next frame are left out.
- It is measured twice: on the frames as recorded (**fine detail**) and
  blurred by `--coarse` pixels (**coarse shapes**). A dither or grain fixed
  to the screen reads 1 in fine detail while the clouds carrying it read 0.
- **leftover after the world warp** against **after the flow warp** (what
  `pops` sees): where the world warp leaves more, the picture isn't fixed to
  the world. A slide is predicted well by the flow warp; a texture
  regenerated in place each frame (a morph) by neither. Morphs also read as
  anchoring near 1, as the pattern has no coherent motion: check this pair.
- **world-stabilized** view: later frames warped back into a reference
  frame. World-fixed content stands still (sharp in the mean), swimming
  crawls. `--video` writes `stabilized.mp4`: recording, stabilized
  recording and both anchoring maps.

Poses: each probe value (or its `camera` or `--pose-key` entry) holds
`position_m` in a frame where the scene is static, `rotation_xyzw` (camera to
that frame; the camera looks along -Z with +Y up) and `fov_y_deg`. Depth
frames: distance along the ray as a 24-bit code in RGB (R high), metres =
10^(code / 16777215 x 14 - 2); code 0 is nothing (sky). For volumes (clouds,
haze) give the far end of what the pixel shows: the nearest opacity is often
featureless haze in front of the structure that carries the texture, and
testing against it reads clouds as swimming. Against the far end the test is
one-sided: anchoring above 0 means slower than anything in the volume could
move. Record with time paused so the camera is the only thing moving; the
depth pairing in the report shows whether both passes followed one path.
Reflections and highlights really do move with the viewer.

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
