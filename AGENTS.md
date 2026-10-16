# pixeldif: notes for agents

A command-line tool that shows agents what changes in video at short time
scales: motion at several lags (fixed camera), unexplained discontinuities
(moving camera) and time slices. Read README.md for the commands.

## Using it to judge realism

- Look at the sheet **and** read `report.json`. The sheet shows where things
  change; the numbers say how much and at what time and spatial scale.
- Compare like with like. Pick the same feature in both sources with
  `--roi-a`/`--roi-b` (the plume, the steam, a patch of grass), give both
  the same lags in time units, and capture the game at a fixed frame rate.
- `motion` and `compare` assume a fixed camera. With a moving camera use
  `pops` (discontinuities) or crop to a region that holds still.
- A `pops` event is evidence, not proof: check its crops. Fast-moving
  objects, transparency and lighting changes can leave residuals too.
- Read the capture checks first. Duplicates or camera jumps mean the
  recording or the camera path is broken. A camera looking straight down
  with an ill-defined up vector, for example, flips its roll between frames.
- Dynamics in physical units: `extent` (front speeds of a spreading
  cloud) and `flow` (median speed and direction coherence inside a region),
  with `--scale` from an object of known size in each source. Compare a
  game capture and a reference at the same metres per pixel in `motion`
  (choose `--width` so they match) so the decorrelation scales line up.
  Read regions in `motion` are working-width pixels or fractions; `extent`
  and `flow` take source pixels or fractions.
- `pops` can't see a slow morph. Optical flow explains a shape that changes
  a little each frame as motion, so procedural clouds that reshape as the
  camera moves leave almost no residual. For "does this change with my
  viewpoint" questions, record the same camera path twice, once with the
  renderer's sampling raised far beyond real-time (the recorder's virtual
  clock makes slow frames fine), and compare `motion --roi` decorrelation:
  the dense version is the genuine change from the motion; anything above
  it is the artefact. Hold keys during a recording with `record --hold`.
- Small pops piling up on objects (see "where pops happen") with a still
  camera are usually shadows re-quantising as the Sun moves, or LOD
  dithering. Crop a few consecutive frames around the hottest pixel to tell.

## Developing it

- Python 3.12 with uv. `uv run pytest` must pass. Every detector change needs
  a known-answer test on a synthetic clip (`synth.py`).
- Outputs are for models: every tile is labelled with what it shows, its lag
  in frames and milliseconds, and its scale. Keep sheets to a size a model
  can read in one image (about 2,500 px wide at most).
- Downloads, captures and output go in the ignored `media/`, `.scratch/` and
  `pixeldif-out/` folders. Never commit video.
- Machine: `navier` (RTX 5060 Ti, 20 cores). ffmpeg and ffprobe are installed.
