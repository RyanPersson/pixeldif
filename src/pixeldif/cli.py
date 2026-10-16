"""pixeldif command line: every command writes a PNG sheet and a JSON report and
prints a short text summary, so an agent can read the numbers and look at one
image."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import motion, pops, slit, synth, viz
from .source import open_source, parse_lag, parse_roi, parse_time

DEFAULT_LAGS = "1,2,4,8,16,32"
DEFAULT_COMPARE_LAGS = "33ms,67ms,133ms,267ms,533ms"


def _out(args, stem: str) -> Path:
    out = Path(args.out) if args.out else Path("pixeldif-out") / stem
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write(out: Path, report: dict) -> Path:
    path = out / "report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def cmd_info(args):
    source = open_source(args.source, args.fps)
    print(json.dumps({"path": str(source.path), "kind": source.kind, "fps": round(source.fps, 4),
                      "size": [source.width, source.height], "frames": source.frame_count,
                      "duration_s": round(source.duration, 3)}, indent=2))


def cmd_motion(args):
    source = open_source(args.source, args.fps)
    lags = [parse_lag(t, source.fps) for t in args.lags.split(",")]
    start = parse_time(args.at, source)
    frames, starts = motion.load_window(source, start, lags, args.window, args.samples, args.width)
    roi = parse_roi(args.roi, frames.shape[2], frames.shape[1])
    sig = motion.signature(frames, source.fps, lags, starts, roi)
    top, bottom, vmax = motion.motion_rows(frames, 0, lags, source.fps, roi, args.gain, width=args.tile, starts=starts)
    table = motion.matrix_tile(sig.decorrelation, lags, source.fps, sig.scales,
                               "decorrelation D(lag, scale): 0 unchanged, 1 replaced")
    heading = (f"pixeldif motion · {source.path.name} · t = {start / source.fps:.2f} s"
               + (f" · roi {roi}" if roi else "") + f" · stats over {sig.samples} base frames")
    out = _out(args, f"motion-{source.path.stem}-{start}")
    sheet = viz.save(viz.grid([top, bottom, [table]], heading), out / "motion.png")
    report = {"command": "motion", "source": str(source.path), "frame": start,
              "time_s": round(start / source.fps, 3), "roi": roi, "heat_vmax": vmax, **sig.as_dict(),
              "sheet": str(sheet)}
    _write(out, report)
    print(f"sheet  {sheet}")
    print(f"report {out / 'report.json'}")
    print(_signature_text(sig))


def _signature_text(sig: motion.Signature, name: str = "") -> str:
    lines = [f"{name}decorrelation by lag (rows) and scale (columns, px {list(sig.scales)}):"]
    for lag, row, mean in zip(sig.lags, sig.decorrelation, sig.mean_abs):
        lines.append(f"  {lag:>3} f {lag / sig.fps * 1000:6.0f} ms  " + " ".join(f"{v:5.2f}" for v in row)
                     + f"   mean |change| {mean * 100:.1f}%")
    return "\n".join(lines)


def cmd_compare(args):
    a = open_source(args.a, args.fps_a)
    b = open_source(args.b, args.fps_b)
    lag_texts = args.lags.split(",")
    lags_a = [parse_lag(t, a.fps) for t in lag_texts]
    lags_b = [parse_lag(t, b.fps) for t in lag_texts]
    start_a, start_b = parse_time(args.at_a, a), parse_time(args.at_b, b)
    frames_a, starts_a = motion.load_window(a, start_a, lags_a, args.window, args.samples, args.width)
    frames_b, starts_b = motion.load_window(b, start_b, lags_b, args.window, args.samples, args.width)
    roi_a = parse_roi(args.roi_a, frames_a.shape[2], frames_a.shape[1])
    roi_b = parse_roi(args.roi_b, frames_b.shape[2], frames_b.shape[1])
    sig_a = motion.signature(frames_a, a.fps, lags_a, starts_a, roi_a)
    sig_b = motion.signature(frames_b, b.fps, lags_b, starts_b, roi_b)
    # One heat scale for both, so brightness compares across sources.
    _, _, vmax_a = motion.motion_rows(frames_a, 0, lags_a, a.fps, roi_a, args.gain, width=args.tile)
    _, _, vmax_b = motion.motion_rows(frames_b, 0, lags_b, b.fps, roi_b, args.gain, width=args.tile)
    vmax = max(vmax_a, vmax_b)
    top_a, bot_a, _ = motion.motion_rows(frames_a, 0, lags_a, a.fps, roi_a, args.gain, vmax, args.tile, "A", starts_a)
    top_b, bot_b, _ = motion.motion_rows(frames_b, 0, lags_b, b.fps, roi_b, args.gain, vmax, args.tile, "B", starts_b)
    ratio = np.log2(np.maximum(sig_b.decorrelation, 1e-3) / np.maximum(sig_a.decorrelation, 1e-3))
    tables = [
        motion.matrix_tile(sig_a.decorrelation, lags_a, a.fps, sig_a.scales, "A decorrelation"),
        motion.matrix_tile(sig_b.decorrelation, lags_b, b.fps, sig_b.scales, "B decorrelation"),
        motion.matrix_tile(ratio, lags_a, a.fps, sig_a.scales, "log2(B / A): red B changes more, blue less",
                           vmax=2.0, diverging=True, fmt="{:+.1f}"),
    ]
    heading = f"pixeldif compare · A {a.path.name} @ {start_a / a.fps:.2f} s · B {b.path.name} @ {start_b / b.fps:.2f} s"
    out = _out(args, f"compare-{a.path.stem}-{b.path.stem}")
    sheet = viz.save(viz.grid([top_a, top_b, bot_a, bot_b, tables], heading), out / "compare.png")
    findings = []
    for i, lag in enumerate(lag_texts):
        for j, s in enumerate(sig_a.scales):
            findings.append((abs(ratio[i, j]), lag, s, ratio[i, j], sig_a.decorrelation[i, j], sig_b.decorrelation[i, j]))
    findings.sort(reverse=True)
    report = {"command": "compare", "a": {"source": str(a.path), "frame": start_a, "roi": roi_a, **sig_a.as_dict()},
              "b": {"source": str(b.path), "frame": start_b, "roi": roi_b, **sig_b.as_dict()},
              "log2_ratio_b_over_a": np.round(ratio, 3).tolist(), "lags": lag_texts, "sheet": str(sheet),
              "largest_differences": [{"lag": lag, "scale_px": s, "log2_ratio": round(float(r), 3),
                                       "a": round(float(da), 3), "b": round(float(db), 3)}
                                      for _, lag, s, r, da, db in findings[:6]]}
    _write(out, report)
    print(f"sheet  {sheet}")
    print(f"report {out / 'report.json'}")
    print(_signature_text(sig_a, "A "))
    print(_signature_text(sig_b, "B "))
    print("largest differences (B vs A):")
    for _, lag, s, r, da, db in findings[:6]:
        print(f"  lag {lag:>6}, scale {s:>2} px: B {'changes' if r > 0 else 'is'} {2 ** abs(r):.1f}x "
              f"{'more' if r > 0 else 'less (steadier)'} (D {da:.2f} -> {db:.2f})")


def cmd_pops(args):
    source = open_source(args.source, args.fps)
    start = parse_time(args.start, source) if args.start else 0
    end = parse_time(args.end, source) if args.end else source.frame_count
    frames = source.read(start, end - start, args.width)
    res, masks, consistent, speeds, fast = pops.residuals(frames, args.preset, args.blur)
    scores = pops.pop_scores(res, masks, fast)
    events = pops.find_events(scores, args.threshold, args.min_area)
    shimmer = np.median(res, axis=0)
    pixels = scores[0].size
    pop_energy = [float(s[s > args.threshold].sum() / pixels) for s in scores]
    shimmer_energy = [float(r.mean()) for r in res]
    out = _out(args, f"pops-{source.path.stem}")
    probes = source.probes()
    x = None
    if args.x_probe:
        def value(frame):
            v = probes.get(frame)
            return v.get(args.x_probe) if isinstance(v, dict) else v
        x = [value(i + 1 + start) for i in range(len(pop_energy))]
        if any(v is None for v in x):
            raise SystemExit(f"--x-probe {args.x_probe}: missing in frames.jsonl for some frames")
    pops.timeline(pop_energy, shimmer_energy, speeds, events, source.fps, start, out / "timeline.png", x,
                  args.x_probe or "time (s)", args.log_x, args.invert_x, args.title or "")

    def note(frame: int) -> str:
        value = probes.get(frame)
        if value is None:
            return ""
        if isinstance(value, dict):
            return ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in value.items())
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    predictor = pops.Predictor(args.preset)
    vmax = max(0.15, max((e.peak for e in events[: args.top]), default=0.15))
    rows = []
    for e in events[: args.top]:
        warped, _, _, _ = predictor.predict(frames[e.frame - 1], frames[e.frame])
        rows.append(pops.event_panel(frames, warped, scores[e.frame - 1], e, vmax, source.fps, start, args.tile,
                                     note(e.frame + start)))
    # Where pops happen over the whole clip: many small pops (scatter, LOD
    # swaps of small objects) never form one large event but pile up here.
    hits = (scores > args.threshold).sum(axis=0).astype(np.float32)
    last = viz.to_float(frames[-1])
    shimmer_row = [
        viz.tile(viz.overlay_heat(viz.to_float(frames[len(frames) // 2]), shimmer, max(0.1, float(np.percentile(shimmer, 99.5)))),
                 "persistent unexplained change", "median residual over the clip (shimmer, aliasing)", args.tile * 2),
        viz.tile(viz.overlay_heat(last, hits, max(2.0, float(hits.max()))),
                 "where pops happen", f"frames with a pop at each pixel (max {int(hits.max())}), on the last frame", args.tile * 2),
    ]
    heading = (f"pixeldif pops · {source.path.name} · {len(frames)} frames from {start / source.fps:.2f} s · "
               f"{len(events)} events (threshold {args.threshold:g})")
    sheet = viz.save(viz.grid(rows + [shimmer_row] if rows else [shimmer_row], heading), out / "pops.png")
    checks = pops.capture_checks(frames, speeds)
    checks = {"duplicate_frames": [f + start for f in checks["duplicate_frames"]],
              "camera_jumps": [dict(j, frame=j["frame"] + start) for j in checks["camera_jumps"]]}
    report = {"command": "pops", "source": str(source.path), "fps": source.fps, "start_frame": start,
              "capture_checks": checks,
              "frames": len(frames), "threshold": args.threshold, "min_area": args.min_area,
              "events": [dict(e.as_dict(source.fps, start), **({"probe": probes[e.frame + start]} if e.frame + start in probes else {}))
                         for e in events[: args.max_events]],
              "event_count": len(events), "pop_energy": [round(v, 6) for v in pop_energy],
              "unexplained_change": [round(v, 5) for v in shimmer_energy],
              "median_flow_px": [round(v, 3) for v in speeds],
              "sheet": str(sheet), "timeline": str(out / "timeline.png")}
    _write(out, report)
    print(f"sheet    {sheet}")
    print(f"timeline {out / 'timeline.png'}")
    print(f"report   {out / 'report.json'}")
    n = len(frames)
    if len(checks["duplicate_frames"]) > 0.02 * n:
        print(f"WARNING: {len(checks['duplicate_frames'])} duplicated frames: the capture may be dropping or repeating frames")
    if checks["camera_jumps"]:
        print(f"WARNING: {len(checks['camera_jumps'])} camera jumps (motion spikes), first at frames "
              f"{[j['frame'] for j in checks['camera_jumps'][:8]]}: the camera path or the capture is not smooth")
    print(f"{len(events)} events above {args.threshold:g}; strongest:")
    for e in events[: min(args.top, 10)]:
        d = e.as_dict(source.fps, start)
        print(f"  frame {d['frame']:>5} ({d['time_s']:7.2f} s)  box {d['box_xywh']}  area {d['area_fraction'] * 100:5.2f}%  peak {d['peak']:.2f}"
              + (f"  {note(d['frame'])}" if note(d['frame']) else ""))


def cmd_slit(args):
    source = open_source(args.source, args.fps)
    start = parse_time(args.start, source) if args.start else 0
    end = parse_time(args.end, source) if args.end else source.frame_count
    frames = source.read(start, end - start, args.width)
    h, w = frames.shape[1:3]
    lines = [slit.parse_line(text, w, h) for text in args.line]
    panels = []
    report = {"command": "slit", "source": str(source.path), "start_frame": start, "frames": len(frames), "lines": []}
    mid = frames[len(frames) // 2]
    marked = mid.copy()
    import cv2
    for k, line in enumerate(lines):
        x0, y0, x1, y1 = line
        cv2.line(marked, (int(x0), int(y0)), (int(x1), int(y1)), (60, 255, 120), 2)
        cv2.putText(marked, str(k + 1), (int(x0) + 4, int(y0) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 120), 2)
        s = slit.sample(frames, line)
        change = slit.change_image(s)
        seams = slit.seams(s)
        panels.append([viz.tile(s, f"slit {k + 1}: time runs down ({len(frames)} frames, {len(frames) / source.fps:.1f} s)",
                                "vertical = still, slanted = moving, horizontal seam = pop", args.tile),
                       viz.tile(viz.colormap(change, max(0.05, float(np.percentile(change, 99.5)))),
                                f"slit {k + 1}: change between frames", f"{len(seams)} seams", args.tile)])
        report["lines"].append({"line": [round(v, 1) for v in line],
                                "seams": [dict(sm, frame=sm["row"] + start, time_s=round((sm["row"] + start) / source.fps, 3))
                                          for sm in seams]})
    rows = [[viz.tile(marked, "slit positions (middle frame)", "", args.tile)]] + panels
    out = _out(args, f"slit-{source.path.stem}")
    sheet = viz.save(viz.grid(rows, f"pixeldif slit · {source.path.name}"), out / "slit.png")
    report["sheet"] = str(sheet)
    _write(out, report)
    print(f"sheet  {sheet}")
    print(f"report {out / 'report.json'}")
    for k, line in enumerate(report["lines"]):
        print(f"slit {k + 1}: {len(line['seams'])} seams" + (f" at frames {[s['frame'] for s in line['seams'][:10]]}" if line["seams"] else ""))


def cmd_record(args):
    from .record import record
    meta = record(args.url, Path(args.out), args.frames, args.fps, args.width, args.height, args.ready,
                  args.setup, args.warmup, args.until, args.canvas, args.timeout, args.headed, args.probe,
                  args.hold)
    print(f"{meta['frames']} frames at {meta['fps']:g} fps in {args.out} ({meta['seconds']} s)"
          + (f"; page error: {meta['page_error']}" if meta["page_error"] else ""))


def cmd_synth(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.kind == "zoom-pop":
        frames, truth = synth.zoom_pop()
    else:
        frames, truth = synth.steam(), []
    from PIL import Image
    for i, f in enumerate(frames):
        Image.fromarray(f).save(out / f"{i:05d}.png")
    (out / "pixeldif.json").write_text(json.dumps({"fps": 60, "truth": truth}, indent=2, default=float) + "\n")
    print(f"{len(frames)} frames in {out}" + (f"; truth {truth}" if truth else ""))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pixeldif", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def common(q, window=True):
        q.add_argument("--fps", type=float, help="frame rate for image folders (default from pixeldif.json or 60)")
        q.add_argument("--width", type=int, default=640, help="working width in pixels (default 640)")
        q.add_argument("--tile", type=int, default=360, help="tile width in the sheet")
        q.add_argument("--out", help="output folder (default pixeldif-out/<command>-<name>)")

    q = sub.add_parser("info", help="frame rate, size and length of a source")
    q.add_argument("source")
    q.add_argument("--fps", type=float)
    q.set_defaults(func=cmd_info)

    q = sub.add_parser("motion", help="differences at several lags from one moment (fixed camera)")
    q.add_argument("source", help="video file or folder of frames")
    q.add_argument("--at", default="0", help="start: seconds, mm:ss or <n>f (default 0)")
    q.add_argument("--lags", default=DEFAULT_LAGS, help=f"comma list: frames, 50ms or 0.5s (default {DEFAULT_LAGS})")
    q.add_argument("--roi", help="x,y,w,h in pixels of the working frame, or fractions")
    q.add_argument("--window", type=float, default=1.0, help="seconds of base frames averaged for the statistics")
    q.add_argument("--samples", type=int, default=8, help="base frames sampled in the window")
    q.add_argument("--gain", type=float, default=2.0, help="contrast of the overlay tiles")
    common(q)
    q.set_defaults(func=cmd_motion)

    q = sub.add_parser("compare", help="motion signatures of two sources side by side (e.g. game vs reference)")
    q.add_argument("a")
    q.add_argument("b")
    q.add_argument("--at-a", default="0")
    q.add_argument("--at-b", default="0")
    q.add_argument("--roi-a")
    q.add_argument("--roi-b")
    q.add_argument("--fps-a", type=float)
    q.add_argument("--fps-b", type=float)
    q.add_argument("--lags", default=DEFAULT_COMPARE_LAGS, help="lags in time units shared by both (default multiples of 1/30 s)")
    q.add_argument("--window", type=float, default=1.0)
    q.add_argument("--samples", type=int, default=8)
    q.add_argument("--gain", type=float, default=2.0)
    q.add_argument("--width", type=int, default=640)
    q.add_argument("--tile", type=int, default=300)
    q.add_argument("--out")
    q.set_defaults(func=cmd_compare)

    q = sub.add_parser("pops", help="changes that camera motion cannot explain (moving camera)")
    q.add_argument("source")
    q.add_argument("--start", help="seconds, mm:ss or <n>f")
    q.add_argument("--end", help="seconds, mm:ss or <n>f")
    q.add_argument("--threshold", type=float, default=0.08, help="pop score threshold (fraction of full scale)")
    q.add_argument("--min-area", type=float, default=0.0004, help="smallest event, fraction of the frame")
    q.add_argument("--preset", default="medium", choices=["ultrafast", "fast", "medium"], help="optical flow quality")
    q.add_argument("--blur", type=float, default=1.2, help="residual blur in pixels; lower keeps tiny pops (scatter)")
    q.add_argument("--top", type=int, default=6, help="events shown in the sheet")
    q.add_argument("--max-events", type=int, default=200, help="events listed in the report")
    q.add_argument("--x-probe", help="plot the timeline against this per-frame probe value (frames.jsonl key)")
    q.add_argument("--log-x", action="store_true", help="logarithmic x axis for the timeline")
    q.add_argument("--invert-x", action="store_true", help="reverse the timeline's x axis (e.g. falling altitude)")
    q.add_argument("--title", help="timeline title")
    common(q)
    q.set_defaults(func=cmd_pops)

    q = sub.add_parser("slit", help="time-slice images along lines (time runs down)")
    q.add_argument("source")
    q.add_argument("--line", action="append", default=None, help="row:Y, col:X or x0,y0,x1,y1 (fractions); repeatable")
    q.add_argument("--start")
    q.add_argument("--end")
    common(q)
    q.set_defaults(func=cmd_slit)

    q = sub.add_parser("record", help="record a web game's canvas at an exact frame rate (virtual clock)")
    q.add_argument("url")
    q.add_argument("out", help="folder for the PNG frames")
    q.add_argument("--frames", type=int, default=120)
    q.add_argument("--fps", type=float, default=60.0)
    q.add_argument("--width", type=int, default=1280, help="browser viewport width")
    q.add_argument("--height", type=int, default=720)
    q.add_argument("--ready", help="JavaScript expression to wait for before setup (e.g. window.spaceState)")
    q.add_argument("--setup", action="append", help="JavaScript to run before recording; repeatable")
    q.add_argument("--warmup", type=int, default=0, help="frames to let pass before recording")
    q.add_argument("--until", help="JavaScript expression to wait for before recording (instead of --warmup)")
    q.add_argument("--canvas", default="canvas", help="CSS selector of the canvas")
    q.add_argument("--timeout", type=float, default=900.0, help="seconds")
    q.add_argument("--headed", action="store_true")
    q.add_argument("--probe", help="JavaScript expression saved per frame to frames.jsonl (e.g. game state)")
    q.add_argument("--hold", action="append", metavar="KEY",
                   help="key held from setup through the recording (Playwright name, e.g. KeyW); repeatable")
    q.set_defaults(func=cmd_record)

    q = sub.add_parser("synth", help="write a synthetic test clip with known answers")
    q.add_argument("kind", choices=["zoom-pop", "steam"])
    q.add_argument("out")
    q.set_defaults(func=cmd_synth)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "line", "unset") is None:
        args.line = ["row:0.5"]
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
