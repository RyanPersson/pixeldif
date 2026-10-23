"""pixeldif command line: every command writes a PNG sheet and a JSON report and
prints a short text summary, so an agent can read the numbers and look at one
image."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import dynamics, motion, pops, slit, swim, synth, viz
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


def _point(text: str, source, w: int, h: int) -> tuple[float, float]:
    """'x,y' as fractions of the frame (both <= 1) or source pixels."""
    x, y = (float(v) for v in text.split(","))
    if x <= 1 and y <= 1:
        return x * (w - 1), y * (h - 1)
    return x * w / source.width, y * h / source.height


def _source_roi(text, source, w: int, h: int):
    """parse_roi, with pixel values given in source pixels."""
    if not text:
        return None
    values = [float(v) for v in text.split(",")]
    if not all(v <= 1.0 for v in values):
        k = w / source.width
        text = ",".join(str(v * k) for v in values)
    return parse_roi(text, w, h)


def cmd_extent(args):
    source = open_source(args.source, args.fps)
    start = parse_time(args.start, source) if args.start else 0
    end = parse_time(args.end, source) if args.end else source.frame_count
    base_index = parse_time(args.base, source) if args.base else start
    frames = source.read(start, end - start, args.width)
    base = source.read(base_index, 1, args.width)[0]
    h, w = frames.shape[1:3]
    origin = _point(args.origin, source, w, h)
    roi = _source_roi(args.roi, source, w, h)
    reaches, masks = dynamics.reach(frames, base, origin, args.threshold, roi, join=args.join)
    unit, k = ("m", args.scale * source.width / w) if args.scale else ("px", 1.0)
    times = [(start + r.frame) / source.fps for r in reaches]
    series = {name: [getattr(r, name) * k for r in reaches] for name in ("left", "right", "up")}
    speeds = {name: dynamics.front_speed(v, source.fps, args.window) for name, v in series.items()}
    out = _out(args, f"extent-{source.path.stem}")
    dynamics.plot_series(series, times, out / "reach.png", f"reach from the source · {source.path.name}", f"reach ({unit})")
    dynamics.plot_series(speeds, times, out / "speed.png", f"front speed ({args.window:g} s fit) · {source.path.name}",
                         f"speed ({unit}/s)")
    picks = sorted({int(round(v)) for v in np.linspace(0, len(frames) - 1, 6)})
    tiles = [viz.tile(dynamics.contour_overlay(frames[i], masks[i], origin), f"t = {times[i]:.2f} s",
                      f"left {series['left'][i]:.0f} right {series['right'][i]:.0f} up {series['up'][i]:.0f} {unit}",
                      args.tile) for i in picks]
    sheet = viz.save(viz.grid([tiles[:3], tiles[3:]], f"pixeldif extent · {source.path.name} · change against t = {base_index / source.fps:.2f} s"),
                     out / "extent.png")
    report = {"command": "extent", "source": str(source.path), "start_frame": start, "base_frame": base_index,
              "origin_px_working": [round(origin[0], 1), round(origin[1], 1)], "working_size": [w, h],
              "threshold": args.threshold, "unit": unit, "metres_per_working_px": k if args.scale else None,
              "frames": [{"frame": start + r.frame, "time_s": round(times[i], 3), "area_px": r.area_px,
                          **{n: round(series[n][i], 2) for n in series},
                          **{f"{n}_speed": round(speeds[n][i], 2) for n in speeds}, "box_xywh": list(r.box)}
                         for i, r in enumerate(reaches)],
              "sheet": str(sheet), "plots": [str(out / "reach.png"), str(out / "speed.png")]}
    _write(out, report)
    print(f"sheet  {sheet}")
    print(f"plots  {out / 'reach.png'} {out / 'speed.png'}")
    print(f"report {out / 'report.json'}")
    step = max(1, int(round(source.fps * args.every)))
    for i in range(0, len(reaches), step):
        print(f"t {times[i]:7.2f} s  reach L {series['left'][i]:6.0f} R {series['right'][i]:6.0f} U {series['up'][i]:6.0f} {unit}"
              f"   speed L {speeds['left'][i]:6.1f} R {speeds['right'][i]:6.1f} U {speeds['up'][i]:6.1f} {unit}/s")


def cmd_flow(args):
    source = open_source(args.source, args.fps)
    start = parse_time(args.start, source) if args.start else 0
    end = parse_time(args.end, source) if args.end else source.frame_count
    frames = source.read(start, end - start, args.width)
    h, w = frames.shape[1:3]
    roi = _source_roi(args.roi, source, w, h)
    every = max(1, args.step)
    stats, fields = dynamics.flow_stats(frames, roi, args.preset, args.min_speed, every)
    unit, k = ("m", args.scale * source.width / w) if args.scale else ("px", 1.0)
    per_s = source.fps * k
    times = [(start + st.frame) / source.fps for st in stats]
    out = _out(args, f"flow-{source.path.stem}")
    dynamics.plot_series({"median speed": [st.median_speed * per_s for st in stats],
                          "90th percentile": [st.p90_speed * per_s for st in stats]},
                         times, out / "speed.png", f"flow speed of moving pixels · {source.path.name}", f"speed ({unit}/s)")
    picks = sorted({int(round(v)) for v in np.linspace(0, len(fields) - 1, 3)})
    vmax = max(0.5, float(np.percentile([st.p90_speed for st in stats], 90)))
    tiles = []
    for i in picks:
        frame_index, field = fields[i]
        image = frames[frame_index]
        if roi is not None:
            x, y, rw, rh = roi
            image = image[y:y + rh, x:x + rw]
        st = stats[i]
        tiles.append([viz.tile(image, f"t = {times[i]:.2f} s", "", args.tile),
                      viz.tile(dynamics.flow_image(field, vmax), "flow: hue = direction, saturation = speed",
                               f"median {st.median_speed * per_s:.1f} {unit}/s, p90 {st.p90_speed * per_s:.1f}, "
                               f"coherence {st.coherence:.2f}", args.tile)])
    sheet = viz.save(viz.grid(tiles, f"pixeldif flow · {source.path.name}"), out / "flow.png")
    summary = {"median_speed": float(np.median([st.median_speed for st in stats])) * per_s,
               "p90_speed": float(np.median([st.p90_speed for st in stats])) * per_s,
               "moving_fraction": float(np.median([st.moving_fraction for st in stats])),
               "coherence": float(np.median([st.coherence for st in stats]))}
    report = {"command": "flow", "source": str(source.path), "start_frame": start, "unit_per_s": f"{unit}/s",
              "roi_working": list(roi) if roi else None, "summary": {n: round(v, 3) for n, v in summary.items()},
              "frames": [{"frame": start + st.frame, "time_s": round(times[i], 3),
                          "median_speed": round(st.median_speed * per_s, 3), "p90_speed": round(st.p90_speed * per_s, 3),
                          "direction_deg": round(st.direction_deg, 1), "coherence": round(st.coherence, 3),
                          "moving_fraction": round(st.moving_fraction, 3)} for i, st in enumerate(stats)],
              "sheet": str(sheet), "plot": str(out / "speed.png")}
    _write(out, report)
    print(f"sheet  {sheet}")
    print(f"plot   {out / 'speed.png'}")
    print(f"report {out / 'report.json'}")
    print(f"moving pixels: median speed {summary['median_speed']:.2f} {unit}/s, p90 {summary['p90_speed']:.2f} {unit}/s, "
          f"coherence {summary['coherence']:.2f} (1 = streaming one way, 0 = churning), moving {summary['moving_fraction'] * 100:.0f}% of the region")


def cmd_swim(args):
    import cv2
    import warnings
    source = open_source(args.source, args.fps)
    depth_source = open_source(args.depth, args.fps)
    start = parse_time(args.start, source) if args.start else 0
    end = min(parse_time(args.end, source) if args.end else source.frame_count, source.frame_count)
    lag = parse_lag(args.lag, source.fps)
    probes = source.probes()
    poses = {i: swim.Pose.from_probe(v, args.pose_key) for i, v in probes.items()}
    depth_poses = {i: swim.Pose.from_probe(v, args.pose_key) for i, v in depth_source.probes().items()}
    poses = {i: p for i, p in poses.items() if p is not None}
    depth_poses = {i: p for i, p in depth_poses.items() if p is not None}
    if not poses or not depth_poses:
        raise SystemExit("swim needs camera poses in frames.jsonl of both recordings (record --probe)")
    # Pair each frame with the depth frame of the same pose (the recordings
    # may start a frame apart); frames without a match are left out.
    steps = [poses[i].difference(poses[i + 1])[0] for i in poses if i + 1 in poses]
    tolerance = 0.02 * (float(np.median(steps)) if steps else 0.0) + 1e-3
    pairing, unmatched, worst = {}, [], [0.0, 0.0]
    depth_ids = sorted(depth_poses)
    for i in range(start, end):
        if i not in poses:
            continue
        j = min(depth_ids, key=lambda k: sum(np.array(poses[i].difference(depth_poses[k])) * (1.0, 1e3)))
        moved, turned = poses[i].difference(depth_poses[j])
        if moved > tolerance or turned > 0.01:
            unmatched.append(i)
            continue
        pairing[i] = j
        worst = [max(worst[0], moved), max(worst[1], turned)]
    w, h = source.scaled_size(args.width)
    frames = source.read(start, end - start, args.width)
    depth_cache = {}

    def depth(i):
        if i not in depth_cache:
            raw = np.asarray(depth_source.read(pairing[i], 1)[0])
            depth_cache[i] = swim.smooth_depth(swim.resize_depth(swim.decode_depth(raw), w, h))
        return depth_cache[i]

    def blurred(frame):
        return cv2.GaussianBlur(frame, (0, 0), args.coarse)

    fine = swim.Analyser(args.preset, args.min_separation, args.min_texture, args.depth_tolerance, args.blur)
    coarse = swim.Analyser(args.preset, args.min_separation, args.min_texture / (1 + args.coarse) ** 2,
                           args.depth_tolerance, args.blur)
    results, coarse_results, pairs = [], [], []
    for t in range(0, len(frames) - lag, args.step):
        i, k = start + t, start + t + lag
        if i not in pairing or k not in pairing:
            continue
        a, b = frames[t], frames[t + lag]
        results.append(fine.pair(a, b, depth(i), depth(k), poses[i], poses[k]))
        coarse_results.append(coarse.pair(blurred(a), blurred(b), depth(i), depth(k), poses[i], poses[k]))
        pairs.append(t)
    if not results:
        raise SystemExit("no frame pairs with poses and depth")

    def median_map(rs):
        stack = np.stack([r.anchoring for r in rs])
        counts = np.isfinite(stack).sum(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(stack, axis=0)
        median[counts < max(2, int(args.min_share * len(rs)))] = np.nan
        return median, stack

    median, stack = median_map(results)
    median_coarse, stack_coarse = median_map(coarse_results)
    leftover_world = np.mean([r.leftover_world for r in results], axis=0)
    leftover_flow = np.mean([r.leftover_flow for r in results], axis=0)

    def summary(values):
        return (float(np.median(values)) if values.size else float("nan"),
                float((values > 0.5).mean()) if values.size else float("nan"))

    series = {"median_anchoring": [], "viewer_share": [], "coarse_median_anchoring": [], "coarse_viewer_share": [],
              "leftover_world": [], "leftover_flow": [], "measured_fraction": []}
    for r, rc in zip(results, coarse_results):
        m, v = summary(r.anchoring[np.isfinite(r.anchoring)])
        mc, vc = summary(rc.anchoring[np.isfinite(rc.anchoring)])
        series["median_anchoring"].append(m)
        series["viewer_share"].append(v)
        series["coarse_median_anchoring"].append(mc)
        series["coarse_viewer_share"].append(vc)
        series["leftover_world"].append(float(r.leftover_world[r.inside].mean()) if r.inside.any() else 0.0)
        series["leftover_flow"].append(float(r.leftover_flow[r.inside].mean()) if r.inside.any() else 0.0)
        series["measured_fraction"].append(float(np.isfinite(r.anchoring).mean()))
    out = _out(args, f"swim-{source.path.stem}")
    x, x_label = [(start + t + lag) / source.fps for t in pairs], "time (s)"
    if args.x_probe:
        def value(frame):
            v = probes.get(frame)
            return v.get(args.x_probe) if isinstance(v, dict) else v
        x, x_label = [value(start + t + lag) for t in pairs], args.x_probe
    swim.timeline(series, x, x_label, out / "timeline.png", args.log_x, args.invert_x, args.title or "")

    def note(frame: int) -> str:
        v = probes.get(frame)
        if isinstance(v, dict) and args.x_probe and args.x_probe in v:
            return f" · {args.x_probe} {v[args.x_probe]:.4g}"
        return ""

    def headline(anchoring):
        a = anchoring[np.isfinite(anchoring)]
        if not a.size:
            return "nothing measurable"
        return f"median {np.median(a):.2f}, {(a > 0.5).mean() * 100:.0f}% above 0.5, on {np.isfinite(anchoring).mean() * 100:.0f}% of the frame"

    tile = args.tile
    rows = []
    picks = sorted({int(round(q)) for q in np.linspace(0, len(results) - 1, min(args.rows, len(results)))})
    vmax = max(0.08, float(np.percentile(np.stack([r.leftover_world for r in results]), 99.5)))
    for n in picks:
        r, rc, t = results[n], coarse_results[n], pairs[n]
        frame = frames[t]
        label = f"frame {start + t} ({(start + t) / source.fps:.2f} s){note(start + t)}"
        rows.append([
            viz.tile(viz.to_float(frame), label, f"to frame {start + t + lag} (lag {lag})", tile),
            viz.tile(swim.anchoring_overlay(frame, r.anchoring), "anchoring, fine detail", headline(r.anchoring), tile),
            viz.tile(swim.anchoring_overlay(frame, rc.anchoring), f"anchoring, coarse (blur {args.coarse:g} px)",
                     headline(rc.anchoring), tile),
            viz.tile(viz.overlay_heat(viz.to_float(frame), r.leftover_world, vmax), "leftover after the world warp",
                     f"mean {r.leftover_world[r.inside].mean():.3f} (0 to {vmax:.2f})", tile),
            viz.tile(viz.overlay_heat(viz.to_float(frame), r.leftover_flow, vmax), "leftover after the flow warp (pops)",
                     f"mean {r.leftover_flow[r.inside].mean():.3f}: what pops can see", tile),
            viz.tile(swim.arrows(frame, r.measured, r.world, r.valid), "fine flow: yellow measured, cyan world",
                     "arrows scaled to the grid", tile),
        ])
    middle = frames[len(frames) // 2]
    rows.append([
        viz.tile(swim.anchoring_overlay(middle, median), "fine detail: median anchoring over the clip",
                 f"{len(results)} pairs · blue 0 world, yellow 0.5, red 1 viewer", tile * 2),
        viz.tile(swim.anchoring_overlay(middle, median_coarse), f"coarse (blur {args.coarse:g} px): median anchoring",
                 "cyan below 0: faster than the world at that depth", tile * 2),
    ])
    # The world-stabilized view: later frames warped back into a reference
    # frame by the world motion. World-fixed texture stays sharp in the mean
    # and still in the change; swimming texture smears and changes.
    ref = start + pairs[len(pairs) // 2] if args.stabilize <= 0 else start
    span = min(args.stabilize if args.stabilize > 0 else 15, end - ref - 1)
    stab = [swim.stabilize(frames[ref - start + k], depth(ref), poses[ref], poses[ref + k])
            for k in range(0, span + 1) if ref + k in poses and ref in pairing]
    if stab:
        images = np.stack([s[0] for s in stab])
        inside = np.logical_and.reduce([s[1] for s in stab])
        stab_change = np.abs(images[1:] - images[0]).max(axis=3).mean(axis=0) if len(images) > 1 else np.zeros((h, w))
        stab_change[~inside] = 0.0
        mean = np.where(inside[..., None], images.mean(axis=0), viz.to_float(frames[ref - start]) * 0.3)
        stab_vmax = max(0.08, float(np.percentile(stab_change[inside], 99.5))) if inside.any() else 0.1
        box = swim.crop_box(median, (max(64, w // 4), max(48, h // 4)))
        bx, by, bw, bh = box
        zoom = tile * 2 // 3
        rows.append([
            viz.tile(mean, f"world-stabilized mean of {len(images)} frames",
                     f"from frame {ref}: world-fixed stays sharp, swimming smears", tile * 2),
            viz.tile(viz.overlay_heat(viz.to_float(frames[ref - start]), stab_change, stab_vmax),
                     "change in the world-stabilized frames", f"mean over {len(images) - 1} frames (0 to {stab_vmax:.2f})", tile),
            viz.tile(viz.to_float(middle[by:by + bh, bx:bx + bw]), "most viewer-anchored area",
                     f"box {list(box)}, magnified", zoom),
            viz.tile(swim.anchoring_overlay(middle[by:by + bh, bx:bx + bw], median[by:by + bh, bx:bx + bw], 0.55),
                     "its fine-detail anchoring", "", zoom),
        ])
    heading = (f"pixeldif swim · {source.path.name} · {len(results)} pairs, lag {lag} · "
               f"depth paired within {worst[0]:.3g} m, {worst[1]:.2g} deg")
    sheet = viz.save(viz.grid(rows, heading), out / "swim.png")
    found = swim.regions(median, args.threshold, args.min_area)
    found_coarse = swim.regions(median_coarse, args.threshold, args.min_area)
    if args.video:
        _swim_video(frames, start, depth, poses, pairing, results, coarse_results, pairs, out / "stabilized.mp4",
                    source.fps, args.video_span)

    def overall(stack_):
        v = stack_[np.isfinite(stack_)]
        return {"median_anchoring": round(float(np.median(v)), 3) if v.size else None,
                "viewer_share": round(float((v > 0.5).mean()), 3) if v.size else None,
                "measured_share": round(float(np.isfinite(stack_).mean()), 3)}

    report = {"command": "swim", "source": str(source.path), "depth": str(depth_source.path), "fps": source.fps,
              "start_frame": start, "lag": lag, "pairs": len(results), "coarse_blur_px": args.coarse,
              "depth_pairing": {"max_position_m": worst[0], "max_rotation_deg": worst[1],
                                "offset_frames": sorted({pairing[i] - i for i in pairing}),
                                "unmatched_frames": unmatched},
              "fine": dict(overall(stack), regions=found[: args.max_regions]),
              "coarse": dict(overall(stack_coarse), regions=found_coarse[: args.max_regions]),
              "leftover_world_mean": round(float(leftover_world.mean()), 4),
              "leftover_flow_mean": round(float(leftover_flow.mean()), 4),
              "series": {k: [None if v != v else round(v, 4) for v in vals] for k, vals in series.items()},
              "x": x, "x_label": x_label, "sheet": str(sheet), "timeline": str(out / "timeline.png")}
    _write(out, report)
    print(f"sheet    {sheet}")
    print(f"timeline {out / 'timeline.png'}")
    if args.video:
        print(f"video    {out / 'stabilized.mp4'}")
    print(f"report   {out / 'report.json'}")
    if unmatched:
        print(f"WARNING: {len(unmatched)} frames have no depth frame at their pose (first {unmatched[:5]}): "
              "record both passes along the same camera path")
    for name, part in (("fine", report["fine"]), (f"coarse (blur {args.coarse:g} px)", report["coarse"])):
        if part["median_anchoring"] is None:
            print(f"{name}: nothing measurable")
            continue
        print(f"{name}: anchoring over {part['measured_share'] * 100:.0f}% of pixels, median {part['median_anchoring']:.2f} "
              f"(0 moves with the world, 1 with the viewer), {part['viewer_share'] * 100:.0f}% above 0.5; "
              f"{len(part['regions'])} regions above {args.threshold:g}")
        for r in part["regions"][:5]:
            print(f"  box {r['box_xywh']}  area {r['area_fraction'] * 100:5.2f}%  median anchoring {r['median_anchoring']:.2f}")
    print(f"leftover after the world warp {report['leftover_world_mean']:.4f}, after the flow warp {report['leftover_flow_mean']:.4f}")


def _swim_video(frames, start, depth, poses, pairing, results, coarse_results, pairs, path, fps, span):
    """A 2x2 video: the recording, the world-stabilized recording (its
    reference resets every `span` frames), and the fine and coarse anchoring."""
    import subprocess
    import cv2
    h, w = frames.shape[1:3]
    by_frame = {start + t: (r, rc) for t, r, rc in zip(pairs, results, coarse_results)}
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w * 2}x{h * 2}",
               "-r", f"{fps:g}", "-i", "-", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "16", str(path)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    fine = coarse = np.full((h, w, 3), 0.1, np.float32)
    for t in range(len(frames)):
        i = start + t
        anchor = start + (t // span) * span
        if anchor in pairing and i in poses:
            stab, inside = swim.stabilize(frames[t], depth(anchor), poses[anchor], poses[i])
            stab[~inside] *= 0.0
        else:
            stab = viz.to_float(frames[t])
        if i in by_frame:
            fine = swim.anchoring_overlay(frames[t], by_frame[i][0].anchoring)
            coarse = swim.anchoring_overlay(frames[t], by_frame[i][1].anchoring)
        panel = np.concatenate([np.concatenate([viz.to_float(frames[t]), stab], axis=1),
                                np.concatenate([fine, coarse], axis=1)], axis=0)
        image = np.ascontiguousarray(viz.to_uint8(panel))
        for (x, y), text in (((8, 20), f"recording, frame {i}"),
                             ((w + 8, 20), f"world-stabilized to frame {anchor}: world-fixed holds still"),
                             ((8, h + 20), "anchoring, fine detail: blue world, red viewer"),
                             ((w + 8, h + 20), "anchoring, coarse shapes")):
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(image, (x - 4, y - th - 5), (x + tw + 4, y + 6), (12, 14, 18), -1)
            cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        proc.stdin.write(image.tobytes())
    proc.stdin.close()
    proc.wait()


def cmd_record(args):
    from .record import record
    meta = record(args.url, Path(args.out), args.frames, args.fps, args.width, args.height, args.ready,
                  args.setup, args.warmup, args.until, args.canvas, args.timeout, args.headed, args.probe,
                  args.hold, not args.no_sandbox)
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

    q = sub.add_parser("extent", help="how far a disturbance spreads from a source point over time (front speeds)")
    q.add_argument("source")
    q.add_argument("--origin", required=True, help="source point x,y (fractions, or source pixels)")
    q.add_argument("--base", help="time of the undisturbed base frame (default: --start)")
    q.add_argument("--start")
    q.add_argument("--end")
    q.add_argument("--roi", help="x,y,w,h limit (fractions, or source pixels)")
    q.add_argument("--threshold", type=float, default=0.12, help="change against the base counted as disturbed (default 0.12)")
    q.add_argument("--scale", type=float, help="metres per source pixel at the subject (reports m and m/s)")
    q.add_argument("--window", type=float, default=0.5, help="seconds per front-speed fit (default 0.5)")
    q.add_argument("--join", type=int, default=40, help="blobs closer than this (working px) count as one disturbance")
    q.add_argument("--every", type=float, default=0.5, help="seconds between printed rows")
    common(q)
    q.set_defaults(func=cmd_extent)

    q = sub.add_parser("flow", help="optical-flow speed and direction inside a region over time")
    q.add_argument("source")
    q.add_argument("--start")
    q.add_argument("--end")
    q.add_argument("--roi", help="x,y,w,h (fractions, or source pixels)")
    q.add_argument("--scale", type=float, help="metres per source pixel at the subject (reports m/s)")
    q.add_argument("--step", type=int, default=1, help="analyse every Nth frame pair")
    q.add_argument("--min-speed", type=float, default=0.15, help="px/frame below which a pixel counts as still")
    q.add_argument("--preset", default="medium", choices=["ultrafast", "fast", "medium"])
    common(q)
    q.set_defaults(func=cmd_flow)

    q = sub.add_parser("swim", help="texture moving with the viewer instead of the world (needs poses and a depth recording)")
    q.add_argument("source", help="folder of frames recorded with per-frame camera poses (record --probe)")
    q.add_argument("--depth", required=True, help="the same camera path recorded in the renderer's depth view")
    q.add_argument("--pose-key", help="key of the pose inside each probe value (default: the value itself, or 'camera')")
    q.add_argument("--start", help="seconds, mm:ss or <n>f")
    q.add_argument("--end", help="seconds, mm:ss or <n>f")
    q.add_argument("--lag", default="1", help="frames between the compared frames (default 1)")
    q.add_argument("--step", type=int, default=1, help="analyse every Nth frame pair")
    q.add_argument("--min-separation", type=float, default=0.5,
                   help="px/frame by which world and viewer motion must differ to measure a pixel (default 0.5)")
    q.add_argument("--min-texture", type=float, default=2e-5, help="smallest structure-tensor eigenvalue to track a pixel")
    q.add_argument("--depth-tolerance", type=float, default=0.25,
                   help="log-distance mismatch at which a point counts as hidden in the next frame (default 0.25)")
    q.add_argument("--min-share", type=float, default=0.2, help="share of pairs a pixel must be measured in for the median map")
    q.add_argument("--coarse", type=float, default=3.0, help="blur (px) for the coarse-shape pass (default 3)")
    q.add_argument("--threshold", type=float, default=0.3, help="median anchoring that makes a region (default 0.3)")
    q.add_argument("--min-area", type=float, default=0.002, help="smallest region, fraction of the frame")
    q.add_argument("--max-regions", type=int, default=50)
    q.add_argument("--rows", type=int, default=4, help="frame pairs shown in the sheet")
    q.add_argument("--stabilize", type=int, default=0, help="frames in the stabilized view (default 15 from the middle)")
    q.add_argument("--video", action="store_true", help="also write stabilized.mp4 (recording | stabilized | anchoring)")
    q.add_argument("--video-span", type=int, default=30, help="frames before the stabilized video's reference resets")
    q.add_argument("--preset", default="medium", choices=["ultrafast", "fast", "medium"])
    q.add_argument("--blur", type=float, default=1.2)
    q.add_argument("--x-probe", help="plot the timeline against this per-frame probe value")
    q.add_argument("--log-x", action="store_true")
    q.add_argument("--invert-x", action="store_true")
    q.add_argument("--title")
    common(q)
    q.set_defaults(func=cmd_swim)

    q = sub.add_parser("record", help="record a web game's canvas at an exact frame rate (virtual clock)")
    q.add_argument("url")
    q.add_argument("out", help="folder for the PNG frames")
    q.add_argument("--frames", type=int, default=120)
    q.add_argument("--fps", type=float, default=60.0)
    q.add_argument("--width", type=int, default=1280, help="browser viewport width")
    q.add_argument("--height", type=int, default=720)
    q.add_argument("--ready", help="JavaScript expression to wait for before setup (e.g. window.gameReady)")
    q.add_argument("--setup", action="append", help="JavaScript to run before recording; repeatable")
    q.add_argument("--warmup", type=int, default=0, help="frames to let pass before recording")
    q.add_argument("--until", help="JavaScript expression to wait for before recording (instead of --warmup)")
    q.add_argument("--canvas", default="canvas", help="CSS selector of the canvas")
    q.add_argument("--timeout", type=float, default=900.0, help="seconds")
    q.add_argument("--headed", action="store_true")
    q.add_argument("--no-sandbox", action="store_true",
                   help="run Chromium without its sandbox (some containers and root need it; only for pages you trust)")
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
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        print(f"pixeldif: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
