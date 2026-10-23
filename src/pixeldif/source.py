"""Frame sources: video files (decoded with ffmpeg) and image-sequence folders.

Frames come back as uint8 RGB arrays of shape (n, h, w, 3), optionally scaled
to a working width. Everything downstream works on these arrays, so a game
capture (a folder of PNGs) and a reference video behave the same.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
# Inputs may be local files only: a video that is really a playlist could
# otherwise make ffmpeg fetch URLs.
INPUT_GUARD = ["-protocol_whitelist", "file"]


def _ffmpeg_path(path: Path) -> str:
    """An absolute path, which ffmpeg can't mistake for an option (a name
    starting with '-') or a protocol ('name:...')."""
    return str(Path(path).absolute())


@dataclass
class Source:
    path: Path
    kind: str  # "video" or "images"
    fps: float
    width: int
    height: int
    frame_count: int
    files: list[Path] | None = None

    def probes(self) -> dict[int, object]:
        """Per-frame values recorded with `record --probe` (frames.jsonl)."""
        path = self.path / "frames.jsonl" if self.kind == "images" else None
        if not path or not path.exists():
            return {}
        out = {}
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                out[int(row["frame"])] = row["value"]
        return out

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps

    def frame_at(self, seconds: float) -> int:
        return int(round(seconds * self.fps))

    def scaled_size(self, width: int | None) -> tuple[int, int]:
        if not width or width >= self.width:
            return self.width, self.height
        height = int(round(self.height * width / self.width / 2)) * 2
        return width, max(height, 2)

    def read(self, start: int, count: int, width: int | None = None) -> np.ndarray:
        """Frames [start, start + count), clamped to the source."""
        start = max(0, min(start, self.frame_count - 1))
        count = max(1, min(count, self.frame_count - start))
        w, h = self.scaled_size(width)
        if self.kind == "images":
            frames = []
            for file in self.files[start:start + count]:
                image = Image.open(file).convert("RGB")
                if image.size != (w, h):
                    image = image.resize((w, h), Image.Resampling.LANCZOS)
                frames.append(np.asarray(image))
            return np.stack(frames)
        # Accurate input seeking: ffmpeg decodes from the previous keyframe and
        # discards frames up to the requested time.
        command = [
            "ffmpeg", "-v", "error", *INPUT_GUARD, "-ss", f"{start / self.fps:.6f}", "-i", _ffmpeg_path(self.path),
            "-frames:v", str(count), "-vf", f"scale={w}:{h}:flags=area",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
        data = subprocess.run(command, check=True, capture_output=True).stdout
        n = len(data) // (w * h * 3)
        if n == 0:
            raise RuntimeError(f"ffmpeg returned no frames for {self.path} at frame {start}")
        return np.frombuffer(data[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)


def open_source(path: str | Path, fps: float | None = None) -> Source:
    path = Path(path)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"no images in {path}")
        meta = path / "pixeldif.json"
        if fps is None and meta.exists():
            fps = json.loads(meta.read_text()).get("fps")
        with Image.open(files[0]) as first:
            width, height = first.size
        return Source(path, "images", float(fps or 60.0), width, height, len(files), files)
    if not path.exists():
        raise FileNotFoundError(path)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", *INPUT_GUARD, "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
         "-of", "json", _ffmpeg_path(path)],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        raise ValueError(f"ffprobe can't read {path}: {probe.stderr.strip() or 'unknown error'}")
    info = json.loads(probe.stdout)
    if not info.get("streams"):
        raise ValueError(f"no video stream in {path}")
    stream = info["streams"][0]

    def rate(text: str) -> float:
        num, _, den = text.partition("/")
        return float(num) / float(den or 1) if float(den or 1) else 0.0

    video_fps = fps or rate(stream.get("avg_frame_rate", "0/1")) or rate(stream.get("r_frame_rate", "30/1"))
    duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0)
    count = int(stream.get("nb_frames") or 0) or int(round(duration * video_fps))
    return Source(path, "video", video_fps, int(stream["width"]), int(stream["height"]), count)


_LAG = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*(f|ms|s)?\s*$")


def parse_lag(text: str, fps: float) -> int:
    """'4' or '4f' frames, '50ms', '0.5s' -> whole frames at this frame rate (>= 1)."""
    match = _LAG.match(text)
    if not match:
        raise ValueError(f"bad lag {text!r}: use frames (4 or 4f), milliseconds (50ms) or seconds (0.5s)")
    value, unit = float(match.group(1)), match.group(2) or "f"
    frames = value if unit == "f" else value * fps / (1000.0 if unit == "ms" else 1.0)
    return max(1, int(round(frames)))


def parse_time(text: str, source: Source) -> int:
    """'12.5' or '12.5s' seconds, '750f' frames, or 'mm:ss.s' -> frame index."""
    text = text.strip()
    if text.endswith("f"):
        return int(float(text[:-1]))
    if ":" in text:
        minutes, _, seconds = text.rpartition(":")
        return source.frame_at(float(minutes or 0) * 60 + float(seconds))
    return source.frame_at(float(text.rstrip("s")))


def parse_roi(text: str | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    """'x,y,w,h' in pixels, or fractions of the frame when all are <= 1."""
    if not text:
        return None
    values = [float(v) for v in text.split(",")]
    if len(values) != 4:
        raise ValueError("roi is x,y,w,h")
    if all(v <= 1.0 for v in values):
        values = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    x, y, w, h = (int(round(v)) for v in values)
    x, y = max(0, x), max(0, y)
    return x, y, max(1, min(w, width - x)), max(1, min(h, height - y))
