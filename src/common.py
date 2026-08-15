"""Shared paths, logging and small helpers for the video preprocessing pipeline."""

import json
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
WORK = DATA / "work"
CROPS = DATA / "crops"
RESULTS = ROOT / "results"
IDENTITIES = RESULTS / "identities"
EVAL = RESULTS / "eval"
LABELS = ROOT / "labels"        # hand-made ground truth: an input, not an output
LOGS = ROOT / "logs"
MODELS = DATA / "models"

# TalkNet and SyncNet are both trained on 25 fps video with 16 kHz mono audio.
FPS = 25
SAMPLE_RATE = 16000


def setup_logging(name: str) -> logging.Logger:
    """Configure root logging to stderr and return a named logger."""
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(LOGS / "pipeline.log"),
        ],
    )
    return logging.getLogger(name)


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into."""
    for d in (RAW, WORK, CROPS, RESULTS, IDENTITIES, EVAL, LOGS, MODELS, LABELS):
        d.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a subprocess, raising with captured stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-4000:]}"
        )
    return proc


def ffprobe(path: Path) -> dict:
    """Return the ffprobe JSON description of a media file."""
    proc = run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ]
    )
    return json.loads(proc.stdout)


def video_info(path: Path) -> dict:
    """Return width, height, fps, frame count and duration for a video file."""
    info = ffprobe(path)
    stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    num, den = stream["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    duration = float(info["format"]["duration"])
    n_frames = stream.get("nb_frames")
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration_s": duration,
        "n_frames": int(n_frames) if n_frames else int(round(duration * fps)),
    }


def read_json(path: Path) -> dict:
    """Load a JSON file."""
    with open(path) as fh:
        return json.load(fh)


def write_json(path: Path, obj: dict) -> None:
    """Write a JSON file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
