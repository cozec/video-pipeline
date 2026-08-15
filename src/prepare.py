"""Stage 1: normalise the source video to 25 fps video + 16 kHz mono audio.

TalkNet and SyncNet both assume exactly 25 fps and 16 kHz. Normalising once up front
means every later stage can index frames and audio samples with a single fixed ratio
(4 MFCC frames per video frame) instead of resampling per track.
"""

import argparse
from pathlib import Path

from common import (
    FPS, RAW, SAMPLE_RATE, WORK, ensure_dirs, run, setup_logging, video_info,
)

logger = setup_logging("prepare")


def prepare(src: Path, name: str, force: bool = False) -> tuple[Path, Path]:
    """Write a 25 fps MP4 and a 16 kHz mono WAV into data/work/. Returns both paths."""
    ensure_dirs()
    video = WORK / f"{name}_25fps.mp4"
    audio = WORK / f"{name}_16k.wav"

    if force or not video.exists():
        logger.info("normalising video to %d fps -> %s", FPS, video)
        run([
            "ffmpeg", "-y", "-i", str(src),
            "-r", str(FPS),
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-an", str(video),
        ])
    else:
        logger.info("reusing %s", video)

    if force or not audio.exists():
        logger.info("extracting %d Hz mono audio -> %s", SAMPLE_RATE, audio)
        run([
            "ffmpeg", "-y", "-i", str(src),
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-vn", "-acodec", "pcm_s16le", str(audio),
        ])
    else:
        logger.info("reusing %s", audio)

    return video, audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="basename used in data/raw")
    parser.add_argument("--force", action="store_true", help="re-encode even if outputs exist")
    args = parser.parse_args()

    src = RAW / f"{args.name}.mp4"
    video, audio = prepare(src, args.name, force=args.force)
    logger.info("normalised video: %s", video_info(video))


if __name__ == "__main__":
    main()
