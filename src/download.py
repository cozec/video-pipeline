"""Stage 0: fetch the source video from YouTube with yt-dlp."""

import argparse
import sys
from pathlib import Path

from common import RAW, ensure_dirs, run, setup_logging, video_info

logger = setup_logging("download")


def download(url: str, name: str) -> Path:
    """Download `url` to data/raw/<name>.mp4, skipping if already present."""
    ensure_dirs()
    out = RAW / f"{name}.mp4"
    if out.exists():
        logger.info("already downloaded: %s", out)
        return out

    logger.info("downloading %s", url)
    run([
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(out),
        url,
    ])
    logger.info("saved %s (%.1f MB)", out, out.stat().st_size / 1e6)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--name", required=True, help="output basename, e.g. sfmta")
    args = parser.parse_args()

    path = download(args.url, args.name)
    logger.info("video info: %s", video_info(path))


if __name__ == "__main__":
    main()
