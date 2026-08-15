"""Stage 6: the two human-checkable outputs.

  - results/<name>_annotated.mp4  boxes, person id and speaking state burned in
  - results/identities/<name>_person_XX.jpg  every shot a person appears in, side by side

The contact sheet is the direct visual proof of cross-shot matching: if a sheet shows two
different faces, the clustering was wrong.
"""

import argparse
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm

from common import (
    FPS, IDENTITIES, RESULTS, WORK, ensure_dirs, read_json, run, setup_logging,
)
from sheets import grid, label

logger = setup_logging("visualize")

GREEN = (60, 220, 60)      # speaking, corroborated by SyncNet
AMBER = (0, 190, 255)      # TalkNet says speaking, SyncNet does not corroborate
RED = (60, 60, 220)        # not speaking
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_all(name: str) -> dict:
    """Load every stage output for one video."""
    return {
        "shots": read_json(RESULTS / f"{name}_shots.json"),
        "tracks": read_json(RESULTS / f"{name}_tracks.json"),
        "asd": read_json(RESULTS / f"{name}_asd.json"),
        "identity": read_json(RESULTS / f"{name}_identity.json"),
    }


def annotate(name: str) -> str:
    """Render the annotated MP4 and mux the original audio back in."""
    ensure_dirs()
    data = load_all(name)
    video_path = WORK / f"{name}_25fps.mp4"
    audio_path = WORK / f"{name}_16k.wav"

    tracks = {t["track_id"]: t for t in data["tracks"]["tracks"]}
    asd = {a["track_id"]: a for a in data["asd"]["asd"]}
    person_of = {a["track_id"]: a["person_id"] for a in data["identity"]["assignments"]}
    shots = data["shots"]["shots"]

    shot_of_frame = {}
    for s in shots:
        for f in range(s["start_frame"], s["end_frame"] + 1):
            shot_of_frame[f] = s["shot_id"]

    # frame -> list of things to draw
    per_frame = defaultdict(list)
    for tid, track in tracks.items():
        scores = asd[tid]["talknet_scores"]
        corroborated = asd[tid]["syncnet_corroborates"]
        for i in range(track["n_frames"]):
            f = track["start_frame"] + i
            score = scores[i] if i < len(scores) else None
            per_frame[f].append({
                "bbox": track["bboxes"][i],
                "person_id": person_of[tid],
                "track_id": tid,
                "score": score,
                "corroborated": corroborated,
            })

    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp = RESULTS / f"{name}_annotated_silent.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (width, height))

    thick = max(2, width // 640)
    scale = width / 1600
    idx = 0
    with tqdm(total=total, desc="annotate") as bar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for d in per_frame.get(idx, []):
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                speaking = d["score"] is not None and d["score"] > 0
                if not speaking:
                    color = RED
                else:
                    color = GREEN if d["corroborated"] else AMBER
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                tag = f"PERSON_{d['person_id']:02d}"
                if d["score"] is not None:
                    tag += f" {d['score']:+.1f}"
                (tw, th), _ = cv2.getTextSize(tag, FONT, scale, thick)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                cv2.putText(frame, tag, (x1 + 3, y1 - 5), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)

            banner = f"shot {shot_of_frame.get(idx, '-')}   frame {idx}   {idx / FPS:6.2f}s"
            cv2.rectangle(frame, (0, 0), (int(560 * scale * 1.6), int(46 * scale * 1.6)), (0, 0, 0), -1)
            cv2.putText(frame, banner, (10, int(32 * scale * 1.6)), FONT, scale * 1.1,
                        (255, 255, 255), thick, cv2.LINE_AA)
            writer.write(frame)
            idx += 1
            bar.update(1)
    cap.release()
    writer.release()

    # OpenCV can only write mp4v here, which is ~10x larger than necessary; re-encode
    # on the mux pass rather than copying it through.
    out = RESULTS / f"{name}_annotated.mp4"
    run(["ffmpeg", "-y", "-i", str(tmp), "-i", str(audio_path),
         "-c:v", "libx264", "-crf", "23", "-preset", "medium", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(out), "-loglevel", "error"])
    tmp.unlink()
    logger.info("wrote %s", out)
    return str(out)


def contact_sheets(name: str, max_per_shot: int = 3) -> list[str]:
    """One sheet per person, sampling face crops from every shot they appear in."""
    ensure_dirs()
    data = load_all(name)
    tracks = {t["track_id"]: t for t in data["tracks"]["tracks"]}
    person_of = {a["track_id"]: a["person_id"] for a in data["identity"]["assignments"]}

    by_person = defaultdict(list)
    for tid, pid in person_of.items():
        by_person[pid].append(tid)

    written = []
    for person in data["identity"]["persons"]:
        pid = person["person_id"]
        tiles = []
        for tid in sorted(by_person[pid]):
            track = tracks[tid]
            cap = cv2.VideoCapture(track["avi"])
            n = track["n_frames"]
            picks = np.linspace(0, max(0, n - 1), min(max_per_shot, n)).astype(int)
            for p in picks:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(p))
                ok, face = cap.read()
                if ok:
                    tiles.append(label(face, f"shot {track['shot_id']} trk{tid}"))
            cap.release()
        if not tiles:
            continue
        sheet = grid(tiles, cols=min(6, len(tiles)), tile_w=160)
        path = IDENTITIES / f"{name}_person_{pid:02d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        written.append(str(path))
    logger.info("wrote %d contact sheets to %s", len(written), IDENTITIES)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()
    if not args.skip_video:
        annotate(args.name)
    contact_sheets(args.name)


if __name__ == "__main__":
    main()
