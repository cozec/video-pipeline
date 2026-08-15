"""End-to-end orchestrator: URL or local file in, manifest + annotated video out.

    python src/pipeline.py --url https://youtu.be/Oq544HEfkLE --name sfmta

Stages cache to disk, so --from-stage restarts partway through without redoing the
expensive detection pass.
"""

import argparse
import time
from collections import defaultdict

from common import (
    FPS, RAW, RESULTS, ensure_dirs, read_json, run, setup_logging, video_info, write_json,
)

logger = setup_logging("pipeline")

STAGES = ["download", "prepare", "shots", "faces", "asd", "identity", "visualize", "manifest"]


def build_manifest(name: str) -> dict:
    """Join every stage output into one results/<name>_manifest.json."""
    shots_data = read_json(RESULTS / f"{name}_shots.json")
    tracks_data = read_json(RESULTS / f"{name}_tracks.json")
    asd_data = read_json(RESULTS / f"{name}_asd.json")
    ident_data = read_json(RESULTS / f"{name}_identity.json")

    asd = {a["track_id"]: a for a in asd_data["asd"]}
    person_of = {a["track_id"]: a["person_id"] for a in ident_data["assignments"]}

    tracks = []
    speaking_frames = defaultdict(int)
    for track in tracks_data["tracks"]:
        tid = track["track_id"]
        a = asd[tid]
        scores = a["talknet_scores"]
        speaking = a["speaking"]
        pid = person_of[tid]
        speaking_frames[pid] += int(sum(speaking))

        frames = []
        for i in range(track["n_frames"]):
            frames.append({
                "frame": track["start_frame"] + i,
                "bbox": track["bboxes"][i],
                "kps": track["kps"][i],
                "det_score": track["det_scores"][i],
                "talknet_score": scores[i] if i < len(scores) else None,
                "speaking": bool(speaking[i]) if i < len(speaking) else None,
            })

        tracks.append({
            "track_id": tid,
            "shot_id": track["shot_id"],
            "person_id": pid,
            "start_frame": track["start_frame"],
            "end_frame": track["end_frame"],
            "start_s": round(track["start_frame"] / FPS, 3),
            "end_s": round((track["end_frame"] + 1) / FPS, 3),
            "n_frames": track["n_frames"],
            "mean_det_score": track["mean_det_score"],
            "mean_face_px": track["mean_face_px"],
            "mean_talknet_score": a["mean_talknet_score"],
            "speaking_ratio": a["speaking_ratio"],
            "syncnet": a["syncnet"] | {"framewise_conf": None},   # per-frame kept in _asd.json
            "syncnet_corroborates": a["syncnet_corroborates"],
            "crop_avi": track["avi"],
            "frames": frames,
        })

    persons = []
    for p in ident_data["persons"]:
        persons.append(p | {
            "speaking_time_s": round(speaking_frames[p["person_id"]] / FPS, 2),
        })

    manifest = {
        "video": shots_data["video"],
        "params": {
            "shots": shots_data["params"],
            "faces": tracks_data["params"],
            "asd": asd_data["params"],
            "identity": ident_data["params"],
        },
        "shots": shots_data["shots"],
        "tracks": tracks,
        "persons": persons,
    }
    path = RESULTS / f"{name}_manifest.json"
    write_json(path, manifest)
    logger.info("manifest: %d shots, %d tracks, %d persons -> %s",
                len(manifest["shots"]), len(tracks), len(persons), path)
    return manifest


def export_shots(name: str) -> None:
    """Optional: write one MP4 per shot by stream copy (the literal 'cut to segments')."""
    from common import WORK
    out_dir = RESULTS / f"{name}_shots"
    out_dir.mkdir(parents=True, exist_ok=True)
    shots = read_json(RESULTS / f"{name}_shots.json")["shots"]
    src = WORK / f"{name}_25fps.mp4"
    for s in shots:
        dst = out_dir / f"shot_{s['shot_id']:03d}.mp4"
        run(["ffmpeg", "-y", "-ss", f"{s['start_s']:.3f}", "-to", f"{s['end_s']:.3f}",
             "-i", str(src), "-c", "copy", str(dst), "-loglevel", "error"])
    logger.info("exported %d shot clips to %s", len(shots), out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--url", help="YouTube URL; omit if data/raw/<name>.mp4 exists")
    parser.add_argument("--from-stage", default="download", choices=STAGES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--export-shots", action="store_true",
                        help="also write one MP4 per shot")
    args = parser.parse_args()

    ensure_dirs()
    start = STAGES.index(args.from_stage)
    todo = STAGES[start:]
    timings = {}

    for stage in todo:
        t0 = time.time()
        if stage == "download":
            if args.url:
                from download import download
                download(args.url, args.name)
            else:
                logger.info("no --url, using data/raw/%s.mp4", args.name)
        elif stage == "prepare":
            from prepare import prepare
            prepare(RAW / f"{args.name}.mp4", args.name)
        elif stage == "shots":
            import shots
            shots.run(args.name)
        elif stage == "faces":
            import faces
            faces.run_stage(args.name)
        elif stage == "asd":
            import asd
            asd.run_stage(args.name, device=args.device)
        elif stage == "identity":
            import identity
            identity.run_stage(args.name)
        elif stage == "visualize":
            import visualize
            visualize.annotate(args.name)
            visualize.contact_sheets(args.name)
        elif stage == "manifest":
            build_manifest(args.name)
        timings[stage] = round(time.time() - t0, 1)
        logger.info("stage %s took %.1fs", stage, timings[stage])

    if args.export_shots:
        export_shots(args.name)

    logger.info("video: %s", video_info(RAW / f"{args.name}.mp4"))
    logger.info("timings (s): %s  total %.1f", timings, sum(timings.values()))
    write_json(RESULTS / f"{args.name}_timings.json", timings)


if __name__ == "__main__":
    main()
