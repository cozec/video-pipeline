"""Stage 5: cross-shot face matching - "if one person shows in different places of the
video, mark them as the same person".

Each face track gets one ArcFace embedding (the renormalised mean of per-frame embeddings).
Tracks are then clustered across the whole video by cosine distance.

The one addition over the reference repos: tracks whose frame ranges overlap are two
different people visible at once, so they are forbidden from merging. Without it a
two-shot of similar-looking people collapses into a single identity.
"""

import argparse

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from common import (
    RESULTS, WORK, ensure_dirs, read_json, setup_logging, write_json,
)

logger = setup_logging("identity")

COSINE_THRESHOLD = 0.55     # agglomerative distance threshold; swept in evaluate.py
MAX_FRAMES_PER_TRACK = 20   # embedding samples per track
MIN_DET_SCORE = 0.6         # only embed confident, well-landmarked faces


def track_embeddings(name: str) -> tuple[list[dict], np.ndarray]:
    """Compute one L2-normalised ArcFace embedding per face track."""
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align

    # FaceAnalysis asserts a detection model is present even when it is never called,
    # so it is loaded and left unused; only the recognition head runs here.
    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1)
    rec = app.models["recognition"]

    tracks = read_json(RESULTS / f"{name}_tracks.json")["tracks"]
    video_path = WORK / f"{name}_25fps.mp4"

    # Pick the frames to embed first, then decode the video once.
    wanted: dict[int, list[tuple[int, int]]] = {}
    for track in tracks:
        scores = np.array(track["det_scores"])
        good = np.where(scores >= MIN_DET_SCORE)[0]
        if len(good) == 0:
            good = np.argsort(scores)[-MAX_FRAMES_PER_TRACK:]
        picks = good[np.linspace(0, len(good) - 1,
                                 min(MAX_FRAMES_PER_TRACK, len(good))).astype(int)]
        for i in picks:
            frame_idx = track["start_frame"] + int(i)
            wanted.setdefault(frame_idx, []).append((track["track_id"], int(i)))

    logger.info("embedding %d tracks from %d frames", len(tracks), len(wanted))
    needed = sorted(wanted)
    cap = cv2.VideoCapture(str(video_path))
    per_track: dict[int, list[np.ndarray]] = {t["track_id"]: [] for t in tracks}
    by_id = {t["track_id"]: t for t in tracks}
    idx, pos = 0, 0
    with tqdm(total=len(needed), desc="embed") as bar:
        while pos < len(needed):
            ok, frame = cap.read()
            if not ok:
                break
            if idx == needed[pos]:
                for track_id, i in wanted[idx]:
                    kps = np.array(by_id[track_id]["kps"][i], dtype=np.float32)
                    aligned = face_align.norm_crop(frame, landmark=kps, image_size=112)
                    feat = rec.get_feat(aligned).flatten()
                    per_track[track_id].append(feat / (np.linalg.norm(feat) + 1e-9))
                pos += 1
                bar.update(1)
            idx += 1
    cap.release()

    embeddings = []
    for track in tracks:
        feats = per_track[track["track_id"]]
        mean = np.mean(feats, axis=0) if feats else np.zeros(512, dtype=np.float32)
        embeddings.append(mean / (np.linalg.norm(mean) + 1e-9))
    return tracks, np.stack(embeddings)


def cluster(tracks: list[dict], embeddings: np.ndarray,
            threshold: float = COSINE_THRESHOLD) -> np.ndarray:
    """Cluster track embeddings into person ids, forbidding temporally overlapping merges."""
    n = len(tracks)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)

    dist = 1.0 - embeddings @ embeddings.T
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, 2.0)

    # Two tracks visible in the same frame cannot be the same person.
    blocked = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tracks[i], tracks[j]
            if a["start_frame"] <= b["end_frame"] and b["start_frame"] <= a["end_frame"]:
                dist[i, j] = dist[j, i] = 1e6
                blocked += 1
    logger.info("blocked %d temporally overlapping track pairs", blocked)

    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold,
        metric="precomputed", linkage="average",
    )
    return model.fit_predict(dist)


def run_stage(name: str, threshold: float = COSINE_THRESHOLD) -> dict:
    """Assign a global person_id to every face track."""
    ensure_dirs()
    tracks, embeddings = track_embeddings(name)
    labels = cluster(tracks, embeddings, threshold=threshold)

    # Number persons by descending screen time so ids are stable and readable.
    frames_per_label: dict[int, int] = {}
    for track, lab in zip(tracks, labels):
        frames_per_label[int(lab)] = frames_per_label.get(int(lab), 0) + track["n_frames"]
    order = sorted(frames_per_label, key=lambda k: -frames_per_label[k])
    remap = {old: new for new, old in enumerate(order)}

    persons: dict[int, dict] = {}
    assignments = []
    for track, lab in zip(tracks, labels):
        pid = remap[int(lab)]
        assignments.append({"track_id": track["track_id"], "person_id": pid})
        p = persons.setdefault(pid, {"person_id": pid, "track_ids": [], "shot_ids": [],
                                     "total_frames": 0})
        p["track_ids"].append(track["track_id"])
        p["shot_ids"].append(track["shot_id"])
        p["total_frames"] += track["n_frames"]

    for p in persons.values():
        p["shot_ids"] = sorted(set(p["shot_ids"]))
        p["n_shots"] = len(p["shot_ids"])
        p["screen_time_s"] = round(p["total_frames"] / 25, 2)

    out = {
        "video": {"name": name},
        "params": {"cosine_threshold": threshold,
                   "max_frames_per_track": MAX_FRAMES_PER_TRACK,
                   "min_det_score": MIN_DET_SCORE},
        "assignments": assignments,
        "persons": [persons[k] for k in sorted(persons)],
    }
    write_json(RESULTS / f"{name}_identity.json", out)
    np.save(WORK / f"{name}_embeddings.npy", embeddings)

    multi = [p for p in out["persons"] if p["n_shots"] > 1]
    logger.info("%d persons from %d tracks; %d appear in more than one shot",
                len(out["persons"]), len(tracks), len(multi))
    for p in out["persons"][:8]:
        logger.info("  person %d: %.1fs over %d shots %s",
                    p["person_id"], p["screen_time_s"], p["n_shots"], p["shot_ids"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--threshold", type=float, default=COSINE_THRESHOLD)
    args = parser.parse_args()
    run_stage(args.name, threshold=args.threshold)


if __name__ == "__main__":
    main()
