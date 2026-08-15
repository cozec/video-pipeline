"""Stage 3: face detection (SCRFD), per-shot IoU tracking, and crop extraction.

The reference diagram calls this "track person"; every downstream stage (crops for TalkNet,
ArcFace embeddings for face matching) is face-based, so tracking is done on faces.

Detection is InsightFace SCRFD rather than the reference repos' S3FD: it is faster and the
same `buffalo_l` pack supplies the 5-point landmarks and the ArcFace embedding used later
for cross-shot matching. The *crop* convention, which is what TalkNet is actually sensitive
to, follows demoTalkNet.crop_video exactly.
"""

import argparse

import cv2
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from tqdm import tqdm

from common import (
    CROPS, FPS, RESULTS, WORK, ensure_dirs, read_json, run, setup_logging, write_json,
)

logger = setup_logging("faces")

# Tracking parameters. syncnet's defaults target film; a news package cuts far faster,
# so min_track and num_failed_det are much shorter here (see README).
IOU_THRES = 0.5
MIN_TRACK = 12          # frames; syncnet uses 100 (4 s), which drops most news shots
NUM_FAILED_DET = 10     # frames a track may coast without a detection
MIN_FACE_SIZE = 60      # px at full resolution
CROP_SCALE = 0.40       # must match TalkNet/SyncNet training crops
DET_THRESH = 0.5


def iou(box_a, box_b) -> float:
    """Intersection over union of two [x1, y1, x2, y2] boxes."""
    x_a, y_a = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x_b, y_b = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detect_faces(video_path, det_size: int = 640) -> dict[int, list[dict]]:
    """Run SCRFD on every frame. Returns {frame_index: [{bbox, kps, score}, ...]}."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_thresh=DET_THRESH, det_size=(det_size, det_size))
    detector = app.models["detection"]

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dets: dict[int, list[dict]] = {}
    idx = 0
    with tqdm(total=total, desc="detect", unit="f") as bar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            bboxes, kpss = detector.detect(frame, max_num=0, metric="default")
            frame_dets = []
            for i in range(bboxes.shape[0]):
                x1, y1, x2, y2, score = bboxes[i]
                frame_dets.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "kps": kpss[i].astype(float).tolist() if kpss is not None else None,
                    "score": float(score),
                })
            dets[idx] = frame_dets
            idx += 1
            bar.update(1)
    cap.release()
    logger.info("detected %d faces over %d frames", sum(len(v) for v in dets.values()), idx)
    return dets


def track_shot(shot_dets: list[dict]) -> list[dict]:
    """Greedy IoU tracking within one shot, following syncnet run_pipeline.track_shot.

    `shot_dets` is a list of per-frame detection lists. Returns tracks with bboxes and
    landmarks linearly interpolated across frames where detection dropped out.
    """
    remaining = [list(fd) for fd in shot_dets]
    tracks = []

    while True:
        track = []
        for frame_faces in remaining:
            for face in list(frame_faces):
                if not track:
                    track.append(face)
                    frame_faces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= NUM_FAILED_DET:
                    if iou(face["bbox"], track[-1]["bbox"]) > IOU_THRES:
                        track.append(face)
                        frame_faces.remove(face)
                        continue
                else:
                    break
        if not track:
            break
        if len(track) <= MIN_TRACK:
            continue

        frames = np.array([f["frame"] for f in track])
        bboxes = np.array([f["bbox"] for f in track])
        scores = np.array([f["score"] for f in track])
        kps = np.array([f["kps"] for f in track])          # (T, 5, 2)

        full = np.arange(frames[0], frames[-1] + 1)
        bboxes_i = np.stack(
            [interp1d(frames, bboxes[:, j])(full) for j in range(4)], axis=1
        )
        scores_i = interp1d(frames, scores)(full)
        kps_i = np.stack(
            [interp1d(frames, kps[:, p, c])(full) for p in range(5) for c in range(2)],
            axis=1,
        ).reshape(-1, 5, 2)

        widths = bboxes_i[:, 2] - bboxes_i[:, 0]
        heights = bboxes_i[:, 3] - bboxes_i[:, 1]
        if max(widths.mean(), heights.mean()) <= MIN_FACE_SIZE:
            continue

        tracks.append({
            "frames": full,
            "bboxes": bboxes_i,
            "scores": scores_i,
            "kps": kps_i,
            "n_detected": len(track),
        })
    return tracks


def _medfilt(values: np.ndarray, kernel: int = 13) -> np.ndarray:
    """Median filter, shrinking the kernel for short tracks.

    scipy.signal.medfilt zero-pads, so a kernel wider than the track would drag the
    smoothed centre towards zero and destroy the crop. Short news tracks hit this.
    """
    k = min(kernel, len(values) if len(values) % 2 == 1 else len(values) - 1)
    return signal.medfilt(values, kernel_size=max(1, k))


def crop_track(video_frames: dict, track: dict, out_stem, audio_path) -> dict:
    """Write a 224x224 face-crop AVI plus its matching WAV, per demoTalkNet.crop_video."""
    bboxes = track["bboxes"]
    size = _medfilt(np.maximum(bboxes[:, 3] - bboxes[:, 1], bboxes[:, 2] - bboxes[:, 0]) / 2)
    cx = _medfilt((bboxes[:, 0] + bboxes[:, 2]) / 2)
    cy = _medfilt((bboxes[:, 1] + bboxes[:, 3]) / 2)

    tmp_avi = out_stem.with_suffix(".t.avi")
    writer = cv2.VideoWriter(str(tmp_avi), cv2.VideoWriter_fourcc(*"XVID"), FPS, (224, 224))
    for i, frame_idx in enumerate(track["frames"]):
        image = video_frames[int(frame_idx)]
        bs = size[i]
        bsi = int(bs * (1 + 2 * CROP_SCALE))
        padded = np.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)),
                        "constant", constant_values=(110, 110))
        my, mx = cy[i] + bsi, cx[i] + bsi
        face = padded[
            int(my - bs):int(my + bs * (1 + 2 * CROP_SCALE)),
            int(mx - bs * (1 + CROP_SCALE)):int(mx + bs * (1 + CROP_SCALE)),
        ]
        writer.write(cv2.resize(face, (224, 224)))
    writer.release()

    wav = out_stem.with_suffix(".wav")
    start_s = float(track["frames"][0]) / FPS
    end_s = float(track["frames"][-1] + 1) / FPS
    run(["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-vn",
         "-acodec", "pcm_s16le", "-ar", "16000",
         "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", str(wav), "-loglevel", "error"])

    avi = out_stem.with_suffix(".avi")
    run(["ffmpeg", "-y", "-i", str(tmp_avi), "-i", str(wav),
         "-c:v", "copy", "-c:a", "copy", str(avi), "-loglevel", "error"])
    tmp_avi.unlink()
    return {"avi": str(avi), "wav": str(wav)}


def run_stage(name: str, det_size: int = 640, redetect: bool = False) -> dict:
    """Detect, track and crop faces for a prepared video."""
    ensure_dirs()
    video_path = WORK / f"{name}_25fps.mp4"
    audio_path = WORK / f"{name}_16k.wav"
    shots = read_json(RESULTS / f"{name}_shots.json")["shots"]

    det_cache = WORK / f"{name}_dets.json"
    if det_cache.exists() and not redetect:
        logger.info("reusing cached detections %s", det_cache.name)
        dets = {int(k): v for k, v in read_json(det_cache)["dets"].items()}
    else:
        dets = detect_faces(video_path, det_size=det_size)
        write_json(det_cache, {"det_size": det_size, "dets": dets})

    # Track within each shot; tracks never cross a shot boundary.
    all_tracks = []
    for shot in shots:
        shot_dets = []
        for f in range(shot["start_frame"], shot["end_frame"] + 1):
            shot_dets.append([{**d, "frame": f} for d in dets.get(f, [])])
        for tr in track_shot(shot_dets):
            tr["shot_id"] = shot["shot_id"]
            all_tracks.append(tr)
    logger.info("%d tracks over %d shots", len(all_tracks), len(shots))

    # One sequential decode pass to gather every frame any track needs.
    needed = sorted({int(f) for tr in all_tracks for f in tr["frames"]})
    logger.info("decoding %d frames for crops", len(needed))
    cap = cv2.VideoCapture(str(video_path))
    frames_by_idx, idx, pos = {}, 0, 0
    while pos < len(needed):
        ok, frame = cap.read()
        if not ok:
            break
        if idx == needed[pos]:
            frames_by_idx[idx] = frame
            pos += 1
        idx += 1
    cap.release()

    records = []
    for i, tr in enumerate(tqdm(all_tracks, desc="crop")):
        stem = CROPS / f"{name}_track_{i:04d}"
        paths = crop_track(frames_by_idx, tr, stem, audio_path)
        records.append({
            "track_id": i,
            "shot_id": tr["shot_id"],
            "start_frame": int(tr["frames"][0]),
            "end_frame": int(tr["frames"][-1]),
            "n_frames": int(len(tr["frames"])),
            "n_detected": tr["n_detected"],
            "mean_det_score": round(float(tr["scores"].mean()), 4),
            "mean_face_px": round(float((tr["bboxes"][:, 2] - tr["bboxes"][:, 0]).mean()), 1),
            "bboxes": np.round(tr["bboxes"], 2).tolist(),
            "kps": np.round(tr["kps"], 2).tolist(),
            "det_scores": np.round(tr["scores"], 4).tolist(),
            **paths,
        })

    out = {
        "video": {"name": name},
        "params": {
            "iou_thres": IOU_THRES, "min_track": MIN_TRACK,
            "num_failed_det": NUM_FAILED_DET, "min_face_size": MIN_FACE_SIZE,
            "crop_scale": CROP_SCALE, "det_thresh": DET_THRESH, "det_size": det_size,
        },
        "tracks": records,
    }
    write_json(RESULTS / f"{name}_tracks.json", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--redetect", action="store_true", help="ignore the detection cache")
    args = parser.parse_args()
    run_stage(args.name, det_size=args.det_size, redetect=args.redetect)


if __name__ == "__main__":
    main()
