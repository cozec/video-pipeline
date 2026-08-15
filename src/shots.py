"""Stage 2: shot boundary detection with TransNetV2, plus a PySceneDetect baseline.

TransNetV2 is the model named in the reference pipeline. PySceneDetect's ContentDetector
is run alongside it purely as a cross-check: where the two disagree is where hand-labelling
effort is worth spending.
"""

import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

from common import (
    FPS, RESULTS, WORK, ensure_dirs, setup_logging, video_info, write_json,
)

logger = setup_logging("shots")


def detect_transnetv2(video_path, threshold: float = 0.5, device: str = "auto") -> list[dict]:
    """Run TransNetV2 over the whole video and return shots as frame/second ranges."""
    import torch
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device=device) if device != "auto" else TransNetV2()
    model.eval()
    with torch.no_grad():
        _, single_frame_pred, _ = model.predict_video(str(video_path), quiet=True)

    preds = single_frame_pred.cpu().numpy() if hasattr(single_frame_pred, "cpu") else single_frame_pred
    preds = np.asarray(preds).reshape(-1)
    scenes = TransNetV2.predictions_to_scenes(preds, threshold=threshold)

    shots = []
    for i, (start, end) in enumerate(scenes):
        shots.append({
            "shot_id": i,
            "start_frame": int(start),
            "end_frame": int(end),          # inclusive
            "start_s": round(float(start) / FPS, 3),
            "end_s": round(float(end + 1) / FPS, 3),
            "n_frames": int(end - start + 1),
            "peak_score": round(float(preds[start:end + 1].max()), 4),
        })
    return shots, preds


def detect_pyscenedetect(video_path, threshold: float = 27.0) -> list[dict]:
    """Baseline shot detection with PySceneDetect ContentDetector."""
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(video_path))
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold))
    manager.detect_scenes(video, show_progress=False)

    shots = []
    for i, (start, end) in enumerate(manager.get_scene_list()):
        shots.append({
            "shot_id": i,
            "start_frame": start.frame_num,
            "end_frame": end.frame_num - 1,
            "start_s": round(start.seconds, 3),
            "end_s": round(end.seconds, 3),
            "n_frames": end.frame_num - start.frame_num,
        })
    return shots


def run(name: str, threshold: float = 0.5, device: str = "auto") -> dict:
    """Detect shots for a prepared video and write results/<name>_shots.json."""
    ensure_dirs()
    video_path = WORK / f"{name}_25fps.mp4"
    info = video_info(video_path)

    logger.info("running TransNetV2 on %s (%d frames)", video_path.name, info["n_frames"])
    shots, preds = detect_transnetv2(video_path, threshold=threshold, device=device)
    logger.info("TransNetV2: %d shots", len(shots))

    logger.info("running PySceneDetect baseline")
    baseline = detect_pyscenedetect(video_path)
    logger.info("PySceneDetect: %d shots", len(baseline))

    out = {
        "video": {"name": name, "path": str(video_path), **info},
        "shots": shots,
        "baseline_pyscenedetect": baseline,
        "params": {"transnetv2_threshold": threshold},
    }
    write_json(RESULTS / f"{name}_shots.json", out)
    np.save(WORK / f"{name}_transnet_preds.npy", preds)

    durations = [s["n_frames"] / FPS for s in shots]
    logger.info(
        "shot durations: min %.2fs median %.2fs max %.2fs",
        min(durations), float(np.median(durations)), max(durations),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run(args.name, threshold=args.threshold, device=args.device)


if __name__ == "__main__":
    main()
