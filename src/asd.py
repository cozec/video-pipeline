"""Stage 4: active speaker detection.

TalkNet is the primary model (as in the reference diagram) and produces a per-frame
speaking score for every face track. SyncNet runs as a second, independent signal: it
measures audio-visual correlation and the AV offset, which both cross-checks TalkNet on
contested tracks and reveals whole-video audio desync that would silently degrade TalkNet.

Both are inference-only ports of the original repos. The vendored model code is unmodified
apart from import paths; the CUDA-hardcoded training wrappers are replaced by the classes
below, which are device-parameterised.
"""

import argparse
import math
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import python_speech_features
import torch
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

from common import MODELS, RESULTS, ensure_dirs, read_json, setup_logging, write_json
from vendor.syncnet.SyncNetInstance import calc_pdist
from vendor.syncnet.SyncNetModel import S as SyncNetS
from vendor.talknet.loss import lossAV
from vendor.talknet.talkNetModel import talkNetModel

logger = setup_logging("asd")

# Multi-scale averaging window lengths, from demoTalkNet.evaluate_network. Repeated
# entries weight the short windows more heavily, which is what the original does.
DURATION_SET = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]
SYNCNET_VSHIFT = 10
SYNCNET_BATCH = 20


SYNCNET_MIN_CONF = 3.0


def corroborates(sync: dict) -> bool:
    """Does SyncNet independently support a "this face is speaking" call?

    An offset pinned at +/-vshift means the search railed at the edge of its window and
    found no genuine alignment, so a high confidence there is not evidence of speech.
    """
    if sync["confidence"] is None or sync["offset_frames"] is None:
        return False
    return sync["confidence"] >= SYNCNET_MIN_CONF and abs(sync["offset_frames"]) < SYNCNET_VSHIFT


def pick_device(requested: str) -> torch.device:
    """Resolve 'auto' to mps when available, else cpu."""
    if requested != "auto":
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _load_state(path):
    """Load a 2019-era checkpoint, falling back for non-tensor pickles."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


class TalkNetScorer(torch.nn.Module):
    """Inference-only TalkNet: replaces talkNet.py, which hardcodes .cuda()."""

    def __init__(self, weights_path, device):
        super().__init__()
        self.device = device
        self.model = talkNetModel()
        self.lossAV = lossAV()

        state = _load_state(weights_path)
        own = self.state_dict()
        loaded, skipped = 0, []
        for key, param in state.items():
            name = key if key in own else key.replace("module.", "")
            if name not in own:
                skipped.append(key)          # lossA / lossV heads: training only
                continue
            if own[name].size() != param.size():
                skipped.append(key)
                continue
            own[name].copy_(param)
            loaded += 1
        logger.info("TalkNet: loaded %d tensors, skipped %d (%s)",
                    loaded, len(skipped), ", ".join(skipped[:4]))
        self.to(device).eval()

    @staticmethod
    def _read_visual(avi_path) -> np.ndarray:
        """Grayscale, 224x224, centre 112x112 crop - exactly demoTalkNet's preprocessing."""
        cap = cv2.VideoCapture(str(avi_path))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224, 224))
            frames.append(face[56:168, 56:168])
        cap.release()
        return np.array(frames)

    @torch.no_grad()
    def score(self, avi_path, wav_path) -> np.ndarray:
        """Return a per-frame speaking score (positive = speaking) for one track."""
        _, audio = wavfile.read(str(wav_path))
        audio_feat = python_speech_features.mfcc(
            audio, 16000, numcep=13, winlen=0.025, winstep=0.010
        )
        video_feat = self._read_visual(avi_path)
        if len(video_feat) == 0:
            return np.zeros(0, dtype=np.float32)

        # 100 audio frames per second against 25 video frames per second.
        length = min(
            (audio_feat.shape[0] - audio_feat.shape[0] % 4) / 100,
            video_feat.shape[0] / 25,
        )
        if length <= 0:
            return np.zeros(len(video_feat), dtype=np.float32)
        audio_feat = audio_feat[: int(round(length * 100)), :]
        video_feat = video_feat[: int(round(length * 25)), :, :]

        all_scores = []
        for duration in DURATION_SET:
            batches = int(math.ceil(length / duration))
            scores = []
            for i in range(batches):
                a = torch.FloatTensor(
                    audio_feat[i * duration * 100:(i + 1) * duration * 100, :]
                ).unsqueeze(0).to(self.device)
                v = torch.FloatTensor(
                    video_feat[i * duration * 25:(i + 1) * duration * 25, :, :]
                ).unsqueeze(0).to(self.device)
                if v.shape[1] == 0 or a.shape[1] == 0:
                    continue
                embed_a = self.model.forward_audio_frontend(a)
                embed_v = self.model.forward_visual_frontend(v)
                embed_a, embed_v = self.model.forward_cross_attention(embed_a, embed_v)
                out = self.model.forward_audio_visual_backend(embed_a, embed_v)
                scores.extend(self.lossAV.forward(out, labels=None))
            all_scores.append(scores)

        n = min(len(s) for s in all_scores)
        stacked = np.array([s[:n] for s in all_scores])
        return np.round(stacked.mean(axis=0), 3).astype(float)


class SyncNetScorer(torch.nn.Module):
    """Inference-only SyncNet, reading the crop AVI directly instead of via JPEG dumps."""

    def __init__(self, weights_path, device):
        super().__init__()
        self.device = device
        self.net = SyncNetS(num_layers_in_fc_layers=1024)
        state = _load_state(weights_path)
        self.net.load_state_dict(state)
        self.net.to(device).eval()

    @torch.no_grad()
    def score(self, avi_path, wav_path) -> dict:
        """Return AV offset (frames), confidence, and median-filtered per-frame confidence."""
        cap = cv2.VideoCapture(str(avi_path))
        images = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            images.append(frame)
        cap.release()

        sample_rate, audio = wavfile.read(str(wav_path))
        mfcc = np.stack([np.array(i) for i in zip(*python_speech_features.mfcc(audio, sample_rate))])
        cct = torch.from_numpy(mfcc[np.newaxis, np.newaxis, :, :].astype(float)).float()

        min_length = min(len(images), int(math.floor(len(audio) / 640)))
        last = min_length - 5
        if last <= 0:
            return {"offset_frames": None, "confidence": None, "min_dist": None,
                    "framewise_conf": []}

        im = np.stack(images, axis=3)[np.newaxis]          # (1, H, W, 3, T)
        imtv = torch.from_numpy(np.transpose(im, (0, 3, 4, 1, 2)).astype(float)).float()

        im_feat, cc_feat = [], []
        for i in range(0, last, SYNCNET_BATCH):
            hi = min(last, i + SYNCNET_BATCH)
            im_in = torch.cat([imtv[:, :, v:v + 5, :, :] for v in range(i, hi)], 0)
            im_feat.append(self.net.forward_lip(im_in.to(self.device)).cpu())
            cc_in = torch.cat([cct[:, :, :, v * 4:v * 4 + 20] for v in range(i, hi)], 0)
            cc_feat.append(self.net.forward_aud(cc_in.to(self.device)).cpu())

        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        dists = calc_pdist(im_feat, cc_feat, vshift=SYNCNET_VSHIFT)
        mdist = torch.mean(torch.stack(dists, 1), 1)
        minval, minidx = torch.min(mdist, 0)

        fdist = np.stack([d[minidx].numpy() for d in dists])
        fconf = torch.median(mdist).numpy() - fdist
        k = min(9, len(fconf) if len(fconf) % 2 == 1 else len(fconf) - 1)
        fconfm = signal.medfilt(fconf, kernel_size=max(1, k))

        return {
            "offset_frames": int(SYNCNET_VSHIFT - minidx.item()),
            "confidence": round(float(torch.median(mdist).item() - minval.item()), 4),
            "min_dist": round(float(minval.item()), 4),
            "framewise_conf": np.round(fconfm, 3).tolist(),
        }


def run_stage(name: str, device: str = "auto", threshold: float = 0.0) -> dict:
    """Score every face track with TalkNet and SyncNet."""
    ensure_dirs()
    dev = pick_device(device)
    logger.info("device: %s", dev)

    tracks = read_json(RESULTS / f"{name}_tracks.json")["tracks"]
    talknet = TalkNetScorer(MODELS / "pretrain_TalkSet.model", dev)
    syncnet = SyncNetScorer(MODELS / "syncnet_v2.model", dev)

    results = []
    for track in tqdm(tracks, desc="asd"):
        scores = talknet.score(track["avi"], track["wav"])
        sync = syncnet.score(track["avi"], track["wav"])
        speaking = (scores > threshold)
        results.append({
            "track_id": track["track_id"],
            "talknet_scores": scores.tolist(),
            "speaking": speaking.tolist(),
            "mean_talknet_score": round(float(scores.mean()), 3) if len(scores) else None,
            "speaking_ratio": round(float(speaking.mean()), 3) if len(scores) else None,
            "syncnet": sync,
            "syncnet_corroborates": corroborates(sync),
        })

    out = {
        "video": {"name": name},
        "params": {"device": str(dev), "threshold": threshold,
                   "duration_set": DURATION_SET, "syncnet_vshift": SYNCNET_VSHIFT},
        "asd": results,
    }
    write_json(RESULTS / f"{name}_asd.json", out)

    offsets = [r["syncnet"]["offset_frames"] for r in results if r["syncnet_corroborates"]]
    if offsets:
        logger.info("SyncNet AV offset over corroborating tracks: median %d frames (%.0f ms), values %s",
                    int(np.median(offsets)), np.median(offsets) / 25 * 1000, sorted(set(offsets)))
    n_speaking = sum(1 for r in results if (r["speaking_ratio"] or 0) > 0.5)
    disagree = [r["track_id"] for r in results
                if (r["speaking_ratio"] or 0) > 0.5 and not r["syncnet_corroborates"]]
    logger.info("%d/%d tracks speaking for >50%% of their frames", n_speaking, len(results))
    logger.info("%d of those are NOT corroborated by SyncNet: %s", len(disagree), disagree)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--device", default="auto", help="auto | mps | cpu")
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()
    run_stage(args.name, device=args.device, threshold=args.threshold)


if __name__ == "__main__":
    main()
