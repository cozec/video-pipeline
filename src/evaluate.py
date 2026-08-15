"""Evaluation against the hand-labelled ground truth in labels/.

Three independent checks:
  shots     boundary precision/recall/F1 for TransNetV2 and the PySceneDetect baseline
  asd       per-frame speaking precision/recall/F1, plus TalkNet vs SyncNet agreement
  identity  pairwise precision/recall/F1 and cluster purity, swept over the cosine threshold
"""

import argparse
from collections import defaultdict

import numpy as np

from common import EVAL, LABELS, RESULTS, read_json, setup_logging, write_json

logger = setup_logging("evaluate")

BOUNDARY_TOL = 2   # frames


def prf(tp: int, fp: int, fn: int) -> dict:
    """Precision, recall and F1 from raw counts."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}


def eval_shots(name: str) -> dict:
    """Boundary detection accuracy for both detectors."""
    labels = read_json(LABELS / "shot_labels.json")
    truth = labels["boundaries"]
    data = read_json(RESULTS / f"{name}_shots.json")

    def score(pred: list[int]) -> dict:
        matched, used = 0, set()
        for p in pred:
            hit = next((t for t in truth
                        if abs(p - t) <= BOUNDARY_TOL and t not in used), None)
            if hit is not None:
                used.add(hit)
                matched += 1
        return prf(matched, len(pred) - matched, len(truth) - matched)

    transnet = sorted(s["start_frame"] for s in data["shots"])[1:]
    baseline = sorted(s["start_frame"] for s in data["baseline_pyscenedetect"])[1:]
    return {
        "n_true_boundaries": len(truth),
        "transnetv2": score(transnet),
        "pyscenedetect_t27": score(baseline),
    }


def eval_asd(name: str) -> dict:
    """Per-frame speaking accuracy over the hand-labelled window."""
    labels = read_json(LABELS / "asd_labels.json")
    asd = {a["track_id"]: a for a in read_json(RESULTS / f"{name}_asd.json")["asd"]}
    tracks = {t["track_id"]: t for t in read_json(RESULTS / f"{name}_tracks.json")["tracks"]}

    tp = fp = fn = tn = 0
    per_track = []
    for entry in labels["tracks"]:
        tid = entry["track_id"]
        track, scores = tracks[tid], asd[tid]["talknet_scores"]
        t_tp = t_fp = t_fn = t_tn = 0
        for span in entry["spans"]:
            for f in range(span["start_frame"], span["end_frame"] + 1):
                i = f - track["start_frame"]
                if not (0 <= i < len(scores)):
                    continue
                pred = scores[i] > 0
                true = bool(span["speaking"])
                if pred and true:
                    t_tp += 1
                elif pred and not true:
                    t_fp += 1
                elif not pred and true:
                    t_fn += 1
                else:
                    t_tn += 1
        tp, fp, fn, tn = tp + t_tp, fp + t_fp, fn + t_fn, tn + t_tn
        per_track.append({"track_id": tid, **prf(t_tp, t_fp, t_fn), "tn": t_tn})

    total = tp + fp + fn + tn
    overall = prf(tp, fp, fn)
    overall["tn"] = tn
    overall["accuracy"] = round((tp + tn) / total, 4) if total else 0.0

    # Agreement between the two models on "is this track speaking at all".
    agree = disagree = 0
    for entry in labels["tracks"]:
        tid = entry["track_id"]
        talknet_says = (asd[tid]["speaking_ratio"] or 0) > 0.5
        if talknet_says == asd[tid]["syncnet_corroborates"]:
            agree += 1
        else:
            disagree += 1

    offsets = [a["syncnet"]["offset_frames"] for a in asd.values() if a["syncnet_corroborates"]]
    return {
        "window": labels.get("window"),
        "n_frames_labelled": total,
        "overall": overall,
        "per_track": per_track,
        "talknet_syncnet_agreement": {
            "agree": agree, "disagree": disagree,
            "rate": round(agree / (agree + disagree), 4) if agree + disagree else None,
        },
        "syncnet_av_offset_frames": {
            "median": int(np.median(offsets)) if offsets else None,
            "median_ms": round(float(np.median(offsets)) / 25 * 1000, 1) if offsets else None,
            "n_corroborating_tracks": len(offsets),
        },
    }


def pairwise_scores(truth: dict[int, str], pred: dict[int, int]) -> dict:
    """Pairwise precision/recall/F1 over all track pairs sharing an identity."""
    ids = sorted(truth)
    tp = fp = fn = 0
    for a_i in range(len(ids)):
        for b_i in range(a_i + 1, len(ids)):
            a, b = ids[a_i], ids[b_i]
            same_true = truth[a] == truth[b]
            same_pred = pred[a] == pred[b]
            if same_true and same_pred:
                tp += 1
            elif same_pred and not same_true:
                fp += 1
            elif same_true and not same_pred:
                fn += 1
    return prf(tp, fp, fn)


def cluster_quality(truth: dict[int, str], pred: dict[int, int]) -> dict:
    """Purity, plus how badly identities are split across clusters."""
    by_cluster = defaultdict(list)
    for tid, c in pred.items():
        if tid in truth:
            by_cluster[c].append(truth[tid])
    correct = sum(max(np.unique(v, return_counts=True)[1]) for v in by_cluster.values())
    total = sum(len(v) for v in by_cluster.values())

    by_person = defaultdict(set)
    for tid, person in truth.items():
        by_person[person].add(pred[tid])
    fragmented = {p: len(cs) for p, cs in by_person.items() if len(cs) > 1}
    impure = {c: sorted(set(v)) for c, v in by_cluster.items() if len(set(v)) > 1}
    return {
        "purity": round(correct / total, 4) if total else 0.0,
        "n_clusters": len(by_cluster),
        "n_true_identities": len(by_person),
        "fragmented_identities": fragmented,
        "impure_clusters": impure,
    }


def eval_identity(name: str, sweep: bool = True) -> dict:
    """Identity clustering accuracy, and how it varies with the cosine threshold."""
    import identity as identity_mod

    labels = read_json(LABELS / "identity_labels.json")
    truth = {int(k): v for k, v in labels["track_person"].items()}
    ident = read_json(RESULTS / f"{name}_identity.json")
    pred = {a["track_id"]: a["person_id"] for a in ident["assignments"]}

    result = {
        "threshold": ident["params"]["cosine_threshold"],
        "n_labelled_tracks": len(truth),
        "pairwise": pairwise_scores(truth, pred),
        "clusters": cluster_quality(truth, pred),
    }

    if sweep:
        tracks = read_json(RESULTS / f"{name}_tracks.json")["tracks"]
        from common import WORK
        embeddings = np.load(WORK / f"{name}_embeddings.npy")
        curve = []
        for th in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00]:
            lab = identity_mod.cluster(tracks, embeddings, threshold=th)
            p = {t["track_id"]: int(l) for t, l in zip(tracks, lab)}
            curve.append({"threshold": th, "n_clusters": int(len(set(lab))),
                          **pairwise_scores(truth, p)})
        result["threshold_sweep"] = curve
        best = max(curve, key=lambda c: c["f1"])
        result["best_threshold"] = best
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--only", choices=["shots", "asd", "identity"])
    args = parser.parse_args()

    out = {}
    for kind, fn in (("shots", eval_shots), ("asd", eval_asd), ("identity", eval_identity)):
        if args.only and args.only != kind:
            continue
        try:
            out[kind] = fn(args.name)
        except FileNotFoundError as exc:
            logger.warning("skipping %s: %s", kind, exc)

    write_json(EVAL / f"{args.name}_metrics.json", out)
    for kind, res in out.items():
        logger.info("%s: %s", kind, res.get("overall") or res.get("pairwise") or res)


if __name__ == "__main__":
    main()
