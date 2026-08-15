# Summary — findings, measurements and failure modes

Test material: **SFMTA "Fare Day Out"**, NBC Bay Area, 167 s, 1920×1080, downloaded with
`yt-dlp` and normalised to 25 fps / 16 kHz (4,172 frames).

Everything below is measured on this machine (M5, 10 cores, 17 GB, macOS 26.5.2), against
hand-labelled ground truth committed in `labels/`.

---

## 1. Headline results

| Check | Metric | Result |
|---|---|---|
| Shot boundaries — TransNetV2 | P / R / F1 | **1.000 / 1.000 / 1.000** (18 of 18, zero false positives) |
| Shot boundaries — PySceneDetect baseline | P / R / F1 | 0.850 / 0.944 / 0.895 (3 FP, 1 FN) |
| Active speaker — per frame | P / R / F1 | **0.977 / 0.992 / 0.984**, accuracy 0.984 (1,215 frames) |
| TalkNet vs SyncNet (labelled tracks) | agreement | **17 / 17** (3 speaking, all corroborated; 14 silent, none corroborated) |
| Cross-shot identity — pairwise | P / R / F1 | **1.000** / 0.667 / 0.800 |
| Cross-shot identity — clusters | purity | **1.000**, 0 impure clusters |

Pipeline found 19 shots, 85 face tracks and 45 distinct people; 6 people appear in more
than one shot.

---

## 2. Shot boundary detection

TransNetV2 is perfect on this video and the PySceneDetect baseline is not, in both directions:

- **f3282** (exterior bus → interior bus) is a real cut that only TransNetV2 found.
- **f3045, f3126, f3151** are PySceneDetect false positives, all caused by fast bus/pan motion
  inside a continuous take.

Building the ground truth honestly mattered here. Rather than trust either detector, I pooled
candidates from TransNetV2 (t=0.5) and PySceneDetect at thresholds 27, 15 and 10 — 42 distinct
candidates — and inspected every candidate the two detectors disagreed on as a
`(b-2, b+2)` frame pair (`results/eval/disputed.jpg`, `recall_a.jpg`, `recall_b.jpg`).

All 23 extra candidates at threshold 15 were rejected on inspection: they are dense-crowd
motion inside the long event b-roll, which is exactly the case TransNetV2 is trained to
ignore. **No cut was missed by both detectors**, so the recall figure is real rather than
circular.

Two long takes (shots 8 and 9, 23.6 s and 30.4 s) initially looked suspicious for missed cuts.
Both detectors agree on them and every interior candidate was rejected — they are genuinely
single takes.

---

## 3. Active speaker detection

**The port is verified, not assumed.** The vendored TalkNet was run on the original repo's own
`demo/001.avi` and compared against the published `demo/001_result.avi`. Speaking/not decisions
match frame for frame at every sampled frame, including a 4-face frame where one person speaks
and three do not (`results/eval/talknet_port_check.jpg`). All 483 model tensors load; only the
4 training-only `lossA`/`lossV` head tensors are skipped.

**Accuracy on the news video:** P 0.977 / R 0.992 / F1 0.984 over 1,215 hand-labelled frames
across 17 tracks in the first 30 s.

The labelled window is deliberately adversarial. It contains:

- a studio anchor speaking continuously (track 0),
- a **split screen** where the anchor talks and the reporter listens (tracks 2 and 1) — the
  case that punishes an audio-only approach, since one voice and two faces are present,
- a reporter piece-to-camera that starts with ~1.7 s of closed mouth before speech (track 3),
- 13 event b-roll faces under the reporter's voice-over, **several of which visibly move their
  mouths** (smiling, talking to each other). These are hard negatives: mouth motion with no
  corresponding audio.

TalkNet gets all of these right. The 15 false positives and 5 false negatives are all at
speech onset/offset boundaries, not whole-track errors.

An independent confirmation of the labelling: TalkNet places track 3's silence run at frames
417–459, and reading the mouth filmstrip by eye put the onset at ~f455–460 — two independent
methods agreeing to within a few frames.

### What SyncNet added

Running SyncNet as a second signal was worth it for two findings that TalkNet alone cannot give:

**1. A calibration constant.** Over the 6 tracks where SyncNet finds genuine audio-visual
alignment, the AV offset is a consistent **+3 frames (120 ms)** — normal broadcast
audio delay, and confirmation that the 25 fps / 16 kHz normalisation is not introducing drift.

**2. A confidence rail that separates real detections from weak ones.** TalkNet's strong
detections (score +1.7 to +2.8) carry SyncNet confidence 5.8–10.6 at offsets 2–4. TalkNet's
*weak* positives on crowd faces (score +0.19 to +1.24) come back with SyncNet offsets pinned at
exactly ±10 — the edge of the search window, meaning no alignment was found at all. A high
confidence at a railed offset is not evidence of speech.

`asd.corroborates()` encodes this: corroboration requires confidence ≥ 3.0 **and**
`|offset| < vshift`. Without the second condition, railed tracks would have been counted as
corroboration and the signal would have been worthless. Uncorroborated positives are drawn
**amber** in the annotated video rather than being silently dropped — the disagreement is
surfaced for a human, not resolved by the pipeline.

**How much it disagrees, over the whole video:** TalkNet calls 17 of 85 tracks speaking;
SyncNet corroborates only 5 of those. The other 12 (tracks 21, 25, 28, 29, 36, 38, 39, 44,
48, 50, 75, 79) are all small crowd faces in the shot 4–8 event b-roll with weak TalkNet
scores. They fall outside the hand-labelled window, so **this is a flag for review, not a
proven error count** — but the same voice-over logic that made the labelled crowd faces
non-speakers applies to them, so they are most likely TalkNet false positives that SyncNet
caught. That is the concrete value of running the second model.

Within the labelled window the two models agree on all 17 tracks, though only 3 of those are
non-trivial: TalkNet calls tracks 0, 2 and 3 speaking and SyncNet corroborates all three; the
remaining 14 are silent tracks that both models agree on.

---

## 4. Cross-shot face matching

This is the part of the brief that says *"if one person shows in different places of the video,
you should mark them as the same person."*

**It works, decisively, for the people the story is about.** Cluster purity is 1.000 with zero
impure clusters — nothing was ever wrongly merged.

Verified independently by the broadcast's own chyrons:

| person_id | who (from the on-screen caption) | shots | screen time | speaking |
|---|---|---|---|---|
| `PERSON_00` | Ginger Conejero Saab (field reporter) | 1, 2, **18** | 42.9 s | 33.2 s |
| `PERSON_02` | Laura Garcia (studio anchor) | 0, 1 | 16.7 s | 16.0 s |
| `PERSON_01` | Julie Kirschbaum (interviewee) | 9 | 30.4 s | 29.9 s |

`PERSON_00` is re-identified in shot 18, **144 seconds and 16 shots after** her first
appearance, and `PERSON_02` is matched across a studio shot and a split-screen shot.

### Why the recall is 0.667, and why the threshold is not the fix

Three same-person pairs were missed. The cosine distances explain it exactly:

| pair | distance | result |
|---|---|---|
| ANCHOR t0–t2 | 0.032 | merged |
| REPORTER t3–t84 | 0.048 | merged |
| REPORTER t1–t3 | 0.067 | merged |
| REPORTER t1–t84 | 0.118 | merged |
| CROWD_HOOP t64–t66 | 0.165 | merged |
| CROWD_GLASSES_REDCAP t61–t69 | 0.210 | merged |
| **CROWD_HOOP t64–t80** | **0.623** | split |
| **CROWD_HOOP t66–t80** | **0.654** | split |
| **CROWD_ORANGECAP t10–t24** | **0.735** | split |
| *closest different-person pair* | *0.669* | — |

The four principal cross-shot pairs sit at 0.03–0.12 against an impostor floor of **0.669** —
5× to 20× of margin, so those matches are not close calls. Even the frontal crowd pairs
(0.165, 0.210) clear the floor by more than 3×.

The three failures are profile-versus-frontal views of small crowd faces, and they land *at or
beyond* the impostor floor: `t10–t24` at 0.735 is literally farther apart than the closest pair
of two *different* people. No decision rule based on a single global distance can separate
those without also merging strangers.

The threshold sweep confirms this is not a tuning problem:

| threshold | 0.35 | 0.45 | 0.55 | 0.65 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|
| precision | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 | 0.600 | 0.375 |
| recall | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 | 0.667 |

**Recall is completely flat** from 0.35 to 1.00 while precision collapses past 0.65. No global
threshold recovers those pairs; loosening only merges the wrong people. The chosen **0.55** sits
in the middle of the safe plateau (0.35–0.65), so it is a robust choice rather than a fitted one.

Fixing these would need a different mechanism, not a different number — pose-aware or
quality-weighted embedding aggregation, or a track-level re-ranking that uses body/clothing
appearance where the face is in profile.

### The overlap constraint earns its place

279 track pairs were blocked from merging because they are visible in the same frame. This is
absent from both reference repos and is what keeps precision at 1.000 in the crowded event
b-roll, where up to 25 tracked faces share a shot.

---

## 5. Ground truth: what was labelled and what was not

Honesty about coverage matters more than a bigger number.

- **Shots** — fully labelled. All 42 pooled candidates adjudicated by eye.
- **Active speaker** — 1,215 frames across 17 tracks in a 30 s window. Genuinely ambiguous
  frames (track 1 f311–331, a brief mouth-opening that could be a smile or an interjection;
  track 3 f456–465, the exact onset of speech) are **excluded rather than guessed**.
- **Identity** — 20 of 85 tracks. Most shot 3–8 b-roll faces are 40–100 px profiles that cannot
  be matched reliably by eye; labelling them by guess would have fabricated the ground truth.
  Because pairwise metrics only consider pairs where *both* tracks are labelled, a singleton in
  the label set asserts only that it differs from the other labelled tracks — a claim that can
  actually be verified — not that it appears nowhere else in the video.

Candidate repeats were re-checked at higher resolution before being committed
(`results/eval/grp_*.jpg`). Two candidate groups (tracks 70/73/75 and track 76) were left
unlabelled because the faces are too occluded to call either way.

---

## 6. Performance

Measured end-to-end on the 169 s backup broadcast, M5, under `caffeinate` (one clean
uninterrupted run, `results/covid_timings.json`):

| Stage | Time | Device |
|---|---|---|
| Download | 3.0 s | — |
| Normalise to 25 fps + 16 kHz | 28.3 s | ffmpeg |
| Shot boundaries (TransNetV2 + baseline) | 28.9 s | CPU |
| Face detection + tracking + crops | 227.6 s | CoreML/CPU |
| TalkNet + SyncNet | 54.7 s | MPS |
| ArcFace embeddings + clustering | 9.9 s | CPU |
| Annotated MP4 + contact sheets | 40.4 s | CPU |
| Manifest | 0.0 s | — |
| **Total** | **392.8 s** | |

**6.5 minutes for a 2.8-minute video** — about 2.3× realtime, dominated by per-frame face
detection at ~26 fps. Detection results are cached to `data/work/<name>_dets.json`, so
retuning tracking parameters does not re-run the detector.

ASD cost scales with the number of face tracks, not video length: 54.7 s for the backup
video's 12 tracks against 172.0 s for the news video's 85 tracks (both 169 s / 167 s long).

**Reproducibility.** The news video was re-run end to end from a cleared detection cache with
the final code; it reproduced 19 shots, 85 tracks and 45 persons, and all three metric sets in
`results/eval/sfmta_metrics.json` came back byte-identical. The numbers in this document are
from that run, not from an accumulation of intermediate ones.

---

## 7. Generalisation check

The pipeline was re-run **completely unchanged** — same thresholds, same tracking parameters,
no re-tuning — on the backup broadcast (*"Wastewater data shows COVID-19 surge"*, 169 s):

```
32 shots · 12 face tracks · 8 people · 3 of them appearing in more than one shot
```

All three multi-shot identities are correct on inspection (`results/eval/covid_identities.jpg`):

| person_id | shots | screen time | note |
|---|---|---|---|
| `PERSON_00` | 3, 31 | 40.0 s | studio anchor — **matched across a heavy red colour grade** in shot 31 |
| `PERSON_01` | 10, 27 | 25.4 s | remote interviewee |
| `PERSON_02` | 4, 15, 21 | 25.2 s | remote interviewee, three separate appearances |

No wrong merges anywhere in the 12 tracks: the second studio anchor (`PERSON_03`) stays
separate, and the three masked lab-worker b-roll faces stay separate from each other and are
all correctly scored as non-speaking.

This is the result that matters most — the thresholds chosen on the first video were not fitted
to it.

---

## 8. Known limitations

1. **Profile faces break re-identification.** Documented in section 4 with distances. Affects
   background/crowd faces, not principals.
2. **Identity ground truth covers 20 of 85 tracks.** The uncovered tracks are exactly the hard
   ones, so the true recall over *all* tracks is likely lower than 0.667.
3. **Tracks cannot span a shot boundary** by construction. A person continuously visible across
   a cut becomes two tracks, re-joined only by face matching — which is the intended design, but
   it means a re-identification failure looks like a new person rather than a broken track.
4. **`min_track = 12` frames trades precision for recall.** Short tracks give TalkNet under half
   a second of context, which is where most of its boundary errors are.
5. **No speaker diarisation.** "Speaking" means *this visible face is the source of the current
   audio*. Off-screen narration is correctly attributed to nobody, but the pipeline does not
   identify who the off-screen voice is.
6. **PySceneDetect is used at default threshold 27** for the baseline. A tuned threshold would
   score better; the comparison shows the out-of-the-box difference.
7. **Phantom faces in news graphics.** On the backup video, SCRFD detected a "face" in the
   rocks and foliage of a `SANTA CRUZ FIRE` title card — det score 0.61, held for 14 frames, so
   it survived both `min_face_size` and `min_track` and became `PERSON_07`
   (`results/eval/covid_phantom.jpg`). It is harmless downstream (TalkNet scores it
   non-speaking, and it clusters alone) but it does inflate the person count. Raising
   `DET_THRESH` from 0.5 to ~0.7 would remove it at some cost to genuine small-face recall;
   a graphics/lower-third mask would be the better fix.
