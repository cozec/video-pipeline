# Video Preprocessing Pipeline

Cuts a long-form video (news broadcast) into shots, tracks every face inside each shot,
decides who is actively speaking, and assigns a **stable person ID across the whole video**
so the same person recognised in different shots gets the same label.

```
Extract frames → Shot boundaries (TransNetV2) ─┬→ Track faces → Landmarks → Face matching → person_id
                                               └→ TalkNet (+ SyncNet) → active speaker
```

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_models.py                       # TalkNet + SyncNet weights

caffeinate -dimsu python src/pipeline.py \
    --url https://youtu.be/Oq544HEfkLE --name sfmta
```

Stages cache to disk, so you can restart partway through:

```bash
python src/pipeline.py --name sfmta --from-stage asd     # skip download/prepare/shots/faces
python src/pipeline.py --name sfmta --export-shots       # also write one MP4 per shot
```

## Outputs

| Path | What it is |
|---|---|
| `results/<name>_manifest.json` | everything joined: shots, tracks (per-frame bbox, 5 landmarks, TalkNet score, speaking flag), person ids, per-person screen/speaking time |
| `results/<name>_annotated.mp4` | boxes burned in, labelled `PERSON_02 +3.0`, with shot number and running time |
| `results/identities/<name>_person_NN.jpg` | contact sheet per person, sampled from **every shot** they appear in — the visual proof that cross-shot matching worked |
| `results/eval/<name>_metrics.json` | accuracy against the hand-labelled ground truth |

Box colours in the annotated video:

- **green** — TalkNet says speaking *and* SyncNet independently corroborates
- **amber** — TalkNet says speaking, SyncNet does not corroborate (treat with suspicion)
- **red** — not speaking

## Stages

| Stage | File | Model |
|---|---|---|
| Normalise to 25 fps / 16 kHz | `src/prepare.py` | ffmpeg |
| Shot boundaries | `src/shots.py` | TransNetV2 (+ PySceneDetect as a baseline) |
| Face detection, tracking, cropping | `src/faces.py` | InsightFace SCRFD-10G |
| Active speaker | `src/asd.py` | TalkNet, cross-checked by SyncNet |
| Cross-shot face matching | `src/identity.py` | ArcFace (`w600k_r50`) + constrained agglomerative clustering |
| Annotated video, contact sheets | `src/visualize.py` | — |
| Accuracy vs ground truth | `src/evaluate.py` | — |

## Design notes

**Everything is normalised to 25 fps and 16 kHz once, up front.** TalkNet and SyncNet were
both trained at exactly those rates, and the audio/video index ratio is a fixed 4:1 (100 MFCC
frames per second against 25 video frames). Normalising once removes a whole class of
off-by-N bugs from every later stage.

**SCRFD instead of the reference repos' S3FD.** The reference diagram names TransNetV2 and
TalkNet but leaves the detector open. InsightFace's `buffalo_l` pack gives detection,
5-point landmarks *and* the ArcFace embedding needed for face matching from one dependency,
and is faster than S3FD. The *crop* convention still follows `demoTalkNet.crop_video`
exactly — `crop_scale=0.40`, 224×224, centre 112×112 at inference — because that is what
TalkNet is actually sensitive to.

**Tracking parameters are retuned for broadcast news.** syncnet's defaults target film:

| param | syncnet | here | why |
|---|---|---|---|
| `min_track` | 100 frames (4 s) | 12 | news cuts fast; 4 s would discard most tracks |
| `num_failed_det` | 25 | 10 | shots are short, don't coast across half of one |
| `min_face_size` | 100 px | 60 px | keeps two-shots and background faces |

**Short tracks break `medfilt`.** `demoTalkNet` median-filters the crop centre with
kernel 13. `scipy.signal.medfilt` zero-pads, so on a track shorter than the kernel the
smoothed centre gets dragged toward zero and the crop lands off the face. With
`min_track=12` this is reachable, so the kernel shrinks to the track length
(`faces._medfilt`).

**Temporally overlapping tracks cannot merge.** Two faces visible in the same frame are two
different people, so their pairwise distance is forced to infinity before clustering
(279 such pairs on the test video). Neither reference repo does this, and without it a
two-shot of similar-looking people collapses into one identity.

**SyncNet as a second opinion, not a tiebreak.** It is scored independently and its verdict
is recorded per track as `syncnet_corroborates`; disagreements are surfaced (amber boxes),
never silently resolved. An AV offset pinned at ±`vshift` means the search railed at the
edge of its window and found no real alignment, so high confidence there is *not* evidence
of speech — `asd.corroborates` checks for this explicitly.

## Verified accuracy

Measured on the 167 s test broadcast against hand-labelled ground truth in `labels/`.
See `summary.md` for the full picture, including where it fails.

| Check | Result |
|---|---|
| Shot boundaries (TransNetV2) | P 1.000 / R 1.000 / **F1 1.000** (18/18, 0 false positives) |
| Shot boundaries (PySceneDetect baseline) | P 0.850 / R 0.944 / F1 0.895 |
| Active speaker, per frame | P 0.977 / R 0.992 / **F1 0.984** over 1,215 labelled frames |
| TalkNet vs SyncNet agreement | 17/17 labelled tracks (3 speaking, all corroborated) |
| Cross-shot identity, pairwise | P **1.000** / R 0.667 / F1 0.800, cluster purity 1.000 |

The TalkNet port is verified against the original repo's own demo clip: our speaking/not
decisions match `demo/001_result.avi` frame for frame, including a 4-face frame
(`results/eval/talknet_port_check.jpg`).

## Requirements

Python **3.12** — `insightface`, `onnxruntime`, `scipy` and `scikit-learn` do not have a
complete wheel set on 3.13/3.14. macOS arm64 with MPS is used automatically where it helps;
pass `--device cpu` to force CPU.

## Licensing

InsightFace's pretrained models are released for **non-commercial research use only**.
TalkNet and SyncNet weights come from their authors' original releases.

## Attribution

Vendored inference code under `src/vendor/` is from
[TaoRuijie/TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) and
[joonson/syncnet_python](https://github.com/joonson/syncnet_python), unmodified apart from
import paths. Their CUDA-hardcoded training wrappers are replaced by the device-parameterised
inference classes in `src/asd.py`.
