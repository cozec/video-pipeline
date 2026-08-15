"""Fetch the TalkNet and SyncNet pretrained weights into data/models/.

InsightFace's buffalo_l pack downloads itself on first use, so it is not listed here.
"""

from common import MODELS, ensure_dirs, run, setup_logging

logger = setup_logging("download_models")

SYNCNET_URL = "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model"
TALKNET_REPO = "AlekseyKorshuk/talknet-asd"
TALKNET_FILE = "pretrain_TalkSet.model"
# TalkNet's own demo clip, used as a regression check on the vendored inference port.
TALKNET_DEMO = "https://raw.githubusercontent.com/TaoRuijie/TalkNet-ASD/main/demo/001.avi"


def fetch() -> dict:
    """Download every model weight not already present. Returns local paths."""
    ensure_dirs()

    syncnet = MODELS / "syncnet_v2.model"
    if not syncnet.exists():
        logger.info("downloading SyncNet weights")
        run(["curl", "-sSL", SYNCNET_URL, "-o", str(syncnet)])

    talknet = MODELS / TALKNET_FILE
    if not talknet.exists():
        logger.info("downloading TalkNet weights from HF %s", TALKNET_REPO)
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=TALKNET_REPO, filename=TALKNET_FILE)
        talknet.write_bytes(open(path, "rb").read())

    demo = MODELS / "talknet_demo_001.avi"
    if not demo.exists():
        logger.info("downloading TalkNet demo clip")
        run(["curl", "-sSL", TALKNET_DEMO, "-o", str(demo)])

    for p in (syncnet, talknet, demo):
        logger.info("%s (%.1f MB)", p.name, p.stat().st_size / 1e6)
    return {"syncnet": syncnet, "talknet": talknet, "demo": demo}


if __name__ == "__main__":
    fetch()
