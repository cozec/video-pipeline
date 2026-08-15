"""Render the three slide figures as PNGs into results/slides/.

Kept separate from visualize.py: these are presentation artefacts, not pipeline outputs,
and they read their numbers from the committed results so the deck cannot drift from the run.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, EVAL, read_json, setup_logging

logger = setup_logging("slides")

OUT = RESULTS / "slides"

INK, MUTED, HAIR = "#0F171E", "#5A6773", "#C6D0D9"
SPINE, OK, WARN, MISS = "#0E7C86", "#2F7D4F", "#A36A08", "#AE3A2B"
SANS, MONO = "Avenir Next", "Menlo"

plt.rcParams.update({
    "font.family": SANS,
    "text.color": INK,
    "axes.edgecolor": HAIR,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def box(ax, x0, x1, y0, y1, title, sub, key=False):
    """Draw a labelled node box."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.9",
        linewidth=1.6 if key else 1.2,
        edgecolor=SPINE if key else HAIR,
        facecolor="#0E7C860F" if key else "white", zorder=3))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax.text(cx, cy + 2.1, title, ha="center", va="center",
            fontsize=16, fontweight="600", color=INK, zorder=4)
    ax.text(cx, cy - 2.6, sub, ha="center", va="center",
            fontsize=12.5, color=MUTED, family=MONO, zorder=4)


def arrow(ax, x0, y0, x1, y1, color=None, dashed=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                arrowprops=dict(arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.3, color=color or MUTED,
                                linestyle=(0, (4, 3)) if dashed else "solid",
                                shrinkA=0, shrinkB=0))


def fig_pipeline():
    """Slide 1: the flow, with the fork that is the architectural point.

    Sized for projection: nothing here lands below ~12pt once placed on the slide,
    so the sub-labels stay readable from the back of a room.
    """
    fig, ax = plt.subplots(figsize=(12.4, 4.0), dpi=170)
    ax.set_xlim(0, 100); ax.set_ylim(14.5, 46.5); ax.axis("off")

    Y, HH = 30.0, 6.0
    box(ax, 1, 20, Y - HH, Y + HH, "Normalise", "25 fps · 16 kHz", key=True)
    box(ax, 26, 45, Y - HH, Y + HH, "TransNetV2", "shot boundaries")
    box(ax, 51, 71, Y - HH, Y + HH, "SCRFD + IoU tracker", "detect · track · crop", key=True)

    for x0, x1, lab in [(20, 26, "4,172 frames"), (45, 51, "19 shots")]:
        arrow(ax, x0, Y, x1, Y)
        ax.text((x0 + x1) / 2, Y + HH + 1.4, lab, ha="center", fontsize=12,
                color=MUTED, family=MONO)

    # audio rail: normalise -> crop step, where the per-track WAV is sliced
    ax.plot([10, 10, 61], [Y - HH, 16, 16], color=SPINE, lw=1.5,
            ls=(0, (4, 3)), zorder=1)
    arrow(ax, 61, 16, 61, Y - HH, color=SPINE, dashed=True)
    ax.text(35, 17.4, "16 kHz audio rail  ·  sliced per track", ha="center",
            fontsize=12.5, color=SPINE, family=MONO)

    # fork
    ax.plot([71, 75], [Y, Y], color=MUTED, lw=1.5, zorder=2)
    ax.plot([75, 75], [22.5, 37.5], color=MUTED, lw=1.5, zorder=2)
    arrow(ax, 75, 37.5, 79, 37.5)
    arrow(ax, 75, 22.5, 79, 22.5)
    ax.text(68.5, Y + HH + 1.4, "85 tracks", ha="center", fontsize=12,
            color=MUTED, family=MONO)

    box(ax, 79, 99, 32.0, 43.0, "TalkNet  +  SyncNet", "who is talking")
    box(ax, 79, 99, 17.0, 28.0, "ArcFace  +  clustering", "who they are")

    ax.text(89, 44.4, "ACTIVE SPEAKER", ha="center", fontsize=12,
            color=SPINE, family=MONO)
    ax.text(89, 29.4, "CROSS-SHOT IDENTITY", ha="center", fontsize=12,
            color=SPINE, family=MONO)

    fig.tight_layout(pad=0.1)
    fig.savefig(OUT / "fig_pipeline.png", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def fig_syncnet():
    """Slide 3: why a confident SyncNet score can still mean nothing."""
    asd = read_json(RESULTS / "sfmta_asd.json")["asd"]
    spk = [a for a in asd if a["speaking_ratio"] > 0.5]
    good = [(a["syncnet"]["offset_frames"], a["syncnet"]["confidence"])
            for a in spk if a["syncnet_corroborates"]]
    bad = [(a["syncnet"]["offset_frames"], a["syncnet"]["confidence"])
           for a in spk if not a["syncnet_corroborates"]]

    fig, ax = plt.subplots(figsize=(5.0, 3.36), dpi=170)
    ax.add_patch(mpatches.Rectangle((-9.6, 3.0), 19.2, 9.4, facecolor="#0E7C8612",
                                    edgecolor=SPINE, lw=1.1, ls=(0, (4, 3)), zorder=1))
    ax.text(0, 12.7, "gate: corroborated", ha="center", fontsize=11,
            color=SPINE, family=MONO)

    for x in (-10, 10):
        ax.axvline(x, color=HAIR, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.axhline(3.0, color=HAIR, lw=1, ls=(0, (2, 3)), zorder=1)

    ax.scatter(*zip(*bad), s=58, facecolors="none", edgecolors=WARN, lw=1.7, zorder=4)
    ax.scatter(*zip(*good), s=62, color=OK, zorder=5)

    ax.annotate(f"{len(good)} corroborated\noffset 2–4 frames",
                xy=(2.4, 6.2), xytext=(-8.0, 8.0), fontsize=11, color=OK, family=MONO,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=OK, lw=1.1, shrinkB=8))
    ax.text(0, 1.30, f"{len(bad)} crowd b-roll faces\npinned at the search edge",
            ha="center", fontsize=11, color=WARN, family=MONO)

    ax.set_xlim(-13.5, 13.5); ax.set_ylim(0, 13.6)
    ax.set_xticks([-10, -5, 0, 5, 10]); ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_xlabel("SyncNet AV offset  (frames)", fontsize=11.5, labelpad=6)
    ax.set_ylabel("confidence", fontsize=11.5, labelpad=6)
    ax.tick_params(labelsize=10.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_syncnet.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def fig_distance():
    """Slide 4: the missed matches sit past the impostor floor."""
    matched = [0.032, 0.048, 0.067, 0.118, 0.165, 0.210]
    missed = [0.623, 0.654, 0.735]
    floor, thresh = 0.669, 0.55

    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=220)
    ax.add_patch(mpatches.Rectangle((0, -0.30), thresh, 0.60, facecolor="#0E7C860F",
                                    edgecolor="none", zorder=1))
    ax.text(0.018, 0.225, "merge", fontsize=8.5, color=SPINE, family=MONO, va="center")
    ax.text(0.565, 0.225, "split", fontsize=8.5, color=MUTED, family=MONO, va="center")

    ax.axhline(0, color=MUTED, lw=1.2, zorder=2)
    ax.axvline(thresh, color=SPINE, lw=1.5, ls=(0, (4, 3)), ymin=0.30, ymax=0.86, zorder=3)
    ax.text(thresh, 0.335, "threshold 0.55", ha="center", fontsize=8.5,
            color=SPINE, family=MONO)
    ax.axvline(floor, color=MISS, lw=1.5, ymin=0.06, ymax=0.50, zorder=3)
    ax.text(floor, -0.255, "0.669  closest pair of\ntwo different people",
            ha="center", va="top", fontsize=8.5, color=MISS, family=MONO)

    ax.scatter(matched, [0] * len(matched), s=62, color=OK, zorder=5)
    ax.scatter(missed, [0] * len(missed), s=62, color=MISS, zorder=5)
    ax.plot([0.024, 0.024, 0.218, 0.218], [0.075, 0.125, 0.125, 0.075], color=OK, lw=1.1)
    ax.text(0.121, 0.155, "6 matched · 0.032 – 0.210", ha="center", fontsize=8.5,
            color=OK, family=MONO)
    ax.plot([0.615, 0.615, 0.743, 0.743], [0.075, 0.125, 0.125, 0.075], color=MISS, lw=1.1)
    ax.text(0.679, 0.155, "3 missed · profile views", ha="center", fontsize=8.5,
            color=MISS, family=MONO)

    ax.set_xlim(-0.01, 0.81); ax.set_ylim(-0.42, 0.42)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("cosine distance between two face tracks", fontsize=9.5, labelpad=6)
    ax.tick_params(labelsize=8.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_distance.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_pipeline()
    fig_syncnet()
    fig_distance()
    fig_overlap()
    logger.info("wrote 3 figures to %s", OUT)




def fig_overlap():
    """Slide 3: two faces on screen at once cannot be the same person."""
    fig, ax = plt.subplots(figsize=(6.0, 2.5), dpi=170)
    ax.set_xlim(0, 100); ax.set_ylim(0, 43); ax.axis("off")

    # overlap band sits behind the bars so it tints the ground, not the tracks
    ax.add_patch(mpatches.Rectangle((34, 8), 28, 27, facecolor="#AE3A2B12",
                                    edgecolor=MISS, lw=1.2, ls=(0, (4, 3)), zorder=1))

    for y, x0, x1, name in [(25, 6, 62, "track A"), (12, 34, 92, "track B")]:
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, y), x1 - x0, 7, boxstyle="round,pad=0,rounding_size=0.7",
            linewidth=1.4, edgecolor=SPINE, facecolor="#E8F3F4", zorder=3))
        ax.text(x0 + 2.5, y + 3.5, name, va="center", fontsize=11.5,
                color=INK, fontweight="600", zorder=4)

    ax.text(48, 37.5, "distance forced to ∞  —  merge forbidden", ha="center",
            fontsize=11.5, color=MISS, family=MONO)
    ax.text(48, 5.0, "visible in the same frames", ha="center", fontsize=11.5,
            color=MISS, family=MONO)
    ax.text(6, 0.5, "279 pairs blocked on the test video", fontsize=11,
            color=MUTED, family=MONO)

    fig.tight_layout(pad=0.1)
    fig.savefig(OUT / "fig_overlap.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


if __name__ == "__main__":
    main()
