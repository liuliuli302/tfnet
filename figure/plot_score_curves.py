"""
Plot frame-level importance score curves for the best-performing video
from SumMe and TVSum datasets.

Outputs:  figure/score_curve_summe.pdf  and  figure/score_curve_tvsum.pdf

Usage:
    python figure/plot_score_curves.py            # default videos
    python figure/plot_score_curves.py \
        --summe_video Excavators_river_crossing \
        --tvsum_video i3wAGJaaktw
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_DIR = os.path.join(BASE, "data", "scores")
OUT_DIR = os.path.join(BASE, "figure")
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 0.2  # best α from parameter sweep


# ── data loading ─────────────────────────────────────────────────────────
def _load_json(rel_path):
    with open(os.path.join(BASE, rel_path), encoding="utf-8") as f:
        return json.load(f)


def load_scores(dataset: str, video_name: str):
    """
    Return (picks, s_llm, s_fs, s_fv, final) as numpy arrays for one video.

    - s_llm : scene-level LLM score (per-frame, step function, normalised to [0,1])
    - s_fs  : frame-scene cosine similarity
    - s_fv  : frame-video cosine similarity
    - final : α · s_llm · s_fs + s_fv
    """
    scene_scores = _load_json(
        f"data/scores/scene_score/deepseek32/{dataset}_scene_scores.json"
    )
    frame_scene = _load_json(
        f"data/scores/frame_scene_contribution/{dataset}.json"
    )
    frame_video = _load_json(
        f"data/scores/frame_video_contribution/{dataset}.json"
    )

    # scene → frame mapping --------------------------------------------------
    vs = scene_scores.get(video_name, {})
    fs_data = frame_scene[video_name]  # list of scenes
    fv_data = frame_video[video_name]  # dict with 'frames' list

    fv_map = {
        int(f["pick"]): float(f["text_sim"])
        for f in fv_data["frames"]
    }

    picks, s_llm, s_fs, s_fv, final = [], [], [], [], []
    for si, scene in enumerate(fs_data):
        score = float(vs.get(str(si), "0")) / 100.0
        for frame in scene["frames"]:
            p = int(frame["pick"])
            f_s = float(frame["sim"])
            f_v = fv_map.get(p, 0.0)
            picks.append(p)
            s_llm.append(score)
            s_fs.append(f_s)
            s_fv.append(f_v)
            final.append(ALPHA * score * f_s + f_v)

    return (
        np.asarray(picks),
        np.asarray(s_llm),
        np.asarray(s_fs),
        np.asarray(s_fv),
        np.asarray(final),
    )


# ── plotting ─────────────────────────────────────────────────────────────
# Global style -----------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# colour palette (colour-blind friendly)
C_LLM   = "#2166AC"   # blue
C_FS    = "#4DAC26"   # green
C_FV    = "#D6604D"   # red
C_FINAL = "#1B1B1B"   # near-black


def plot_video(dataset: str, video_name: str, out_path: str):
    picks, s_llm, s_fs, s_fv, final = load_scores(dataset, video_name)

    # time axis: convert frame indices to seconds (sampled at 2 fps)
    t = picks / 15.0  # original videos at ~15 fps → pick/15 ≈ seconds

    fig, axes = plt.subplots(
        4, 1, figsize=(7.0, 5.0), sharex=True,
        gridspec_kw={"hspace": 0.12}
    )

    # helper to draw one curve
    def _draw(ax, y, colour, label, fill=True):
        ax.plot(t, y, color=colour, linewidth=0.9, label=label)
        if fill:
            ax.fill_between(t, 0, y, color=colour, alpha=0.12)
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
        ax.tick_params(direction="in", which="both")

    _draw(axes[0], s_llm,  C_LLM,  r"$S_{\mathrm{LLM}}$")
    _draw(axes[1], s_fs,   C_FS,   r"$S_{FS}$")
    _draw(axes[2], s_fv,   C_FV,   r"$S_{FV}$")
    _draw(axes[3], final,  C_FINAL, r"$F(p_j)$")

    axes[-1].set_xlabel("Time (s)")

    # dataset-specific subtitle
    ds_label = "SumMe" if dataset == "summe" else "TVSum"
    fig.suptitle(
        f"{ds_label}  —  {video_name.replace('_', ' ')}",
        fontsize=11, fontweight="bold", y=0.98
    )

    fig.align_ylabels(axes)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved  {out_path}")


# ── main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summe_video", default="Excavators_river_crossing",
                        help="SumMe video name")
    parser.add_argument("--tvsum_video", default="i3wAGJaaktw",
                        help="TVSum video name (YouTube id)")
    args = parser.parse_args()

    plot_video("summe", args.summe_video,
               os.path.join(OUT_DIR, "score_curve_summe.pdf"))
    plot_video("tvsum", args.tvsum_video,
               os.path.join(OUT_DIR, "score_curve_tvsum.pdf"))


if __name__ == "__main__":
    main()
