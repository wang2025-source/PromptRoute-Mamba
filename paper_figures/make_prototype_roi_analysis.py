"""Create publication assets for prototype identity and ROI distributions."""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent / "simulated_prototype_routing_00038N"
CASE = ROOT / "vi_low_light"
OUT = CASE / "roi_prototype_analysis"
ROI = (168, 112, 382, 368)  # x0, y0, x1, y1; pedestrians and vehicle region

COLORS = [
    "#3B5BA7", "#4E8BC9", "#45A7A1", "#67B567",
    "#D5B143", "#E58B3F", "#D95A55", "#8A5AA9",
]


def roi_overlay(src, dst):
    im = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(im)
    x0, y0, x1, y1 = ROI
    for k in range(3):
        draw.rectangle((x0-k, y0-k, x1+k, y1+k), outline="#FFD43B", width=1)
    im.save(dst, compress_level=1)


def save_identity(prob_path, stem):
    prob = np.load(prob_path)
    ids = np.argmax(prob, axis=-1)
    rgb = (np.asarray([tuple(int(COLORS[k][j:j+2], 16) for j in (1, 3, 5))
                       for k in range(8)], np.uint8)[ids])
    Image.fromarray(rgb, "RGB").save(OUT / f"{stem}_color_640x480.png", compress_level=1)
    np.save(OUT / f"{stem}_integer.npy", ids.astype(np.uint8))


def save_legend():
    fig, ax = plt.subplots(figsize=(6.4, .58), constrained_layout=True)
    ax.axis("off")
    handles = [Patch(facecolor=COLORS[k], edgecolor="none", label=f"P{k+1}") for k in range(8)]
    ax.legend(handles=handles, ncol=8, loc="center", frameon=False,
              handlelength=1.25, handleheight=.9, columnspacing=1.05, fontsize=9)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"prototype_identity_legend.{ext}", dpi=600,
                    bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)


def save_distribution():
    pv = np.load(CASE / "prob_visible_8ch.npy")
    pi = np.load(CASE / "prob_infrared_8ch.npy")
    x0, y0, x1, y1 = ROI
    mv = pv[y0:y1, x0:x1].mean(axis=(0, 1))
    mi = pi[y0:y1, x0:x1].mean(axis=(0, 1))
    x = np.arange(8)
    width = .36

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.linewidth": .8, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(5.6, 3.3), constrained_layout=True)
    bv = ax.bar(x - width/2, mv, width, label="Visible", color="#2563A6")
    bi = ax.bar(x + width/2, mi, width, label="Infrared", color="#E55C45")
    ax.set_xticks(x, [f"P{k}" for k in range(1, 9)])
    ax.set_ylabel("Assignment probability")
    ax.set_xlabel("Learned prototype")
    ax.set_ylim(0, max(.35, float(max(mv.max(), mi.max()) * 1.18)))
    ax.grid(axis="y", color="#D9DEE5", lw=.6, alpha=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    # Keep labels compact; only annotate each modality's dominant prototype.
    for bars, vals in ((bv, mv), (bi, mi)):
        k = int(np.argmax(vals))
        b = bars[k]
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+.008,
                f"{vals[k]:.2f}", ha="center", va="bottom", fontsize=8)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"prototype_assignment_selected_roi.{ext}", dpi=600,
                    bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig)
    np.savetxt(OUT / "prototype_assignment_selected_roi.csv",
               np.column_stack([np.arange(1, 9), mv, mi]), delimiter=",",
               header="prototype,visible_probability,infrared_probability", comments="")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    roi_overlay(CASE / "visible.png", OUT / "visible_with_selected_roi_640x480.png")
    roi_overlay(CASE / "infrared.png", OUT / "infrared_with_selected_roi_640x480.png")
    save_identity(CASE / "prob_visible_8ch.npy", "prototype_id_visible")
    save_identity(CASE / "prob_infrared_8ch.npy", "prototype_id_infrared")
    save_legend()
    save_distribution()
    (OUT / "README.txt").write_text(
        "Representative case: vi_low_light. Yellow ROI=(168,112)-(382,368).\n"
        "Identity maps use the same discrete P1-P8 colors as the supplied legend.\n"
        "The grouped bar chart averages the full 8-channel assignments inside the same ROI.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
