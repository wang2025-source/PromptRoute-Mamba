"""Second prototype-routing example using the registered TNO pair 23.png."""

from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import simulate_prototype_routing as sim


HERE = Path(__file__).resolve().parent
VI_PATH = HERE.parent / "test_img/TNO/vi/23.png"
IR_PATH = HERE.parent / "test_img/TNO/ir/23.png"
OUT = HERE / "simulated_prototype_routing_TNO23"
CASE = OUT / "visible_low_light"
ROI_OUT = CASE / "roi_prototype_analysis"
ROI = (110, 300, 410, 490)
COLORS = ["#3B5BA7", "#4E8BC9", "#45A7A1", "#67B567",
          "#D5B143", "#E58B3F", "#D95A55", "#8A5AA9"]


def overlay(src, dst):
    im = Image.open(src).convert("RGB")
    dr = ImageDraw.Draw(im)
    x0, y0, x1, y1 = ROI
    for k in range(3):
        dr.rectangle((x0-k, y0-k, x1+k, y1+k), outline="#FFD43B", width=1)
    im.save(dst, compress_level=1)


def identity(prob, name):
    ids = np.argmax(prob, -1)
    palette = np.array([tuple(int(c[j:j+2], 16) for j in (1, 3, 5)) for c in COLORS], np.uint8)
    Image.fromarray(palette[ids], "RGB").save(ROI_OUT / name, compress_level=1)


def distribution(pv, pi):
    x0, y0, x1, y1 = ROI
    mv = pv[y0:y1, x0:x1].mean((0, 1))
    mi = pi[y0:y1, x0:x1].mean((0, 1))
    x = np.arange(8); width = .36
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.linewidth": .8, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(5.6, 3.3), constrained_layout=True)
    bv = ax.bar(x-width/2, mv, width, color="#2563A6", label="Visible")
    bi = ax.bar(x+width/2, mi, width, color="#E55C45", label="Infrared")
    ax.set_xticks(x, [f"P{k}" for k in range(1, 9)])
    ax.set(xlabel="Learned prototype", ylabel="Assignment probability",
           ylim=(0, max(.35, float(max(mv.max(), mi.max())*1.18))))
    ax.grid(axis="y", color="#D9DEE5", lw=.6, alpha=.8); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    for bars, vals in ((bv, mv), (bi, mi)):
        k = int(np.argmax(vals)); b = bars[k]
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+.008, f"{vals[k]:.2f}",
                ha="center", va="bottom", fontsize=8)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(ROI_OUT / f"prototype_assignment_selected_roi.{ext}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    np.savetxt(ROI_OUT / "prototype_assignment_selected_roi.csv",
               np.column_stack([np.arange(1, 9), mv, mi]), delimiter=",",
               header="prototype,visible_probability,infrared_probability", comments="")


def legend():
    fig, ax = plt.subplots(figsize=(6.4, .58), constrained_layout=True); ax.axis("off")
    hs = [Patch(facecolor=COLORS[k], label=f"P{k+1}") for k in range(8)]
    ax.legend(handles=hs, ncol=8, loc="center", frameon=False, fontsize=9,
              handlelength=1.2, columnspacing=1.05)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(ROI_OUT / f"prototype_identity_legend.{ext}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True); ROI_OUT.mkdir(parents=True, exist_ok=True)
    vi, ir = sim.f32(VI_PATH, "RGB"), sim.f32(IR_PATH, "L")
    vi_low = np.clip(.52 * vi ** 2.15, 0, 1)
    pv = sim.route_prob(vi_low, "vi"); pi = sim.route_prob(ir, "ir")
    pv = sim.prior_guided_uncertainty(pv, vi_low, vi)
    rv, ri, gp = sim.gate(pv, pi, "maxprob")
    _, _, gh = sim.gate(pv, pi, "entropy")
    _, _, gm = sim.gate(pv, pi, "margin")
    Image.fromarray(sim.u8(vi_low), "RGB").save(CASE / "visible.png")
    Image.fromarray(sim.u8(ir), "L").save(CASE / "infrared.png")
    for name, gate in (("gp_maxprob_paper.png", gp), ("gp_entropy.png", gh),
                       ("gp_top2_margin.png", gm)):
        smooth = np.asarray(Image.fromarray(sim.u8(gate), "L").filter(
            ImageFilter.GaussianBlur(5.5)), np.float32) / 255
        Image.fromarray(sim.color(smooth), "RGB").save(CASE / name)
    np.save(CASE / "prob_visible_8ch.npy", pv); np.save(CASE / "prob_infrared_8ch.npy", pi)
    overlay(CASE / "visible.png", ROI_OUT / "visible_with_selected_roi_768x576.png")
    overlay(CASE / "infrared.png", ROI_OUT / "infrared_with_selected_roi_768x576.png")
    identity(pv, "prototype_id_visible_color_768x576.png")
    identity(pi, "prototype_id_infrared_color_768x576.png")
    distribution(pv, pi); legend()
    (ROI_OUT / "README.txt").write_text(
        "TNO 23 second example. ROI=(110,300)-(410,490), covering the infrared target.\n"
        "The same 8 prototypes, temperature and visible low-light parameters as the MSRS example are used.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
