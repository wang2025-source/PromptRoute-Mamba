from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "reliability_components_00038N"
VI_PATH = ROOT / "MSRS-main/train/vi/00038N.png"
IR_PATH = ROOT / "MSRS-main/train/ir/00038N.png"
SEED = 3801


def to_float(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def to_u8(x):
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def gray(rgb):
    if rgb.ndim == 2:
        return rgb
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def smooth(x, radius=9):
    z = Image.fromarray(to_u8(np.clip(x, 0, 1)), mode="L")
    return np.asarray(z.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0


def robust01(x, lo=2, hi=98):
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / max(b - a, 1e-6), 0, 1)


def structure(x):
    gy, gx = np.gradient(x)
    return robust01(smooth(np.sqrt(gx * gx + gy * gy), 5), 3, 97)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def corrupt(arr, sigma=0.075, blur=1.65, seed=SEED):
    rng = np.random.default_rng(seed)
    noisy = np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 1)
    mode = "RGB" if noisy.ndim == 3 else "L"
    return to_float(Image.fromarray(to_u8(noisy), mode=mode).filter(ImageFilter.GaussianBlur(blur)))


def colorize(x, vmin=0.0, vmax=1.0, cmap="coolwarm"):
    norm = np.clip((x - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    return (plt.colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)


def save_rgb(name, arr):
    Image.fromarray(arr if arr.dtype == np.uint8 else to_u8(arr), mode="RGB").save(
        OUT / name, compress_level=1
    )


def save_gray(name, arr):
    Image.fromarray(arr if arr.dtype == np.uint8 else to_u8(arr), mode="L").save(
        OUT / name, compress_level=1
    )


def make_plots():
    # Smooth curves pass through the intended endpoints while markers retain the
    # five discrete corruption levels used in the planned experiment.
    x = np.array([0, .02, .04, .06, .08])
    yv = np.array([0, .045, .092, .143, .196])
    yi = np.array([0, -.041, -.087, -.132, -.181])
    xd = np.linspace(0, .08, 400)
    cv = np.polyfit(x, yv, 3)
    ci = np.polyfit(x, yi, 3)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.linewidth": .8, "xtick.major.width": .8, "ytick.major.width": .8,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(4.5, 3.35), constrained_layout=True)
    ax.axhline(0, color="#555555", lw=.8, ls="--", zorder=0)
    ax.plot(xd, np.polyval(cv, xd), color="#E55C45", lw=2.2, label="Visible corrupted")
    ax.plot(xd, np.polyval(ci, xd), color="#2563A6", lw=2.2, label="Infrared corrupted")
    ax.scatter(x, yv, s=31, facecolor="white", edgecolor="#E55C45", lw=1.6, zorder=3)
    ax.scatter(x, yi, s=31, facecolor="white", edgecolor="#2563A6", lw=1.6, zorder=3)
    ax.set(xlim=(-.002, .082), ylim=(-.22, .23), xlabel=r"Corruption severity $\sigma$",
           ylabel=r"Mean gate shift $\Delta\bar{G}_p$")
    ax.grid(True, color="#D9DEE5", lw=.6, alpha=.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"e_reliability_calibration_smooth.{ext}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    labels = ["MaxProb", "Entropy", "Top-2 Margin"]
    vals = [94.8, 88.9, 91.6]
    fig, ax = plt.subplots(figsize=(4.5, 3.35), constrained_layout=True)
    bars = ax.bar(labels, vals, color=["#2563A6", "#8B9097", "#7656B5"], width=.58)
    ax.set_ylim(80, 100)
    ax.set_ylabel("Routing agreement (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9DEE5", lw=.6, alpha=.75)
    ax.set_axisbelow(True)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"f_measure_comparison_predicted.{ext}", dpi=600,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vi_img = Image.open(VI_PATH).convert("RGB")
    ir_img = Image.open(IR_PATH).convert("L")
    vi = to_float(vi_img)
    ir = to_float(ir_img)
    h, w = ir.shape

    # Pixel-derived modality evidence. Visible evidence emphasizes recoverable
    # texture and exposure; infrared evidence emphasizes thermal saliency and
    # stable structure. The final gate follows sigmoid(r_i-r_v), matching Eq. 48.
    vg = gray(vi)
    v_texture = structure(vg)
    v_exposure = np.exp(-((vg - .48) / .34) ** 2)
    r_v = smooth(.72 * v_texture + .28 * v_exposure, 8)

    i_hot = robust01(smooth(ir, 4), 55, 99.5)
    i_structure = structure(ir)
    r_i = smooth(.68 * i_hot + .32 * i_structure, 8)
    logit_clean = 3.0 * (r_i - r_v)
    gp_clean = sigmoid(logit_clean)

    vi_corr = corrupt(vi, sigma=.075, blur=1.65, seed=SEED)
    ir_corr = corrupt(ir, sigma=.075, blur=1.65, seed=SEED + 1)
    gp_vcorr = sigmoid(logit_clean + 1.05)
    gp_icorr = sigmoid(logit_clean - 1.05)

    mask = np.zeros((h, w), dtype=np.float32)
    x0, y0, x1, y1 = 168, 112, 382, 368
    mask[y0:y1, x0:x1] = 1.0
    local_vi = vi.copy()
    local_vi[y0:y1, x0:x1] = corrupt(
        vi[y0:y1, x0:x1], sigma=.10, blur=2.1, seed=SEED + 2
    )
    soft_mask = smooth(mask, 12)
    delta_r = 1.18 * soft_mask
    gp_local = sigmoid(logit_clean + delta_r)
    delta_gp = gp_local - gp_clean

    # A-C components, all exactly 640×480.
    save_rgb("a_visible_clean_640x480.png", to_u8(vi))
    save_gray("a_infrared_clean_640x480.png", to_u8(ir))
    save_rgb("a_gp_clean_640x480.png", colorize(gp_clean))
    save_rgb("b_visible_corrupted_640x480.png", to_u8(vi_corr))
    save_gray("b_infrared_unchanged_640x480.png", to_u8(ir))
    save_rgb("b_gp_visible_corrupted_640x480.png", colorize(gp_vcorr))
    save_rgb("c_visible_unchanged_640x480.png", to_u8(vi))
    save_gray("c_infrared_corrupted_640x480.png", to_u8(ir_corr))
    save_rgb("c_gp_infrared_corrupted_640x480.png", colorize(gp_icorr))

    # D components, again exact native resolution.
    save_rgb("d_local_visible_corruption_640x480.png", to_u8(local_vi))
    save_gray("d_known_corruption_mask_640x480.png", to_u8(mask))
    save_rgb("d_delta_r_map_640x480.png", colorize(delta_r, 0, 1.2))
    save_rgb("d_gp_local_corruption_640x480.png", colorize(gp_local))
    save_rgb("d_delta_gp_map_640x480.png", colorize(delta_gp, -.35, .35))

    # Extra evidence maps useful during final typesetting.
    save_gray("aux_visible_reliability_640x480.png", to_u8(robust01(r_v)))
    save_gray("aux_infrared_reliability_640x480.png", to_u8(robust01(r_i)))
    make_plots()

    readme = f"""Reliability validation components for MSRS 00038N

All A-D raster components are derived from the original registered pair and are
exactly {w}x{h} pixels. Corruption is deterministic (seed={SEED}). The predicted
gate maps use G_p = sigmoid(3*(r_i-r_v)+corruption_shift), consistent with the
relative-gating principle in Eq. 48. They are visualization placeholders, not
measured network activations.

E and F are supplied as 600-dpi PNG plus editable SVG/PDF. Values in E/F are
predicted placeholders and should be replaced by measured results before the
figure is described as an experiment.
"""
    (OUT / "README.txt").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
