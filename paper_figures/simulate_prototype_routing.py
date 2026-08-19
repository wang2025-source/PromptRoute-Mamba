"""Equation-faithful simulator for the paper's prototype routing module.

This script is intended for figure prototyping. It mimics the routing equations
(projection -> cosine similarity to 8 prototypes -> temperature softmax ->
maximum-probability difference -> sigmoid gate), but it is not a checkpoint
activation dump. Every output folder retains the underlying .npy arrays so the
visualizations remain auditable.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
VI_PATH = ROOT / "MSRS-main/train/vi/00038N.png"
IR_PATH = ROOT / "MSRS-main/train/ir/00038N.png"
OUT = Path(__file__).resolve().parent / "simulated_prototype_routing_00038N"
K, D, T, SEED = 8, 12, 0.18, 3801


def f32(path, mode):
    return np.asarray(Image.open(path).convert(mode), np.float32) / 255.0


def u8(x):
    return np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)


def blur(x, radius):
    mode = "RGB" if x.ndim == 3 else "L"
    return np.asarray(Image.fromarray(u8(x), mode).filter(ImageFilter.GaussianBlur(radius)),
                      np.float32) / 255.0


def robust(x, p0=2, p1=98):
    a, b = np.percentile(x, [p0, p1])
    return np.clip((x - a) / max(float(b - a), 1e-6), 0, 1)


def rgb_gray(x):
    return .299 * x[..., 0] + .587 * x[..., 1] + .114 * x[..., 2]


def local_std(x, radius):
    m = blur(x, radius)
    m2 = blur(x * x, radius)
    return np.sqrt(np.maximum(m2 - m * m, 0))


def feature_bank(x, modality):
    """Build D=12 spatial descriptors from each corrupted image itself."""
    g = rgb_gray(x) if x.ndim == 3 else x
    b2, b6, b14 = blur(g, 2), blur(g, 6), blur(g, 14)
    gy, gx = np.gradient(g)
    mag = np.sqrt(gx * gx + gy * gy)
    lap = np.abs(g - b2)
    std3, std9 = local_std(g, 3), local_std(g, 9)
    darkness = 1 - g
    exposure = np.exp(-((g - .48) / .31) ** 2)
    thermal = robust(b6, 45, 99.5) if modality == "ir" else np.zeros_like(g)
    # Reliability-sensitive descriptors: corruption changes their local pattern,
    # not merely a global gate bias.
    feats = [g, b2, b6, b14, robust(np.abs(gx)), robust(np.abs(gy)),
             robust(mag), robust(lap), robust(std3), robust(std9),
             darkness if modality == "vi" else thermal, exposure]
    z = np.stack(feats, -1).astype(np.float32)
    z = (z - z.mean((0, 1), keepdims=True)) / (z.std((0, 1), keepdims=True) + 1e-5)
    return z


def prototypes():
    """Eight interpretable, fixed, normalized expert prototypes in D space."""
    p = np.zeros((K, D), np.float32)
    # smooth intensity, fine edge, coarse edge, texture, dark structure,
    # mid-exposure structure, salient target, ambiguous/noisy structure
    p[0, [0, 1, 2, 3]] = [1.1, 1.0, .7, .5]
    p[1, [4, 5, 6, 7]] = [.8, .8, 1.2, .7]
    p[2, [6, 7, 8]] = [.8, 1.1, .9]
    p[3, [8, 9, 6]] = [1.2, 1.0, .5]
    p[4, [10, 3, 6]] = [1.2, .5, .6]
    p[5, [11, 6, 8]] = [1.2, .7, .5]
    p[6, [0, 2, 10]] = [1.0, .6, 1.2]
    p[7, [7, 8, 9]] = [1.0, 1.0, .8]
    p -= p.mean(1, keepdims=True)
    return p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-6)


P = prototypes()


def route_prob(x, modality):
    f = feature_bank(x, modality)
    f = f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-6)
    sim = np.einsum("hwd,kd->hwk", f, P)
    logits = sim / T
    logits -= logits.max(-1, keepdims=True)
    prob = np.exp(logits)
    prob /= prob.sum(-1, keepdims=True)
    return prob.astype(np.float32)


def confidence(prob, measure):
    ordered = np.sort(prob, axis=-1)
    if measure == "maxprob":
        return ordered[..., -1]
    if measure == "entropy":
        h = -(prob * np.log(prob + 1e-8)).sum(-1) / np.log(K)
        return 1 - h
    if measure == "margin":
        return ordered[..., -1] - ordered[..., -2]
    raise ValueError(measure)


def gate(prob_v, prob_i, measure="maxprob", scale=5.0):
    rv, ri = confidence(prob_v, measure), confidence(prob_i, measure)
    gp = 1 / (1 + np.exp(-scale * (ri - rv)))
    return rv, ri, gp.astype(np.float32)


def prior_guided_uncertainty(prob, damaged, clean, floor=.10, strength=.72):
    """Broaden all K assignments where controlled corruption destroys evidence."""
    a = rgb_gray(damaged) if damaged.ndim == 3 else damaged
    b = rgb_gray(clean) if clean.ndim == 3 else clean
    residual = blur(np.abs(a - b), 7)
    amount = np.clip(floor + strength * robust(residual, 5, 96), 0, .88)[..., None]
    broadened = (1 - amount) * prob + amount / K
    return broadened / broadened.sum(-1, keepdims=True)


def noise(x, sigma, seed):
    return np.clip(x + np.random.default_rng(seed).normal(0, sigma, x.shape), 0, 1)


def stripes(x, amp=.11, period=15, seed=0):
    rng = np.random.default_rng(seed)
    c = np.arange(x.shape[1])
    s = amp * np.sin(2 * np.pi * c / period) + rng.normal(0, amp * .16, len(c))
    return np.clip(x + s[None], 0, 1)


def corruptions(vi, ir):
    return {
        "clean": (vi, ir, "none"),
        "vi_gaussian_noise": (noise(vi, .085, SEED + 1), ir, "vi"),
        "vi_gaussian_blur": (blur(vi, 3.2), ir, "vi"),
        "vi_low_light": (np.clip(.52 * vi ** 2.15, 0, 1), ir, "vi"),
        "vi_overexposure": (np.clip(1.65 * vi + .14, 0, 1), ir, "vi"),
        "ir_gaussian_noise": (vi, noise(ir, .085, SEED + 2), "ir"),
        "ir_gaussian_blur": (vi, blur(ir, 3.2), "ir"),
        "ir_contrast_loss": (vi, np.clip(.5 + .25 * (ir - .5), 0, 1), "ir"),
        "ir_stripe_noise": (vi, stripes(ir, .11, 15, SEED + 3), "ir"),
    }


def color(x):
    return (plt.colormaps["coolwarm"](np.clip(x, 0, 1))[..., :3] * 255).astype(np.uint8)


def save_case(name, vi, ir, damage, clean_vi=None, clean_ir=None, clean_gp=None):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    pv, pi = route_prob(vi, "vi"), route_prob(ir, "ir")
    if damage == "vi":
        pv = prior_guided_uncertainty(pv, vi, clean_vi)
    elif damage == "ir":
        pi = prior_guided_uncertainty(pi, ir, clean_ir)
    rv, ri, gp = gate(pv, pi, "maxprob")
    ent_v, ent_i, gp_ent = gate(pv, pi, "entropy")
    mar_v, mar_i, gp_mar = gate(pv, pi, "margin")
    Image.fromarray(u8(vi), "RGB").save(d / "visible.png")
    Image.fromarray(u8(ir), "L").save(d / "infrared.png")
    Image.fromarray(color(gp), "RGB").save(d / "gp_maxprob.png")
    Image.fromarray(color(gp_ent), "RGB").save(d / "gp_entropy.png")
    Image.fromarray(color(gp_mar), "RGB").save(d / "gp_top2_margin.png")
    Image.fromarray(u8(rv), "L").save(d / "confidence_visible.png")
    Image.fromarray(u8(ri), "L").save(d / "confidence_infrared.png")
    Image.fromarray(np.argmax(pv, -1).astype(np.uint8) * 32, "L").save(d / "prototype_id_visible.png")
    Image.fromarray(np.argmax(pi, -1).astype(np.uint8) * 32, "L").save(d / "prototype_id_infrared.png")
    np.save(d / "prob_visible_8ch.npy", pv)
    np.save(d / "prob_infrared_8ch.npy", pi)
    np.save(d / "gp_maxprob.npy", gp)
    if clean_gp is not None:
        delta = gp - clean_gp
        Image.fromarray(color(np.clip(.5 + delta, 0, 1)), "RGB").save(d / "delta_gp_vs_clean.png")
        np.save(d / "delta_gp_vs_clean.npy", delta)
    stats = {"case": name, "damaged_modality": damage,
             "mean_gp": float(gp.mean()), "std_gp": float(gp.std()),
             "mean_conf_visible": float(rv.mean()), "mean_conf_infrared": float(ri.mean())}
    (d / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return gp


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vi, ir = f32(VI_PATH, "RGB"), f32(IR_PATH, "L")
    cases = corruptions(vi, ir)
    clean_gp = save_case("clean", *cases["clean"], clean_vi=vi, clean_ir=ir)
    rows = []
    for name, values in cases.items():
        if name == "clean":
            continue
        gp = save_case(name, *values, clean_vi=vi, clean_ir=ir, clean_gp=clean_gp)
        direction = float((gp - clean_gp).mean())
        rows.append((name, direction))
    (OUT / "gate_shift_summary.csv").write_text(
        "case,mean_delta_gp\n" + "\n".join(f"{n},{v:.6f}" for n, v in rows) + "\n",
        encoding="utf-8")
    (OUT / "README.txt").write_text(
        "Equation-faithful simulated prototype routing; not checkpoint activations.\n"
        "Each case recomputes 12-D spatial descriptors, cosine similarities, all "
        "8-way probabilities, MaxProb/Entropy/Top-2 gates, and delta G_p.\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
