"""Fuse an infrared/visible test set and optionally report eight metrics."""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from net import AdvancedBaseFusion, Mamba_Decoder, Mamba_Encoder, SpatialChannelDetailFusion
from utils.Evaluator import Evaluator
from utils.img_read_save import image_read_cv2, img_save


def parse_args():
    parser = argparse.ArgumentParser(description="Test PromptRoute-Mamba on paired infrared/visible images.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing ir/ and vi/ subfolders.")
    parser.add_argument("--output-dir", type=Path, default=Path("test_result/MSRS"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-metrics", action="store_true", help="Save fused images without computing metrics.")
    return parser.parse_args()


def build_models(device):
    modules = (
        Mamba_Encoder(inp_channels=1, dim=64), Mamba_Decoder(out_channels=1, dim=64),
        AdvancedBaseFusion(dim=64), SpatialChannelDetailFusion(dim=64),
    )
    return tuple(nn.DataParallel(module).to(device) for module in modules)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    ir_dir, vi_dir = args.input_dir / "ir", args.input_dir / "vi"
    names = sorted(path.name for path in ir_dir.iterdir() if path.is_file() and (vi_dir / path.name).is_file())
    if not names:
        raise RuntimeError("No aligned pairs were found in input-dir/ir and input-dir/vi.")

    device = torch.device(args.device)
    encoder, decoder, base_fusion, detail_fusion = build_models(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    encoder.load_state_dict(checkpoint["DIDF_Encoder"])
    decoder.load_state_dict(checkpoint["DIDF_Decoder"])
    base_fusion.load_state_dict(checkpoint["BaseFuseLayer"])
    detail_fusion.load_state_dict(checkpoint["DetailFuseLayer"])
    for module in (encoder, decoder, base_fusion, detail_fusion):
        module.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for name in names:
            infrared = image_read_cv2(str(ir_dir / name), mode="GRAY")[None, None, ...] / 255.0
            visible = image_read_cv2(str(vi_dir / name), mode="GRAY")[None, None, ...] / 255.0
            infrared = torch.from_numpy(infrared).float().to(device)
            visible = torch.from_numpy(visible).float().to(device)
            visible_base, visible_detail, _ = encoder(visible)
            infrared_base, infrared_detail, _ = encoder(infrared)
            fused_base = base_fusion(visible_base, infrared_base)
            fused_detail = detail_fusion(visible_detail, infrared_detail)
            fused, _ = decoder(fused_base, fused_detail)
            fused = (fused - fused.amin()) / (fused.amax() - fused.amin()).clamp_min(1e-8)
            img_save(np.squeeze((fused * 255).cpu().numpy()), Path(name).stem, str(args.output_dir))

    print(f"Saved {len(names)} fused images to {args.output_dir}")
    if args.no_metrics:
        return
    totals = np.zeros(8)
    for name in names:
        infrared = image_read_cv2(str(ir_dir / name), "GRAY")
        visible = image_read_cv2(str(vi_dir / name), "GRAY")
        fused = image_read_cv2(str(args.output_dir / f"{Path(name).stem}.png"), "GRAY")
        totals += np.array([
            Evaluator.EN(fused), Evaluator.SD(fused), Evaluator.SF(fused), Evaluator.MI(fused, infrared, visible),
            Evaluator.SCD(fused, infrared, visible), Evaluator.VIFF(fused, infrared, visible),
            Evaluator.Qabf(fused, infrared, visible), Evaluator.SSIM(fused, infrared, visible),
        ])
    values = totals / len(names)
    print("EN\tSD\tSF\tMI\tSCD\tVIF\tQabf\tSSIM")
    print("\t".join(f"{value:.2f}" for value in values))


if __name__ == "__main__":
    main()
