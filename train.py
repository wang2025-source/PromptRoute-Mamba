"""Two-stage training entry point for PromptRoute-Mamba."""

import argparse
import datetime
from pathlib import Path
import time

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net import AdvancedBaseFusion, Mamba_Decoder, Mamba_Encoder, SpatialChannelDetailFusion
from utils.dataset import H5Dataset
from utils.loss import Fusionloss, cc, safe_ssim_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train PromptRoute-Mamba in two stages.")
    parser.add_argument("--data", type=Path, required=True, help="HDF5 file created by dataprocessing.py.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--stage1-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--log-dir", type=Path, default=Path("runs"))
    parser.add_argument("--name", default="PromptRoute-Mamba")
    return parser.parse_args()


def build_models(device):
    modules = (
        Mamba_Encoder(inp_channels=1, dim=64),
        Mamba_Decoder(out_channels=1, dim=64),
        AdvancedBaseFusion(dim=64),
        SpatialChannelDetailFusion(dim=64),
    )
    return tuple(nn.DataParallel(module).to(device) for module in modules)


def main():
    args = parse_args()
    if args.stage1_epochs < 1 or args.stage1_epochs >= args.epochs:
        raise ValueError("--stage1-epochs must be in [1, epochs - 1].")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    device = torch.device(args.device)
    modules = build_models(device)
    encoder, decoder, base_fusion, detail_fusion = modules
    optimizers = [torch.optim.Adam(module.parameters(), lr=args.lr) for module in modules]
    schedulers = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5) for optimizer in optimizers]
    loader = DataLoader(
        H5Dataset(args.data), batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
    )

    fusion_criterion = Fusionloss()
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    scaler = GradScaler(enabled=device.type == "cuda")
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    writer = SummaryWriter(args.log_dir / f"{args.name}_{timestamp}")
    global_step = 0

    for epoch in range(args.epochs):
        stage_one = epoch < args.stage1_epochs
        encoder.train(stage_one)
        decoder.train(stage_one)
        base_fusion.train(not stage_one)
        detail_fusion.train(not stage_one)
        epoch_start = time.time()
        for batch_index, (visible, infrared) in enumerate(loader, start=1):
            visible = visible.to(device, non_blocking=True)
            infrared = infrared.to(device, non_blocking=True)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=device.type == "cuda"):
                if stage_one:
                    visible_base, visible_detail, _ = encoder(visible)
                    infrared_base, infrared_detail, _ = encoder(infrared)
                    visible_hat, _ = decoder(visible_base, visible_detail)
                    infrared_hat, _ = decoder(infrared_base, infrared_detail)
                    visible_loss = 5 * safe_ssim_loss(visible_hat, visible, window_size=11).mean() + mse(visible_hat, visible)
                    infrared_loss = 5 * safe_ssim_loss(infrared_hat, infrared, window_size=11).mean() + mse(infrared_hat, infrared)
                    decomposition_loss = cc(visible_detail, infrared_detail).square() + 0.5 * (1 - cc(visible_base, infrared_base))
                    gradient_loss = l1(kornia.filters.SpatialGradient()(visible), kornia.filters.SpatialGradient()(visible_hat))
                    loss = visible_loss + infrared_loss + 2 * decomposition_loss + 5 * gradient_loss
                else:
                    with torch.no_grad():
                        visible_base, visible_detail, _ = encoder(visible)
                        infrared_base, infrared_detail, _ = encoder(infrared)
                    fused_base = base_fusion(visible_base, infrared_base)
                    fused_detail = detail_fusion(visible_detail, infrared_detail)
                    fused, _ = decoder(fused_base, fused_detail)
                    fusion_loss, _, _ = fusion_criterion(visible, infrared, fused)
                    prototypes = base_fusion.module.dusc.prototypes.squeeze(0)
                    normalized = F.normalize(prototypes.float(), dim=-1, eps=1e-6).to(prototypes.dtype)
                    similarity = normalized @ normalized.t()
                    identity = torch.eye(similarity.size(0), device=device, dtype=similarity.dtype)
                    orthogonal_loss = ((similarity - identity) ** 2).sum()
                    loss = fusion_loss + 5 * orthogonal_loss

            scaler.scale(loss).backward()
            active = (0, 1) if stage_one else (2, 3)
            for index in active:
                scaler.unscale_(optimizers[index])
                nn.utils.clip_grad_norm_(modules[index].parameters(), max_norm=0.01)
                scaler.step(optimizers[index])
            scaler.update()
            global_step += 1
            writer.add_scalar("loss/total", loss.item(), global_step)
            if stage_one:
                writer.add_scalar("loss/reconstruction_visible", visible_loss.item(), global_step)
                writer.add_scalar("loss/reconstruction_infrared", infrared_loss.item(), global_step)
            else:
                writer.add_scalar("loss/fusion", fusion_loss.item(), global_step)
                writer.add_scalar("loss/orthogonal", orthogonal_loss.item(), global_step)
            print(f"\rEpoch {epoch + 1:02d}/{args.epochs:02d} | batch {batch_index:04d}/{len(loader):04d} | loss {loss.item():.5f}", end="")

        for index in ((0, 1) if stage_one else (2, 3)):
            schedulers[index].step()
        print(f" | {time.time() - epoch_start:.1f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{args.name}_{timestamp}.pth"
    torch.save({
        "DIDF_Encoder": encoder.state_dict(), "DIDF_Decoder": decoder.state_dict(),
        "BaseFuseLayer": base_fusion.state_dict(), "DetailFuseLayer": detail_fusion.state_dict(),
    }, checkpoint_path)
    writer.close()
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
