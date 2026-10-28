# -*- coding: utf-8 -*-
from net import Mamba_Encoder, Mamba_Decoder, AdvancedBaseFusion, SpatialChannelDetailFusion
from utils.dataset import H5Dataset
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.loss import Fusionloss, cc, safe_ssim_loss 
import kornia
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler 
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()
model_str = 'PromptRoute-Mamba'
architecture_version = 'paper-equations-v1'

num_epochs = 100
epoch_gap = 30

# Section 4.1.2: run independently with seeds 0, 1, 2, 3, and 4.
seed = int(os.environ.get('PROMPTROUTE_SEED', '0'))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

lr = 1e-4 
weight_decay = 0
batch_size = 2 

coeff_mse_loss_VF = 1. 
coeff_mse_loss_IF = 1.
coeff_decomp = 2.      
coeff_tv = 5.
coeff_ssim = 5.0
coeff_ortho = 5.0

clip_grad_norm_value = 0.01
optim_step = 4
optim_gamma = 0.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DIDF_Encoder = nn.DataParallel(Mamba_Encoder(inp_channels=1, dim=64)).to(device)
DIDF_Decoder = nn.DataParallel(Mamba_Decoder(out_channels=1, dim=64)).to(device)
BaseFuseLayer = nn.DataParallel(AdvancedBaseFusion(dim=64)).to(device)
DetailFuseLayer = nn.DataParallel(SpatialChannelDetailFusion(dim=64)).to(device)

trainable_parameters = [
    tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    for module in (DIDF_Encoder, DIDF_Decoder, BaseFuseLayer, DetailFuseLayer)
]

optimizer1 = torch.optim.Adam(DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()
scaler = GradScaler() 

train_data = os.environ.get("PROMPTROUTE_TRAIN_H5", os.path.join("data", "MSRS_train_imgsize_256_stride_100.h5"))
trainloader = DataLoader(H5Dataset(train_data),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=8)

loader = {'train': trainloader }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
log_dir = f"runs/{model_str}_seed{seed}_{timestamp}"
writer = SummaryWriter(log_dir)

global_step = 0
prev_time = time.time()

for epoch in range(num_epochs):
    # Section 4.1.2: one epoch-based schedule across both stages.
    epoch_lr = lr * optim_gamma ** (epoch // optim_step)
    for optimizer in (optimizer1, optimizer2, optimizer3, optimizer4):
        for param_group in optimizer.param_groups:
            param_group['lr'] = epoch_lr
    # Section 3.6: only the modules assigned to the current stage are trainable.
    stage_one = epoch < epoch_gap
    # Keep the fixed Laplacian kernels non-trainable in both stages.
    for parameters, enabled in zip(
        trainable_parameters, (stage_one, stage_one, not stage_one, not stage_one)
    ):
        for parameter in parameters:
            parameter.requires_grad_(enabled)
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()
        
        
        if epoch < epoch_gap:
            DIDF_Encoder.train()
            DIDF_Decoder.train()
            BaseFuseLayer.eval()   
            DetailFuseLayer.eval()
        else:
            DIDF_Encoder.eval()    
            DIDF_Decoder.eval()
            BaseFuseLayer.train()
            DetailFuseLayer.train()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        with autocast():
            if epoch < epoch_gap: 
                
                feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
                feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
                data_VIS_hat, _ = DIDF_Decoder(feature_V_B, feature_V_D)
                data_IR_hat, _ = DIDF_Decoder(feature_I_B, feature_I_D)

                cc_loss_B = cc(feature_V_B, feature_I_B)
                cc_loss_D = cc(feature_V_D, feature_I_D)
                
                
                mse_loss_V = coeff_ssim * torch.mean(safe_ssim_loss(data_VIS_hat, data_VIS, window_size=11)) + MSELoss(data_VIS, data_VIS_hat)
                mse_loss_I = coeff_ssim * torch.mean(safe_ssim_loss(data_IR_hat, data_IR, window_size=11)) + MSELoss(data_IR, data_IR_hat)
                
                Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS), kornia.filters.SpatialGradient()(data_VIS_hat))

                # Equation (43): cross-modal and intra-modal correlation penalties.
                cc_loss_V_BD = cc(feature_V_B, feature_V_D)
                cc_loss_I_BD = cc(feature_I_B, feature_I_D)
                loss_decomp = (
                    cc_loss_D ** 2
                    + 0.5 * (1.0 - cc_loss_B)
                    + 0.5 * (cc_loss_V_BD ** 2 + cc_loss_I_BD ** 2)
                )
                loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * mse_loss_I + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss
                
            else:  
                
                with torch.no_grad():
                    feature_V_B, feature_V_D, _ = DIDF_Encoder(data_VIS)
                    feature_I_B, feature_I_D, _ = DIDF_Encoder(data_IR)
                
                feature_F_B = BaseFuseLayer(feature_V_B, feature_I_B)
                feature_F_D = DetailFuseLayer(feature_V_D, feature_I_D)
                
                data_Fuse, feature_F = DIDF_Decoder(feature_F_B, feature_F_D)  

                
                fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)
                
                # ==========================================================
                
                # ==========================================================
                prototypes = BaseFuseLayer.module.dusc.prototypes.squeeze(0) 
                
                p_norm = F.normalize(prototypes.float(), dim=-1, eps=1e-6).to(prototypes.dtype)
                
                sim_matrix = torch.matmul(p_norm, p_norm.t())
                identity = torch.eye(sim_matrix.size(0)).to(sim_matrix.device)
                
                
                loss_ortho = torch.sum((sim_matrix - identity) ** 2)

                
                loss = fusionloss + coeff_ortho * loss_ortho
                # ==========================================================

        scaler.scale(loss).backward()
        
        
        if epoch < epoch_gap:
            scaler.unscale_(optimizer1)
            scaler.unscale_(optimizer2)
            nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value)
            nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value)
            scaler.step(optimizer1)
            scaler.step(optimizer2)
        else:
            scaler.unscale_(optimizer3)
            scaler.unscale_(optimizer4)
            nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value)
            nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value)
            scaler.step(optimizer3)
            scaler.step(optimizer4)
            
        scaler.update()

        global_step += 1
        writer.add_scalar('Loss/total', loss.item(), global_step)
        
        if epoch < epoch_gap:
            writer.add_scalar('Loss/mse_visible', mse_loss_V.item(), global_step)
            writer.add_scalar('Loss/mse_infrared', mse_loss_I.item(), global_step)
        else:
            writer.add_scalar('Loss/fusion', fusionloss.item(), global_step)
            writer.add_scalar('Loss/ortho_penalty', loss_ortho.item(), global_step)

        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (epoch, num_epochs, i, len(loader['train']), loss.item(), time_left,)
        )

    if epoch >= epoch_gap:
        torch.cuda.empty_cache()
    
os.makedirs("models", exist_ok=True)
checkpoint = {
    'architecture_version': architecture_version,
    'seed': seed,
    'epochs': num_epochs,
    'stage1_epochs': epoch_gap,
    'DIDF_Encoder': DIDF_Encoder.state_dict(),
    'DIDF_Decoder': DIDF_Decoder.state_dict(),
    'BaseFuseLayer': BaseFuseLayer.state_dict(),
    'DetailFuseLayer': DetailFuseLayer.state_dict(),
}
torch.save(checkpoint, os.path.join("models", f"{model_str}_seed{seed}_{timestamp}.pth"))
writer.close()
