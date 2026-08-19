import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm import Mamba
except ImportError:
    print("请先安装 mamba_ssm: pip install mamba-ssm causal-conv1d")

# =============================================================================

# =============================================================================
class LocalSpatialVariance(nn.Module):
    def __init__(self, window_size=5):
        super().__init__()
        self.pad = window_size // 2
        
        self.pool = nn.AvgPool2d(window_size, stride=1, padding=0)
        
    def forward(self, x):
        x_f = x.float()
        
        x_pad = F.pad(x_f, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        var = self.pool(x_pad**2) - self.pool(x_pad)**2
        return torch.clamp(var, min=1e-6).to(x.dtype)

class SpatialBoundaryDetector(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
        self.weight = nn.Parameter(torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0), requires_grad=False)
    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.weight.repeat(c, 1, 1, 1).to(x.device)
        edge = torch.abs(F.conv2d(x.float(), weight.float(), padding=1, groups=c))
        return torch.clamp(edge / 0.2, 0, 1).to(x.dtype)

class SpatialAdaptiveDynamicFilter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight_gen = nn.Sequential(nn.Conv2d(dim, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 9, 1))
    def forward(self, x):
        b, c, h, w = x.shape
        unfolded = F.unfold(x, kernel_size=3, padding=1).view(b, c, 9, h, w)
        w_logits = torch.clamp(self.weight_gen(x).unsqueeze(1), -15.0, 15.0)
        dynamic_weights = F.softmax(w_logits, dim=2) 
        return torch.sum(unfolded * dynamic_weights, dim=2)

# =============================================================================

# =============================================================================
class StarGatedRefinement(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, dim * 4, 1)
        self.dw_conv = nn.Conv2d(dim * 2, dim * 2, 3, padding=1, groups=dim * 2)
        self.proj_out = nn.Conv2d(dim * 2, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x_high = self.proj_in(x)
        x1, x2 = x_high.chunk(2, dim=1)
        x_star = self.act(self.dw_conv(x1)) * torch.clamp(x2, -20.0, 20.0)
        return x + self.proj_out(x_star)

# =============================================================================

# =============================================================================
class LaplacianPyramidEdgeBooster(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        kernel = [[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]
        self.laplacian = nn.Parameter(torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.edge_gate = nn.Sequential(nn.Conv2d(dim, dim // 2, 1), nn.GELU(), nn.Conv2d(dim // 2, dim, 1), nn.Sigmoid())

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.laplacian.repeat(c, 1, 1, 1).to(x.device)
        edge_response = torch.abs(F.conv2d(x.float(), weight.float(), padding=1, groups=c)).to(x.dtype)
        enhancement = self.edge_gate(edge_response)
        return x + x * enhancement

# =============================================================================

# =============================================================================
class ContrastiveComplementaryAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.diff_net = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 1),
            nn.Sigmoid()
        )
    def forward(self, v, i, fused):
        diff = torch.abs(v - i)
        attn = self.diff_net(diff)
        return fused * (1.0 + attn)

# =============================================================================

# =============================================================================
class StarGatedMultiScaleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        sub_dim = dim // 4 
        self.conv1 = nn.Sequential(nn.Conv2d(dim, sub_dim, 1), nn.LeakyReLU(0.1))
        self.conv3 = nn.Sequential(nn.Conv2d(dim, sub_dim, 3, padding=1), nn.LeakyReLU(0.1))
        self.conv5 = nn.Sequential(nn.Conv2d(dim, sub_dim, 5, padding=2), nn.LeakyReLU(0.1))
        self.conv7 = nn.Sequential(nn.Conv2d(dim, sub_dim, 7, padding=3), nn.LeakyReLU(0.1))
        self.fuse = nn.Conv2d(dim, dim, 1)
        self.star_refiner = StarGatedRefinement(dim)

    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        out = self.fuse(out) + x
        return self.star_refiner(out)  

# =============================================================================

# =============================================================================
class MaxInformationChannelAmplifier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.amp_network = nn.Sequential(nn.Linear(dim * 2, dim // 2), nn.ReLU(), nn.Linear(dim // 2, dim), nn.Sigmoid())
    def forward(self, fused, v, i):
        
        v_f, i_f = v.float(), i.float()
        var_v = torch.clamp(torch.mean(v_f**2, dim=[2, 3]) - torch.mean(v_f, dim=[2, 3])**2, min=1e-6)
        var_i = torch.clamp(torch.mean(i_f**2, dim=[2, 3]) - torch.mean(i_f, dim=[2, 3])**2, min=1e-6)
        
        info_cat = torch.cat([var_v, var_i], dim=1).to(v.dtype)
        amp_factor = self.amp_network(info_cat).unsqueeze(-1).unsqueeze(-1)
        return fused * (1.0 + amp_factor)

class FeatureDistributionStretcher(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local_contrast = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim), 
            nn.InstanceNorm2d(dim, affine=True, eps=1e-5), 
            nn.GELU(), 
            nn.Conv2d(dim, dim, 1), 
            nn.Sigmoid()
        )
        self.global_stretch = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, dim, 1), nn.Sigmoid())
    def forward(self, x):
        local_detail = self.local_contrast(x)
        x_enhanced = x + x * local_detail
        mean = torch.mean(x_enhanced, dim=[2, 3], keepdim=True)
        stretch = self.global_stretch(x_enhanced) * 1.5 + 1.0 
        return mean + (x_enhanced - mean) * stretch

class BoundaryAwareRoutingBase(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local_var = LocalSpatialVariance(window_size=5)
        self.boundary_extractor = SpatialBoundaryDetector()
        self.fuse_conv = nn.Conv2d(dim*2, dim, 1)

    def forward(self, v, i):
        e_v, e_i = self.local_var(v), self.local_var(i)
        e_v_safe, e_i_safe = e_v + 1e-2, e_i + 1e-2
        m = torch.max(e_v_safe, e_i_safe).detach()
        b_map = torch.clamp(self.boundary_extractor(v) + self.boundary_extractor(i), 0, 1)
        w_v_hard, w_i_hard = (e_v_safe / m) ** 8, (e_i_safe / m) ** 8
        gate_hard = w_v_hard / (w_v_hard + w_i_hard + 1e-6)
        w_v_soft, w_i_soft = (e_v_safe / m) ** 2, (e_i_safe / m) ** 2
        gate_soft = w_v_soft / (w_v_soft + w_i_soft + 1e-6)
        final_gate = gate_hard * (1.0 - b_map) + gate_soft * b_map
        routed_feat = v * final_gate + i * (1.0 - final_gate)
        return self.fuse_conv(torch.cat([v + routed_feat, i + routed_feat], dim=1))

class GatedLinearCrossAttentionInjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dynamic_filter = SpatialAdaptiveDynamicFilter(dim)
        self.gated_attention = nn.Sequential(nn.Conv2d(dim*2, dim, 1), nn.Sigmoid())

    def forward(self, v, i):
        
        edge_v = self.dynamic_filter(v)
        edge_i = self.dynamic_filter(i)
        
        
        attn_gate = self.gated_attention(torch.cat([edge_v, edge_i], dim=1))
        
        # ==========================================================
        
        
        # ==========================================================
        abs_edge_v = torch.abs(edge_v)
        abs_edge_i = torch.abs(edge_i)
        
        
        edge_v_safe = abs_edge_v + 1e-4
        edge_i_safe = abs_edge_i + 1e-4
        
        
        m = torch.max(edge_v_safe, edge_i_safe).detach()
        
        # ==========================================================
        
        
        # ==========================================================
        base_v = torch.clamp(edge_v_safe / m, min=0.0, max=1.0)
        base_i = torch.clamp(edge_i_safe / m, min=0.0, max=1.0)
        
        
        w_v = base_v ** 8
        w_i = base_i ** 8
        
        hard_gate = w_v / (w_v + w_i + 1e-6)
        final_gate = hard_gate * attn_gate
        
        return v * final_gate + i * (1.0 - final_gate)

class CrossSequenceMambaFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cross_mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=1)
        self.norm = nn.LayerNorm(dim)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.i_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, v, i):
            b, c, h, w = v.shape
            v_flat = rearrange(v, 'b c h w -> b (h w) c')
            i_flat = rearrange(i, 'b c h w -> b (h w) c')
            
            
            
            cat_seq = torch.stack([v_flat, i_flat], dim=2).view(b, h * w * 2, c)
            cat_seq = self.norm(cat_seq)
            
            out_seq = self.cross_mamba(cat_seq)
            
            
            out_seq = out_seq.view(b, h * w, 2, c)
            v_out_seq = out_seq[:, :, 0, :]
            i_out_seq = out_seq[:, :, 1, :]
            
            v_out = rearrange(v_out_seq, 'b (h w) c -> b c h w', h=h, w=w)
            i_out = rearrange(i_out_seq, 'b (h w) c -> b c h w', h=h, w=w)
            
            return v + self.i_proj(i_out), i + self.v_proj(v_out)

class BaseDetailSynergisticIntegration(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.detail_to_base = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.Sigmoid())
        self.base_to_detail = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.Sigmoid())
        self.fuse = nn.Sequential(nn.Conv2d(dim*2, dim, 1), nn.GELU())

    def forward(self, base, detail):
        base_refined = base + base * self.detail_to_base(detail)
        detail_refined = detail * self.base_to_detail(base)
        return self.fuse(torch.cat([base_refined, detail_refined], dim=1))

class MDTA(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1, eps=1e-6)
        k = torch.nn.functional.normalize(k, dim=-1, eps=1e-6)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)

class ECA(nn.Module):
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y).expand_as(x)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
    def forward(self, x): return self.proj(x)

class DVSSBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.ldc = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False), nn.GELU(), nn.Conv2d(dim, dim, 1, bias=False))
        self.ln1 = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand//2)
        self.merge_proj = nn.Linear(dim * 4, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.eca = ECA()
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        local_feat = self.ldc(x)
        x_norm = rearrange(x, 'b c h w -> b h w c')
        x_norm = self.ln1(x_norm)
        x_h = rearrange(x_norm, 'b h w c -> b (h w) c')
        x_w = rearrange(x_norm, 'b h w c -> b (w h) c')
        x_h_rev = torch.flip(x_h, dims=[1])
        x_w_rev = torch.flip(x_w, dims=[1])
        x_cat = torch.cat([x_h, x_w, x_h_rev, x_w_rev], dim=0) 
        out_cat = self.mamba(x_cat)
        out_h, out_w, out_h_rev, out_w_rev = out_cat.chunk(4, dim=0) 
        out_h_rev = torch.flip(out_h_rev, dims=[1])
        out_w_rev = torch.flip(out_w_rev, dims=[1])
        out_h = rearrange(out_h, 'b (h w) c -> b h w c', h=h, w=w)
        out_w = rearrange(out_w, 'b (w h) c -> b h w c', h=h, w=w)
        out_h_rev = rearrange(out_h_rev, 'b (h w) c -> b h w c', h=h, w=w)
        out_w_rev = rearrange(out_w_rev, 'b (w h) c -> b h w c', h=h, w=w)
        mamba_merged = torch.cat([out_h, out_w, out_h_rev, out_w_rev], dim=-1) 
        essm_out = self.merge_proj(mamba_merged)
        essm_out = rearrange(self.out_proj(essm_out), 'b h w c -> b c h w')
        mid_feat = x + self.layer_scale_1 * essm_out
        mid_norm = rearrange(mid_feat, 'b c h w -> b h w c')
        mid_norm = rearrange(self.ln2(mid_norm), 'b h w c -> b c h w')
        attn_feat = self.eca(mid_norm) + mid_feat 
        out = attn_feat + self.layer_scale_2 * local_feat
        return out

class MultiScaleInvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.expand_conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, bias=False), nn.ReLU6(inplace=True))
        self.dw3 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False)
        self.dw5 = nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2, groups=hidden_dim, bias=False)
        self.scale_fuse = nn.Sequential(nn.Conv2d(hidden_dim * 2, hidden_dim, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.project_conv = nn.Conv2d(hidden_dim, oup, 1, bias=False)
    def forward(self, x):
        h = self.expand_conv(x)
        h3 = self.dw3(h)
        h5 = self.dw5(h)
        h_fused = self.scale_fuse(torch.cat([h3, h5], dim=1))
        return self.project_conv(h_fused)

class DetailNode(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta_phi = MultiScaleInvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = MultiScaleInvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = MultiScaleInvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, z1, z2):
        shuffled = self.shffleconv(torch.cat((z1, z2), dim=1))
        z1_new, z2_new = shuffled[:, :32], shuffled[:, 32:]
        z2_new = z2_new + self.theta_phi(z1_new)
        
        rho_safe = torch.tanh(self.theta_rho(z2_new)) * 2.0
        z1_new = z1_new * torch.exp(rho_safe) + self.theta_eta(z2_new)
        
        return z1_new, z2_new

class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=2): 
        super().__init__()
        self.net = nn.Sequential(*[DetailNode() for _ in range(num_layers)])
    def forward(self, x):
        z1, z2 = x[:, :32], x[:, 32:]
        for layer in self.net: z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


# =============================================================================

# =============================================================================
class DeepUnfoldedSemanticClustering(nn.Module):
    def __init__(self, dim, num_clusters=8, temperature=0.1):
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature
        
        
        self.prototypes = nn.Parameter(torch.randn(1, num_clusters, dim))
        nn.init.orthogonal_(self.prototypes)
        
        
        
        self.prior_prompt_net = nn.Sequential(
            nn.Conv2d(2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        
        self.proj_v = nn.Conv2d(dim, dim, 1)
        self.proj_i = nn.Conv2d(dim, dim, 1)
        self.fuse_conv = nn.Conv2d(dim * 2, dim, 3, padding=1)
        self.act = nn.GELU()

    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, v, i):
        v_f, i_f = v.float(), i.float()
        b, c, h, w = v_f.shape
        
        # ==========================================================
        
        # ==========================================================
        
        v_pad = F.pad(v_f, (1, 1, 1, 1), mode='reflect')
        v_var = F.avg_pool2d(v_pad**2, 3, stride=1) - F.avg_pool2d(v_pad, 3, stride=1)**2
        v_prior = torch.clamp(v_var, min=1e-6)
        v_prior = v_prior / (torch.amax(v_prior, dim=[2, 3], keepdim=True) + 1e-6) 
        
        
        i_prior = F.avg_pool2d(i_f, 3, stride=1, padding=1)
        i_prior = i_prior / (torch.amax(i_prior, dim=[2, 3], keepdim=True) + 1e-6) 
        
        
        prior_map = torch.cat([v_prior.mean(dim=1, keepdim=True), i_prior.mean(dim=1, keepdim=True)], dim=1)
        prior_prompt = self.prior_prompt_net(prior_map)

        # ==========================================================
        
        # ==========================================================
        
        v_flat = rearrange(self.proj_v(v_f + prior_prompt), 'b c h w -> b (h w) c')
        i_flat = rearrange(self.proj_i(i_f + prior_prompt), 'b c h w -> b (h w) c')

        v_norm = F.normalize(v_flat, dim=-1, eps=1e-6)
        i_norm = F.normalize(i_flat, dim=-1, eps=1e-6)
        p_norm = F.normalize(self.prototypes.float(), dim=-1, eps=1e-6)

        sim_v = torch.einsum('b n c, d k c -> b n k', v_norm, p_norm)
        sim_i = torch.einsum('b n c, d k c -> b n k', i_norm, p_norm)

        prob_v = F.softmax(sim_v / self.temperature, dim=-1)
        prob_i = F.softmax(sim_i / self.temperature, dim=-1)

        
        conf_diff = prob_i.max(dim=-1)[0] - prob_v.max(dim=-1)[0]
        routing_gate = torch.sigmoid(conf_diff * 5.0).unsqueeze(-1)

        
        v_orig_flat = rearrange(v_f, 'b c h w -> b (h w) c')
        i_orig_flat = rearrange(i_f, 'b c h w -> b (h w) c')
        fused_flat = v_orig_flat * (1.0 - routing_gate) + i_orig_flat * routing_gate
        fused_clustered = rearrange(fused_flat, 'b (h w) c -> b c h w', h=h, w=w)

        return self.act(self.fuse_conv(torch.cat([v + fused_clustered, i + fused_clustered], dim=1)))

# =============================================================================

# =============================================================================
class AdvancedBaseFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ba_routing = BoundaryAwareRoutingBase(dim)
        self.dusc = DeepUnfoldedSemanticClustering(dim=dim, num_clusters=8)
        self.tcmf_fusion = CrossSequenceMambaFusion(dim)
        self.mica_amp = MaxInformationChannelAmplifier(dim)
        self.cca = ContrastiveComplementaryAttention(dim)
        
        self.fuse_conv = nn.Sequential(nn.Conv2d(dim * 2, dim, 3, padding=1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        self.refine_block = DVSSBlock(dim=dim)

    
    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, v, i):
        
        v_route, i_route = v.float(), i.float()
        
        aligned_feat = self.ba_routing(v_route, i_route)
        v_route = v_route + aligned_feat
        i_route = i_route + aligned_feat
        
        cluster_feat = self.dusc(v_route, i_route)
        v_mamba, i_mamba = self.tcmf_fusion(v_route + cluster_feat, i_route + cluster_feat)
        fused = self.fuse_conv(torch.cat([v_mamba, i_mamba], dim=1))
        
        fused_amped = self.mica_amp(fused, v_route, i_route)
        fused_cca = self.cca(v_route, i_route, fused_amped)
        
        return self.refine_block(fused_cca)

class SpatialChannelDetailFusion(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.ms_block = StarGatedMultiScaleBlock(dim)
        self.glca_injection = GatedLinearCrossAttentionInjection(dim)
        
        self.proj_1 = nn.Conv2d(dim * 2, dim, 1)
        self.dwconv = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.dwconv_d = nn.Conv2d(dim, dim, 7, padding=9, groups=dim, dilation=3)
        self.proj_2 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        
        self.edge_scale = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5) 
        self.mdta_refiner = MDTA(dim=dim)
        self.lpeb = LaplacianPyramidEdgeBooster(dim)
        
    
    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, v, i):
        
        v_f, i_f = v.float(), i.float()
        
        x = torch.cat([v_f, i_f], dim=1)
        u = self.proj_1(x)
        attn = self.dwconv(u)
        attn = self.dwconv_d(attn)
        attn = self.act(attn)
        w_v = self.sigmoid(self.proj_2(attn))
        
        fused_detail = w_v * v_f + (1.0 - w_v) * i_f
        glca_edge = self.glca_injection(v_f, i_f)
        fused_detail = fused_detail + glca_edge * self.edge_scale
        
        out = self.ms_block(fused_detail)
        out = out + self.mdta_refiner(out) 
        out = self.lpeb(out)
        return out

class Mamba_Encoder(nn.Module):
    def __init__(self, inp_channels=1, dim=64, num_blocks=[2, 1]): 
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.eca = ECA(k_size=3) 
        self.encoder_level1 = nn.Sequential(*[DVSSBlock(dim=dim) for _ in range(num_blocks[0])])
        self.baseFeature = nn.Sequential(*[DVSSBlock(dim=dim) for _ in range(num_blocks[1])])
        self.detailFeature = DetailFeatureExtraction(num_layers=2)
    @torch.amp.autocast('cuda', enabled=False)         
    def forward(self, inp_img):
        inp_img = inp_img.float()
        x = self.patch_embed(inp_img)
        x = self.eca(x) 
        out_enc_level1 = self.encoder_level1(x)
        return self.baseFeature(out_enc_level1), self.detailFeature(out_enc_level1), out_enc_level1

class Mamba_Decoder(nn.Module):
    def __init__(self, out_channels=1, dim=64, num_blocks=2, bias=False):
        super().__init__()
        self.bdsi = BaseDetailSynergisticIntegration(dim)
        self.fds_stretcher = FeatureDistributionStretcher(dim)
        self.ms_reconstruction = StarGatedMultiScaleBlock(dim)
        
        self.encoder_level2 = nn.Sequential(*[DVSSBlock(dim=dim) for _ in range(num_blocks)])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3, stride=1, padding=1, bias=bias), nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
        )
        self.sigmoid = nn.Sigmoid()              
        
    
    @torch.amp.autocast('cuda', enabled=False)
    def forward(self, base_feature, detail_feature):
        base_feature = base_feature.float()
        detail_feature = detail_feature.float()
        out_enc_level0 = self.bdsi(base_feature, detail_feature)
        out_enc_level0 = self.fds_stretcher(out_enc_level0)
        
        out_enc_level0 = self.ms_reconstruction(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        
        return self.sigmoid(self.output(out_enc_level1)), out_enc_level0