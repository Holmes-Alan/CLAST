import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as Func
from mamba_ssm import Mamba
from timm.models.vision_transformer import Mlp

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class MHSA(nn.Module):
    def __init__(self, n_dims, width=16, height=16):
        super(MHSA, self).__init__()

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):  # b, c, 16,16
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, C, -1)
        k = self.key(x).view(n_batch, C, -1)
        v = self.value(x).view(n_batch, C, -1)

        content_content = torch.bmm(q.permute(0, 2, 1), k)

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)

        return out
        

class CrossAttN_v8(nn.Module):
    def __init__(self, in_planes, clip_dim):
        super(CrossAttN_v8, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, 1, 1, 0)
        self.g = nn.Conv2d(in_planes, in_planes, 1, 1, 0)
        self.h = nn.Conv2d(in_planes, in_planes, 1, 1, 0)
        self.output = nn.Conv2d(in_planes, in_planes, 1, 1, 0)
        return

    def forward(self, F_c, F_s):
        b, c = F_c.shape[0], F_c.shape[1]

        # T_c = F_c
        F = self.f(F_c)
        F = F.view(b, F.shape[1], -1)
        G = self.g(F_s)
        G = G.view(b, G.shape[1], -1)
        # G = G / G.norm(dim=2, keepdim=True)
        H = self.h(F_s)
        H = H.view(b, H.shape[1], -1)
        # H = H / H.norm(dim=2, keepdim=True)
        S = torch.bmm(F.permute(0, 2, 1), G) # b, d_s, d_c
        S = Func.softmax(S, dim=-1)
        result = torch.bmm(H, S.permute(0, 2, 1)) # b, d_c, h*w
        result = result.view(b, result.shape[1], F_c.shape[2], F_c.shape[3])
        result = self.output(result)

        return result
        


class CA_SA_v4(nn.Module):
    def __init__(self, in_planes, out_planes, clip_dim=512, max_sample=256 * 256):
        super(CA_SA_v4, self).__init__()
        self.clip_dim = clip_dim
        self.fs = nn.Sequential(
            nn.Conv2d(clip_dim, in_planes, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        )
        self.prepare_out = nn.Conv2d(in_planes, out_planes, 1, 1, 0)
        self.attn1 = CrossAttN_v8(in_planes, clip_dim)  # 1st order
        self.attn2 = CrossAttN_v8(in_planes, clip_dim)  # 2ed order
        return

    def forward(self, F_clip_s, F_content):
        b, c, h, w = F_content.shape
        # F_clip_s: # b, 512, 16, 16
        F_clip_s = self.fs(F_clip_s)
        F_c = mean_variance_norm(F_content)
        mean_s, std_s = calc_mean_std(F_clip_s)
        F_s = (F_clip_s - mean_s.expand(F_clip_s.size())) / std_s.expand(F_clip_s.size())
        result = self.attn1(F_c, F_s) + self.attn2(F_c, torch.pow(F_s, 2)) + F_c
        result = result.view(b, -1, h, w).contiguous()
        result = mean_variance_norm(result) * std_s.expand(F_content.size()) + mean_s.expand(F_content.size())
        result = self.prepare_out(result)
        return result
    

class Mamba_v1(nn.Module):
    def __init__(self, in_planes, out_planes, clip_dim=512, max_sample=256 * 256):
        super(Mamba_v1, self).__init__()
        self.clip_dim = clip_dim
        self.fs = nn.Sequential(
            nn.Conv2d(clip_dim, in_planes, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        )
        self.prepare_out = nn.Conv2d(in_planes, out_planes, 1, 1, 0)
        self.dim = in_planes
        self.mamba_attn = Mamba(
            d_model=in_planes*2, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

    def forward(self, F_clip_s, F_content):
        b, c, h, w = F_content.shape
        # F_clip_s: # b, 512, 16, 16
        F_clip_s = self.fs(F_clip_s)
        F_c = mean_variance_norm(F_content)
        mean_s, std_s = calc_mean_std(F_clip_s)
        F_s = (F_clip_s - mean_s.expand(F_clip_s.size())) / std_s.expand(F_clip_s.size())
        F_c_resize = Func.interpolate(F_c, size=[16, 16], mode='bicubic')
        combine_feat = torch.cat((F_c_resize, F_s), dim=1).reshape(b, 2*c, -1). permute(0, 2, 1)
        result = self.mamba_attn(combine_feat)
        result = result.permute(0, 2, 1).reshape(b, 2*c, 16, 16)
        result = Func.interpolate(result.chunk(2, dim=1)[0], size=[h, w], mode='bicubic') + F_c
        # result = result.view(b, -1, h, w).contiguous()
        result = mean_variance_norm(result) * std_s.expand(F_content.size()) + mean_s.expand(F_content.size())
        result = self.prepare_out(result)
        return result
    

class Mamba_v2(nn.Module):
    def __init__(self, in_planes, out_planes, clip_dim=512, max_sample=256 * 256):
        super(Mamba_v2, self).__init__()
        self.clip_dim = clip_dim
        self.fs = nn.Sequential(
            nn.Linear(clip_dim, 6*in_planes)
        )
        self.norm1 = nn.LayerNorm(in_planes, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(in_planes, elementwise_affine=False, eps=1e-6)
        self.prepare_out = nn.Conv2d(in_planes, out_planes, 1, 1, 0)
        self.dim = in_planes
        self.attn = Mamba(
            d_model=in_planes, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=in_planes, hidden_features=in_planes, act_layer=approx_gelu, drop=0)

    def forward(self, F_clip_s, F_content):
        b, c, h, w = F_content.shape
        # F_clip_s: # b, 512*6
        F_content = F_content.reshape(b, c, -1). permute(0, 2, 1) # b, hw, c
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.fs(F_clip_s).chunk(6, dim=1)
        F_content = F_content + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(F_content), shift_msa, scale_msa))
        result = F_content + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(F_content), shift_mlp, scale_mlp))

        result = result.permute(0, 2, 1).reshape(b, c, h, w)
        result = self.prepare_out(result)
        return result
        

class Mamba_v3(nn.Module):
    def __init__(self, out_planes):
        super(Mamba_v3, self).__init__()
        self.fs = nn.Sequential(
            nn.Linear(512, 6*256)
        )
        self.pixel_unshuffle = nn.PixelUnshuffle(8)
        self.pixel_shuffle = nn.PixelShuffle(8)
        self.norm1 = nn.LayerNorm(256, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(256, elementwise_affine=False, eps=1e-6)

        self.attn = Mamba(
            d_model=256, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=256, hidden_features=256, act_layer=approx_gelu, drop=0)

    def forward(self, F_clip_s, F_content):
        b, c, h, w = F_content.shape

        F_content_new = self.pixel_unshuffle(F_content) # b, 256, 4, 4
        F_content_new = F_content_new.reshape(b, 256, -1).permute(0, 2, 1) # b, 16, 256
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.fs(F_clip_s).chunk(6, dim=1)
        F_content_new = F_content_new + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(F_content_new), shift_msa, scale_msa))
        result = F_content_new + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(F_content_new), shift_mlp, scale_mlp))

        result = result.permute(0, 2, 1).reshape(b, 256, 4, 4) # b, 256, 4, 4
        result = self.pixel_shuffle(result) # b, 4, 32, 32
        return result
    

class Transform_CA_SA_v4(nn.Module):
    def __init__(self, isDisable=False):
        super(Transform_CA_SA_v4, self).__init__()
        in_planes = [64, 128, 320, 512]
        self.isDisable = isDisable
        self.MHSA = MHSA(512)
        self.sanet0 = CA_SA_v4(in_planes=in_planes[0], out_planes=in_planes[0])
        self.sanet1 = CA_SA_v4(in_planes=in_planes[0] + in_planes[1], out_planes=in_planes[1])
        self.sanet2 = CA_SA_v4(in_planes=in_planes[0] + in_planes[1] + in_planes[2], out_planes=in_planes[2])

        return

    def get_key(self, feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
            results.append(mean_variance_norm(feats[last_layer_idx]))
            return torch.cat(results, dim=1)
        else:
            return mean_variance_norm(feats[last_layer_idx])

    def edit_style(self, F_s, mean, std):
        mean = mean.expand_as(F_s)
        std = std.expand_as(F_s)
        return (1 + std) * (F_s) + mean

    def forward(self, F_clip_c, F_clip_s, F_c):
        if self.isDisable: return F_c  # if is disable style transfer
        F_clip_s = F_clip_s.permute(0, 2, 1).unsqueeze(-1)  # b, 512, 1, 1
        F_clip_s = F_clip_s.repeat(1, 1, 16, 16)  # b, 512, 16, 16
        F_clip_s = self.MHSA(F_clip_s) + F_clip_s# b, 512, 16, 16

        F0 = self.sanet0(F_clip_s=F_clip_s, F_content=self.get_key(F_c, 0))
        F1 = self.sanet1(F_clip_s=F_clip_s, F_content=self.get_key(F_c, 1))
        F2 = self.sanet2(F_clip_s=F_clip_s, F_content=self.get_key(F_c, 2))
        F_out = [F0, F1, F2]
        return F_out  # feat # [3, 512, 64, 64]


# class Transform_CA_SA_mamba(nn.Module):
#     def __init__(self, isDisable=False):
#         super(Transform_CA_SA_mamba, self).__init__()
#         in_planes = [64, 128, 320, 512]
#         self.isDisable = isDisable
#         self.MHSA = MHSA(512)
#         self.sanet0 = Mamba_v1(in_planes=in_planes[0], out_planes=in_planes[0])
#         self.sanet1 = Mamba_v1(in_planes=in_planes[0] + in_planes[1], out_planes=in_planes[1])
#         self.sanet2 = Mamba_v1(in_planes=in_planes[0] + in_planes[1] + in_planes[2], out_planes=in_planes[2])

#         return

#     def get_key(self, feats, last_layer_idx, need_shallow=True):
#         if need_shallow and last_layer_idx > 0:
#             results = []
#             _, _, h, w = feats[last_layer_idx].shape
#             for i in range(last_layer_idx):
#                 results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
#             results.append(mean_variance_norm(feats[last_layer_idx]))
#             return torch.cat(results, dim=1)
#         else:
#             return mean_variance_norm(feats[last_layer_idx])

#     def edit_style(self, F_s, mean, std):
#         mean = mean.expand_as(F_s)
#         std = std.expand_as(F_s)
#         return (1 + std) * (F_s) + mean

#     def forward(self, F_clip_c, F_clip_s, F_c):
#         if self.isDisable: return F_c  # if is disable style transfer
#         F_clip_s = F_clip_s.permute(0, 2, 1).unsqueeze(-1)  # b, 512, 1, 1
#         F_clip_s = F_clip_s.repeat(1, 1, 16, 16)  # b, 512, 16, 16
#         F_clip_s = self.MHSA(F_clip_s) + F_clip_s# b, 512, 16, 16

#         F0 = self.sanet0(F_clip_s=F_clip_s, F_content=self.get_key(F_c, 0))
#         F1 = self.sanet1(F_clip_s=F_clip_s, F_content=self.get_key(F_c, 1))
#         F2 = self.sanet2(F_clip_s=F_clip_s, F_content=self.get_key(F_c, 2))
#         F_out = [F0, F1, F2]
#         return F_out  # feat # [3, 512, 64, 64]
    

class Transform_CA_SA_mamba(nn.Module):
    def __init__(self, isDisable=False):
        super(Transform_CA_SA_mamba, self).__init__()
        in_planes = [64, 128, 320, 512]
        self.isDisable = isDisable
        self.sanet0 = Mamba_v2(in_planes=in_planes[0], out_planes=in_planes[0])
        self.sanet1 = Mamba_v2(in_planes=in_planes[0] + in_planes[1], out_planes=in_planes[1])
        self.sanet2 = Mamba_v2(in_planes=in_planes[0] + in_planes[1] + in_planes[2], out_planes=in_planes[2])

    def get_key(self, feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
            results.append(mean_variance_norm(feats[last_layer_idx]))
            return torch.cat(results, dim=1)
        else:
            return mean_variance_norm(feats[last_layer_idx])


    def forward(self, F_clip_c, F_clip_s, F_c):
        if self.isDisable: return F_c  # if is disable style transfer
        F_clip_s = F_clip_s.squeeze(1)  # b, 512, 1, 1

        F0 = self.sanet0(F_clip_s=F_clip_s, F_content=self.get_key(F_c, 0))
        F1 = self.sanet1(F_clip_s=F_clip_s, F_content=self.get_key(F_c, 1))
        F2 = self.sanet2(F_clip_s=F_clip_s, F_content=self.get_key(F_c, 2))
        F_out = [F0, F1, F2]
        return F_out  # feat # [3, 512, 64, 64]
    

class txt_img_clip(nn.Module):
    def __init__(self):
        super(txt_img_clip, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)
        )

    def forward(self, F_clip_s):
        mean, std = self.mlp(F_clip_s)
        F_clip_s = F_clip_s * (1 + std) + mean
        return F_clip_s


