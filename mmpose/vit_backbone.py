"""
Plain ViT backbone for ViTPose, registered into mmpose's BACKBONES registry.

Needed because the base Docker image (stanfordnmbl/mmpose:0.1, mmpose ~v0.13,
June 2021) predates ViTPose and does not ship this backbone. Adding this file
to opencap-core/mmpose/ is sufficient: the mmpose Dockerfile does
  COPY mmpose /mmpose
so the file lands at /mmpose/vit_backbone.py inside the container.

Reference: https://github.com/ViTAE-Transformer/ViTPose
"""

import math
from functools import partial

import torch
import torch.nn as nn

from mmcv.runner import load_checkpoint
from mmpose.models.builder import BACKBONES
from mmpose.utils import get_root_logger


# ---------------------------------------------------------------------------
# Stochastic depth
# ---------------------------------------------------------------------------

def _drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep_prob
    return x / keep_prob * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return _drop_path(x, self.drop_prob, self.training)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        attn = self.attn_drop((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path_rate=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(256, 192), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.grid_h = img_size[0] // patch_size[0]
        self.grid_w = img_size[1] // patch_size[1]
        self.num_patches = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)   # (B, N, C)


# ---------------------------------------------------------------------------
# ViT backbone — registered for mmpose
# ---------------------------------------------------------------------------

@BACKBONES.register_module()
class ViT(nn.Module):
    """Plain Vision Transformer backbone used by ViTPose.

    Args:
        img_size (tuple): Input image size (H, W). Default: (256, 192).
        patch_size (int): Patch size. Default: 16.
        in_chans (int): Input channels. Default: 3.
        embed_dim (int): Embedding dimension. Default: 768.
        depth (int): Number of transformer blocks. Default: 12.
        num_heads (int): Number of attention heads. Default: 12.
        mlp_ratio (float): MLP hidden-dim multiplier. Default: 4.
        qkv_bias (bool): Add bias to QKV. Default: False.
        drop_rate (float): Dropout rate. Default: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ratio (int): Kept for config compatibility (unused when 1). Default: 1.
        use_checkpoint (bool): Use gradient checkpointing. Default: False.
        pretrained (str, optional): Pretrained checkpoint path.
    """

    def __init__(self,
                 img_size=(256, 192),
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 ratio=1,
                 use_checkpoint=False,
                 pretrained=None):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.pretrained = pretrained
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # ViTPose-B uses a fixed-zero cls_token that is never saved in the
        # checkpoint.  pos_embed covers N+1 positions: [cls, p1…pN].
        # cls_token is created on-the-fly in forward() to avoid any state_dict
        # key mismatch when loading the official checkpoint.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path_rate=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.last_norm = norm_layer(embed_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, N, C)

        # Prepend a fixed-zero cls_token (never trained, not in checkpoint)
        # then add the full N+1 positional embedding.
        cls = x.new_zeros(B, 1, self.embed_dim)          # (B, 1, C) zeros
        x = torch.cat([cls, x], dim=1)                   # (B, N+1, C)
        x = self.pos_drop(x + self.pos_embed)

        for blk in self.blocks:
            if self.use_checkpoint:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)

        # Discard cls_token; reshape patch sequence → spatial feature map (B, C, H, W)
        x = x[:, 1:]
        h, w = self.patch_embed.grid_h, self.patch_embed.grid_w
        x = x.reshape(B, h, w, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        return [x]   # mmpose expects a list of feature tensors
