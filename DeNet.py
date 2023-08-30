# -*- coding:utf-8 -*-
"""
@Author：JS
@Time：2023/1/23 22:06
@File：Swin_Res2Net.py
@Mission：
"""
import math
import timm
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from utils.checkpoint import load_checkpoint
from mmseg.utils import get_root_logger

from utils.module import Attention, PreNorm, FeedForward, CrossAttention
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, resnet34, resnet50
from timm.models import resnetv2
from thop import profile

from PraNet.Res2Net_v1b import res2net50_v1b_26w_4s

import warnings

warnings.filterwarnings('ignore')

# 当Swin-base作为Transformer分支时
# groups = 32
# 当Swin-tiny作为Transformer分支时
groups = 24


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features).cuda()
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features).cuda()
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# window partition函数是用于对张量划分窗口, 指定窗口大小。将原本的张量从[B H W C], 划分成[num_windows*B, window_size, window_size, C]
# 其中num_windows = H*W / window_size*window_size, 即窗口的个数。
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


# window reverse函数是window_partition的逆过程(将shape返回为(B, C, H, W))
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# 定义W-MSA、SW-MSA模块           (当mask不为None时,执行SW-MSA)
class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # 使用线性映射得到q k v
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # 计算自注意力之后使输出映射到想要的维数
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        # print(f"WindowAttention的输出shape{x.shape}")
        return x


# SwinTransformerBlock就是包括W-MSA和SW-MSA的两个Transformer Block
class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # 要滑动的尺寸大小需要在0-window_size之间
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size     # window_size=7,若feature map对window_size不能够整除,则需要进行pad
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size

        # 分别在左右、上下填充0
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift                        # 如果shift_size>0,则进行窗口移动
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        print(f"x_windows shape {x_windows.shape}")

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchRecover(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=dim // 2, num_groups=groups),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.permute(0, 1, 2)  # B, C, L
        x = x.reshape(B, C, H, W)  # B, C, L-->B, C, H, W
        x = self.up(x)  # B, C, H, W-->B, C//2, 2H, 2W

        x = x.reshape(B, C // 2, -1)  # B, C//2, 2H*2W
        x = x.permute(0, 2, 1)  # B, 2H*2W, C//2

        return x  # 由B, H*W, C-->B, 2H*2W, C//2


# 定义Patch Merging模块(通过Patch Merging实现分辨率减半、通道数翻倍)
class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 能够实现channel数翻倍
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape  # 输入PatchMerging模块的shape:[B, H*W, C]
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)  # [B, H*W, C]-->[B, H, W, C]

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))  # 中间两个参数表示对W维度进行填充(左右填充),后两个参数对H维度进行填充(上下填充)

        x0 = x[:, 0::2, 0::2, :]  # [B, H, W, C]-->[B H/2 W/2 C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H, W, C]-->[B H/2 W/2 C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H, W, C]-->[B H/2 W/2 C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H, W, C]-->[B H/2 W/2 C]
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)  # [B H/2*W/2 4*C]-->[B H/2*W/2 2*C]

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 up=True):
        super().__init__()
        self.window_size = window_size      # window_size = 7
        self.shift_size = window_size // 2  # shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.up = up

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,  # qk_scale就是根号dk
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer             # 指定使用PatchMerging进行降采样
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA                     # 生成用于计算SW-MSA的attention mask矩阵
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            if self.up:
                Wh, Ww = (H + 1) // 2, (W + 1) // 2
            else:
                Wh, Ww = H * 2, W * 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


# 定义PatchEmbedding模块(将image切分成一个个小patch,其中patch_size=4)
class PatchEmbed(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)  # patch_size=(4, 4)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 使用Conv进行PatchEmbedding
        # 若input.shape=[1, 3, 256, 256]  PatchEmbed的shape变化: [1, 3, 256, 256]-->[1, 96, 64, 64]
        # 注意: 因为是patch_size作为stride,所以不同的patch_size可以得到不同size的feature map
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()  # input.shape=[1, 3, 256, 256], H=256, W=256

        # 此处判断input的H和W是否可以对patch_size整除
        if W % self.patch_size[1] != 0:  # 如果W / self.patch_size[1]的余数不为0
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))  # 那么就在输入feature map的右边进行pad
        if H % self.patch_size[0] != 0:  # 如果H / self.patch_size[0]的余数不为0
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))  # 那么就在输入feature map的下边进行pad

        x = self.proj(x)  # [1, 3, 256, 256]-->[1, 96, 64, 64]
        print(f"x shape {x.shape}")
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)  # Wh=64, Ww=64
            x = x.flatten(2).transpose(1, 2)  # [1, 96, 64, 64]-->[1, 96, 64*64]-->[1, 64*64, 96]

            # 在进行LayerNorm前需要将shape变换为[b, h*w, c]
            x = self.norm(x)

            # [1, 64*64, 96]-->[1, 96, 64*64]-->[1, 96, 64, 64]
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x  # PatchEmbed返回的特征图shape为[b, embed_dim, h/patch_size, w/patch_size]


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=128,
                 depths=(2, 2, 18, 2),
                 num_heads=(4, 8, 16, 32),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.5,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)
        print(f"x shape {x.shape}")

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()


# up_conv模块使用nearest上采样来恢复图像分辨率(用在两个常规Decoder模块、同时代替SwinDeconder中的patch-expanding)
class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),  # num_channels表示channel总数,num_groups表示要分的组数
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


# 定义两个常规的Decoder模块
class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        # up_conv模块使用nearest上采样来恢复图像分辨率
        # 当输入dim=96时
        # self.up = up_conv(in_channels, out_channels)  # groups参数为GroupNorm中的num_groups
        # 当输入dim=128时
        self.up = up_conv(in_channels, out_channels)  # groups参数为GroupNorm中的num_groups

        # 另一Upsample方式: nn.ConvTranspose2d为转置卷积
        # self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x2 = self.att_block(x1, x2)
        x1 = torch.cat((x2, x1), dim=1)
        x1 = self.conv_relu(x1)
        return x1


# 保证跳跃连接中的shape一致
class Conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(Conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


# 基于Swin-Transformer的三个Decoder模块(每个Decoder模块堆叠两个Transformer block)
class SwinDecoder(nn.Module):

    def __init__(self,
                 embed_dim,
                 patch_size=4,
                 depths=2,
                 num_heads=6,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False):
        super(SwinDecoder, self).__init__()

        self.patch_norm = patch_norm

        # split image into non-overlapping patches

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        # build layers
        self.layer = BasicLayer(
            dim=embed_dim // 2,
            depth=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint)

        self.up = up_conv(embed_dim, embed_dim // 2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x):
        """Forward function."""

        identity = x
        B, C, H, W = x.shape
        x = self.up(x)  # B , C//2, 2H, 2W
        x = x.reshape(B, C // 2, H * W * 4)
        x = x.permute(0, 2, 1)

        x_out, H, W, x, Wh, Ww = self.layer(x, H * 2, W * 2)

        x = x.permute(0, 2, 1)
        x = x.reshape(B, C // 2, H, W)
        # B, C//4 2H, 2W
        x = self.conv_relu(x)

        return x


# 定义decoder模块
class Swin_Decoder(nn.Module):
    def __init__(self, in_channels, depths, num_heads):
        super(Swin_Decoder, self).__init__()
        self.up = SwinDecoder(in_channels, depths=depths, num_heads=num_heads)

        # self.up1 = nn.Upsample(scale_factor=2)
        # self.up2 = nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    # 每个decoder模块中接收两个输入,一个是同级encoder经TIF的输出由跳跃连接传递过来,另一个就是更深一层encoder经TIF的输出传递过来
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # x1 = self.up2(x1)
        # x2 = self.att_block(x1, x2)
        x2 = self.conv2(x2)
        x1 = torch.cat((x2, x1), dim=1)
        out = self.conv_relu(x1)
        return out


# 定义TIF模块中的Transformer模块(就是常规的Transformer Block)
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# 用于CNN分支的空间注意力模块做完MaxPool和AvgPool后对channel维度进行调整
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# 用于CNN分支的空间注意力模块中的MaxPool和AvgPool
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


# 定义特征融合中的Residual模块(就是一个常规的Resnet模块)
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        # 常规残差模块中的第一个1*1conv
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        # 残差模块中间层的3*3conv
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        # 残差模块中的最后一个1*1conv
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        # 是否需要加残差支路
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


# 定义尝试三中的ECA-Block
class ECA_Block(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


# 定义替换CBAM的SCSEModule
class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


# 定义特征融合模块TIF
class Cross_Att(nn.Module):
    def __init__(self, r_dim, e_dim, reduction, ch_int, ch_out, drop_rate=0.):  # e_dim表示Transformer分支, r_dim表示CNN分支
        super().__init__()
        # spatial attention for CNN分支
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)

        # channel attention for e, use SE Block
        # self.fc1 = nn.Conv2d(e_dim, e_dim // r_2, kernel_size=1)      # 使用1*1Conv代替原SE模块中的全连接层
        # self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Conv2d(e_dim // r_2, e_dim, kernel_size=1)
        # self.sigmoid = nn.Sigmoid()

        # 尝试三中的通道注意力ECA-Block
        self.eca = ECA_Block(e_dim, k_size=3)
        self.sigmoid = nn.Sigmoid()
        # 尝试四中的空间注意力SCSEModule          # SCSEModule里有参数reduction,遵循SEBlock中的设计(1, 2, 4, 8)
        self.scse = SCSEModule(r_dim, reduction)

        # bi-linear modelling for both
        self.W_g = Conv(r_dim, ch_int, 1, bn=True, relu=False)       # W_g表示CNN分支
        self.W_x = Conv(e_dim, ch_int, 1, bn=True, relu=False)       # W_x表示Transformer分支
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.residual = Residual(r_dim + e_dim + ch_int, ch_out)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

    def forward(self, r, e):  # r表示CNN分支, e表示Transformer分支
        # bilinear pooling
        W_g = self.W_g(r)
        print(f"w_g shape {W_g.shape}")
        W_x = self.W_x(e)
        print(f"w_x shape {W_x.shape}")
        bp = self.W(W_g * W_x)
        print(f"bp shape {bp.shape}")

        # # 对于e分支添加的通道注意力模块(channel attention)
        # t_in = e
        # # print(t_in.shape)  # [1, 128, 64, 64]
        # e = e.mean((2, 3), keepdim=True)
        # # print(x.shape)  # [1, 128, 64, 64]-->[1, 128, 1, 1]
        # e = self.fc1(e)
        # # print(x.shape)  # [1, 128, 1, 1]-->[1, 64, 1, 1]
        # e = self.relu(e)
        # e = self.fc2(e)
        # # print(x.shape)  # [1, 64, 1, 1]-->[1, 128, 1, 1]
        # e = self.sigmoid(e) * t_in
        # # print(e.shape)  # [1, 128, 1, 1]-->[1, 128, 64, 64]

        # 对于r分支添加的空间注意力模块(spatial attention)
        # c_in = r
        # # print(c_in.shape)     # [1, 96, 32, 32]
        # g = self.compress(r)
        # # print(g.shape)        # [1, 96, 32, 32]-->[1, 2, 32, 32]
        # g = self.spatial(g)
        # # print(g.shape)        # [1, 2, 32, 32]-->[1, 1, 32, 32]
        # r = self.sigmoid(g) * c_in
        # # print(r.shape)        # [1, 1, 32, 32]-->[1, 96, 32, 32]

        # 用于替换SEBlock的ECA模块
        e = self.eca(e)
        # print(f"e shape {e.shape}")
        r = self.scse(r)
        # print(f"r shape {r.shape}")

        fuse = self.residual(torch.cat([r, e, bp], 1))

        # 添加Residual模块之后的返回值
        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse


class DoubleConv(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=groups),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, dim, n_class, in_ch=3):
        super().__init__()
        # swin-tiny
        self.encoder = SwinTransformer(depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), drop_path_rate=0.2,
                                       patch_size=4, embed_dim=96)
        # swin-base
        # self.encoder = SwinTransformer(depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), drop_path_rate=0.5,
        #                                patch_size=4, embed_dim=128)

        # swin-small
        # self.encoder = SwinTransformer(depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24), drop_path_rate=0.2,
        #                                patch_size=4, embed_dim=96)

        # Transformer分支使用swin-tiny预训练权重(strict=False时)
        # checkpoint = torch.load('../checkpoints/swin_tiny_patch4_window7_224.pth')
        # self.encoder.load_state_dict(checkpoint, strict=False)

        self.encoder.init_weights('../checkpoints/swin_tiny_patch4_window7_224.pth')

        # CNN分支选择resnet
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        drop_rate = 0.2
        self.drop = nn.Dropout2d(drop_rate)

        # 定义decoder模块,一共5层(包括3层基于swin-transformer的Swin_Decoder, 2层使用nearest插值进行Upsample的Decoder)
        self.layer1 = Swin_Decoder(8 * dim, 2, 8)
        self.layer2 = Swin_Decoder(4 * dim, 2, 4)
        self.layer3 = Swin_Decoder(2 * dim, 2, 2)
        self.layer4 = Decoder(dim, dim, dim // 2)
        self.layer5 = Decoder(dim // 2, dim // 2, dim // 4)

        # 定义使用Conv的下采样模块
        self.down1 = nn.Conv2d(in_ch, dim // 4, kernel_size=1, stride=1, padding=0)
        self.down2 = DoubleConv(dim // 4, dim // 2)
        self.final = nn.Conv2d(dim // 4, n_class, kernel_size=1, stride=1, padding=0)

        # 采用深层监督output1
        self.loss1 = nn.Sequential(
            nn.Conv2d(dim * 8, n_class, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=32)
        )

        # 采用深层监督output2(解码器的最后一层的输出)
        self.loss2 = nn.Sequential(
            nn.Conv2d(dim, n_class, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=4)
        )

        # 采用深度监督swin-transformer编码器分支的stage1
        self.loss3 = nn.Sequential(
            nn.Conv2d(dim, n_class, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Upsample(scale_factor=4)
        )
        # CNN分支的维数
        r_dim = 256          # 原来为dim_s
        # Transformer分支的维数(patch_size=4时)
        e_dim = 96           # 原来为dim_l

        # TIF模块中对于两个不同尺度的分支输出的channel进行concat,其中tb是两个不同尺度输出的channel拼接之后的总通道数
        tb = r_dim + e_dim
        self.change1 = Conv_block(tb, dim)  # 这个dim就是小尺度分支中的输入dim, 即128
        # 其中,每经过一个encoder模块,两个不同尺度分支的size减半,channel数翻倍。所以总通道数也会翻倍
        self.change2 = Conv_block(tb * 2, dim * 2)
        self.change3 = Conv_block(tb * 4, dim * 4)
        self.change4 = Conv_block(tb * 8, dim * 8)

        self.cross_att_1 = Cross_Att(r_dim * 1, e_dim * 1, reduction=1, ch_int=r_dim * 1, ch_out=e_dim * 1, drop_rate=0.1)
        self.cross_att_2 = Cross_Att(r_dim * 2, e_dim * 2, reduction=2, ch_int=r_dim * 2, ch_out=e_dim * 2, drop_rate=0.1)
        self.cross_att_3 = Cross_Att(r_dim * 4, e_dim * 4, reduction=4, ch_int=r_dim * 4, ch_out=e_dim * 4, drop_rate=0.1)
        self.cross_att_4 = Cross_Att(r_dim * 8, e_dim * 8, reduction=8, ch_int=r_dim * 8, ch_out=e_dim * 8, drop_rate=0.1)

    def forward(self, x):
        # Swin-Transformer分支
        out = self.encoder(x)

        # CNN分支/ResNet34分支(先实验的resnet34分支,因为resnet34与resnet50channel数有较大不同,如实验resnet50时要进行修改)
        out_conv = self.resnet.conv1(x)                    # [1, 3, 224, 224]-->[1, 64, 112, 112]
        out_conv = self.resnet.bn1(out_conv)
        out_conv = self.resnet.relu(out_conv)
        out_conv = self.resnet.maxpool(out_conv)           # [1, 64, 112, 112]-->[1, 64, 56, 56]
        # print(f"out_conv shape {out_conv.shape}")

        out3_1 = self.resnet.layer1(out_conv)              # [1, 64, 56, 56]-->[1, 64, 56, 56]
        out3_1 = self.drop(out3_1)
        # print(f"out3_1 shape {out3_1.shape}")

        out3_2 = self.resnet.layer2(out3_1)                # [1, 256, 56, 56]-->[1, 512, 28, 28]
        out3_2 = self.drop(out3_2)
        # print(f"out3_2 shape {out3_2.shape}")

        out3_3 = self.resnet.layer3(out3_2)                # [1, 512, 28, 28]-->[1, 1024, 14, 14]
        out3_3 = self.drop(out3_3)
        # print(f"out3_3 shape {out3_3.shape}")

        out3_4 = self.resnet.layer4(out3_3)                # [1, 1024, 14, 14]-->[2, 2048, 7, 7]
        out3_4 = self.drop(out3_4)
        # print(f"out3_4 shape {out3_4.shape}")

        # Swin-Transformer分支: e1为stage1输出, e2为stage2输出, e3为stage3输出, e4为stage4输出
        e1, e2, e3, e4 = out[0], out[1], out[2], out[3]
        print(f"e1 shape {e1.shape}, e2 shape {e2.shape}, e3 shape {e3.shape}, e4 shape {e4.shape}")

        # 尝试将深度监督之一添加在Swin-Encoder分支中的stage1
        loss3 = self.loss3(e1)
        print(f"loss3 shape {loss3.shape}")

        # CNN分支: r1为stage1输出, r2为stage2输出, r3为stage3输出, r4为stage4输出
        r1, r2, r3, r4 = out3_1, out3_2, out3_3, out3_4
        print(f"r1 shape {out3_1.shape}, r2 shape {out3_2.shape}, r3 shape {out3_3.shape}, r4 shape {out3_4.shape}")

        # 经过TIF模块
        e1_r1 = self.cross_att_1(r1, e1)
        # print(f"e1_r1 shape {e1_r1.shape}")
        e2_r2 = self.cross_att_2(r2, e2)
        # print(f"e2_r2 shape {e2_r2.shape}")
        e3_r3 = self.cross_att_3(r3, e3)
        # print(f"e3_r3 shape {e3_r3.shape}")
        e4_r4 = self.cross_att_4(r4, e4)
        # print(f"e4_r4 shape {e4_r4.shape}")

        # 添加的深度监督一
        loss1 = self.loss1(e4_r4)
        # print(f"loss1 shape {loss1.shape}")

        # ds1、ds2是中间两个下采样模块
        ds1 = self.down1(x)
        # print(f"ds1 shape {ds1.shape}")
        ds2 = self.down2(ds1)
        # print(f"ds2 shape {ds2.shape}")

        # d1、d2、d3是三个基于Swin-Transformer的Decoder模块
        d1 = self.layer1(e4_r4, e3_r3)
        print(f"d1 shape {d1.shape}")
        d2 = self.layer2(d1, e2_r2)
        print(f"d2 shape {d2.shape}")
        d3 = self.layer3(d2, e1_r1)
        print(f"d3 shape {d3.shape}")
        loss2 = self.loss2(d3)
        # print(f"loss2 shape {loss2.shape}")

        # d4、d5就是两个普通的Decoder模块
        d4 = self.layer4(d3, ds2)
        print(f"d4 shape {d4.shape}")
        d5 = self.layer5(d4, ds1)
        print(f"d5 shape {d5.shape}")
        o = self.final(d5)

        # return o, loss1, loss2
        return o, loss1, loss3


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable

    x = Variable(torch.rand(2, 3, 224, 224)).cuda()
    model = UNet(96, 1, in_ch=3).cuda()
    y = model(x)
    # print('Output shape:', y[-1].shape)
    print('Output shape:', y[0].shape)
    # flops, params = profile(model, (x,))
    # print('flops: ', flops, 'params: ', params)

    # 测试Cross_Att的正确性
    # x1 = torch.randn((4, 96, 56, 56))
    # x2 = torch.randn((4, 64, 56, 56))
    # model = Cross_Att(64, 96, 1, 64, 96, drop_rate=0.1)
    # outputs = model(x2, x1)
    # print(outputs.shape)