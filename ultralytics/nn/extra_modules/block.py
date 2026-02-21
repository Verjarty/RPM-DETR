import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.modules.utils import _pair

from torch import Tensor
from torch.jit import Final
import math
import numpy as np
from functools import partial

from typing import Optional, Callable, Optional, Dict, Union, List
from einops import rearrange, reduce
from collections import OrderedDict


from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose
from ..modules.block import get_activation, ConvNormLayer, BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck
from .upsample import FAU

from ultralytics.utils.torch_utils import fuse_conv_and_bn, make_divisible

__all__ = ['Ghost_HGBlock', 'Rep_HGBlock',
            'BasicBlock_DualConv', 'DRFD',
           'FAU', 'PrismChannelAttentionModule',
           'BasicBlock_DualConv_PCA','BasicBlock_PCA'
           ]

######################################## HGBlock with RepConv and GhostConv start ########################################

class Ghost_HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = GhostConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y

class RepLightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = RepConv(c2, c2, k, g=math.gcd(c1, c2), act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))

class Rep_HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = RepLightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y

######################################## HGBlock with RepConv and GhostConv end ########################################


######################################## DualConv start ########################################

class DualConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, g=4):
        """
        Initialize the DualConv class.
        :param input_channels: the number of input channels
        :param output_channels: the number of output channels
        :param stride: convolution stride
        :param g: the value of G used in DualConv
        """
        super(DualConv, self).__init__()
        # Group Convolution
        self.gc = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=g, bias=False)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, input_data):
        """
        Define how DualConv processes the input images or input feature maps.
        :param input_data: input images or input feature maps
        :return: return output feature maps
        """
        return self.gc(input_data) + self.pwc(input_data)

class EDLAN(nn.Module):
    def __init__(self, c, g=4) -> None:
        super().__init__()
        self.m = nn.Sequential(DualConv(c, c, 1, g=g), DualConv(c, c, 1, g=g))
    
    def forward(self, x):
        return self.m(x)

class CSP_EDLAN(nn.Module):
    # CSP Efficient Dual Layer Aggregation Networks
    def __init__(self, c1, c2, n=1, g=4, e=0.5) -> None:
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(EDLAN(self.c, g=g) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class BasicBlock_DualConv(BasicBlock):
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        
        self.branch2b = DualConv(ch_out, ch_out, g=4)

######################################## DualConv end ########################################

######################################## Deep feature downsampling start ########################################

class Cut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)     # x = [B, out_channels, H/2, W/2]
        x = self.batch_norm(x)
        return x

class SRFD(nn.Module):
    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        out_c14 = int(out_channels / 4)  # out_channels / 4
        out_c12 = int(out_channels / 2)  # out_channels / 2

        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        self.conv_init = nn.Conv2d(in_channels, out_c14, kernel_size=7, stride=1, padding=3)

        # original size to 2x downsampling layer
        self.conv_1 = nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14)
        self.conv_x1 = nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12)
        self.batch_norm_x1 = nn.BatchNorm2d(out_c12)
        self.cut_c = Cut(out_c14, out_c12)
        self.fusion1 = nn.Conv2d(out_channels, out_c12, kernel_size=1, stride=1)

        # 2x to 4x downsampling layer
        self.conv_2 = nn.Conv2d(out_c12, out_channels, kernel_size=3, stride=1, padding=1, groups=out_c12)
        self.conv_x2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.batch_norm_x2 = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.cut_r = Cut(out_c12, out_channels)
        self.fusion2 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        x = self.conv_init(x)  # x = [B, C/4, H, W]

    # original size to 2x downsampling layer
        c = x                   # c = [B, C/4, H, W]
        # CutD
        c = self.cut_c(c)       # c = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]
        # ConvD
        x = self.conv_1(x)      # x = [B, C/4, H, W] --> [B, C/2, H/2, W/2]
        x = self.conv_x1(x)     # x = [B, C/2, H/2, W/2]
        x = self.batch_norm_x1(x)
        # Concat + conv
        x = torch.cat([x, c], dim=1)    # x = [B, C, H/2, W/2]
        x = self.fusion1(x)     # x = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]

    # 2x to 4x downsampling layer
        r = x                   # r = [B, C/2, H/2, W/2]
        x = self.conv_2(x)      # x = [B, C/2, H/2, W/2] --> [B, C, H/2, W/2]
        m = x                   # m = [B, C, H/2, W/2]
        # ConvD
        x = self.conv_x2(x)     # x = [B, C, H/4, W/4]
        x = self.batch_norm_x2(x)
        # MaxD
        m = self.max_m(m)       # m = [B, C, H/4, W/4]
        m = self.batch_norm_m(m)
        # CutD
        r = self.cut_r(r)       # r = [B, C, H/4, W/4]
        # Concat + conv
        x = torch.cat([x, r, m], dim=1)  # x = [B, C*3, H/4, W/4]
        x = self.fusion2(x)     # x = [B, C*3, H/4, W/4] --> [B, C, H/4, W/4]
        return x                # x = [B, C, H/4, W/4]

# Deep feature downsampling
class DRFD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cut_c = Cut(in_channels=in_channels, out_channels=out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv_x = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.act_x = nn.GELU()
        self.batch_norm_x = nn.BatchNorm2d(out_channels)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):       # input: x = [B, C, H, W]
        c = x                   # c = [B, C, H, W]
        x = self.conv(x)        # x = [B, C, H, W] --> [B, 2C, H, W]
        m = x                   # m = [B, 2C, H, W]

        # CutD
        c = self.cut_c(c)       # c = [B, C, H, W] --> [B, 2C, H/2, W/2]

        # ConvD
        x = self.conv_x(x)      # x = [B, 2C, H, W] --> [B, 2C, H/2, W/2]
        x = self.act_x(x)
        x = self.batch_norm_x(x)

        # MaxD
        m = self.max_m(m)       # m = [B, 2C, H/2, W/2]
        m = self.batch_norm_m(m)

        # Concat + conv
        x = torch.cat([c, x, m], dim=1)  # x = [B, 6C, H/2, W/2]
        x = self.fusion(x)      # x = [B, 6C, H/2, W/2] --> [B, 2C, H/2, W/2]

        return x                # x = [B, 2C, H/2, W/2]

######################################## Deep feature downsampling end ########################################

################################# PrismChannelAttentionModule start ##################################################
class PrismChannelAttentionModule(nn.Module):
    """
    一个融合 GAP, GMP, 和标准差 (Std Dev) 的通道注意力模块。

    流程:
    1. 并行计算 Global Average Pooling (GAP), Global Max Pooling (GMP), 和 Standard Deviation。
    2. 三个统计量分别通过独立的全连接层（用1x1卷积实现）将维度降至 C/2。
    3. 将三个降维后的向量拼接在一起，形成一个 C*1.5 维的向量。
    4. 通过最终的全连接层将维度恢复至 C。
    5. 使用 Sigmoid 生成注意力权重，并与原特征图相乘。
    """
    def __init__(self, channels):
        super(PrismChannelAttentionModule, self).__init__()

        if channels <= 1:
            raise ValueError("通道数必须大于1")
            
        # 降维后的通道数
        reduced_channels = channels // 2
        # 防止 reduced_channels 为 0
        if reduced_channels < 1:
            reduced_channels = 1
        
        # 1. GAP, GMP 的池化层
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 2. 为每个统计量准备独立的降维全连接层
        self.fc_gap = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.fc_gmp = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)
        self.fc_std = nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False)

        # 3. 拼接后的升维全连接层
        self.final_fc = nn.Conv2d(reduced_channels * 3, channels, kernel_size=1, bias=False)
        
        # 4. 激活函数
        self.relu = nn.Hardswish(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()

        # --- 步骤 1: 并行特征提取 ---
        # 全局平均池化
        y_avg = self.avg_pool(x)
        # 全局最大池化
        y_max = self.max_pool(x)
        # 标准差: 在 H 和 W 维度上计算
        # keepdim=True 保持 [B, C, 1, 1] 的形状以适配1x1卷积
        y_std = torch.std(x, dim=[2, 3], keepdim=True)

        # --- 步骤 2: 独立降维 ---
        y_avg_reduced = self.fc_gap(y_avg)
        y_max_reduced = self.fc_gmp(y_max)
        y_std_reduced = self.fc_std(y_std)

        # --- 步骤 3: 特征拼接 ---
        y_cat = torch.cat([y_avg_reduced, y_max_reduced, y_std_reduced], dim=1)
        
        # --- 步骤 4: 信息融合与升维 ---
        # 在融合前加入非线性激活，让模型学习更复杂的关系
        y_cat_activated = self.relu(y_cat)
        y_fused = self.final_fc(y_cat_activated)

        # --- 步骤 5 & 6: 生成注意力权重并与原特征图相乘 ---
        attention_weights = self.sigmoid(y_fused)
        
        # 使用广播机制 (broadcasting) 将权重应用到原特征图上
        return x * attention_weights
    
################################# PrismChannelAttentionModule end ##################################################
###################################### BasicBlock_PCA start ###################################################

from ultralytics.nn.modules.block import BasicBlock

class BasicBlock_PCA(BasicBlock):
    
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.pca=PrismChannelAttentionModule(ch_out)
    
    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.pca(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = out + short
        out = self.act(out)

        return out
        
############################################ BasicBlock_PCA end ###################################################
################################## BasicBlock_DualConv_PCA start ###################################################

from ultralytics.nn.extra_modules.block import BasicBlock_DualConv

class BasicBlock_DualConv_PCA(BasicBlock_DualConv):
    
    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__(ch_in, ch_out, stride, shortcut, act, variant)
        self.pca=PrismChannelAttentionModule(ch_out)
    
    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.pca(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        out = out + short
        out = self.act(out)

        return out
        
################################## BasicBlock_DualConv_PCA end ###################################################
