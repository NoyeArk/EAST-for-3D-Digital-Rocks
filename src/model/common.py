import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


def fill(x):
    b, c, h, w = x.size()
    pad_h = 8 - h % 8
    pad_w = 8 - w % 8
    y = F.pad(x, [0, pad_w, 0, pad_h])
    return y


class SpaceToDepth(nn.Module):
    def __init__(self, bs):
        super().__init__()
        self.bs = bs

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(N, C * (self.bs**2), H // self.bs, W // self.bs)
        return x


class Denoiser(nn.Module):
    def __init__(self, conv, channel, n_feat, act=nn.ReLU(True), bn=False):
        super(Denoiser, self).__init__()
        self.down2 = SpaceToDepth(2)
        self.down4 = SpaceToDepth(4)

        self.top1 = BasicBlock(conv, channel * 16, n_feat, 3, bn=bn)
        self.top2 = ResBlock(conv, n_feat, 3, act=act, bn=bn)
        self.top3 = BasicBlock(conv, n_feat, n_feat, 3, bn=bn)

        self.bottom1 = BasicBlock(conv, channel * 4, n_feat, 3, bn=bn)
        self.bottom_gate = conv(n_feat // 4 + n_feat, n_feat, 1)
        self.bottom2 = ResBlock(conv, n_feat, 3, act=act, bn=bn)
        self.bottom3 = BasicBlock(conv, n_feat, n_feat, 3, bn=bn)

        self.main1 = BasicBlock(conv, channel, n_feat, 3, bn=bn)
        self.main_gate = conv(n_feat + n_feat // 4, n_feat, 1)
        self.main2 = ResBlock(conv, n_feat, 3, act=act, bn=bn)
        self.main3 = BasicBlock(conv, n_feat, n_feat, 3, bn=bn)

        self.end = conv(n_feat, channel, 3)

    def forward(self, x):
        b, c, h, w = x.size()
        x = fill(x)
        top_x = self.down4(x)
        bottom_x = self.down2(x)

        top_x = self.top1(top_x)
        top_x = self.top2(top_x)
        top_x = self.top3(top_x)
        top_x = F.pixel_shuffle(top_x, 2)

        bottom_x = self.bottom1(bottom_x)
        bottom_x = torch.cat((bottom_x, top_x), 1)
        bottom_x = self.bottom_gate(bottom_x)
        bottom_x = self.bottom2(bottom_x)
        bottom_x = self.bottom3(bottom_x)
        bottom_x = F.pixel_shuffle(bottom_x, 2)

        x = self.main1(x)
        x = torch.cat((x, bottom_x), 1)
        x = self.main_gate(x)
        x = self.main2(x)
        x = self.main3(x)

        x = self.end(x)
        x = x[:, :, :h, :w]
        return x


class MeanShift(nn.Conv3d):
    """
    对输入张量执行均值/标准差归一化或反归一化操作的3D卷积层。
    主要用于在网络前后平移、标准化输入值范围，常用于图像增强或复原任务。

    Args:
        rgb_range (float): 输入数值的范围（如255）。
        rgb_mean (tuple): 均值，默认为(0.4516,)。可设置通道均值（如ImageNet: 0.4488, 0.4371, 0.4040）。
        rgb_std (tuple): 标准差，默认为(1.0,)，可设置每个通道的标准差。
        sign (int): -1代表归一化（减均值），+1代表反归一化（加均值）。
    """

    def __init__(
        self, rgb_range, rgb_mean=(0.4516,), rgb_std=(1.0,), sign=-1
    ):  # rgbmean 0.4488, 0.4371, 0.4040; train0.4516

        super(MeanShift, self).__init__(1, 1, kernel_size=1)
        # 将标准差转为Tensor
        std = torch.Tensor(rgb_std)
        # 配置weight为单位阵除以std，实现归一化（或重缩放）
        self.weight.data = torch.eye(1).view(1, 1, 1, 1, 1) / std.view(1, 1, 1, 1, 1)
        # 配置bias为±rgb_range * 均值 / std，实现加/减均值功能
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        # 冻结参数，使该层不参与训练
        for p in self.parameters():
            p.requires_grad = False


def get_mean_std(data):
    r = data[:, 0, :, :]
    g = data[:, 1, :, :]
    b = data[:, 2, :, :]
    r_std, r_mean = torch.std_mean(r)
    g_std, g_mean = torch.std_mean(g)
    b_std, b_mean = torch.std_mean(b)
    return (r_mean, g_mean, b_mean), (r_std, g_std, b_std)


class BasicBlock(nn.Module):
    def __init__(
        self,
        conv,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=False,
        bn=True,
        act=nn.ReLU(True),
    ):
        super(BasicBlock, self).__init__()
        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        self.body = nn.Sequential(*m)

    def forward(self, x):
        return self.body(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        conv,
        n_feats,
        kernel_size,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
        res_scale=1,
    ):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


"""
reference: http://www.multisilicon.com/blog/a25332339.html
"""


class PixelShuffle3d(nn.Module):
    """
    3D版本的像素重组（PixelShuffle）操作，用于将特征图的通道数转换为空间分辨率。
    主要用于上采样阶段，可将通道数映射为空间上的扩大，如(1, C, D, H, W) -> (1, C/r^3, D*r, H*r, W*r)。
    """

    def __init__(self, scale):
        """
        初始化PixelShuffle3d模块。

        Args:
            scale (int): 上采样的放大因子（如2或4）。
        """
        super().__init__()
        self.scale = scale

    def forward(self, input):
        """
        前向传播函数，将输入特征图的通道信息重排至depth/height/width维实现上采样。

        Args:
            input (Tensor): 输入5D张量，形状为(batch_size, channels, D, H, W)，
                            其中channels必须能被scale**3整除。

        Returns:
            Tensor: 上采样后的5D张量，shape为(batch_size, channels//scale**3, D*scale, H*scale, W*scale)
        """
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale**3  # 输出通道数

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        # [B, nOut*r*r*r, D, H, W] -> [B, nOut, r, r, r, D, H, W]
        input_view = input.contiguous().view(
            batch_size,
            nOut,
            self.scale,
            self.scale,
            self.scale,
            in_depth,
            in_height,
            in_width,
        )

        # 把r三个维度分别插入到D, H, W，转换为[B, nOut, D, r, H, r, W, r]
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        # 合并每个空间维度和对应的scale维度，形状变为[B, nOut, D*r, H*r, W*r]
        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 8 * n_feats, 3, bias))
                m.append(PixelShuffle3d(2))
                if bn:
                    m.append(nn.BatchNorm3d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 18 * n_feats, 3, bias))
            m.append(nn.PixelShuffle3d(3))
            if bn:
                m.append(nn.BatchNorm3d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


# Up conv   described in https://distill.pub/2016/deconv-checkerboard/
class Upconv(nn.Sequential):
    def __init__(
        self, scale, n_feats, mode="nearest", act=False, bias=True
    ):  # nearest/trilinear

        # m = [default_conv(n_feats, n_feats, 3, bias=bias)]  # LRConv
        m = []

        if scale == 4:
            m.append(nn.Upsample(scale_factor=(4, 4, 4), mode=mode))
            m.append(default_conv(n_feats, n_feats, 3, bias=bias))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
            elif act == "leakyrelu":
                m.append(nn.LeakyReLU(True))
        elif (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Upsample(scale_factor=(2, 2, 2), mode=mode))
                m.append(default_conv(n_feats, n_feats, 3, bias=bias))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))
                elif act == "leakyrelu":
                    m.append(nn.LeakyReLU(True))
        elif scale == 3:
            m.append(nn.Upsample(scale_factor=(scale, scale, scale), mode=mode))
            m.append(default_conv(n_feats, n_feats, 3, bias=bias))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
            elif act == "leakyrelu":
                m.append(nn.LeakyReLU(True))
        else:
            raise NotImplementedError

        # m.append(default_conv(n_feats, n_feats, 3, bias=bias))  # HRConv

        super(Upconv, self).__init__(*m)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)
