import math
import torch.nn as nn
import torch.nn.functional as F

from model import common
from model.east_block import ELAB


def make_model(args, parent=False):
    return EAST(args)


# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CALayer, self).__init__()
        # feature channel downscale and upscale --> channel weight
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_du = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, 1, padding=0, bias=True),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            # common.SiLU(),
            nn.Conv3d(channels // reduction, channels, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg = self.avg_pool(x)
        y = self.conv_du(avg)
        return x * y


class CABlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(CABlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            # common.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.calayer = CALayer(out_channels, reduction=reduction)

    def forward(self, x):
        y = self.calayer(self.conv(x)) + x
        return y


class ResGroup(nn.Module):
    def __init__(self, n_resblocks, channels, r_expand, window_sizes, group_index):
        """
        Args:
            n_resblocks: 残差块数量
            channels: 通道数量
            r_expand: 扩展比例
            window_sizes: 窗口大小
            group_index: 组索引
        """
        super(ResGroup, self).__init__()

        # 通道压缩后用于通道注意力的中间通道数
        ca_channels = channels // 2

        res_group = []
        # 按照设定的残差块数量，循环堆叠ELAB模块
        for j in range(n_resblocks):
            # ELAB模块：每隔一个残差块执行分组位移（偶数索引位移，奇数不变，增强空间关系）
            # j % 2 * group_index ： 仅偶数残差块启用位移，移位步长随group_index变化
            res_group.append(
                ELAB(
                    channels, channels, r_expand, j % 2 * group_index, window_sizes
                )  # i % 2/5, noshift=0；奇数个不动，偶数个位移 j%2*group_index
                # ELAB(channels, channels, r_expand, 0, window_sizes)
            )
        # 1×1卷积进行通道压缩，准备进入通道注意力模块
        res_group.append(nn.Conv3d(channels, ca_channels, kernel_size=1, padding=0))
        # 通道注意力模块，提升网络对有效特征的关注
        res_group.append(CABlock(ca_channels, ca_channels))
        # 1×1卷积恢复通道数，准备与输入张量残差连接
        res_group.append(nn.Conv3d(ca_channels, channels, kernel_size=1, padding=0))
        # 用nn.Sequential串联组成完整的ResGroup
        self.res_group = nn.Sequential(*res_group)

    def forward(self, x):
        res = self.res_group(x)
        return x + res


class EAST(nn.Module):
    def __init__(self, args):
        super(EAST, self).__init__()

        self.scale = args.scale[0]
        self.colors = args.n_colors

        self.window_sizes = list(
            map(lambda x: int(x), args.window_sizes.split("-"))
        )  # args.window_sizes  [2,4,8],[4,8,8]
        self.m_east = args.n_resblocks  # args.m_east   36
        self.g_east = args.n_resgroups  #
        self.c_east = args.n_feats  # args.c_east  180
        self.r_expand = 1  # args.r_expand

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [
            nn.Conv3d(self.colors, self.c_east, kernel_size=3, stride=1, padding=1)
        ]

        # define body module
        m_body = []  # add cablock

        for i in range(self.g_east):
            m_body.append(
                ResGroup(
                    self.m_east,  # 残差块数量
                    self.c_east,  # 通道数量
                    self.r_expand,  # 扩展比例
                    self.window_sizes,  # 窗口大小
                    i % 4 + 1,  # 组索引
                )
            )  # 1,2,3,4
        # m_body.append(nn.Conv3d(self.c_east, self.c_east, kernel_size=3, stride=1, padding=1))
        # m_body.append(nn.Conv3d(self.c_east, self.c_east, kernel_size=1, stride=1, padding=0))
        # m_body.append(CABlock(self.c_east, self.c_east, reduction=16))  # add cablock

        # define tail module
        m_tail = [
            nn.Conv3d(
                self.c_east,
                args.n_colors * self.scale**3,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            common.PixelShuffle3d(self.scale),
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (Tensor): 输入的5D张量, 形状为 (batch, channels, depth, height, width)

        Returns:
            Tensor: 超分辨率输出, 截取至输入的缩放后空间尺寸，形状为(batch, channels, D*scale, H*scale, W*scale)
        """
        D, H, W = x.shape[2:]  # 记录输入的深度、高度、宽度
        x = self.check_image_size(x)  # 保证输入大小可以整除window_size（必要padding）

        x = self.sub_mean(x)  # 去均值归一化
        x = self.head(x)  # 头部卷积

        res = self.body(x)  # 通过主体网络（多个残差组）
        res = res + x  # 残差连接(classic residual learning)

        x = self.tail(res)  # 尾部卷积+PixelShuffle3d 实现上采样
        x = self.add_mean(x)  # 恢复均值

        # 网络输出的空间尺寸可能大于scale倍的输入，为保持和输入一致，需裁剪回原始scale倍尺寸
        return x[:, :, 0 : D * self.scale, 0 : H * self.scale, 0 : W * self.scale]

    def check_image_size(self, x):
        """
        检查并将输入张量x的空间尺寸pad到能够被所有window_size整除，以便后续分块处理。

        Args:
            x (Tensor): 输入5D张量，维度为(batch, channels, depth, height, width)

        Returns:
            Tensor: 如果输入尺寸不能被总window_size整除，则在depth/height/width后三维进行padding
        Padding mode: 'replicate'方式，即用边缘值复制填充
        """

        _, _, d, h, w = x.size()
        # 计算所有window_size的最小公倍数作为目标块大小
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            # 累乘并约去公约数，得到最终的最小公倍数
            wsize = (
                wsize * self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
            )

        # 计算需要pad的数量（如果能整除则不pad，否则补齐到能整除为止）
        mod_pad_d = (wsize - d % wsize) % wsize
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize

        # 填充的顺序：(左pad, 右pad) - pytorch pad顺序是(w前, w后, h前, h后, d前, d后)
        # 这里只在每一维度的tail（后面）填充，不在开头pad
        # 填充模式'replicate': 用边界值填充  例123->(123333)
        # 也可用'circular'(环形填充 123->123123)、'reflect'(镜像填充123->12321)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d), "replicate")
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find("tail") == -1:
                        raise RuntimeError(
                            "While copying the parameter named {}, "
                            "whose dimensions in the model are {} and "
                            "whose dimensions in the checkpoint are {}.".format(
                                name, own_state[name].size(), param.size()
                            )
                        )
            elif strict:
                if name.find("tail") == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))

        # normmean = 49.6082 # 194.4797
        # # 修改偏置参数的值
        # submean = torch.full_like(own_state['sub_mean.bias'].data, -normmean)
        # own_state['sub_mean.bias'].copy_(submean)
        # # 修改偏置参数的值
        # addmean = torch.full_like(own_state['add_mean.bias'].data, normmean)
        # own_state['add_mean.bias'].copy_(addmean)
        # # print(own_state['sub_mean.bias'].requires_grad)
        # print(own_state['sub_mean.bias'])
