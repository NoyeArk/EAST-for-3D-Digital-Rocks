"""
DRSRD / shuffled3D 数据集：目录为 shuffled3D_train_HR、shuffled3D_train_LR_default_X2/X4 等，
HR 文件为 0001.mat，LR 文件为 0001x4.mat（直接在 LR 目录下，无 X4 子目录）。
"""

import os
import glob
from data import srdata


class DRSRD(srdata.SRData):
    def __init__(self, args, name="DRSRD", train=True, benchmark=False):
        # 解析数据范围参数，支持类似 "1-800/1-100" 的格式
        data_range = [r.split("-") for r in args.data_range.split("/")]
        if train:
            data_range = data_range[0]  # 训练时用第一个区间
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]  # 只测试时且只有一个区间，直接用
            else:
                data_range = data_range[1]  # 测试时用第二个区间（如果有）

        # 转换为起始和结束索引
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        # 初始化父类 SRData
        super(DRSRD, self).__init__(args, name=name, train=train, benchmark=benchmark)

    def _set_filesystem(self, dir_data):
        # 设置数据根目录
        self.apath = dir_data
        # HR 和 LR 目录根据 train/test 区分
        if self.train:
            self.dir_hr = os.path.join(self.apath, "shuffled3D_train_HR")
            self.dir_lr = os.path.join(self.apath, "shuffled3D_train_LR_default_X4")
        else:
            self.dir_hr = os.path.join(self.apath, "shuffled3D_test_HR")
            self.dir_lr = os.path.join(self.apath, "shuffled3D_test_LR_default_X4")
        # 文件扩展名（高低分辨率用 .mat 格式）
        self.ext = (".mat", ".mat")

    def _scan(self):
        # 获取全部 HR 文件路径并排序
        names_hr = sorted(glob.glob(os.path.join(self.dir_hr, "*" + self.ext[0])))
        # 按 begin 和 end 裁剪（索引从1开始）
        names_hr = names_hr[self.begin - 1 : self.end]
        # 为每种 scale 创建独立的 LR 路径列表
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            # 取 HR 文件名（无扩展名）
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                # 构造对应的 LR 目录（支持多倍率）
                lr_dir = os.path.join(
                    self.apath,
                    (
                        "shuffled3D_train_LR_default_X{}".format(s)
                        if self.train
                        else "shuffled3D_test_LR_default_X{}".format(s)
                    ),
                )
                # 目标 LR 文件名示例：0001x4.mat
                lr_path = os.path.join(
                    lr_dir, "{}{}{}{}".format(filename, "x", s, self.ext[1])
                )
                names_lr[si].append(lr_path)
        # 返回 HR 路径列表和 LR 列表（嵌套，分别对应不同 scale）
        return names_hr, names_lr
