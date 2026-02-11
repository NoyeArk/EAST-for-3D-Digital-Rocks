"""
DRSRD / shuffled3D 数据集：目录为 shuffled3D_train_HR、shuffled3D_train_LR_default_X2/X4 等，
HR 文件为 0001.mat，LR 文件为 0001x4.mat（直接在 LR 目录下，无 X4 子目录）。
"""
import os
import glob
from data import srdata


class DRSRD(srdata.SRData):
    def __init__(self, args, name="DRSRD", train=True, benchmark=False):
        data_range = [r.split("-") for r in args.data_range.split("/")]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(DRSRD, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        self.apath = dir_data
        if self.train:
            self.dir_hr = os.path.join(self.apath, "shuffled3D_train_HR")
            self.dir_lr = os.path.join(self.apath, "shuffled3D_train_LR_default_X4")
        else:
            self.dir_hr = os.path.join(self.apath, "shuffled3D_test_HR")
            self.dir_lr = os.path.join(self.apath, "shuffled3D_test_LR_default_X4")
        self.ext = (".mat", ".mat")

    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, "*" + self.ext[0]))
        )
        names_hr = names_hr[self.begin - 1 : self.end]
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                lr_dir = os.path.join(
                    self.apath,
                    "shuffled3D_train_LR_default_X{}".format(s)
                    if self.train
                    else "shuffled3D_test_LR_default_X{}".format(s),
                )
                lr_path = os.path.join(
                    lr_dir, "{}{}{}{}".format(filename, "x", s, self.ext[1])
                )
                names_lr[si].append(lr_path)
        return names_hr, names_lr
