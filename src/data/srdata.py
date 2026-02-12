import os
import glob
import random
import pickle
import h5py
from data import common

import numpy as np
import imageio
import torch
import torch.utils.data as data


class SRData(data.Dataset):
    def __init__(self, args, name="", train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = "train" if train else "test"
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = args.model == "VDSR"
        self.scale = args.scale
        self.idx_scale = 0

        self._set_filesystem(args.dir_data)  # set filesystem structure

        # if file extension is not img, creat binary file folder
        if args.ext.find("img") < 0:
            path_bin = os.path.join(self.apath, "bin")
            os.makedirs(path_bin, exist_ok=True)
            self.ext = (".mat", ".mat")

        list_hr, list_lr = self._scan()  # hrname & lrname list

        # if read image return filename list, else convert all files to binary file
        if args.ext.find("img") >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find("sep") >= 0:
            os.makedirs(self.dir_hr.replace(self.apath, path_bin), exist_ok=True)
            for s in self.scale:
                os.makedirs(
                    os.path.join(
                        self.dir_lr.replace(self.apath, path_bin), "X{}".format(s)
                    ),
                    exist_ok=True,
                )

            self.images_hr, self.images_lr = [], [[] for _ in self.scale]

            # 遍历所有 HR 图像文件路径，将其转换为二进制 .pt 文件路径并保存
            for h in list_hr:
                # 将原始 HR 路径替换为 bin 目录，并把后缀替换为 .pt（PyTorch 二进制格式）
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], ".pt")
                self.images_hr.append(b)  # 添加到 HR 二进制文件列表
                # 检查 .pt 文件是否存在，如不存在则从原始文件读取内容并保存为二进制 .pt 文件
                self._check_and_load(args.ext, h, b, verbose=True)

            for i, ll in enumerate(list_lr):
                for l in ll:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], ".pt")
                    self.images_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True)

        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, "*" + self.ext[0]))
        )  # list all hrfiles' name
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                names_lr[si].append(
                    os.path.join(
                        self.dir_lr, "X{}/{}x{}{}".format(s, filename, s, self.ext[1])
                    )
                )

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, "HR")
        self.dir_lr = os.path.join(self.apath, "LR")  # (self.apath, 'LR_bicubic')
        # if self.input_large: self.dir_lr += 'L'
        self.ext = (".JPEG", ".JPEG")
        # self.ext = ('.mat', '.mat')  # '.png'

    # if the files are binary, load them
    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find("reset") >= 0:
            if verbose:
                print("Making a binary: {}".format(f))
            with open(f, "wb") as _f:
                # pickle.dump(imageio.imread(img), _f)
                pickle.dump(h5py.File(img, "r")["temp"][:], _f)

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        pair = self.get_patch(lr, hr)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)
        return pair_t[0], pair_t[1], filename

    def __len__(self):
        if self.train:
            return len(self.images_hr)
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        # 获取当前索引（如果是训练集，支持循环取模）
        idx = self._get_index(idx)

        # 获取对应的高分辨率文件(hr)和低分辨率文件(lr)路径
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        # 提取数据文件的主文件名，不含扩展名
        filename, _ = os.path.splitext(os.path.basename(f_hr))

        # 如果扩展名为'img'或为benchmark数据集，则读取原始图片文件
        if self.args.ext == "img" or self.benchmark:
            hr = imageio.imread(f_hr)
            lr = imageio.imread(f_lr)
            # 将二维图片扩展为三维数组：仿造补齐为3D立体体积，第三维度为patch_size（hr）或patch_size//scale（lr），内容完全一致
            hr = np.tile(hr[np.newaxis, :, :], (self.args.patch_size, 1, 1))
            lr = np.tile(
                lr[np.newaxis, :, :],
                (self.args.patch_size // self.scale[self.idx_scale], 1, 1),
            )
        # 如果扩展方式为'sep...'（即二进制文件），用pickle读取
        elif self.args.ext.find("sep") >= 0:
            # 以二进制形式加载高分辨率数据
            with open(f_hr, "rb") as _f:
                hr = pickle.load(_f)
            # 以二进制形式加载低分辨率数据
            with open(f_lr, "rb") as _f:
                lr = pickle.load(_f)

        # 返回低分辨率、高清、文件名
        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        if self.train:
            lr, hr = common.get_patch(
                lr,
                hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
            )
            if not self.args.no_augment:
                lr, hr = common.augment(lr, hr)
            if self.args.noise:
                lr = common.add_noise(lr)
            # print(lr[0])
            # imageio.imwrite('noiselr.png', lr[0])
        else:
            ic, ih, iw = lr.shape
            # lr = common.add_noise(lr)  # selfsupervised训练和测试加
            hr = hr[0 : ic * scale, 0 : ih * scale, 0 : iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)
