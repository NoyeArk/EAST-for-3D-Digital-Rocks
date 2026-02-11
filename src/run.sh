#!/bin/bash

# 训练数据路径，可以根据需要覆盖
data_dir='/home/zql/code/EAST-for-3D-Digital-Rocks/DRP-211/DRSRD Dataset/DRSRD1_3D/DRSRD1_3D/shuffled3D'

# 用法说明
usage() {
    echo "Usage: $0 [train|test]"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

MODE=$1

if [ "$MODE" == "train" ]; then
    # 训练命令
    CUDA_VISIBLE_DEVICES=1 uv run main.py \
        --dir_data "$data_dir" \          # 训练和测试数据的主目录
        --data_train DRSRD \              # 指定用于训练的数据集名称
        --data_test DRSRD \               # 指定用于测试的数据集名称
        --data_range '1-1600/1-200' \     # 训练/测试使用的数据序号范围（训练1-1600，测试1-200）
        --model EAST \                    # 使用的模型名称（EAST）
        --save '0211train' \              # 结果保存文件夹名
        --n_GPUs 1 \                      # 使用的GPU数量
        --scale 4 \                       # 超分辨放大倍数
        --save_results \                  # 是否保存结果
        --epochs 10 \                     # 总训练轮数
        --batch_size 8 \                  # 批量大小
        --patch_size 64 \                 # 输入patch的尺寸
        --warm_up 10 \                    # warm up步数，如果为0则不使用
        --gclip 20 \                      # 梯度裁剪阈值（0为不裁剪）
        --reset \                         # 是否重置训练/重新开始
        --window_sizes 2-4-8 \            # 滑动窗口的尺寸配置
        --lr 1e-4 \                       # 学习率
        --test_every 2 \                  # 每多少个epoch进行一次测试
        --n_feats 180 \                   # 特征层数
        --n_resgroups 7 \                 # 残差组数量
        --n_resblocks 5 \                 # 每组的残差块数量
        --print_every 50 \                # 每多少次输出一次训练状态
        --loss 1*Charbonnier+2*HF \       # 损失函数类型及权重
        --optimizer AdamW \               # 优化器类型
        --noise                           # 是否在输入中加噪声数据增强
elif [ "$MODE" == "test" ]; then
    # 测试命令
    CUDA_VISIBLE_DEVICES=1 uv run main.py \
        --model EAST \                            # 使用的模型名称（EAST）
        --n_GPUs 1 \                              # 使用的GPU数量
        --dir_data "$data_dir" \                  # 测试数据主目录
        --scale 4 \                               # 超分辨放大倍数
        --pre_train ../models/model_best.pt \     # 预训练模型文件路径
        --save 0211test \                         # 测试结果保存文件夹
        --data_range 1-300/1-300 \                # 测试用数据范围（训练/测试）
        --n_feats 180 \                           # 特征层数
        --n_resgroups 7 \                         # 残差组数量
        --n_resblocks 5 \                         # 每组残差块数量
        --patch_size 64 \                         # 输入patch的尺寸
        --test_only \                             # 只进行测试
        --save_results                            # 保存测试结果
else
    usage
fi