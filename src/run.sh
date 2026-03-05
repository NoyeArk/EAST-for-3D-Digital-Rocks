#!/bin/bash

# 训练数据路径，可以根据需要覆盖
data_dir='/home/zql/code/refsr/EAST-for-3D-Digital-Rocks/DRP-211/DRSRD Dataset/DRSRD1_3D/DRSRD1_3D/shuffled3D'

# 用法说明
usage() {
    echo "Usage: $0 [train|test]"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

MODE=$1

# 参数说明
# --dir_data: 数据集目录
# --data_train: 训练数据集
# --data_test: 测试数据集
# --data_range: 数据集范围
# --model: 模型
# --save: 保存模型
# --n_GPUs: 使用GPU数量
# --scale: 缩放比例
# --save_results: 保存结果
# --epochs: 训练轮数
# --batch_size: 批量大小
# --patch_size: 补丁大小
# --warm_up: 预热轮数
# --gclip: 梯度裁剪
# --reset: 重置模型
# --window_sizes: 窗口大小
# --lr: 学习率
# --test_every: 测试间隔
# --n_feats: 特征数量
# --n_resgroups: 残差组数量
# --n_resblocks: 残差块数量
# --print_every: 打印间隔
# --loss: 损失函数
# --optimizer: 优化器
# --noise: 添加噪声z

if [ "$MODE" == "train" ]; then
    # 训练命令（注意：行末 \ 后不能有空格或注释，否则会断开命令）
    CUDA_VISIBLE_DEVICES=0 uv run main.py \
        --dir_data "$data_dir" \
        --data_train DRSRD \
        --data_test DRSRD \
        --data_range '1-1600/1-200' \
        --model SRCNN \
        --save 'SRCNNtrain' \
        --n_GPUs 1 \
        --scale 4 \
        --save_results \
        --epochs 10 \
        --batch_size 8 \
        --patch_size 64 \
        --warm_up 10 \
        --gclip 20 \
        --reset \
        --window_sizes '2-4-8' \
        --lr 1e-4 \
        --test_every 2 \
        --n_feats 180 \
        --n_resgroups 7 \
        --n_resblocks 5 \
        --print_every 50 \
        --loss '1*Charbonnier+2*HF' \
        --optimizer AdamW \
        --noise
elif [ "$MODE" == "test" ]; then
    # 测试命令
    CUDA_VISIBLE_DEVICES=1 uv run main.py \
        --model SRCNN \
        --n_GPUs 1 \
        --dir_data "$data_dir" \
        --scale 4 \
        --pre_train ../experiment/EDSRtrain/model/model_best.pt \
        --save SRCNNtest \
        --data_range '1-200/1-200' \
        --n_feats 180 \
        --n_resgroups 7 \
        --n_resblocks 5 \
        --patch_size 64 \
        --test_only \
        --save_results
else
    usage
fi