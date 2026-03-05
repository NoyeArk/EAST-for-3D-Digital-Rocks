#!/usr/bin/env python3
"""
从 experiment 下各模型训练目录的 log.txt 中解析 loss 和 PSNR，分别绘制两个图。
"""

from __future__ import annotations

import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False


def parse_log(log_path: Path) -> tuple[list, list, list, list]:
    """
    解析 log.txt，提取每个 epoch 的 loss 和 PSNR

    Returns:
        epochs_loss: epoch 列表（有 loss 的）
        losses: Total loss 列表
        epochs_psnr: epoch 列表（有 PSNR 的）
        psnrs: PSNR 列表
    """
    epochs_loss = []
    losses = []
    epochs_psnr = []
    psnrs = []

    loss_pattern = re.compile(
        r"\[\d+/\d+\]\s+\[Charbonnier: [\d.]+\]\[HF: [\d.]+\]\[Total: ([\d.]+)\]"
    )
    # PSNR 格式: [DRSRD x4] PSNR: X.XXX (Best: Y.YYY @epoch N)
    # 注意：@epoch 是 Best 所在的 epoch，不是当前评估的 epoch。当前评估的 epoch 应为上一个 [Epoch N] 对应的 N
    psnr_pattern = re.compile(r"\[DRSRD x\d+\]\s+PSNR: ([\d.]+)")
    epoch_pattern = re.compile(r"\[Epoch (\d+)\]")

    current_epoch = None
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            loss_match = loss_pattern.search(line)
            if loss_match and current_epoch is not None:
                total_loss = float(loss_match.group(1))
                # 只保留每个 epoch 的最后一个 loss（1600/1600）
                if "1600/1600" in line:
                    epochs_loss.append(current_epoch)
                    losses.append(total_loss)

            psnr_match = psnr_pattern.search(line)
            if psnr_match and current_epoch is not None:
                # 使用 current_epoch：Evaluation 紧接在完成该 epoch 训练之后，此时 current_epoch 即为刚完成的 epoch
                psnr_val = float(psnr_match.group(1))
                epochs_psnr.append(current_epoch)
                psnrs.append(psnr_val)

    return epochs_loss, losses, epochs_psnr, psnrs


def main():
    experiment_dir = Path(__file__).resolve().parent
    train_dirs = sorted(
        [
            d
            for d in experiment_dir.iterdir()
            if d.is_dir() and "train" in d.name.lower()
        ]
    )

    if not train_dirs:
        print("未找到训练目录（名称包含 'train'）")
        return

    # 收集各模型数据
    model_data = {}
    for train_dir in train_dirs:
        log_path = train_dir / "log.txt"
        if not log_path.exists():
            print(f"跳过 {train_dir.name}: 无 log.txt")
            continue

        epochs_loss, losses, epochs_psnr, psnrs = parse_log(log_path)
        if not epochs_loss and not epochs_psnr:
            print(f"跳过 {train_dir.name}: log.txt 中无有效数据")
            continue

        model_name = (
            train_dir.name.replace("train", "").replace("Train", "") or train_dir.name
        )
        model_data[model_name] = {
            "epochs_loss": epochs_loss,
            "losses": losses,
            "epochs_psnr": epochs_psnr,
            "psnrs": psnrs,
        }
        print(
            f"已解析 {model_name}: {len(epochs_loss)} 个 epoch loss, {len(psnrs)} 个 PSNR"
        )

    if not model_data:
        print("无有效数据可绘制")
        return

    # 颜色和线型
    colors = plt.cm.tab10.colors
    linestyles = ["-", "--", "-.", ":"]

    # 图1: Loss vs Epoch
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i, (model_name, data) in enumerate(model_data.items()):
        ax1.plot(
            data["epochs_loss"],
            data["losses"],
            label=model_name,
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            linewidth=1.5,
        )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Total Loss", fontsize=12)
    ax1.set_title("Training Loss vs Epoch", fontsize=14)
    ax1.legend(loc="best", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    fig1.tight_layout()
    loss_fig_path = experiment_dir / "loss_vs_epoch.png"
    fig1.savefig(loss_fig_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {loss_fig_path}")
    plt.close(fig1)

    # 图2: PSNR vs Epoch
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i, (model_name, data) in enumerate(model_data.items()):
        if data["epochs_psnr"] and data["psnrs"]:
            ax2.plot(
                data["epochs_psnr"],
                data["psnrs"],
                label=model_name,
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=1.5,
            )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("PSNR (dB)", fontsize=12)
    ax2.set_title("PSNR vs Epoch", fontsize=14)
    ax2.legend(loc="best", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    fig2.tight_layout()
    psnr_fig_path = experiment_dir / "psnr_vs_epoch.png"
    fig2.savefig(psnr_fig_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {psnr_fig_path}")
    plt.close(fig2)

    print("绘制完成。")


if __name__ == "__main__":
    main()
