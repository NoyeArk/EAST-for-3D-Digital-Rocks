#!/usr/bin/env python3
"""
将 experiment 下各 test 目录中 results-xxx 里的 .mat 文件转换为 .tif，
保存到同目录下的 results-xxx-tif 中。
"""

from pathlib import Path

import scipy.io as scio
import tifffile


def convert_mat_to_tif(mat_path: Path, tif_path: Path, data_key: str = "temp") -> bool:
    """
    将单个 .mat 文件转换为 .tif

    Args:
        mat_path: .mat 文件路径
        tif_path: 输出 .tif 文件路径
        data_key: .mat 中数据变量的键名

    Returns:
        是否转换成功
    """
    try:
        data = scio.loadmat(mat_path)
        if data_key not in data:
            keys = [k for k in data.keys() if not k.startswith("__")]
            data_key = keys[0] if keys else None
            if data_key is None:
                print(f"  跳过 {mat_path.name}: 无有效数据变量")
                return False
        arr = data[data_key]
        tif_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(tif_path, arr)
        return True
    except Exception as e:
        print(f"  错误 {mat_path}: {e}")
        return False


def main():
    experiment_dir = Path(__file__).resolve().parent.parent / "experiment"

    # 只处理 test 目录
    test_dirs = sorted(
        [d for d in experiment_dir.iterdir() if d.is_dir() and "test" in d.name.lower()]
    )

    total_converted = 0
    for test_dir in test_dirs:
        # 查找 results-xxx 目录（排除 results-xxx-tif）
        results_dirs = [
            d
            for d in test_dir.iterdir()
            if d.is_dir()
            and d.name.startswith("results-")
            and not d.name.endswith("-tif")
        ]

        for results_dir in results_dirs:
            mat_files = list(results_dir.glob("*.mat"))
            if not mat_files:
                continue

            # 输出目录: results-DRSRD -> results-DRSRD-tif
            out_dir = test_dir / f"{results_dir.name}-tif"
            out_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"{test_dir.name}/{results_dir.name} -> {out_dir.name}: {len(mat_files)} 个文件"
            )
            for mat_path in mat_files:
                tif_path = out_dir / mat_path.with_suffix(".tif").name
                if convert_mat_to_tif(mat_path, tif_path):
                    total_converted += 1

    print(f"\n共转换 {total_converted} 个文件")


if __name__ == "__main__":
    main()
