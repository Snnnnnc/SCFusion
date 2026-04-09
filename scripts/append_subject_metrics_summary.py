#!/usr/bin/env python3
"""
在 results_new 下每个模型的 subject_metrics.csv 末尾追加：
按被试的 acc / precision / f1 / recall 的 均值、方差、最大值、最小值。
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results_new"
METRICS = ["accuracy", "precision", "recall", "f1"]


def process_one_csv(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    if df.empty or not all(m in df.columns for m in METRICS):
        print(f"  跳过（无数据或缺少列）: {csv_path}")
        return

    rows = []
    for stat_name, func in [
        ("mean", np.mean),
        ("var", np.var),
        ("min", np.min),
        ("max", np.max),
    ]:
        row = {"subject_id": stat_name}
        for m in METRICS:
            row[m] = func(df[m])
        if "num_samples" in df.columns:
            row["num_samples"] = ""
        rows.append(row)

    # 保持与原有列一致
    out_df = pd.DataFrame(rows)
    out_df = out_df[[c for c in df.columns if c in out_df]]
    df_combined = pd.concat([df, out_df], ignore_index=True)
    df_combined.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  ✓ 已追加汇总行: {csv_path.name}")


def main():
    if not RESULTS_ROOT.exists():
        print(f"目录不存在: {RESULTS_ROOT}")
        return
    for d in sorted(RESULTS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        csv_path = d / "subject_metrics.csv"
        if not csv_path.exists():
            continue
        print(f"处理: {d.name}")
        process_one_csv(csv_path)
    print("完成。")


if __name__ == "__main__":
    main()
