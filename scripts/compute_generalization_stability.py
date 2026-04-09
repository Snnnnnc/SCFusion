#!/usr/bin/env python3
"""
根据 results_new 下 allmix、physio、imu 三个模型的 checkpoint_log.csv，
计算全程（全部 epoch）val_loss / val_acc / val_f1 的 mean 和 std（泛化稳定性），
输出到 results_new 下两个新文件：generalization_stability_full.csv、generalization_stability_full.txt。
"""
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results_new"
COLS = ["val_loss", "val_acc", "val_f1"]
OUT_CSV = "generalization_stability_full.csv"
OUT_TXT = "generalization_stability_full.txt"

# 模型目录名（含 checkpoint_log.csv 的文件夹）
MODEL_DIRS = {
    "allmix": "MotionSickness_AllMixClassificationModel_PhysioFusionNet_v1_allmix_fold1_seed42",
    "physio": "MotionSickness_PhysioClassificationModel_PhysioFusionNet_v1_physio_fold1_seed42",
    "imu": "MotionSickness_IMUClassificationModel_PhysioFusionNet_v1_imu_fold1_seed42",
}


def main():
    rows = []
    for model_name, dir_name in MODEL_DIRS.items():
        log_path = RESULTS_ROOT / dir_name / "checkpoint_log.csv"
        if not log_path.exists():
            print(f"跳过（不存在）: {log_path}")
            continue
        df = pd.read_csv(log_path)
        # 按 epoch 去重保留最后一条（防止 resume 导致重复 epoch 0）
        df = df.drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch")
        if df.empty:
            print(f"跳过（无数据）: {model_name}")
            continue
        for c in COLS:
            if c not in df.columns:
                continue
            mean = df[c].mean()
            std = df[c].std()
            if pd.isna(std):
                std = 0.0
            rows.append({"model": model_name, "指标": c, "mean": mean, "std": std})
        print(f"  {model_name}: 全程 {len(df)} 个 epoch 已统计")

    out_csv = RESULTS_ROOT / OUT_CSV
    out_txt = RESULTS_ROOT / OUT_TXT
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"已写入: {out_csv}")

    # 同时写一份可读的 txt（与表格图一致：指标 / mean / std）
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("泛化稳定性（全程全部 epoch 的 val 指标 mean ± std）\n\n")
        for model_name in MODEL_DIRS:
            sub = df_out[df_out["model"] == model_name]
            if sub.empty:
                continue
            f.write(f"--- {model_name} ---\n")
            f.write("指标\tmean\tstd\n")
            for _, r in sub.iterrows():
                f.write("{}\t{:.4f}\t{:.4f}\n".format(r["指标"], r["mean"], r["std"]))
            f.write("\n")
    print(f"已写入: {out_txt}")


if __name__ == "__main__":
    main()
