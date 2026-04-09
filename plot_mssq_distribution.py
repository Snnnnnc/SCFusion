#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""绘制 MSSQ 得分分布：直方图 + KDE 曲线"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# 设置中文字体（macOS 常用）
import matplotlib
for font in ["PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS"]:
    if font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
        plt.rcParams["font.sans-serif"] = [font]
        break
plt.rcParams["axes.unicode_minus"] = False

def main():
    df = pd.read_csv("被试记录_MSSQ得分.csv", encoding="utf-8-sig")
    scores = df["MSSQ_S_总分"].dropna().values
    n = len(scores)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # 直方图
    counts, bins, patches = ax.hist(
        scores,
        bins=12,
        density=True,
        color="steelblue",
        alpha=0.65,
        edgecolor="white",
        linewidth=0.8,
        label="直方图",
    )
    # KDE
    kde = stats.gaussian_kde(scores)
    x_kde = np.linspace(scores.min(), scores.max(), 200)
    ax.plot(
        x_kde,
        kde(x_kde),
        color="crimson",
        linewidth=2,
        label="KDE 曲线",
    )
    ax.set_xlabel("MSSQ-S 总分", fontsize=11)
    ax.set_ylabel("密度", fontsize=11)
    ax.set_title(f"MSSQ 得分分布 (n={n})", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig("MSSQ_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("已保存: MSSQ_score_distribution.png")

if __name__ == "__main__":
    main()
