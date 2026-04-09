#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析原始数据中评分的分布情况。
从 data/xxx/xxx/evt.bdf 读取事件，提取评分（编码 10–19 → 评分 0–9），统计并绘制评分分布。
依赖：mne, numpy, pandas, matplotlib, scipy
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

try:
    import mne
except ImportError:
    mne = None

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent


def _read_annotations_bdf(tal_data):
    """从 BDF TAL 原始数据解析出 (onset, duration, description)，与 neuracle read_annotations_bdf 一致。"""
    pat = re.compile(r'([+-]\d+\.?\d*)(\x15(\d+\.?\d*))?(\x14.*?)\x14\x00')
    tals = bytearray()
    for chan in tal_data:
        this_chan = np.asarray(chan).ravel()
        if this_chan.dtype == np.int32:
            this_chan = this_chan.view(np.uint8).reshape(-1, 4)[:, :3].ravel()
            tals.extend(this_chan.tolist())
        else:
            for s in this_chan:
                i = int(s)
                tals.extend([i % 256, i // 256])
    triggers = pat.findall(tals.decode('latin-1', errors='ignore'))
    events = []
    for ev in triggers:
        onset = float(ev[0])
        duration = float(ev[2]) if ev[2] else 0.0
        for description in ev[3].split('\x14')[1:]:
            if description:
                try:
                    events.append([onset, duration, int(description)])
                except ValueError:
                    pass
    return events


def read_events_from_evt_bdf(evt_path: Path, data_srate: int = 1000):
    """
    仅从 evt.bdf 读取事件（不读 data.bdf）。
    返回 list of [onset_sample_idx, duration, code]；若需与 data 对齐，data 采样率默认为 1000。
    """
    if mne is None:
        return []
    evt_path = Path(evt_path)
    if not evt_path.exists():
        return []
    try:
        raw_evt = mne.io.read_raw_bdf(str(evt_path), preload=False, verbose=False)
        n_times = int(raw_evt.n_times)
        try:
            tal_data = raw_evt._read_segment_file(
                [], [], 0, 0, n_times, None, None
            )
        except Exception:
            idx = np.empty(0, int)
            tal_data = raw_evt._read_segment_file(
                np.empty((0, n_times)), idx, 0, 0, n_times, np.ones((len(idx), 1)), None
            )
        events_list = _read_annotations_bdf(tal_data[0])
        # onset 在 TAL 里是秒，转为采样点（与 data.bdf 同率）
        out = []
        for onset_sec, duration, code in events_list:
            onset_sample = int(round(onset_sec * data_srate))
            out.append([onset_sample, int(duration), int(code)])
        return out
    except Exception:
        return []


def extract_rating_scores(events: list):
    """从事件列表中提取评分：编码 10–19 → 评分 0–9。"""
    scores = []
    for _onset, _dur, code in events:
        if 10 <= code <= 19:
            scores.append(code % 10)
    return scores


def find_last_start_onset(events: list):
    """最后一个事件 100（实验开始）的 onset（采样点）。"""
    for i in range(len(events) - 1, -1, -1):
        if events[i][2] == 100:
            return events[i][0]
    return None


def find_evt_bdf_paths(data_root: Path):
    """枚举 data_root 下所有 evt.bdf 路径。"""
    data_root = Path(data_root)
    return sorted(data_root.rglob("evt.bdf"))


def load_all_ratings(data_root: Path, trim_after_start: bool = True, data_srate: int = 1000, data_source_label: str = None):
    """
    从 data_root 下所有 evt.bdf 读取事件，提取评分（可选：仅保留“开始事件 100”之后的评分）。
    返回 (per_session: [(path, scores, data_source)], all_scores: list)。
    """
    data_root = Path(data_root)
    evt_paths = find_evt_bdf_paths(data_root)
    label = data_source_label if data_source_label else str(data_root)
    per_session = []
    all_scores = []
    for evt_path in evt_paths:
        events = read_events_from_evt_bdf(evt_path, data_srate=data_srate)
        if not events:
            continue
        if trim_after_start:
            start_onset = find_last_start_onset(events)
            if start_onset is not None:
                events = [e for e in events if e[0] >= start_onset]
        scores = extract_rating_scores(events)
        if not scores:
            continue
        try:
            rel = evt_path.relative_to(data_root)
        except ValueError:
            rel = evt_path
        per_session.append((str(rel), scores, label))
        all_scores.extend(scores)
    return per_session, all_scores


def plot_rating_distribution(all_scores: list, output_path: Path, n_sessions: int):
    """绘制评分分布：直方图 + KDE。"""
    scores = np.array(all_scores, dtype=float)
    if scores.size == 0:
        print("无评分数据，跳过绘图")
        return
    for font in ["PingFang SC", "Heiti SC", "STHeiti", "SimHei", "Arial Unicode MS"]:
        if font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
            plt.rcParams["font.sans-serif"] = [font]
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(-0.5, 10, 1)
    ax.hist(
        scores,
        bins=bins,
        density=True,
        color="steelblue",
        alpha=0.65,
        edgecolor="white",
        linewidth=0.8,
        label="直方图",
    )
    kde = stats.gaussian_kde(scores)
    x_kde = np.linspace(0, 9, 200)
    ax.plot(x_kde, kde(x_kde), color="crimson", linewidth=2, label="KDE 曲线")
    ax.set_xlabel("评分", fontsize=11)
    ax.set_ylabel("密度", fontsize=11)
    ax.set_title(f"原始数据评分分布（共 {len(scores)} 个评分，{n_sessions} 个 session）", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="从 evt.bdf 读取评分事件并统计分布，支持多数据路径合并")
    parser.add_argument(
        "extra_roots",
        nargs="*",
        default=[],
        help="额外数据根目录，与项目 data/ 合并统计（例如: /Volumes/LENOVO_USB_HDD/MS/data）",
    )
    args = parser.parse_args()

    if mne is None:
        print("请先安装 mne: pip install mne")
        return

    # 数据根：项目 data/ + 用户指定的额外路径
    roots = [PROJECT_ROOT / "data"]
    for r in args.extra_roots:
        p = Path(r)
        if p.exists():
            roots.append(p)
        else:
            print(f"警告: 路径不存在，已跳过: {r}")

    all_per_session = []
    all_scores = []
    by_source = []  # (label, per_session, scores) 用于分源统计

    for data_root in roots:
        label = "project_data" if data_root == PROJECT_ROOT / "data" else str(data_root)
        if not data_root.exists():
            print(f"跳过不存在目录: {data_root}")
            continue
        print(f"正在扫描: {data_root} …")
        per_session, scores = load_all_ratings(data_root, trim_after_start=True, data_source_label=label)
        n = len(per_session)
        total_s = len(scores)
        print(f"  -> {n} 个 session, {total_s} 条评分")
        all_per_session.extend(per_session)
        all_scores.extend(scores)
        by_source.append((label, per_session, scores))

    n_sessions = len(all_per_session)
    if not all_scores:
        print("未提取到任何评分，请确认各数据目录下存在 evt.bdf 且包含评分事件（编码 10–19）。")
        return

    # 合并分布统计
    counter = Counter(all_scores)
    total = len(all_scores)
    score_order = sorted(counter.keys())
    dist_rows = [
        {"评分": s, "次数": counter[s], "占比": counter[s] / total}
        for s in score_order
    ]
    dist_df = pd.DataFrame(dist_rows)
    out_csv = PROJECT_ROOT / "raw_rating_distribution.csv"
    dist_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n合并评分分布已保存: {out_csv}")
    print("\n合并评分分布统计:")
    print(dist_df.to_string(index=False))

    # 各数据源分布简要
    print("\n--- 各数据源简要 ---")
    for label, per_sess, scores in by_source:
        c = Counter(scores)
        total_s = len(scores)
        print(f"  {label}: {len(per_sess)} sessions, {total_s} 条评分, 平均分={np.mean(scores):.3f}")

    # 绘图（合并）
    out_png = PROJECT_ROOT / "raw_rating_distribution.png"
    plot_rating_distribution(all_scores, out_png, n_sessions)

    # 各 session 汇总（含数据源）
    summary_rows = []
    for name, scores, source in all_per_session:
        summary_rows.append({
            "data_source": source,
            "session": name,
            "评分数量": len(scores),
            "平均分": np.mean(scores),
            "最小": min(scores),
            "最大": max(scores),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = PROJECT_ROOT / "raw_rating_per_session.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\n各 session 评分汇总已保存: {summary_csv}")


if __name__ == "__main__":
    main()
