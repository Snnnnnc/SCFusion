#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算被试 MSSQ-S (Motion Sickness Susceptibility Questionnaire-Short form) 得分
根据问卷作答情况，按童年/成年两时段分别计算后按公式汇总
"""

import pandas as pd

from typing import Tuple

# 事件1-4对应0-3分，事件5不列入计算
SCORE_MAP = {1: 0, 2: 1, 3: 2, 4: 3}  # 从未/很少/有时/经常 -> 0/1/2/3

# 童年时期列索引（4-12列，即第4到第12列，pandas 0-based为3-11）
# 成年时期列索引（13-21列，即第13到第21列，pandas 0-based为12-20）
CHILDHOOD_COL_START = 3   # 第4列
CHILDHOOD_COL_END = 12    # 含第12列
ADULT_COL_START = 12      # 第13列
ADULT_COL_END = 21        # 含第21列


def calc_period_score(series: pd.Series) -> Tuple[float, int]:
    """
    计算某一时段（童年或成年）的总分和有效题数。
    事件1-4计分(0-3)，事件5不参与计分也不计入有效题数。
    返回 (总分, 有效题数)。有效题数为0时，返回 (0, 0)，该时段平均分按0处理。
    """
    total = 0
    valid_count = 0
    for v in series:
        if pd.isna(v):
            continue
        try:
            x = int(float(v))
        except (ValueError, TypeError):
            continue
        if x in SCORE_MAP:
            total += SCORE_MAP[x]
            valid_count += 1
        # x==5 不参与
    return total, valid_count


def calc_mssq_total(child_total: float, child_valid: int, adult_total: float, adult_valid: int) -> float:
    """
    MSSQ-S Total = ( (Childhood总分/Childhood有效题数) + (Adult总分/Adult有效题数) ) × 9
    当某时段有效题数为0时，该时段贡献为0。
    """
    child_avg = child_total / child_valid if child_valid > 0 else 0.0
    adult_avg = adult_total / adult_valid if adult_valid > 0 else 0.0
    return (child_avg + adult_avg) * 9


def main():
    input_path = "被试记录.csv"
    output_path = "被试记录_MSSQ得分.csv"

    df = pd.read_csv(input_path, encoding="utf-8")

    # 确定童年/成年列：按列名包含"5、"或"6、"来区分（与用户描述一致）
    all_cols = df.columns.tolist()
    childhood_cols = [c for c in all_cols if "5、" in str(c)]
    adult_cols = [c for c in all_cols if "6、" in str(c)]

    # 若按列名筛不到，则按位置：第4-12列童年，第13-21列成年
    if len(childhood_cols) != 9:
        childhood_cols = all_cols[CHILDHOOD_COL_START:CHILDHOOD_COL_END]
    if len(adult_cols) != 9:
        adult_cols = all_cols[ADULT_COL_START:ADULT_COL_END]

    results = []
    for idx, row in df.iterrows():
        child_series = row[childhood_cols]
        adult_series = row[adult_cols]
        child_total, child_valid = calc_period_score(child_series)
        adult_total, adult_valid = calc_period_score(adult_series)
        mssq_total = calc_mssq_total(child_total, child_valid, adult_total, adult_valid)

        # 保留原表前几列信息（序号、性别、年龄段等）
        info = {df.columns[i]: row.iloc[i] for i in range(min(3, len(df.columns)))}
        results.append({
            **info,
            "童年总分": child_total,
            "童年有效题数": child_valid,
            "成年总分": adult_total,
            "成年有效题数": adult_valid,
            "MSSQ_S_总分": round(mssq_total, 2),
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已计算 {len(out_df)} 位被试的 MSSQ 得分，结果已保存至: {output_path}")
    print(out_df[["序号", "童年总分", "童年有效题数", "成年总分", "成年有效题数", "MSSQ_S_总分"]].head(10).to_string())


if __name__ == "__main__":
    main()
