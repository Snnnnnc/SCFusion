import csv
import math
import pandas as pd

# Golding (2006) MSSQ-Short 百分位映射系数（拟合多项式）
# y = a*x + b*x^2 + c*x^3 + d*x^4
a = 5.1160923
b = -0.055169904
c = -0.00067784495
d = 1.0714752e-005

def raw_to_percentile(x_raw):
    """
    用 Golding 拟合公式把 MSSQ raw 分数映射到百分位
    x_raw: MSSQ-Short raw 总分（MSA + MSB）
    返回：百分位数（0-100）
    """
    y = a * x_raw + b * (x_raw**2) + c * (x_raw**3) + d * (x_raw**4)
    if y < 0:
        y = 0
    if y > 100:
        y = 100
    return y

def classify_susceptibility(percentile, low_p=33, high_p=67):
    """
    根据百分位数分类为低 / 中 / 高 易感
    默认： ≤ 33% 为低，≥ 67% 为高，其余为中
    """
    if percentile <= low_p:
        return "Low"
    elif percentile >= high_p:
        return "High"
    else:
        return "Medium"

def process_mssq(input_csv, output_csv):
    """
    input_csv 应包含每位被试的 MSA_raw, num_not_experienced_A, MSB_raw, num_not_experienced_B
    脚本会校正曝光、计算 raw_total、映射百分位、分类
    """
    rows = []
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for rec in reader:
            subj = rec["SubjectID"]
            # 童年部分
            MSA_raw = float(rec["MSA_raw"])
            na_A = int(rec["NotExperienced_A"])
            # 成年部分
            MSB_raw = float(rec["MSB_raw"])
            na_B = int(rec["NotExperienced_B"])
            # 校正曝光（如果有 NotExperienced 项）
            # MSSQ-Short 文档要求：
            # MSA = (raw_sum_A) × 9 / (9 - na_A)
            # MSB = (raw_sum_B) × 9 / (9 - na_B)
            # 但要避免除以零
            if na_A >= 9:
                adj_MSA = 0.0
            else:
                adj_MSA = MSA_raw * 9.0 / (9 - na_A)
            if na_B >= 9:
                adj_MSB = 0.0
            else:
                adj_MSB = MSB_raw * 9.0 / (9 - na_B)
            raw_total = adj_MSA + adj_MSB
            pct = raw_to_percentile(raw_total)
            category = classify_susceptibility(pct)
            rows.append({
                "SubjectID": subj,
                "MSA_raw": MSA_raw,
                "NotExperienced_A": na_A,
                "Adj_MSA": round(adj_MSA, 3),
                "MSB_raw": MSB_raw,
                "NotExperienced_B": na_B,
                "Adj_MSB": round(adj_MSB, 3),
                "Raw_Total": round(raw_total, 3),
                "Percentile": round(pct, 1),
                "Susceptibility": category
            })
    # 输出到 CSV / Excel
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print("输出已写入", output_csv)
    return df

# 测试示例
if __name__ == "__main__":
    # 假设 input_mssq.csv 格式如下：
    # SubjectID, MSA_raw, NotExperienced_A, MSB_raw, NotExperienced_B
    # S1, 5, 1, 4, 0
    # S2, 8, 0, 6, 2
    # ...
    output = process_mssq("input_mssq.csv", "output_mssq_classified.csv")
    print(output)
