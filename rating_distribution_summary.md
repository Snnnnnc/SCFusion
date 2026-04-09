# BDF 事件评分分布汇总报告

## 数据来源与规则

- **源文件**: `data/xxx/xxx/evt.bdf`（被试评分的 BDF 事件文件）
- **事件编码**:
  - `100` = 实验开始
  - `200` = 实验结束
  - `10`–`19` = 被试评分，对应 **0–9 分**（即 `code - 10`）

分析时仅统计 **最后一次“开始(100)”之后** 的评分事件（10–19），以排除实验前的误触。

### 多数据路径合并

支持将 **项目 data/** 与 **外部路径**（如 `/Volumes/LENOVO_USB_HDD/MS/data`）一起统计并汇总：

```bash
# 仅项目 data/
python analyze_raw_rating_distribution.py

# 项目 data/ + 外部路径，合并统计
python analyze_raw_rating_distribution.py /Volumes/LENOVO_USB_HDD/MS/data
```

---

## 整体评分分布（所有 session 汇总）

| 评分 | 次数 | 占比 |
|------|------|------|
| 0 | 435 | **47.28%** |
| 1 | 238 | **25.87%** |
| 2 | 157 | **17.07%** |
| 3 | 45 | **4.89%** |
| 4 | 45 | **4.89%** |
| 5–9 | 0 | 0% |

- **总评分数**: 920
- **有效 session 数**: 86（每个 session 至少有一个评分）
- **评分范围**: 0–4 分（未出现 5–9 分）

---

## 简要结论

1. **分布高度左偏**：约 73% 的评分为 0 或 1，说明多数时段为无/轻度不适。
2. **高分段缺失**：5–9 分在本数据集中未出现，不适程度集中在中低分段。
3. **单 session**：多数 session 约有 9–12 次评分，与实验设计中的多次评分一致。

---

## 生成的文件

- `raw_rating_distribution.csv`：各评分（0–9）的次数与占比（合并所有数据源）
- `raw_rating_per_session.csv`：每个 session 的评分数量、平均分、最小/最大分，以及 **data_source** 列标识来源（project_data 或外部路径）
- `raw_rating_distribution.png`：合并后的评分分布直方图 + KDE

**合并项目 data 与 USB 盘数据后重新统计**（需已安装 `mne`）：

```bash
python analyze_raw_rating_distribution.py /Volumes/LENOVO_USB_HDD/MS/data
```

仅统计项目 data 时直接运行：

```bash
python analyze_raw_rating_distribution.py
```
