# 舒适度预测闭环系统 - 部署说明

本目录包含了将训练好的舒适度预测模型部署到车辆驾驶（仿真）系统中的核心代码。系统设计为准实时处理，每 10 秒输出一次舒适度评分（0-4分）。

## 系统组件

1.  **`main.py`**: 部署服务入口，负责启动和整合所有组件。
2.  **`comfort_predictor.py`**: 核心服务类，协调采集、处理和推理流程。
3.  **`data_collector.py`**: 数据采集接口。支持模拟数据生成（测试用）和硬件采集接口（待扩展）。
4.  **`data_buffer.py`**: 滑动窗口缓冲区，负责 10 秒数据的累积和同步。
5.  **`data_processor.py`**: 数据处理流水线，包含采样对齐、IMU冲突计算、生理信号预处理（滤波、去伪迹等）和归一化。
6.  **`model_manager.py`**: 模型加载和推理引擎。
7.  **`config.py`**: 部署参数配置。
8.  **`utils.py`**: 日志记录、结果保存等工具函数。

## 运行要求

-   Python 3.8+
-   PyTorch
-   NumPy, SciPy
-   `models/` 目录下的模型定义文件
-   训练好的模型 Checkpoint (`.pkl`)

## 使用方法

### 1. 快速启动（使用模拟随机数据测试）

可以使用 `--simulated` 标志启动模拟数据模式，用于验证整个软件链路：

```bash
python deployment/main.py \
    --checkpoint_path ./results/path_to_your_model/checkpoint_best.pkl \
    --normalization_stats_path ./data/normalization_stats.pkl \
    --simulated
```

### 2. 使用原始 BDF 数据回放

可以使用 `--original` 标志并指定被试和地图，从 `data/` 目录中读取真实的历史 BDF 数据进行回放预测：

```bash
python deployment/main.py \
    --checkpoint_path ./results/path_to_your_model/checkpoint_best.pkl \
    --normalization_stats_path ./data/normalization_stats.pkl \
    --original \
    --subject "被试姓名" \
    --map "地图编号"
```

### 3. 参数说明

-   `--checkpoint_path`: **(必填)** 指向训练好的 `.pkl` 文件。
-   `--normalization_stats_path`: 归一化统计量路径。
-   `--original`: 开启原始数据回放模式。
-   `--subject`: 被试姓名（如 `cyz`）。
-   `--map`: 地图编号（如 `01`）。
-   `--collect_interval`: 数据采集频率（秒），默认 0.1s。
-   `--window_length`: 预测窗口点数，默认 2500。

## 闭环控制集成

在 `deployment/main.py` 的 `main()` 函数中，定义了 `on_prediction` 回调函数。这是与车辆控制系统对接的核心入口：

```python
def on_prediction(result):
    score = result['score']
    # 在此处添加与 CARLA 或实际车辆控制器的交互逻辑
    if score >= 3:
        # 触发驾驶模式切换策略
        pass
```

## 数据流程图

```mermaid
flowchart LR
    Sensor[传感器] --> Buffer[缓冲区]
    Buffer -- 10s数据 --> Processor[预处理流水线]
    Processor -- Patches --> Model[推理引擎]
    Model -- 评分 --> Output[结果输出/闭环控制]
```
