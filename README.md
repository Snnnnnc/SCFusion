# Motion Sickness Classification with Multimodal Physiological Signals

基于EEG和ECG多模态生理信号融合的晕动症状分类项目

## 项目概述

本项目使用EEG（脑电图）和ECG（心电图）两种生理信号进行多模态融合，通过深度学习模型对晕动症状进行0-10分的分类预测。

## 项目结构

```
motion_sickness_classification/
├── README.md                           # 项目说明文档
├── main.py                             # 主程序入口
├── configs.py                          # 配置文件
├── experiment.py                       # 实验管理
├── trainer.py                          # 训练器
├── dataset.py                          # 数据集处理
├── generate_result.py                  # 结果生成
├── test.py                             # 测试脚本
├── base/                               # 基础组件
│   ├── __init__.py
│   ├── dataset.py                      # 基础数据集类
│   ├── experiment.py                   # 基础实验类
│   ├── trainer.py                      # 基础训练器
│   ├── scheduler.py                    # 学习率调度器
│   ├── loss_function.py                # 损失函数
│   ├── checkpointer.py                 # 检查点管理
│   ├── logger.py                       # 日志管理
│   ├── parameter_control.py            # 参数控制
│   ├── utils.py                        # 工具函数
│   └── transforms3D.py                 # 数据变换
├── models/                             # 模型定义
│   ├── __init__.py
│   ├── model.py                        # 主模型定义
│   ├── backbone.py                     # 特征提取骨干网络
│   ├── temporal_convolutional_model.py # 时序卷积网络
│   ├── multimodal_fusion.py            # 多模态融合模块
│   ├── cross_attention.py              # 交叉注意力机制
│   ├── dense_coattention.py            # 密集协同注意力
│   └── transformer.py                  # Transformer组件
├── data/                               # 数据目录
│   ├── raw/                            # 原始数据
│   ├── processed/                      # 预处理数据
│   └── splits/                         # 数据分割
├── checkpoints/                        # 模型检查点
├── results/                            # 结果输出
├── logs/                               # 日志文件
└── requirements.txt                    # 依赖包列表
```

## 文件详细说明

### 核心文件

- **main.py**: 程序入口，解析命令行参数，启动训练
- **configs.py**: 所有配置参数，包括模型参数、训练参数、数据路径等
- **experiment.py**: 实验管理，负责模型初始化、数据加载、训练流程控制
- **trainer.py**: 训练器，实现具体的训练、验证、测试逻辑
- **dataset.py**: 数据集类，处理EEG和ECG数据的加载和预处理

### 模型文件

- **models/model.py**: 主模型类 `PhysioFusionNet`，整合所有组件
- **models/backbone.py**: 生理信号特征提取器，包括EEG和ECG的专用网络
- **models/temporal_convolutional_model.py**: 时序卷积网络，捕获时间序列特征
- **models/multimodal_fusion.py**: 多模态融合模块，实现EEG和ECG的特征融合
- **models/cross_attention.py**: 交叉注意力机制，实现模态间的交互
- **models/dense_coattention.py**: 密集协同注意力，增强模态融合效果

### 基础组件

- **base/dataset.py**: 基础数据集类，提供通用的数据加载接口
- **base/trainer.py**: 基础训练器，提供通用的训练框架
- **base/experiment.py**: 基础实验类，提供实验管理框架
- **base/scheduler.py**: 学习率调度器，支持多种调度策略
- **base/loss_function.py**: 损失函数，包括分类损失和融合损失
- **base/checkpointer.py**: 检查点管理，保存和加载模型状态
- **base/logger.py**: 日志管理，记录训练过程和结果
- **base/parameter_control.py**: 参数控制，支持渐进式参数释放
- **base/utils.py**: 工具函数，包括评估指标、数据转换等

## 数据要求

### 数据格式

#### EEG数据
- **格式**: `.mat` 或 `.npy` 文件
- **维度**: `[subjects, channels, time_points]`
- **通道数**: 建议64通道或更多
- **采样率**: 建议1000Hz或更高
- **时间长度**: 建议每个样本至少30秒

#### ECG数据
- **格式**: `.mat` 或 `.npy` 文件
- **维度**: `[subjects, channels, time_points]`
- **通道数**: 建议3-12通道
- **采样率**: 建议500Hz或更高
- **时间长度**: 与EEG数据同步

#### 标签数据
- **格式**: `.csv` 或 `.npy` 文件
- **内容**: 晕动症状评分 (0-10分)
- **维度**: `[subjects, 1]`

### 数据组织结构

```
data/
├── raw/
│   ├── eeg/
│   │   ├── subject_001.mat
│   │   ├── subject_002.mat
│   │   └── ...
│   ├── ecg/
│   │   ├── subject_001.mat
│   │   ├── subject_002.mat
│   │   └── ...
│   └── labels/
│       └── motion_sickness_scores.csv
├── processed/
│   ├── eeg_features/
│   ├── ecg_features/
│   └── dataset_info.pkl
└── splits/
    ├── train_indices.npy
    ├── val_indices.npy
    └── test_indices.npy
```

### 数据预处理要求

1. **EEG预处理**:
   - 滤波 (0.5-50Hz)
   - 去伪迹
   - 标准化
   - 分段 (30秒片段)

2. **ECG预处理**:
   - 滤波 (0.5-40Hz)
   - R波检测
   - 心率变异性特征提取
   - 标准化

3. **数据对齐**:
   - EEG和ECG时间同步
   - 标签对应

## 使用方法

### 环境配置

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python main.py \
    --modality eeg ecg \
    --model_name PhysioFusionNet \
    --num_epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --cross_validation 1 \
    --num_folds 5
```

### 测试模型

```bash
python test.py \
    --checkpoint_path checkpoints/best_model.pth \
    --test_data_path data/processed/test_data.npy
```

## 模型架构

### 主要组件

1. **EEG特征提取器**: 基于CNN的脑电信号特征提取
2. **ECG特征提取器**: 基于CNN的心电信号特征提取
3. **时序建模**: TCN捕获时间序列依赖关系
4. **多模态融合**: 交叉注意力机制融合EEG和ECG特征
5. **分类器**: 全连接层进行0-10分分类

### 创新点

- **生理信号专用骨干网络**: 针对EEG和ECG信号特点设计
- **多模态交叉注意力**: 实现EEG和ECG的深度交互
- **时序建模**: 捕获生理信号的动态变化
- **端到端训练**: 从原始信号到分类结果的完整流程

## 评估指标

- **准确率 (Accuracy)**
- **精确率 (Precision)**
- **召回率 (Recall)**
- **F1分数**
- **混淆矩阵**
- **分类报告**

## 注意事项

1. **数据隐私**: 确保生理数据的安全性和隐私保护
2. **数据质量**: 确保EEG和ECG数据的质量和完整性
3. **标签一致性**: 确保晕动症状评分的一致性和可靠性
4. **计算资源**: 建议使用GPU进行训练，特别是对于大规模数据

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。 