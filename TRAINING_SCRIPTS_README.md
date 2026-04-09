# 模型训练脚本文件结构说明

## 📁 文件结构概览

```
motion_sickness_classification/
├── main.py                          # 主入口脚本
├── experiment.py                    # 实验管理类
├── trainer.py                       # 训练器实现
├── dataset.py                       # 数据集类
├── configs.py                       # 配置文件
│
├── base/                            # 基础组件模块
│   ├── experiment.py                # 基础实验类
│   ├── trainer.py                   # 基础训练器抽象类
│   ├── loss_function.py             # 损失函数
│   ├── scheduler.py                 # 学习率调度器
│   ├── checkpointer.py              # 检查点管理
│   ├── parameter_control.py         # 参数控制（渐进式释放）
│   └── utils.py                     # 工具函数
│
└── models/                          # 模型定义
    ├── comfort_model.py             # 舒适度分类模型（主要使用）
    ├── patch_encoder.py             # Patch编码器
    ├── cross_attention.py           # 交叉注意力模块
    └── ...
```

---

## 📄 核心文件详解

### 1. `main.py` - 主入口脚本

**路径**: `./main.py`

**功能**: 程序入口，解析命令行参数并启动训练流程

**主要函数**:
- 无类/函数定义，直接执行
- 使用 `argparse` 解析所有训练参数
- 创建 `Experiment` 实例并运行

**关键参数**:
- `-dataset_path`: 数据集路径
- `-model_name`: 模型名称（ComfortClassificationModel/PhysioFusionNet/CAN）
- `-num_classes`: 分类类别数（默认5，0-4分）
- `-batch_size`: 批次大小
- `-learning_rate`: 学习率
- `-num_epochs`: 训练轮数
- `-eeg_channels`, `-ecg_channels`: 通道数
- `-patch_length`, `-num_patches`: Patch相关参数

---

### 2. `experiment.py` - 实验管理类

**路径**: `./experiment.py`

**类**: `Experiment(GenericExperiment)`

**主要方法**:

#### `__init__(self, args)`
- 初始化实验参数
- 设置模型、训练相关参数

#### `prepare(self)`
- 准备实验环境
- 加载/创建数据集信息
- 初始化数据加载器
- 计算均值和标准差

#### `run(self)`
- **核心训练流程**:
  1. 初始化损失函数
  2. 遍历每个fold（交叉验证）
  3. 初始化模型
  4. 初始化数据加载器
  5. 初始化训练器
  6. 初始化参数控制器
  7. 初始化检查点管理器
  8. 执行训练
  9. 执行测试

#### `init_model(self)`
- 根据 `model_name` 初始化对应模型
- 支持: `ComfortClassificationModel`, `PhysioFusionNet`, `CAN`

#### `init_dataloader(self, fold)`
- 加载数据
- **数据拆分**: 70% 训练集, 15% 验证集, 15% 测试集（简单顺序拆分）
- 创建 DataLoader

#### `init_dataset(self, data, ...)`
- 创建 `PhysiologicalDataset` 实例

#### `subset_data(self, data, indices)`
- 根据索引创建数据子集

---

### 3. `trainer.py` - 训练器实现

**路径**: `./trainer.py`

**类**: `Trainer(GenericVideoTrainer)`

**主要方法**:

#### `__init__(self, **kwargs)`
- 初始化训练器
- 设置最佳模型信息字典

#### `init_optimizer_and_scheduler(self, epoch=0)`
- 初始化优化器（Adam）
- 初始化学习率调度器（MyWarmupScheduler）

#### `fit(self, dataloader_dict, checkpoint_controller, parameter_controller)`
- **主训练循环**:
  - 遍历每个epoch
  - 调用 `train()` 训练
  - 调用 `validate()` 验证
  - 更新学习率
  - 保存最佳模型
  - 早停机制

#### `train(self, dataloader_dict, epoch)`
- **训练一个epoch**:
  - 遍历训练数据
  - 前向传播
  - 计算loss（支持样本权重）
  - 反向传播
  - 更新参数
  - 计算指标（accuracy, precision, recall, f1）

#### `validate(self, dataloader_dict, epoch)`
- **验证一个epoch**:
  - 评估模式
  - 计算验证集loss和指标

#### `test(self, checkpoint_controller, predict_only=1, **kwargs)`
- **测试模型**:
  - 加载最佳模型
  - 计算测试集指标
  - 保存混淆矩阵
  - 保存预测结果

#### `move_batch_to_device(self, batch)`
- 将batch数据移动到指定设备（CPU/GPU）

#### `save_test_results(self, record_dict, save_path)`
- 保存测试结果（混淆矩阵、指标、预测值）

---

### 4. `dataset.py` - 数据集类

**路径**: `./dataset.py`

**类1**: `PhysiologicalDataset(Dataset)`

**主要方法**:

#### `__init__(self, data_dict, modality, ...)`
- 初始化数据集
- 检测数据格式（patches格式 vs 连续信号格式）
- 加载权重数据

#### `preprocess_data(self)`
- 预处理数据（滤波、归一化）

#### `preprocess_eeg(self, eeg_data)` / `preprocess_ecg(self, ecg_data)`
- 预处理EEG/ECG信号
- 支持patches格式和连续信号格式

#### `__getitem__(self, idx)`
- 返回一个样本:
  - `eeg`: EEG patches数据
  - `ecg`: ECG patches数据
  - `label`: 标签
  - `weight`: 样本权重（如果存在）

#### `__len__(self)`
- 返回数据集大小

**类2**: `DataArranger`

**主要方法**:

#### `load_data(self)`
- 加载训练数据
- **支持两种格式**:
  1. 新格式: 从 `training_dataset` 目录加载 `eeg_patches.npy`, `ecg_patches.npy`, `labels.npy`, `weights.npy`
  2. 旧格式: 从 `data/raw/` 目录加载

---

### 5. `base/loss_function.py` - 损失函数

**路径**: `./base/loss_function.py`

**类**: `ClassificationLoss(nn.Module)`

**主要方法**:

#### `__init__(self, loss_type='cross_entropy', num_classes=11, class_weights=None)`
- 初始化损失函数
- 支持: `cross_entropy`, `focal`, `label_smoothing`

#### `forward(self, predictions, targets, sample_weights=None)`
- **计算损失**:
  - 如果提供 `sample_weights`，使用加权平均
  - 否则使用标准损失
  - 支持类别权重和样本权重同时使用

**其他损失函数**:
- `FocalLoss`: 处理类别不平衡
- `LabelSmoothingLoss`: 标签平滑
- `CombinedLoss`: 组合多种损失

---

### 6. `base/scheduler.py` - 学习率调度器

**路径**: `./base/scheduler.py`

**类1**: `MyWarmupScheduler(_LRScheduler)`

**功能**:
- Warmup阶段: 前N个epoch线性增加学习率
- Plateau阶段: 验证指标不提升时降低学习率
- 支持最小学习率限制

**类2**: `GradualWarmupScheduler(_LRScheduler)`
- 渐进式warmup调度器

---

### 7. `base/checkpointer.py` - 检查点管理

**路径**: `./base/checkpointer.py`

**类**: `Checkpointer`

**主要方法**:

#### `save_checkpoint(self, epoch, is_best=False)`
- 保存检查点:
  - 模型状态
  - 优化器状态
  - 调度器状态
  - 最佳模型信息
  - 如果是最佳模型，额外保存 `_best.pkl`

#### `load_checkpoint(self)`
- 加载检查点
- 恢复训练状态

#### `init_csv_logger(self, args, config)`
- 初始化CSV日志记录器
- 记录训练过程

---

### 8. `base/parameter_control.py` - 参数控制

**路径**: `./base/parameter_control.py`

**类**: `ResnetParamControl`

**功能**: 渐进式参数释放
- 初始阶段冻结部分层
- 在指定epoch逐步解冻
- 用于迁移学习场景

**主要方法**:
- `setup_parameter_groups()`: 设置参数组
- `release_param()`: 释放参数
- `should_release_parameters()`: 判断是否应该释放

---

### 9. `base/trainer.py` - 基础训练器

**路径**: `./base/trainer.py`

**类**: `GenericVideoTrainer(ABC)`

**功能**: 训练器基类，定义接口

**主要方法**:
- `get_parameters()`: 获取可训练参数
- `init_optimizer_and_scheduler()`: 初始化优化器和调度器（抽象方法）

---

### 10. `base/experiment.py` - 基础实验类

**路径**: `./base/experiment.py`

**类**: `GenericExperiment(ABC)`

**功能**: 实验基类，定义通用接口

**主要属性**:
- 实验参数（路径、设备、随机种子等）
- 训练参数（学习率、批次大小等）
- 交叉验证参数

---

### 11. `base/utils.py` - 工具函数

**路径**: `./base/utils.py`

**主要函数**:
- `load_pickle(file_path)`: 加载pickle文件
- `save_pickle(data, file_path)`: 保存pickle文件
- `compute_metrics(predictions, targets)`: 计算评估指标
- `set_random_seed(seed)`: 设置随机种子
- `count_parameters(model)`: 统计模型参数数量
- `save_model_checkpoint()` / `load_model_checkpoint()`: 模型检查点操作

---

## 🔄 训练流程

```
main.py
  └─> Experiment.prepare()
       ├─> 加载/创建数据集信息
       ├─> 初始化 DataArranger
       └─> 计算均值和标准差
  └─> Experiment.run()
       ├─> 初始化损失函数 (ClassificationLoss)
       ├─> 遍历每个fold:
       │    ├─> init_model() - 初始化模型
       │    ├─> init_dataloader() - 初始化数据加载器
       │    ├─> Trainer() - 创建训练器
       │    ├─> ResnetParamControl() - 参数控制器
       │    ├─> Checkpointer() - 检查点管理器
       │    ├─> Trainer.fit() - 训练循环
       │    │    ├─> train() - 训练一个epoch
       │    │    ├─> validate() - 验证
       │    │    └─> 保存最佳模型
       │    └─> Trainer.test() - 测试
```

---

## 📊 数据流

```
DataArranger.load_data()
  └─> 加载 eeg_patches.npy, ecg_patches.npy, labels.npy, weights.npy
       └─> 返回 data 字典

PhysiologicalDataset
  └─> __getitem__(idx)
       └─> 返回: {'eeg': tensor, 'ecg': tensor, 'label': tensor, 'weight': tensor}

DataLoader
  └─> 批量加载数据
       └─> batch: {'eeg': [B, N_patches, C, L], 'ecg': [B, N_patches, C, L], 
                   'label': [B], 'weight': [B]}

Trainer.train()
  └─> model(eeg_patches, ecg_patches)
       └─> outputs: [B, num_classes]
       └─> criterion(outputs, targets, sample_weights)
            └─> loss (加权)
```

---

## 🎯 关键特性

1. **样本权重支持**: Loss计算时考虑每个样本的权重
2. **多种模型支持**: ComfortClassificationModel, PhysioFusionNet, CAN
3. **检查点管理**: 自动保存/加载最佳模型
4. **学习率调度**: Warmup + Plateau reduction
5. **渐进式参数释放**: 支持迁移学习
6. **早停机制**: 防止过拟合
7. **详细日志**: CSV格式记录训练过程

---

## 📝 使用示例

```bash
python main.py \
    -dataset_path ./data/training_dataset_valid \
    -model_name ComfortClassificationModel \
    -modality eeg ecg \
    -num_classes 5 \
    -eeg_channels 59 \
    -ecg_channels 1 \
    -patch_length 250 \
    -num_patches 10 \
    -encoding_dim 256 \
    -num_heads 8 \
    -batch_size 32 \
    -learning_rate 1e-4 \
    -num_epochs 100 \
    -early_stopping 20
```

