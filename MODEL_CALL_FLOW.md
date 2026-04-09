# 模型调用流程说明

## 📊 数据维度

根据 `data/training_dataset_valid/` 目录：
- **eeg_patches.npy**: `(8581, 10, 59, 250)` - 8581个窗口，每个窗口10个patches，59个EEG通道，250个时间点
- **ecg_patches.npy**: `(8581, 10, 1, 250)` - 8581个窗口，每个窗口10个patches，1个ECG通道，250个时间点
- **labels.npy**: `(8581,)` - 8581个标签（0-4分）
- **weights.npy**: `(8581,)` - 8581个样本权重

## 🔄 模型调用流程

### 1. 数据加载 (`dataset.py`)

```python
DataArranger.load_data()
  └─> 加载 eeg_patches.npy, ecg_patches.npy, labels.npy, weights.npy
       └─> 返回 data 字典: {'eeg': [...], 'ecg': [...], 'labels': [...], 'weights': [...]}

PhysiologicalDataset.__getitem__(idx)
  └─> 返回样本: {
        'eeg': tensor (10, 59, 250),    # (num_patches, channels, patch_length)
        'ecg': tensor (10, 1, 250),     # (num_patches, channels, patch_length)
        'label': tensor ([label]),      # 标签
        'weight': tensor ([weight])     # 权重
      }
```

### 2. 模型初始化 (`experiment.py`)

```python
Experiment.init_model()
  └─> 如果 model_name == "ComfortClassificationModel":
       └─> ComfortClassificationModel(
             eeg_channels=59,           # ✅ 从main.py参数获取
             ecg_channels=1,            # ✅ 从main.py参数获取
             patch_length=250,           # ✅ 从main.py参数获取 (1s @ 250Hz)
             num_patches=10,             # ✅ 从main.py参数获取
             encoding_dim=256,           # ✅ 从main.py参数获取
             num_heads=8,                # ✅ 从main.py参数获取
             num_classes=5,              # ✅ 从main.py参数获取
             attention_output_mode='global',
             hidden_dims=[512, 256, 128],
             dropout=0.1
           )
```

### 3. 模型前向传播 (`models/comfort_model.py`)

```python
ComfortClassificationModel.forward(eeg_patches, ecg_patches)
  │
  ├─> 输入: 
  │    - eeg_patches: (batch, 10, 59, 250)
  │    - ecg_patches: (batch, 10, 1, 250)
  │
  ├─> 步骤1: Patch编码 (PatchEncoder1D)
  │    ├─> 对每个patch (p=0..9):
  │    │    ├─> eeg_patch = eeg_patches[:, p, :, :]  # (batch, 59, 250)
  │    │    ├─> ecg_patch = ecg_patches[:, p, :, :]  # (batch, 1, 250)
  │    │    ├─> eeg_enc = eeg_encoder(eeg_patch)     # (batch, 256)
  │    │    └─> ecg_enc = ecg_encoder(ecg_patch)     # (batch, 256)
  │    └─> 堆叠: 
  │         - eeg_features: (batch, 10, 256)
  │         - ecg_features: (batch, 10, 256)
  │
  ├─> 步骤2: Cross-attention融合 (BidirectionalCrossAttention)
  │    ├─> eeg_enhanced = eeg_to_ecg(eeg_features, ecg_features)
  │    ├─> ecg_enhanced = ecg_to_eeg(ecg_features, eeg_features)
  │    ├─> fused = concat([eeg_enhanced, ecg_enhanced])  # (batch, 10, 512) 或 (batch, 512)
  │    └─> fused = fusion(fused)  # (batch, 256) [如果output_mode='global']
  │
  └─> 步骤3: 分类网络 (classifier)
       └─> logits = classifier(fused)  # (batch, 5)
```

### 4. 训练流程 (`trainer.py`)

```python
Trainer.train()
  └─> 遍历batch:
       ├─> batch = {
             'eeg': (batch_size, 10, 59, 250),
             'ecg': (batch_size, 10, 1, 250),
             'label': (batch_size,),
             'weight': (batch_size,)
           }
       ├─> outputs = model(batch['eeg'], batch['ecg'])  # (batch_size, 5)
       ├─> loss = criterion(outputs, targets, sample_weights=weights)  # ✅ 使用样本权重
       └─> loss.backward() → optimizer.step()
```

## ✅ 已修复的问题

### 1. `main.py` 参数修复

- ✅ **默认模型**: `ComfortClassificationModel` (之前是 `PhysioFusionNet`)
- ✅ **EEG通道数**: 59 (之前是64)
- ✅ **ECG通道数**: 1 (之前是3)
- ✅ **采样率**: 250Hz (之前是1000/500Hz)
- ✅ **窗口长度**: 2500 (10s @ 250Hz，之前是3000)
- ✅ **步长**: 750 (3s @ 250Hz，之前是1500)
- ✅ **新增参数**:
  - `-patch_length`: 250 (1s @ 250Hz)
  - `-num_patches`: 10
  - `-encoding_dim`: 256
  - `-attention_output_mode`: 'global'
  - `-dropout`: 0.1
  - `-hidden_dims`: '512,256,128'
- ✅ **num_heads默认值**: 8 (之前是4)
- ✅ **apply_filter默认值**: 0 (数据已预处理)

### 2. `experiment.py` 模型调用

- ✅ 正确调用 `ComfortClassificationModel`
- ✅ 所有参数从 `args` 正确传递
- ✅ 处理 `hidden_dims` 字符串解析

### 3. 模型实现确认

- ✅ `PatchEncoder1D`: 1D CNN编码器，将 `(C, patch_length)` → `(encoding_dim,)`
- ✅ `BidirectionalCrossAttention`: 双向交叉注意力融合
- ✅ `classifier`: 全连接层分类网络

## 🎯 完整训练命令示例

```bash
conda activate py39

python main.py \
    -dataset_path ./data/training_dataset_valid \
    -model_name ComfortClassificationModel \
    -modality eeg ecg \
    -num_classes 5 \
    -eeg_channels 59 \
    -ecg_channels 1 \
    -eeg_sampling_rate 250 \
    -ecg_sampling_rate 250 \
    -patch_length 250 \
    -num_patches 10 \
    -encoding_dim 256 \
    -num_heads 8 \
    -attention_output_mode global \
    -hidden_dims "512,256,128" \
    -dropout 0.1 \
    -batch_size 32 \
    -learning_rate 1e-4 \
    -num_epochs 100 \
    -early_stopping 20 \
    -normalize_data 1 \
    -apply_filter 0
```

## 📝 关键点总结

1. **数据格式**: 已窗口化和patch化，直接使用patches格式
2. **模型架构**: 
   - Patch编码 (1D CNN) → Cross-attention → 分类器
   - 所有组件都在 `models/` 目录下实现
3. **参数传递**: 所有参数通过 `main.py` → `experiment.py` → `ComfortClassificationModel` 正确传递
4. **样本权重**: Loss计算时已考虑样本权重
5. **默认值**: 所有默认值已根据实际数据维度调整


