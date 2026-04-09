# 继续训练指南

## 功能说明

代码现在支持从保存的checkpoint继续训练，可以：
- ✅ 加载模型权重
- ✅ 恢复optimizer和scheduler状态
- ✅ 恢复训练历史（train_losses, validate_losses）
- ✅ 从上次停止的epoch继续训练
- ✅ 支持加载最新模型或最佳模型

## 使用方法

### 方法1：从最新checkpoint继续训练（推荐）

```bash
python main.py -resume 1 ...
```

这会：
- 加载 `checkpoint.pkl`（最新保存的模型）
- 从上次训练的epoch继续
- 恢复所有训练状态

### 方法2：从最佳模型继续训练

```bash
python main.py -resume 1 -resume_from_best 1 ...
```

这会：
- 加载 `checkpoint_best.pkl`（验证集上表现最好的模型）
- 从最佳模型的epoch继续训练
- 恢复所有训练状态

### 方法3：从头开始训练

```bash
python main.py -resume 0 ...
```

或者不指定 `-resume` 参数（默认为0）

## 参数说明

- `-resume`: 
  - `0` (默认): 从头开始训练
  - `1`: 从checkpoint继续训练

- `-resume_from_best`:
  - `0` (默认): 加载最新checkpoint (`checkpoint.pkl`)
  - `1`: 加载最佳模型checkpoint (`checkpoint_best.pkl`)
  - 仅在 `-resume 1` 时有效

## 完整示例

### 示例1：从最新checkpoint继续训练50个epoch

```bash
python main.py \
    -resume 1 \
    -num_epochs 100 \
    -dataset_path ./data/training_dataset_valid \
    ...
```

假设之前训练了50个epoch，现在会从epoch 51继续训练到epoch 100。

### 示例2：从最佳模型继续训练

```bash
python main.py \
    -resume 1 \
    -resume_from_best 1 \
    -num_epochs 100 \
    ...
```

这会加载验证集上表现最好的模型，然后继续训练。

## 保存的内容

Checkpoint会保存以下内容：
- ✅ 模型权重 (`model_state_dict`)
- ✅ Optimizer状态 (`optimizer_state_dict`)
- ✅ Scheduler状态 (`scheduler_state_dict`)
- ✅ 最佳模型信息 (`best_epoch_info`)
- ✅ 训练历史 (`train_losses`, `validate_losses`)
- ✅ Early stopping计数器
- ✅ 训练完成状态 (`fit_finished`)
- ✅ Parameter controller状态

## 恢复的内容

当resume时，会恢复：
- ✅ 模型权重
- ✅ Optimizer状态（学习率、动量等）
- ✅ Scheduler状态（学习率调度器的状态）
- ✅ 训练历史（用于绘制完整的训练曲线）
- ✅ 最佳模型信息
- ✅ Early stopping计数器
- ✅ 从 `epoch + 1` 继续训练

## 注意事项

1. **确保路径正确**：checkpoint文件路径基于 `save_path`、`experiment_name`、`model_name`、`stamp`、`fold` 和 `seed` 生成

2. **模型架构一致**：继续训练时，模型架构必须与保存时一致，否则无法加载权重

3. **训练参数**：可以修改训练参数（如学习率、batch_size等），但建议保持一致以获得最佳效果

4. **CSV日志**：训练日志会追加到现有的 `checkpoint_log.csv` 文件中

5. **最佳模型**：如果选择加载最佳模型，会从最佳模型的epoch继续，而不是最新epoch

## 检查点文件位置

Checkpoint文件保存在：
```
{save_path}/{experiment_name}_{model_name}_{stamp}_fold{fold}_seed{seed}/
├── checkpoint.pkl          # 最新checkpoint
├── checkpoint_best.pkl     # 最佳模型checkpoint
└── checkpoint_log.csv      # 训练日志
```

## 故障排除

### 问题1：找不到checkpoint文件
```
Checkpoint file ... not found. Starting from scratch.
```
**解决**：检查 `save_path`、`experiment_name`、`stamp`、`fold` 等参数是否与之前训练时一致

### 问题2：模型架构不匹配
```
RuntimeError: Error(s) in loading state_dict
```
**解决**：确保模型架构（层数、维度等）与保存时完全一致

### 问题3：Optimizer状态加载失败
```
⚠️  加载Optimizer状态失败: ...
```
**解决**：代码会自动使用新的optimizer，训练可以继续，但学习率等状态会重置

## 示例场景

### 场景1：训练中断后继续
```bash
# 第一次训练（训练到epoch 50时中断）
python main.py -num_epochs 100 ...

# 继续训练（从epoch 51继续到100）
python main.py -resume 1 -num_epochs 100 ...
```

### 场景2：想从最佳模型继续微调
```bash
# 从最佳模型继续训练更多epoch
python main.py -resume 1 -resume_from_best 1 -num_epochs 150 ...
```

### 场景3：调整学习率后继续训练
```bash
# 从checkpoint继续，但使用新的学习率
python main.py -resume 1 -learning_rate 1e-5 ...
```

