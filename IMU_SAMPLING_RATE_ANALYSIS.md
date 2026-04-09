# IMU数据采样率分析报告

## 检查结果

### 1. BDF文件原始采样率

**从实际BDF文件读取：**
- ✅ **BDF文件采样率：1000 Hz**
- ✅ 所有通道（包括IMU）都在同一个BDF文件中，共享相同的采样率
- ✅ IMU通道：`['GYR-X', 'GYR-Y', 'GYR-Z', 'ACC-X', 'ACC-Y', 'ACC-Z', 'MAG-X', 'MAG-Y', 'MAG-Z']`

**代码验证：**
```python
# readbdfdata函数从BDF文件读取
fs = raw.info['sfreq']  # 从MNE读取的采样率
eeg['srate'] = fs       # 保存为1000.0 Hz
```

### 2. 数据处理流程

**build_eeg_dataset.py中的处理：**
1. 从BDF文件读取数据，采样率 = 1000 Hz
2. **没有对IMU数据进行下采样**
3. 直接提取IMU通道并保存为 `imu.npy`
4. 保存的采样率信息在 `{session}.npz` 文件中：`srate = 1000`

**关键代码：**
```python
# build_eeg_dataset.py:594
srate = int(eeg.get('srate', cfg.sample_rate_hz))  # 从BDF读取，默认1000

# save_modalities函数直接保存，无下采样
arr = data_trim[np.array(chan_idx, dtype=int), :]
np.save(out_dir / f"imu.npy", arr)  # 直接保存，采样率仍为1000 Hz
```

### 3. 数据验证

**已保存的IMU数据：**
- 文件：`processed/zzh/20251109163847_zzh_01_zzh/_modalities/imu.npy`
- 形状：`(9, 372736)` - 9个通道，372736个时间点
- NPZ文件中的采样率：`srate = 1000 Hz`

**计算验证：**
- 如果采样率是1000 Hz，时长 = 372736 / 1000 = 372.736秒 ≈ 6.21分钟
- 如果采样率是100 Hz，时长 = 372736 / 100 = 3727.36秒 ≈ 62.12分钟（不合理）

### 4. 产品说明书 vs 实际数据

**可能的情况：**

#### 情况A：IMU传感器原生采样率是100Hz，但记录时上采样到1000Hz
- **说明**：为了与EEG数据（1000Hz）同步，IMU数据可能被上采样
- **证据**：
  - BDF文件中所有通道采样率都是1000Hz
  - IMU数据与其他通道（EEG、ECG）在同一个文件中
  - 数据长度与1000Hz采样率一致

#### 情况B：产品说明书指的是IMU传感器的物理采样率
- **说明**：IMU传感器本身以100Hz采样，但在记录到BDF时被插值/上采样到1000Hz
- **影响**：数据中可能包含插值点，但时间戳和采样率都是1000Hz

#### 情况C：存在下采样步骤但代码中未实现
- **说明**：理论上应该下采样到100Hz，但当前代码没有实现
- **影响**：如果确实需要100Hz，需要添加下采样步骤

## 结论

### ✅ 当前状态

1. **BDF文件中的采样率：1000 Hz**（已确认）
2. **保存的imu.npy数据采样率：1000 Hz**（与BDF一致）
3. **没有下采样步骤**（代码中未实现）

### ⚠️ 需要注意

1. **产品说明书上的100Hz**可能是：
   - IMU传感器的物理采样率
   - 建议使用的采样率（需要下采样）
   - 或者记录时已经上采样到1000Hz以匹配EEG

2. **如果确实需要100Hz采样率**：
   - 需要在 `build_eeg_dataset.py` 中添加下采样步骤
   - 或者在使用IMU数据时进行下采样

3. **当前vestibular_model.py中的使用**：
   - 代码中使用 `dt = 1.0 / srate`，其中 `srate = 1000`
   - 如果实际应该是100Hz，需要修改为 `srate = 100`

## 建议

1. **确认产品说明书**：
   - 100Hz是指物理采样率还是记录采样率？
   - 是否在记录时已经上采样到1000Hz？

2. **如果需要100Hz**：
   - 在 `build_eeg_dataset.py` 的 `save_modalities` 函数中添加下采样
   - 或者在 `vestibular_model.py` 中读取时下采样

3. **验证方法**：
   - 检查IMU数据的频谱，看是否有100Hz以上的有效信息
   - 如果100Hz以上主要是噪声，说明原始采样率可能是100Hz

## 代码位置

- **BDF读取**：`EEG_trigger/neuracle_lib/readbdfdata.py:85` - `fs = raw.info['sfreq']`
- **数据保存**：`data/preprocessing/build_eeg_dataset.py:502` - 直接保存，无下采样
- **采样率获取**：`data/preprocessing/build_eeg_dataset.py:594` - `srate = int(eeg.get('srate', 1000))`
- **前庭模型使用**：`models/observer_model.py:320` - `dt = 1.0 / srate`（srate=1000）

