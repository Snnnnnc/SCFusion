# IMU数据提取逻辑说明

## 数据提取流程

### 1. 原始数据读取
- **函数**: `read_session_bdf()` (在 `data/preprocessing/build_eeg_dataset.py`)
- **数据源**: BDF文件 (`data.bdf`)
- **读取库**: `neuracle_lib.readbdfdata.readbdfdata()`
- **处理**: 直接读取BDF文件，**不做任何插值或重采样**

### 2. 数据裁剪
- **函数**: `trim_before_start()`
- **操作**: 裁剪开始事件（事件100）之前的数据
- **处理**: 简单的数组切片操作 `data[:, start_index:]`，**不做插值**

### 3. 模态拆分
- **函数**: `split_modalities_by_ch_names()` 和 `save_modalities()`
- **IMU通道识别**: 根据通道名称识别IMU通道
  - `'GYR-X'`, `'GYR-Y'`, `'GYR-Z'` - 陀螺仪（角速度）
  - `'ACC-X'`, `'ACC-Y'`, `'ACC-Z'` - 加速度计
  - `'MAG-X'`, `'MAG-Y'`, `'MAG-Z'` - 磁力计
- **保存操作**: 直接从裁剪后的数据中提取对应通道索引，然后保存为 `imu.npy`
  ```python
  arr = data_trim[np.array(chan_idx, dtype=int), :]
  np.save(out_dir / f"imu.npy", arr)
  ```
- **处理**: **直接提取，不做任何插值、重采样或变换**

## 数据保存格式

### 文件位置
- `processed/{subject}/{session_id}/_modalities/imu.npy`

### 数据格式
- **形状**: `(9, N)` - 9个通道 × N个时间点
- **数据类型**: `float64`
- **采样率**: 1000 Hz（与原始BDF文件一致）

### 通道顺序
根据 `modalities.json` 文件，IMU数据的通道顺序为：
1. `GYR-X` - 陀螺仪X轴（角速度）
2. `GYR-Y` - 陀螺仪Y轴（角速度）
3. `GYR-Z` - 陀螺仪Z轴（角速度）
4. `ACC-X` - 加速度计X轴
5. `ACC-Y` - 加速度计Y轴
6. `ACC-Z` - 加速度计Z轴
7. `MAG-X` - 磁力计X轴
8. `MAG-Y` - 磁力计Y轴
9. `MAG-Z` - 磁力计Z轴

## 关键结论

✅ **保存的是原始数据**：
- IMU数据从BDF文件中直接提取，没有经过任何插值、重采样或变换
- 只进行了简单的裁剪（去除开始事件之前的数据）
- 采样率保持原始1000 Hz

❌ **没有进行的处理**：
- 没有插值
- 没有重采样
- 没有滤波
- 没有归一化
- 没有坐标变换

## 数据质量检查结果

根据 `analyze_imu.py` 的分析结果：
- ✅ 数据完整，无NaN和Inf值
- ⚠️ 发现 `MAG-Z` 通道为常数（7.233234），可能需要检查传感器或数据采集过程
- ✅ 其他通道数据正常变化

## 相关代码文件

1. **数据提取**: `data/preprocessing/build_eeg_dataset.py`
   - `read_session_bdf()`: 读取BDF文件
   - `trim_before_start()`: 裁剪数据
   - `split_modalities_by_ch_names()`: 识别IMU通道
   - `save_modalities()`: 保存IMU数据

2. **数据分析**: `analyze_imu.py`
   - 分析IMU数据质量
   - 绘制时间序列图
   - 生成统计报告

