# 基于数据集的MPC控制序列生成器

## 概述

本项目实现了一个基于真实自动驾驶车辆轨迹数据的MPC控制序列生成器。与传统的简单平滑函数不同，该系统直接使用数据集中现成的速度序列作为参考，生成更加真实和实用的控制序列。

## 主要特性

- **真实数据驱动**: 使用Ultra-AV数据集中的真实自动驾驶车辆轨迹
- **10Hz数据频率**: 适配数据集的10Hz采样频率（0.1s间隔）
- **多数据集支持**: 支持CATS、Vanderbilt、Ohio、Waymo等多个数据集
- **MPC控制**: 基于模型预测控制的智能控制策略
- **平滑控制**: 优化的控制平滑性，避免剧烈变化
- **性能分析**: 完整的速度跟踪性能和控制质量分析

## 文件结构

```
├── dataset_mpc_controller.py    # 主要实现文件
├── test_dataset_mpc.py         # 测试脚本
├── example_dataset_mpc.py      # 使用示例
└── DATASET_MPC_README.md       # 本文档
```

## 数据集格式

系统支持的数据集包含以下字段：

| 字段名 | 描述 | 单位 |
|--------|------|------|
| Trajectory_ID | 轨迹ID | - |
| Time_Index | 时间戳 | s |
| Speed_FAV | 跟随车辆速度 | m/s |
| Acc_FAV | 跟随车辆加速度 | m/s² |
| Pos_FAV | 跟随车辆位置 | m |

## 核心组件

### 1. DatasetLoader
数据集加载器，支持：
- 自动发现可用数据集
- 加载CSV格式的轨迹数据
- 提取速度剖面信息

### 2. VehicleModel
车辆动力学模型：
- 简化的车辆动力学方程
- 考虑阻力、滚动阻力等因素
- 适配10Hz数据频率

### 3. DatasetMPCController
基于数据集的MPC控制器：
- 增强的PID控制策略
- 考虑加速度参考
- 控制平滑性优化
- 控制变化率限制

### 4. DatasetControlGenerator
控制序列生成器：
- 根据真实轨迹生成控制序列
- 状态序列预测
- 控制序列验证
- 性能分析

## 使用方法

### 基本使用

```python
from dataset_mpc_controller import DatasetLoader, VehicleModel, DatasetMPCController, DatasetControlGenerator

# 1. 初始化组件
data_root = "path/to/dataset"
loader = DatasetLoader(data_root)
vehicle_model = VehicleModel()
mpc_controller = DatasetMPCController(vehicle_model, horizon=5, dt=0.1)
control_generator = DatasetControlGenerator(vehicle_model, mpc_controller)

# 2. 加载数据
df = loader.load_trajectory_data("CATS", "step3_ACC.csv")

# 3. 提取速度剖面
speed_profile = loader.extract_speed_profile(
    df, 
    trajectory_id=0, 
    start_time=0.0, 
    duration=30.0
)

# 4. 生成控制序列
control_data = control_generator.generate_control_sequence(speed_profile)

# 5. 验证和可视化
is_valid, warnings = control_generator.validate_control_sequence(control_data)
control_generator.plot_control_sequence(control_data)
```

### 多轨迹处理

```python
# 获取所有轨迹ID
trajectory_ids = df['Trajectory_ID'].unique()

for traj_id in trajectory_ids[:5]:  # 处理前5个轨迹
    speed_profile = loader.extract_speed_profile(df, trajectory_id=traj_id, duration=20.0)
    control_data = control_generator.generate_control_sequence(speed_profile)
    # 处理结果...
```

### 不同数据集

```python
# 测试不同数据集
datasets = [
    ("CATS", "step3_ACC.csv"),
    ("Vanderbilt", "step2_two_vehicle_ACC.csv"),
    ("Ohio", "step3_single_vehicle.csv")
]

for dataset_name, file_name in datasets:
    df = loader.load_trajectory_data(dataset_name, file_name)
    # 处理数据...
```

## 性能指标

系统提供以下性能指标：

### 速度跟踪性能
- 最大速度误差
- 平均速度误差
- RMS速度误差

### 控制质量
- 控制变化率
- 控制平滑性
- 控制范围验证

### 数据质量
- 数据完整性检查
- 时间间隔验证
- 数据范围检查

## 配置参数

### MPC控制器参数
```python
mpc_controller = DatasetMPCController(
    vehicle_model, 
    horizon=5,      # 预测时域
    dt=0.1         # 时间步长（10Hz）
)
```

### 车辆模型参数
```python
vehicle_model = VehicleModel(
    mass=1500.0,           # 车辆质量
    drag_coeff=0.3,       # 阻力系数
    rolling_resistance=0.02  # 滚动阻力
)
```

### PID控制参数
- kp = 0.25  # 比例增益
- ki = 0.08  # 积分增益
- kd = 0.05  # 微分增益
- max_change = 0.2  # 最大控制变化率

## 输出格式

控制序列输出包含：

```python
control_data = {
    'time': time_points,                    # 时间序列
    'control_sequence': control_array,      # 控制序列 [throttle, brake, steer]
    'state_sequence': state_array,          # 状态序列 [x, y, vx, vy, yaw, yaw_rate]
    'target_speed': target_speeds,          # 目标速度
    'target_acceleration': target_accs,     # 目标加速度
    'data_source': 'real_trajectory'       # 数据源标识
}
```

## 可视化功能

系统提供丰富的可视化功能：

1. **控制信号图**: 油门、刹车、转向控制
2. **速度跟踪图**: 目标速度vs实际速度
3. **速度误差图**: 跟踪误差分析
4. **加速度对比图**: 目标加速度vs实际加速度
5. **轨迹图**: 车辆运动轨迹
6. **控制平滑性图**: 控制变化率分析

## 依赖项

```
numpy
pandas
matplotlib
cvxpy (可选，用于高级MPC优化)
```

## 安装依赖

```bash
pip install numpy pandas matplotlib
pip install cvxpy  # 可选，用于高级MPC优化
```

## 运行示例

```bash
# 运行测试
python test_dataset_mpc.py

# 运行示例
python example_dataset_mpc.py
```

## 注意事项

1. **数据路径**: 确保数据集路径正确
2. **内存使用**: 大数据集可能需要较多内存
3. **计算时间**: 长轨迹可能需要较长计算时间
4. **控制平滑性**: 系统已优化控制平滑性，但极端情况下可能仍有突变

## 故障排除

### 常见问题

1. **数据集加载失败**
   - 检查数据路径是否正确
   - 确认CSV文件格式正确

2. **控制序列验证失败**
   - 检查控制参数是否合理
   - 调整PID参数

3. **性能不佳**
   - 调整MPC参数
   - 检查车辆模型参数

### 调试模式

```python
# 启用详细输出
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

### 自定义车辆模型
```python
class CustomVehicleModel(VehicleModel):
    def dynamics(self, state, control, dt):
        # 自定义动力学模型
        pass
```

### 自定义控制器
```python
class CustomMPCController(DatasetMPCController):
    def solve_mpc(self, state, ref_speed, ref_acc, ref_steer):
        # 自定义控制策略
        pass
```

## 贡献

欢迎提交问题和改进建议！

## 许可证

本项目遵循MIT许可证。
