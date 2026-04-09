# 运动病分类实验系统 - 新架构说明

## 架构概述

根据您的要求，我们重新设计了文件结构，采用三层架构：

```
第1层：速度规划模块 (speed_planner.py)
第2层：MPC控制模块 (mpc_controller.py)  
第3层：CARLA交互接口 (carla_interface.py)
主协调器：整合三层架构 (experiment_orchestrator.py)
```

## 文件结构

```
├── path_test.py                    # 原有代码（保留）
├── speed_planner.py               # 第1层：速度规划模块
├── mpc_controller.py              # 第2层：MPC控制模块
├── carla_interface.py             # 第3层：CARLA交互接口
├── experiment_orchestrator.py     # 主协调器
├── config/
│   ├── __init__.py
│   └── experiment_configs.py      # 实验配置
└── ARCHITECTURE_README.md         # 本文档
```

## 三层架构详细说明

### 第1层：速度规划模块 (`speed_planner.py`)

**功能**：根据实验场景设计期望的速度变化模式，生成预期速度序列

**主要类**：
- `SpeedPlanner`: 速度规划器主类

**主要方法**：
- `generate_longitudinal_profile()`: 生成纵向速度剖面
- `generate_lateral_profile()`: 生成横向速度剖面
- `generate_custom_profile()`: 生成自定义速度剖面
- `validate_profile()`: 验证速度剖面的合理性
- `plot_speed_profile()`: 绘制速度剖面图

**特点**：
- 支持多种速度剖面类型（纵向、横向、自定义）
- 内置平滑过渡函数，避免速度跳变
- 包含扰动参数（bump）支持
- 自动验证速度剖面的合理性

### 第2层：MPC控制模块 (`mpc_controller.py`)

**功能**：根据预期速度，结合车辆动力学模型和MPC控制生成车辆控制序列

**主要类**：
- `VehicleModel`: 简化的车辆动力学模型
- `MPCController`: 模型预测控制器
- `ControlSequenceGenerator`: 控制序列生成器

**主要方法**：
- `solve_mpc()`: 求解MPC优化问题
- `generate_control_sequence()`: 生成完整的控制序列
- `validate_control_sequence()`: 验证控制序列的合理性
- `plot_control_sequence()`: 绘制控制序列图

**特点**：
- 基于cvxpy的MPC优化求解
- 考虑车辆动力学约束
- 支持舒适性权重调整
- 包含备用简单控制策略

### 第3层：CARLA交互接口 (`carla_interface.py`)

**功能**：输入控制序列，直接与CARLA环境交互实现可视化和运动状态洗出，同步收集被试反馈

**主要类**：
- `CarlaEnvironment`: CARLA环境管理器
- `DataLogger`: 数据记录器
- `FeedbackCollector`: 被试反馈收集器
- `CarlaInterface`: 主接口类

**主要方法**：
- `initialize()`: 初始化CARLA环境
- `run_experiment()`: 运行实验
- `apply_control()`: 应用控制输入
- `get_vehicle_state()`: 获取车辆状态
- `cleanup()`: 清理资源

**特点**：
- 自动环境配置（天气、时间、地图）
- 智能生成点选择
- 实时数据记录
- 被试反馈收集框架
- 资源自动清理

### 主协调器 (`experiment_orchestrator.py`)

**功能**：整合三层架构，协调完整的实验流程

**主要类**：
- `ExperimentConfig`: 实验配置类
- `ExperimentOrchestrator`: 实验协调器

**主要方法**：
- `run_complete_experiment()`: 运行完整实验流程
- `run_multiple_experiments()`: 运行多个实验
- `_generate_speed_profile()`: 生成速度剖面
- `_generate_control_sequence()`: 生成控制序列
- `_run_carla_experiment()`: 运行CARLA实验
- `_analyze_results()`: 结果分析和可视化

**特点**：
- 端到端实验流程管理
- 自动验证和错误处理
- 结果分析和可视化
- 支持批量实验

## 使用方法

### 1. 运行单个实验

```python
from experiment_orchestrator import ExperimentOrchestrator, create_longitudinal_experiment_config

# 创建实验配置
config = create_longitudinal_experiment_config()

# 创建协调器并运行实验
orchestrator = ExperimentOrchestrator(config)
success = orchestrator.run_complete_experiment()
```

### 2. 运行多个实验

```python
from experiment_orchestrator import ExperimentOrchestrator
from config.experiment_configs import get_longitudinal_comfort_test_config, get_lateral_comfort_test_config

# 创建多个实验配置
configs = [
    get_longitudinal_comfort_test_config(),
    get_lateral_comfort_test_config()
]

# 运行多个实验
orchestrator = ExperimentOrchestrator()
results = orchestrator.run_multiple_experiments(configs)
```

### 3. 使用预定义配置

```python
from config.experiment_configs import get_experiment_config, list_available_configs

# 查看可用配置
print(list_available_configs())

# 获取特定配置
config = get_experiment_config("longitudinal_comfort")
```

## 实验流程

1. **速度规划**：根据实验场景生成目标速度序列
2. **控制生成**：使用MPC控制器生成车辆控制序列
3. **CARLA交互**：在CARLA环境中执行控制序列
4. **数据收集**：记录车辆状态和被试反馈
5. **结果分析**：生成分析图表和报告

## 配置系统

通过 `config/experiment_configs.py` 可以方便地配置：

- **速度剖面参数**：持续时间、速度范围、扰动参数
- **车辆参数**：车型、质量、动力学参数
- **MPC参数**：预测时域、权重矩阵
- **CARLA参数**：地图、天气、时间
- **实验参数**：输出目录、数据保存选项

## 优势

1. **模块化设计**：每层职责清晰，便于维护和扩展
2. **配置灵活**：通过配置文件轻松调整实验参数
3. **错误处理**：完善的错误处理和验证机制
4. **数据完整性**：自动记录所有相关数据
5. **可扩展性**：易于添加新的实验场景和控制策略

## 与原代码的关系

- `path_test.py` 保留原有代码，作为参考
- 新架构完全独立，可以并行使用
- 新架构提供了更好的模块化和可维护性
- 支持更复杂的实验场景和控制策略

