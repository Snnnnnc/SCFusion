"""
实验配置文件
定义各种实验场景的配置参数
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class SpeedProfileConfig:
    """速度剖面配置"""
    profile_type: str  # "longitudinal", "lateral", "custom"
    duration: float
    dt: float = 0.05
    
    # 纵向剖面参数
    start_speed: float = 0.0  # m/s
    max_speed: float = 15.0   # m/s
    acceleration_phases: List[Dict] = None
    
    # 横向剖面参数
    constant_speed: float = 11.1  # m/s (40 km/h)
    steer_amplitude: float = 0.3
    steer_frequency: float = 0.05  # Hz
    
    # 扰动参数
    bump_enabled: bool = False
    bump_start: float = 210.0
    bump_end: float = 300.0
    bump_freq: float = 0.5
    bump_amp: float = 0.4


@dataclass
class VehicleConfig:
    """车辆配置"""
    model: str = "model3"
    mass: float = 1500.0  # kg
    drag_coefficient: float = 0.3
    rolling_resistance: float = 0.02
    max_engine_force: float = 3000.0  # N
    max_brake_force: float = 8000.0   # N


@dataclass
class MPCConfig:
    """MPC控制器配置"""
    horizon: int = 10
    dt: float = 0.05
    
    # 权重矩阵
    state_weights: List[float] = None  # [x, y, vx, vy, yaw, yaw_rate]
    control_weights: List[float] = None  # [throttle, brake, steer]
    terminal_weights: List[float] = None
    
    def __post_init__(self):
        if self.state_weights is None:
            self.state_weights = [1.0, 1.0, 10.0, 1.0, 1.0, 1.0]
        if self.control_weights is None:
            self.control_weights = [0.1, 0.1, 0.01]
        if self.terminal_weights is None:
            self.terminal_weights = [10.0, 10.0, 100.0, 10.0, 10.0, 10.0]


@dataclass
class CarlaConfig:
    """CARLA环境配置"""
    map_name: str = "Town07"
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0
    
    # 天气配置
    weather: Dict = None
    
    # 时间配置
    time_of_day: float = 12.0  # 小时
    
    def __post_init__(self):
        if self.weather is None:
            self.weather = {
                "cloudiness": 0.0,
                "precipitation": 0.0,
                "wind_intensity": 0.0,
                "fog_density": 0.0
            }


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    experiment_id: str
    speed_profile: SpeedProfileConfig
    vehicle: VehicleConfig
    mpc: MPCConfig
    carla: CarlaConfig
    
    # 输出配置
    output_dir: str = "./experiment_results"
    save_data: bool = True
    generate_plots: bool = True
    
    # 反馈收集配置
    collect_feedback: bool = True
    feedback_interval: float = 5.0  # 秒


# 预定义实验配置
def get_longitudinal_comfort_test_config() -> ExperimentConfig:
    """纵向舒适性测试配置"""
    return ExperimentConfig(
        experiment_id="longitudinal_comfort_test",
        speed_profile=SpeedProfileConfig(
            profile_type="longitudinal",
            duration=300.0,
            start_speed=0.0,
            max_speed=15.0,
            bump_enabled=True
        ),
        vehicle=VehicleConfig(model="model3", mass=1500.0),
        mpc=MPCConfig(horizon=10, dt=0.05),
        carla=CarlaConfig(map_name="Town07", host="localhost", port=2000)
    )


def get_lateral_comfort_test_config() -> ExperimentConfig:
    """横向舒适性测试配置"""
    return ExperimentConfig(
        experiment_id="lateral_comfort_test",
        speed_profile=SpeedProfileConfig(
            profile_type="lateral",
            duration=300.0,
            constant_speed=11.1,  # 40 km/h
            steer_amplitude=0.3,
            steer_frequency=0.05
        ),
        vehicle=VehicleConfig(model="model3", mass=1500.0),
        mpc=MPCConfig(horizon=10, dt=0.05),
        carla=CarlaConfig(map_name="Town03", host="localhost", port=2000)
    )


def get_aggressive_driving_config() -> ExperimentConfig:
    """激进驾驶测试配置"""
    return ExperimentConfig(
        experiment_id="aggressive_driving_test",
        speed_profile=SpeedProfileConfig(
            profile_type="longitudinal",
            duration=180.0,
            start_speed=0.0,
            max_speed=20.0,  # 更高的最大速度
            bump_enabled=False
        ),
        vehicle=VehicleConfig(model="model3", mass=1500.0),
        mpc=MPCConfig(
            horizon=8,  # 更短的预测时域
            dt=0.05,
            state_weights=[1.0, 1.0, 5.0, 1.0, 1.0, 1.0],  # 降低速度跟踪权重
            control_weights=[0.05, 0.05, 0.005]  # 降低控制权重
        ),
        carla=CarlaConfig(map_name="Town07", host="localhost", port=2000)
    )


def get_gentle_driving_config() -> ExperimentConfig:
    """温和驾驶测试配置"""
    return ExperimentConfig(
        experiment_id="gentle_driving_test",
        speed_profile=SpeedProfileConfig(
            profile_type="longitudinal",
            duration=300.0,
            start_speed=0.0,
            max_speed=12.0,  # 较低的最大速度
            bump_enabled=True
        ),
        vehicle=VehicleConfig(model="prius", mass=1400.0),  # 更轻的车辆
        mpc=MPCConfig(
            horizon=15,  # 更长的预测时域
            dt=0.05,
            state_weights=[1.0, 1.0, 15.0, 1.0, 1.0, 1.0],  # 更高的速度跟踪权重
            control_weights=[0.2, 0.2, 0.02]  # 更高的控制权重
        ),
        carla=CarlaConfig(map_name="Town07", host="localhost", port=2000)
    )


def get_custom_speed_profile_config(profile_segments: List[Dict]) -> ExperimentConfig:
    """自定义速度剖面配置"""
    return ExperimentConfig(
        experiment_id="custom_speed_profile_test",
        speed_profile=SpeedProfileConfig(
            profile_type="custom",
            duration=300.0,
            acceleration_phases=profile_segments
        ),
        vehicle=VehicleConfig(model="model3", mass=1500.0),
        mpc=MPCConfig(horizon=10, dt=0.05),
        carla=CarlaConfig(map_name="Town07", host="localhost", port=2000)
    )


# 实验配置集合
EXPERIMENT_CONFIGS = {
    "longitudinal_comfort": get_longitudinal_comfort_test_config(),
    "lateral_comfort": get_lateral_comfort_test_config(),
    "aggressive_driving": get_aggressive_driving_config(),
    "gentle_driving": get_gentle_driving_config(),
}


def get_experiment_config(config_name: str) -> ExperimentConfig:
    """获取预定义实验配置"""
    if config_name in EXPERIMENT_CONFIGS:
        return EXPERIMENT_CONFIGS[config_name]
    else:
        raise ValueError(f"未知的实验配置: {config_name}")


def list_available_configs() -> List[str]:
    """列出可用的实验配置"""
    return list(EXPERIMENT_CONFIGS.keys())


if __name__ == "__main__":
    # 测试配置
    print("可用的实验配置:")
    for config_name in list_available_configs():
        print(f"  - {config_name}")
    
    # 示例：获取纵向舒适性测试配置
    config = get_experiment_config("longitudinal_comfort")
    print(f"\n纵向舒适性测试配置:")
    print(f"  实验ID: {config.experiment_id}")
    print(f"  持续时间: {config.speed_profile.duration}s")
    print(f"  地图: {config.carla.map_name}")
    print(f"  车辆: {config.vehicle.model}")

