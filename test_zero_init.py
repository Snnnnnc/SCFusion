"""
测试零向量初始化是否会导致NaN
"""
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from models.vestibular_model import VestibularModel, gravity_conflict

def test_zero_initialization():
    """测试零向量初始化的情况"""
    print("=" * 60)
    print("测试零向量初始化")
    print("=" * 60)
    
    dt = 1.0 / 250.0
    model = VestibularModel(dt=dt)
    model.reset()
    
    print(f"\n初始状态:")
    print(f"  omega_est: {model.omega_est}")
    print(f"  f_hat: {model.f_hat}")
    
    # 测试第一次调用
    print(f"\n第一次调用（使用非零输入）:")
    acc_world_G = np.array([0.1, -0.8, -0.6])
    gyro_head_dps = np.array([0.75, -3.67, -0.57])
    
    result = model.step(acc_world_G, gyro_head_dps, gravity_switch=0)
    
    print(f"  输入 acc_world_G: {acc_world_G}")
    print(f"  输入 gyro_head_dps: {gyro_head_dps}")
    print(f"\n  外部模型输出:")
    print(f"    oto_out: {result['oto_out']}")
    print(f"    scc_out: {result['scc_out']}")
    print(f"\n  内部模型输出:")
    print(f"    oto_int_out: {result['oto_int_out']}")
    print(f"    scc_int_out: {result['scc_int_out']}")
    print(f"\n  误差信号:")
    print(f"    e_scc: {result['e_scc']}")
    print(f"    e_oto: {result['e_oto']}")
    print(f"    e_v: {result['e_v']}")
    print(f"\n  内部状态更新:")
    print(f"    omega_est: {result['omega_est']}")
    print(f"    f_hat: {result['f_hat']}")
    print(f"    k_out: {result['k_out']}")
    
    # 检查NaN
    has_nan = False
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            if np.isnan(value).any():
                print(f"\n  ⚠️  {key} 包含 NaN!")
                has_nan = True
        elif isinstance(value, (int, float)):
            if np.isnan(value):
                print(f"\n  ⚠️  {key} 是 NaN!")
                has_nan = True
    
    if not has_nan:
        print(f"\n  ✅ 第一次调用没有NaN")
    
    # 测试gravity_conflict在零向量情况下的行为
    print(f"\n测试 gravity_conflict 函数:")
    print(f"  情况1: 两个非零向量")
    u1 = np.array([1.0, 2.0, 3.0])
    v1 = np.array([0.5, 1.0, 1.5])
    ev1 = gravity_conflict(u1, v1)
    print(f"    u: {u1}, v: {v1}")
    print(f"    e_v: {ev1}, NaN: {np.isnan(ev1).any()}")
    
    print(f"\n  情况2: 一个零向量，一个非零向量")
    u2 = np.array([0.0, 0.0, 0.0])
    v2 = np.array([1.0, 2.0, 3.0])
    ev2 = gravity_conflict(u2, v2)
    print(f"    u: {u2}, v: {v2}")
    print(f"    e_v: {ev2}, NaN: {np.isnan(ev2).any()}")
    
    print(f"\n  情况3: 两个零向量")
    u3 = np.array([0.0, 0.0, 0.0])
    v3 = np.array([0.0, 0.0, 0.0])
    ev3 = gravity_conflict(u3, v3)
    print(f"    u: {u3}, v: {v3}")
    print(f"    e_v: {ev3}, NaN: {np.isnan(ev3).any()}")
    
    print(f"\n  情况4: oto_out 非零，oto_int_out 为零（第一次调用的情况）")
    u4 = np.array([-0.4, 33.5, -17.0])  # 模拟 oto_out
    v4 = np.array([0.0, 0.0, 0.0])      # 模拟 oto_int_out（第一次调用时为零）
    ev4 = gravity_conflict(u4, v4)
    print(f"    u: {u4}, v: {v4}")
    print(f"    e_v: {ev4}, NaN: {np.isnan(ev4).any()}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_zero_initialization()

