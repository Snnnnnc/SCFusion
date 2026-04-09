"""
诊断前庭模型滤波器问题的脚本
检查：
1. 滤波器参数和稳定性
2. 输入数据范围
3. 滤波器输出的中间值
4. 状态变量的变化
"""

import numpy as np
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models.vestibular_model import (
    VestibularModel, 
    ContinuousZPKFilter, 
    MultiAxisZPKFilter,
    OtolithZPK3,
    SCC3External,
    SCC3Internal
)
from data.preprocessing.process_vestibular_conflicts import load_imu_data, extract_imu_channels


def check_filter_stability(zeros_s, poles_s, gain_s, dt, filter_name):
    """检查滤波器的稳定性"""
    print(f"\n{'='*60}")
    print(f"检查滤波器: {filter_name}")
    print(f"{'='*60}")
    
    print(f"连续域参数:")
    print(f"  零点: {zeros_s}")
    print(f"  极点: {poles_s}")
    print(f"  增益: {gain_s}")
    print(f"  采样周期: {dt} s")
    
    # 双线性变换
    T = dt
    eps_bilinear = 1e-10
    
    zeros_z = []
    for z in zeros_s:
        denominator = 2.0 - T*z
        if abs(denominator) < eps_bilinear:
            denominator = np.sign(denominator) * eps_bilinear if denominator != 0 else eps_bilinear
        zeros_z.append((2.0 + T*z) / denominator)
    
    poles_z = []
    for p in poles_s:
        denominator = 2.0 - T*p
        if abs(denominator) < eps_bilinear:
            denominator = np.sign(denominator) * eps_bilinear if denominator != 0 else eps_bilinear
        poles_z.append((2.0 + T*p) / denominator)
    
    print(f"\n离散域参数 (双线性变换后):")
    print(f"  零点: {zeros_z}")
    print(f"  极点: {poles_z}")
    
    # 计算极点幅度
    pole_magnitudes = [abs(p) for p in poles_z]
    max_pole_magnitude = max(pole_magnitudes) if pole_magnitudes else 0.0
    
    print(f"\n极点幅度:")
    for i, (p, mag) in enumerate(zip(poles_z, pole_magnitudes)):
        status = "✅ 稳定" if mag < 1.0 else "❌ 不稳定"
        print(f"  极点 {i+1}: {p:.6f}, 幅度={mag:.6f} {status}")
    
    if max_pole_magnitude > 1.0:
        print(f"\n⚠️  警告: 滤波器不稳定！最大极点幅度={max_pole_magnitude:.6f} > 1.0")
        return False
    else:
        print(f"\n✅ 滤波器稳定 (最大极点幅度={max_pole_magnitude:.6f} < 1.0)")
        return True


def test_filter_with_data(filter, input_data, filter_name, max_samples=1000):
    """用实际数据测试滤波器"""
    print(f"\n{'='*60}")
    print(f"测试滤波器: {filter_name}")
    print(f"{'='*60}")
    
    n_samples = min(len(input_data), max_samples)
    input_data = input_data[:n_samples]
    
    print(f"输入数据统计:")
    print(f"  样本数: {n_samples}")
    print(f"  范围: [{np.min(input_data):.6f}, {np.max(input_data):.6f}]")
    print(f"  均值: {np.mean(input_data):.6f}")
    print(f"  标准差: {np.std(input_data):.6f}")
    print(f"  NaN数量: {np.isnan(input_data).sum()}")
    print(f"  Inf数量: {np.isinf(input_data).sum()}")
    
    # 重置滤波器
    filter.reset()
    
    # 逐步处理
    outputs = []
    state_history = []
    explosion_points = []
    
    for i, x in enumerate(input_data):
        # 记录状态
        if hasattr(filter, 'state'):
            state_history.append(filter.state.copy())
        elif hasattr(filter, 'filters'):  # MultiAxisZPKFilter
            state_history.append([f.state.copy() for f in filter.filters])
        
        # 处理
        try:
            y = filter.step(x)
            outputs.append(y)
            
            # 检查状态是否爆炸
            if hasattr(filter, 'state'):
                state_mag = np.max(np.abs(filter.state))
                if state_mag > 1e10:
                    explosion_points.append((i, state_mag, filter.state.copy()))
            elif hasattr(filter, 'filters'):
                max_state_mag = max([np.max(np.abs(f.state)) for f in filter.filters])
                if max_state_mag > 1e10:
                    explosion_points.append((i, max_state_mag, [f.state.copy() for f in filter.filters]))
        except Exception as e:
            print(f"❌ 处理失败 (样本 {i}): {e}")
            break
    
    outputs = np.array(outputs)
    
    print(f"\n输出数据统计:")
    print(f"  范围: [{np.min(outputs):.6f}, {np.max(outputs):.6f}]")
    print(f"  均值: {np.mean(outputs):.6f}")
    print(f"  标准差: {np.std(outputs):.6f}")
    print(f"  NaN数量: {np.isnan(outputs).sum()}")
    print(f"  Inf数量: {np.isinf(outputs).sum()}")
    
    if explosion_points:
        print(f"\n⚠️  发现 {len(explosion_points)} 个状态爆炸点:")
        for idx, mag, state in explosion_points[:5]:  # 只显示前5个
            print(f"  样本 {idx}: 状态幅度={mag:.2e}")
    
    # 检查状态增长趋势
    if state_history:
        if isinstance(state_history[0], list):  # MultiAxisZPKFilter
            state_magnitudes = [max([np.max(np.abs(s)) for s in states]) for states in state_history]
        else:
            state_magnitudes = [np.max(np.abs(s)) for s in state_history]
        
        print(f"\n状态变化趋势:")
        print(f"  初始状态幅度: {state_magnitudes[0]:.6e}")
        print(f"  最终状态幅度: {state_magnitudes[-1]:.6e}")
        print(f"  最大状态幅度: {max(state_magnitudes):.6e}")
        
        # 检查是否持续增长
        if len(state_magnitudes) > 100:
            early_avg = np.mean(state_magnitudes[:100])
            late_avg = np.mean(state_magnitudes[-100:])
            if late_avg > early_avg * 10:
                print(f"  ⚠️  状态持续增长: 早期平均={early_avg:.6e}, 后期平均={late_avg:.6e}")
    
    return outputs, state_history


def main():
    """主函数"""
    print("="*60)
    print("前庭模型滤波器诊断")
    print("="*60)
    
    # 1. 检查滤波器参数和稳定性
    dt = 0.01
    f_oto = 2.0
    tau_scc = 5.7
    tau_a = 80.0
    f_scc = 2.0
    
    # OTO滤波器
    oto_zeros = []
    oto_poles = [-2.0 * np.pi * f_oto]
    oto_gain = 2.0 * np.pi * f_oto
    check_filter_stability(oto_zeros, oto_poles, oto_gain, dt, "OTO滤波器")
    
    # SCC外部滤波器
    scc_ext_zeros = [0.0, 0.0]
    scc_ext_poles = [-1.0 / tau_scc, -1.0 / tau_a, -2.0 * np.pi * f_scc]
    scc_ext_gain = 2.0 * np.pi * f_scc
    check_filter_stability(scc_ext_zeros, scc_ext_poles, scc_ext_gain, dt, "SCC外部滤波器")
    
    # SCC内部滤波器
    scc_int_zeros = [0.0]
    scc_int_poles = [-1.0 / tau_scc, -2.0 * np.pi * f_scc]
    scc_int_gain = 2.0 * np.pi * f_scc
    check_filter_stability(scc_int_zeros, scc_int_poles, scc_int_gain, dt, "SCC内部滤波器")
    
    # 2. 加载实际数据
    print(f"\n{'='*60}")
    print("加载实际IMU数据")
    print(f"{'='*60}")
    
    # 找一个有问题的session
    project_root = Path(__file__).parent
    processed_root = project_root / "processed"
    
    # 查找zzh被试的session
    zzh_dir = processed_root / "zzh"
    if not zzh_dir.exists():
        print(f"❌ processed/zzh目录不存在: {zzh_dir}")
        print("请指定一个有效的会话目录")
        return
    
    # 查找第一个session
    sessions = [d for d in zzh_dir.iterdir() if d.is_dir() and (d / "_modalities" / "imu.npy").exists()]
    if not sessions:
        print(f"❌ 未找到有效的session目录")
        return
    
    session_dir = sessions[0]
    print(f"使用session: {session_dir.name}")
    
    try:
        imu_data, channel_info = load_imu_data(session_dir)
        gyro_dps, acc_G = extract_imu_channels(imu_data, channel_info['channel_names'])
        
        print(f"✅ 成功加载数据")
        print(f"  角速度形状: {gyro_dps.shape}")
        print(f"  加速度形状: {acc_G.shape}")
        print(f"  角速度范围: [{np.min(gyro_dps):.2f}, {np.max(gyro_dps):.2f}] deg/s")
        print(f"  加速度范围: [{np.min(acc_G):.2f}, {np.max(acc_G):.2f}] G")
        
        # 转换为rad/s
        gyro_rad = gyro_dps * np.pi / 180.0
        print(f"  角速度范围(rad/s): [{np.min(gyro_rad):.6f}, {np.max(gyro_rad):.6f}] rad/s")
        
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 测试各个滤波器
    print(f"\n{'='*60}")
    print("测试各个滤波器")
    print(f"{'='*60}")
    
    # 测试OTO滤波器（使用加速度的3个通道，返回3维向量）
    oto_filter = OtolithZPK3(f_oto, dt)
    print(f"\n{'='*60}")
    print(f"测试滤波器: OTO滤波器 (3轴)")
    print(f"{'='*60}")
    print(f"输入数据统计 (ACC):")
    print(f"  形状: {acc_G.shape}")
    print(f"  范围: [{np.min(acc_G):.6f}, {np.max(acc_G):.6f}] G")
    print(f"  均值: {np.mean(acc_G):.6f} G")
    print(f"  标准差: {np.std(acc_G):.6f} G")
    
    oto_filter.reset()
    n_test = min(1000, len(acc_G))
    oto_outputs = []
    for i in range(n_test):
        output = oto_filter.step(acc_G[i, :])
        oto_outputs.append(output)
    oto_outputs = np.array(oto_outputs)
    print(f"\n输出数据统计:")
    print(f"  形状: {oto_outputs.shape}")
    print(f"  范围: [{np.min(oto_outputs):.6f}, {np.max(oto_outputs):.6f}]")
    print(f"  均值: {np.mean(oto_outputs):.6f}")
    print(f"  标准差: {np.std(oto_outputs):.6f}")
    print(f"  NaN数量: {np.isnan(oto_outputs).sum()}")
    print(f"  Inf数量: {np.isinf(oto_outputs).sum()}")
    
    # 测试SCC外部滤波器（使用角速度的3个通道）
    scc_ext_filter = SCC3External(tau_scc, tau_a, f_scc, dt)
    print(f"\n{'='*60}")
    print(f"测试滤波器: SCC外部滤波器 (3轴)")
    print(f"{'='*60}")
    print(f"输入数据统计 (GYR, rad/s):")
    print(f"  形状: {gyro_rad.shape}")
    print(f"  范围: [{np.min(gyro_rad):.6f}, {np.max(gyro_rad):.6f}] rad/s")
    print(f"  均值: {np.mean(gyro_rad):.6f} rad/s")
    print(f"  标准差: {np.std(gyro_rad):.6f} rad/s")
    
    scc_ext_filter.reset()
    scc_ext_outputs = []
    for i in range(n_test):
        output = scc_ext_filter.step(gyro_rad[i, :])
        scc_ext_outputs.append(output)
    scc_ext_outputs = np.array(scc_ext_outputs)
    print(f"\n输出数据统计:")
    print(f"  形状: {scc_ext_outputs.shape}")
    print(f"  范围: [{np.min(scc_ext_outputs):.6f}, {np.max(scc_ext_outputs):.6f}]")
    print(f"  均值: {np.mean(scc_ext_outputs):.6f}")
    print(f"  标准差: {np.std(scc_ext_outputs):.6f}")
    print(f"  NaN数量: {np.isnan(scc_ext_outputs).sum()}")
    print(f"  Inf数量: {np.isinf(scc_ext_outputs).sum()}")
    
    # 测试SCC内部滤波器
    scc_int_filter = SCC3Internal(tau_scc, f_scc, dt)
    print(f"\n{'='*60}")
    print(f"测试滤波器: SCC内部滤波器 (3轴)")
    print(f"{'='*60}")
    
    scc_int_filter.reset()
    scc_int_outputs = []
    for i in range(n_test):
        output = scc_int_filter.step(gyro_rad[i, :])
        scc_int_outputs.append(output)
    scc_int_outputs = np.array(scc_int_outputs)
    print(f"\n输出数据统计:")
    print(f"  形状: {scc_int_outputs.shape}")
    print(f"  范围: [{np.min(scc_int_outputs):.6f}, {np.max(scc_int_outputs):.6f}]")
    print(f"  均值: {np.mean(scc_int_outputs):.6f}")
    print(f"  标准差: {np.std(scc_int_outputs):.6f}")
    print(f"  NaN数量: {np.isnan(scc_int_outputs).sum()}")
    print(f"  Inf数量: {np.isinf(scc_int_outputs).sum()}")
    
    # 检查状态
    print(f"\n{'='*60}")
    print(f"检查滤波器状态")
    print(f"{'='*60}")
    print(f"OTO滤波器状态幅度:")
    for i, f in enumerate(oto_filter.filt.filters):
        state_mag = np.max(np.abs(f.state)) if len(f.state) > 0 else 0.0
        print(f"  轴 {i}: {state_mag:.6e}")
    
    print(f"\nSCC外部滤波器状态幅度:")
    for i, f in enumerate(scc_ext_filter.filt.filters):
        state_mag = np.max(np.abs(f.state)) if len(f.state) > 0 else 0.0
        print(f"  轴 {i}: {state_mag:.6e}")
    
    print(f"\nSCC内部滤波器状态幅度:")
    for i, f in enumerate(scc_int_filter.filt.filters):
        state_mag = np.max(np.abs(f.state)) if len(f.state) > 0 else 0.0
        print(f"  轴 {i}: {state_mag:.6e}")
    
    # 4. 完整运行前庭模型，检查状态变化
    print(f"\n{'='*60}")
    print(f"完整运行前庭模型 (前5000个样本)")
    print(f"{'='*60}")
    
    from models.vestibular_model import VestibularModel
    
    vestibular_model = VestibularModel(dt=dt)
    vestibular_model.reset()
    
    # 运行部分数据
    n_test_full = min(5000, len(gyro_dps))
    print(f"运行 {n_test_full} 个样本...")
    explosion_points = []
    state_history = []
    output_history = {'e_scc': [], 'e_oto': [], 'k_out': []}
    
    for i in range(n_test_full):
        # 检查状态
        if i % 500 == 0 or i < 10:
            # OTO状态
            oto_state_mags = [np.max(np.abs(f.state)) for f in vestibular_model.oto.filt.filters]
            scc_ext_state_mags = [np.max(np.abs(f.state)) for f in vestibular_model.scc.filt.filters]
            scc_int_state_mags = [np.max(np.abs(f.state)) for f in vestibular_model.scc_int.filt.filters]
            max_state = max(oto_state_mags + scc_ext_state_mags + scc_int_state_mags)
            state_history.append((i, max_state))
            
            if max_state > 1e10:
                explosion_points.append((i, max_state))
                print(f"⚠️  样本 {i}: 状态爆炸! 最大状态幅度={max_state:.2e}")
        
        # 运行一步
        try:
            result = vestibular_model.step(acc_G[i, :], gyro_dps[i, :])
            
            # 记录输出
            output_history['e_scc'].append(np.max(np.abs(result['e_scc'])))
            output_history['e_oto'].append(np.max(np.abs(result['e_oto'])))
            output_history['k_out'].append(np.max(np.abs(result['k_out'])))
            
            # 检查输出
            if i % 1000 == 0:
                e_scc_mag = np.max(np.abs(result['e_scc']))
                e_oto_mag = np.max(np.abs(result['e_oto']))
                k_out_mag = np.max(np.abs(result['k_out']))
                if e_scc_mag > 1e6 or e_oto_mag > 1e6 or k_out_mag > 1e6:
                    print(f"⚠️  样本 {i}: 输出异常大! e_scc={e_scc_mag:.2e}, e_oto={e_oto_mag:.2e}, k_out={k_out_mag:.2e}")
        except Exception as e:
            print(f"❌ 样本 {i} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print(f"\n状态变化总结:")
    if state_history:
        early_states = [s for i, s in state_history if i < 500]
        late_states = [s for i, s in state_history if i >= n_test_full - 500]
        if early_states and late_states:
            print(f"  早期平均状态幅度: {np.mean(early_states):.6e}")
            print(f"  后期平均状态幅度: {np.mean(late_states):.6e}")
            print(f"  最大状态幅度: {max([s for _, s in state_history]):.6e}")
    
    print(f"\n输出变化总结:")
    for key, values in output_history.items():
        if values:
            print(f"  {key}: 范围=[{np.min(values):.2e}, {np.max(values):.2e}], 均值={np.mean(values):.2e}")
    
    if explosion_points:
        print(f"\n⚠️  发现 {len(explosion_points)} 个状态爆炸点")
    else:
        print(f"\n✅ 未发现状态爆炸")
    
    print(f"\n{'='*60}")
    print("诊断完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

