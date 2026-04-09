"""
诊断前庭模型反馈回路问题
检查初始阶段的误差信号和反馈增益
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from models.vestibular_model import VestibularModel
from data.preprocessing.process_vestibular_conflicts import load_imu_data, extract_imu_channels


def main():
    project_root = Path(__file__).parent
    processed_root = project_root / "processed"
    
    # 查找session
    zzh_dir = processed_root / "zzh"
    sessions = [d for d in zzh_dir.iterdir() if d.is_dir() and (d / "_modalities" / "imu.npy").exists()]
    if not sessions:
        print("❌ 未找到有效的session目录")
        return
    
    session_dir = sessions[0]
    print(f"使用session: {session_dir.name}")
    
    # 加载数据
    imu_data, channel_info = load_imu_data(session_dir)
    gyro_dps, acc_G = extract_imu_channels(imu_data, channel_info.get('channel_names', []))
    
    srate = 100
    dt = 1.0 / srate
    
    # 创建模型
    vestibular_model = VestibularModel(dt=dt)
    vestibular_model.reset()
    
    print(f"\n反馈增益参数:")
    print(f"  kww={vestibular_model.kww}")
    print(f"  kfw={vestibular_model.kfw}")
    print(f"  kfg={vestibular_model.kfg}")
    print(f"  kaa={vestibular_model.kaa}")
    print(f"  kwg={vestibular_model.kwg}")
    print(f"  kwvw={vestibular_model.kwvw}")
    
    # 检查前100个样本
    print(f"\n{'='*60}")
    print(f"检查前100个样本的反馈回路")
    print(f"{'='*60}")
    
    print(f"{'样本':<8} {'e_scc_mag':<15} {'e_oto_mag':<15} {'e_v_mag':<15} {'omega_est_mag':<15} {'f_hat_mag':<15} {'k_out_mag':<15}")
    print("-" * 100)
    
    for i in range(min(100, len(gyro_dps))):
        result = vestibular_model.step(acc_G[i, :], gyro_dps[i, :])
        
        e_scc_mag = np.linalg.norm(result['e_scc'])
        e_oto_mag = np.linalg.norm(result['e_oto'])
        e_v_mag = np.linalg.norm(result['e_v'])
        omega_est_mag = np.linalg.norm(vestibular_model.omega_est)
        f_hat_mag = np.linalg.norm(vestibular_model.f_hat)
        k_out_mag = np.linalg.norm(result['k_out'])
        
        if i < 20 or i % 10 == 0:
            print(f"{i:<8} {e_scc_mag:<15.6e} {e_oto_mag:<15.6e} {e_v_mag:<15.6e} {omega_est_mag:<15.6e} {f_hat_mag:<15.6e} {k_out_mag:<15.6e}")
        
        # 检查是否异常
        if e_scc_mag > 1e6 or e_oto_mag > 1e6 or omega_est_mag > 1e6 or f_hat_mag > 1e6:
            print(f"\n⚠️  样本 {i}: 异常大的值!")
            print(f"  e_scc: {result['e_scc']}")
            print(f"  e_oto: {result['e_oto']}")
            print(f"  omega_est: {vestibular_model.omega_est}")
            print(f"  f_hat: {vestibular_model.f_hat}")
            break
    
    # 检查滤波器状态
    print(f"\n{'='*60}")
    print(f"检查滤波器状态（前100个样本后）")
    print(f"{'='*60}")
    
    oto_state_mags = [np.max(np.abs(f.state)) for f in vestibular_model.oto.filt.filters]
    scc_ext_state_mags = [np.max(np.abs(f.state)) for f in vestibular_model.scc.filt.filters]
    scc_int_state_mags = [np.max(np.abs(f.state)) for f in vestibular_model.scc_int.filt.filters]
    oto_int_state_mags = [np.max(np.abs(f.state)) for f in vestibular_model.oto_int.filt.filters]
    
    print(f"OTO外部滤波器状态幅度: {oto_state_mags}")
    print(f"SCC外部滤波器状态幅度: {scc_ext_state_mags}")
    print(f"SCC内部滤波器状态幅度: {scc_int_state_mags}")
    print(f"OTO内部滤波器状态幅度: {oto_int_state_mags}")
    print(f"最大状态幅度: {max(oto_state_mags + scc_ext_state_mags + scc_int_state_mags + oto_int_state_mags):.6e}")


if __name__ == "__main__":
    main()

