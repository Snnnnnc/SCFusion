"""
调试前庭模型计算中的NaN问题
"""
import numpy as np
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from models.vestibular_model import VestibularModel

def test_vestibular_model():
    """测试前庭模型，检查NaN产生的位置"""
    print("=" * 60)
    print("测试前庭模型计算")
    print("=" * 60)
    
    # 加载一些真实的IMU数据来测试
    data_dir = project_root / "data" / "preprocessed"
    imu_file = data_dir / "imu_concatenated.npy"
    
    if not imu_file.exists():
        print(f"❌ 找不到IMU数据文件: {imu_file}")
        return
    
    print(f"📂 加载IMU数据: {imu_file}")
    imu_data = np.load(imu_file)  # (18, N)
    print(f"  数据形状: {imu_data.shape}")
    
    # 提取前6个通道（原始IMU数据）
    gyro_dps = imu_data[0:3, :].T  # (N, 3)
    acc_G = imu_data[3:6, :].T     # (N, 3)
    
    print(f"  角速度形状: {gyro_dps.shape}")
    print(f"  加速度形状: {acc_G.shape}")
    
    # 检查输入数据中的NaN
    gyro_nan = np.isnan(gyro_dps).sum()
    acc_nan = np.isnan(acc_G).sum()
    print(f"\n输入数据检查:")
    print(f"  角速度NaN数量: {gyro_nan}")
    print(f"  加速度NaN数量: {acc_nan}")
    
    if gyro_nan > 0 or acc_nan > 0:
        print("  ⚠️  输入数据包含NaN，这可能是问题的根源！")
        # 找到第一个NaN的位置
        if gyro_nan > 0:
            nan_idx = np.where(np.isnan(gyro_dps))[0]
            print(f"  角速度第一个NaN位置: 索引 {nan_idx[0] if len(nan_idx) > 0 else 'N/A'}")
        if acc_nan > 0:
            nan_idx = np.where(np.isnan(acc_G))[0]
            print(f"  加速度第一个NaN位置: 索引 {nan_idx[0] if len(nan_idx) > 0 else 'N/A'}")
    
    # 创建前庭模型
    dt = 1.0 / 250.0  # 250Hz
    print(f"\n🔄 创建前庭模型 (dt={dt:.6f}s)...")
    vestibular_model = VestibularModel(dt=dt)
    vestibular_model.reset()
    
    # 测试前1000个时间步
    N_test = min(1000, len(gyro_dps))
    print(f"  测试前 {N_test} 个时间步...")
    
    nan_detected = False
    for i in range(N_test):
        try:
            result = vestibular_model.step(
                acc_world_G=acc_G[i],
                gyro_head_dps=gyro_dps[i],
                gravity_switch=0
            )
            
            # 检查输出中的NaN
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    if np.isnan(value).any() or np.isinf(value).any():
                        if not nan_detected:
                            print(f"\n⚠️  在时间步 {i} 检测到NaN/Inf:")
                            nan_detected = True
                        nan_count = np.isnan(value).sum()
                        inf_count = np.isinf(value).sum()
                        print(f"  {key}: NaN={nan_count}, Inf={inf_count}, shape={value.shape}")
                        
                        # 如果是冲突信号，打印详细信息
                        if key in ['e_scc', 'e_oto', 'e_v', 'k_out']:
                            print(f"    值: {value}")
                            print(f"    输入 acc_world_G: {acc_G[i]}")
                            print(f"    输入 gyro_head_dps: {gyro_dps[i]}")
                            
                            # 检查中间变量
                            if 'oto_out' in result:
                                print(f"    oto_out: {result['oto_out']}")
                            if 'oto_int_out' in result:
                                print(f"    oto_int_out: {result['oto_int_out']}")
                            if 'scc_out' in result:
                                print(f"    scc_out: {result['scc_out']}")
                            if 'scc_int_out' in result:
                                print(f"    scc_int_out: {result['scc_int_out']}")
                            
                            break  # 只打印第一个NaN
        except Exception as e:
            print(f"\n❌ 在时间步 {i} 发生错误: {e}")
            import traceback
            traceback.print_exc()
            break
    
    if not nan_detected:
        print(f"\n✅ 前 {N_test} 个时间步未检测到NaN/Inf")
    
    # 检查模型参数
    print(f"\n📊 模型参数检查:")
    print(f"  kww: {vestibular_model.kww}")
    print(f"  kwg: {vestibular_model.kwg}")
    print(f"  kwvw: {vestibular_model.kwvw}")
    print(f"  kill_value: {vestibular_model.kill_value}")
    
    # 检查K_block可能的除零问题
    if vestibular_model.kww == 0:
        print("  ⚠️  kww == 0，K_block中可能出现除零错误！")
    if vestibular_model.kww + vestibular_model.kwvw == 0:
        print("  ⚠️  kww + kwvw == 0，K_block中可能出现除零错误！")
    
    print("\n" + "=" * 60)
    print("调试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_vestibular_model()

