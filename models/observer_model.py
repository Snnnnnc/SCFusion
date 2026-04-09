# Vestibular Observer Model - simplified implementation
# This code implements a workable observer-style vestibular model based on the
# structure described in "Mathematical models for dynamic, multisensory spatial orientation perception".
# It is simplified for clarity and experimentation, and includes:
#  - SCC (half) high-pass approximation via input - lowpass(input)
#  - Otolith handling using specific force f = g - a (IMU acc input as 'f')
#  - Internal model states: g_hat (gravity vector), omega_hat (angular velocity estimate), a_hat (linear accel est)
#  - Sensory conflict signals (ea, ef, eomega) and proportional feedback updates using typical gains from the paper
#  - A run() method to simulate over a time series of IMU inputs
#
# Note: This is an engineering, not a neuroscience-accurate, implementation. It's intended for
#       generating features (sensory conflicts and estimated states) to feed into an IMU-based ML model.
#
# Author: ChatGPT (for user)
# Dependencies: numpy, matplotlib

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
available_fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
preferred_fonts = ['PingFang SC', 'Heiti TC', 'STHeiti', 'Songti SC', 'SimHei']
for f in preferred_fonts:
    if f in available_fonts:
        matplotlib.rcParams['font.sans-serif'] = [f]
        matplotlib.rcParams['axes.unicode_minus'] = False
        break

class VestibularObserver:
    def __init__(self, dt=0.01, gains=None, tau_lp=1.0):
        """
        dt: timestep (s)
        gains: dict with keys K_omega, K_au, K_au_perp, K_f, K_fomega
        tau_lp: time constant for low-pass used to compute SCC high-pass (s)
        """
        self.dt = dt
        # default gains from Table 2 (vestibular-only subset) in the reviewed paper
        default_gains = {
            "K_omega": 8.0,       # semicircular canal error -> angular velocity estimate
            "K_au": -2.0,         # otolith error in utricular plane -> linear accel estimate
            "K_au_perp": -4.0,    # otolith error perp to utricular plane -> linear accel estimate
            "K_f": 4.0,           # otolith rotation error -> gravity estimate (1/s)
            "K_fomega": 8.0       # otolith rotation error -> angular velocity estimate (1/s)
        }
        if gains is None:
            gains = default_gains
        self.gains = gains
        self.tau_lp = tau_lp
        
        # internal states (initialized later in reset)
        self.reset()

    def reset(self, init_g=None, init_omega=None, init_a=None):
        # gravity estimate (body frame) - unit vector pointing 'down' in body coords
        if init_g is None:
            self.g_hat = np.array([0.0, 0.0, 1.0])  # assume +Z head-up; gravity toward +Z in this convention
        else:
            self.g_hat = init_g.astype(float)
        # angular velocity estimate
        if init_omega is None:
            self.omega_hat = np.zeros(3)
        else:
            self.omega_hat = init_omega.astype(float)
        # linear acceleration estimate (in body frame)
        if init_a is None:
            self.a_hat = np.zeros(3)
        else:
            self.a_hat = init_a.astype(float)
        # internal lowpass states for SCC (to compute HPF)
        self.lp_scc_input = np.zeros(3)
        self.lp_scc_hat = np.zeros(3)
        # history buffers
        self.history = {"t":[], "omega_meas":[], "omega_hat":[], "g_meas":[], "g_hat":[],
                        "ea_norm":[], "ef_norm":[], "eomega_norm":[]}

    def _lowpass_step(self, lp, x, tau):
        """Exponential lowpass discrete update: lp += (dt/tau)*(x - lp)"""
        alpha = self.dt / (tau + 1e-12)
        return lp + alpha * (x - lp)

    def _scc_afference(self, omega):
        """Approximate SCC afference (high-pass of angular velocity) using hp = input - lowpass(input)"""
        # update lowpass of raw omega (physically SCC acts like HP, so we approximate)
        self.lp_scc_input = self._lowpass_step(self.lp_scc_input, omega, self.tau_lp)
        aff = omega - self.lp_scc_input
        return aff

    def _otolith_afference(self, specific_force):
        """
        Otolith afference = specific force f = g - a (vector).
        We assume imu accelerometer measurement is 'specific_force' = g - a (consistent with sensor physics).
        """
        return specific_force

    def _compute_ef_rotation_error(self, f_meas, f_hat):
        """
        Compute a rotation-vector-like error ef that would rotate f_hat toward f_meas.
        Use cross product as small-angle approximation: ef ≈ cross(f_hat_norm, f_meas_norm) * angle_mag
        We'll return a 3-vector in body frame.
        """
        # normalize to avoid scale issues
        n1 = np.linalg.norm(f_hat)
        n2 = np.linalg.norm(f_meas)
        if n1 < 1e-8 or n2 < 1e-8:
            return np.zeros(3)
        u1 = f_hat / n1
        u2 = f_meas / n2
        # rotation axis ~ cross(u1,u2), angle ~ atan2(norm(cross), dot)
        cross = np.cross(u1, u2)
        cross_norm = np.linalg.norm(cross)
        dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
        angle = np.arctan2(cross_norm, dot)
        if cross_norm < 1e-8:
            return np.zeros(3)
        axis = cross / cross_norm
        # rotation vector:
        ef = axis * angle
        return ef

    def step(self, omega_meas, specific_force_meas):
        """
        One simulation step.
        - omega_meas: measured angular velocity (rad/s), 3-vector (body frame)
        - specific_force_meas: accelerometer specific force f = g - a (m/s^2), 3-vector (body frame)
        Returns current estimates and conflicts.
        """
        # measured afferences
        alpha_scc = self._scc_afference(omega_meas)
        alpha_oto = self._otolith_afference(specific_force_meas)

        # internal model expected afferences
        alpha_scc_hat = self._scc_afference(self.omega_hat)  # reuse hp implementation on omega_hat
        f_hat = self.g_hat - self.a_hat  # expected otolith specific force

        # sensory conflicts
        ea = alpha_oto - f_hat          # otolith vector difference
        eomega = alpha_scc - alpha_scc_hat  # SCC vector difference
        ef = self._compute_ef_rotation_error(alpha_oto, f_hat)  # rotation vector error aligning otoliths

        # projections for utricular plane vs perpendicular (simple split using utricular plane approx)
        # For simplicity, take utricular plane as body X-Y plane; perpendicular is Z
        # Project ea into utricular plane and perpendicular component
        ea_utr = np.array([ea[0], ea[1], 0.0])
        ea_perp = np.array([0.0, 0.0, ea[2]])

        # feedback updates (simple proportional-integrator-like update)
        K = self.gains
        # update angular velocity estimate
        # apply SCC error and ef contributions
        d_omega = K["K_omega"] * eomega + K["K_fomega"] * ef
        self.omega_hat = self.omega_hat + d_omega * self.dt

        # update gravity estimate using ef (rotation error) scaled by K_f
        # treat g_hat as a vector; rotate slightly by ef
        # small-angle approximation: rotate g_hat by cross(ef, g_hat)
        self.g_hat = self.g_hat + (K["K_f"] * np.cross(ef, self.g_hat)) * self.dt
        # renormalize gravity magnitude to 1 (unit gravity direction)
        g_norm = np.linalg.norm(self.g_hat)
        if g_norm > 1e-8:
            self.g_hat = self.g_hat / g_norm

        # update linear acceleration estimate (a_hat) from otolith conflict projections
        d_a = K["K_au"] * ea_utr + K["K_au_perp"] * ea_perp
        self.a_hat = self.a_hat + d_a * self.dt

        # store some history info
        self.history["t"].append(len(self.history["t"]) * self.dt)
        self.history["omega_meas"].append(np.linalg.norm(omega_meas))
        self.history["omega_hat"].append(np.linalg.norm(self.omega_hat))
        self.history["g_meas"].append(np.linalg.norm(specific_force_meas))  # magnitude of f (g-a)
        self.history["g_hat"].append(np.linalg.norm(self.g_hat))
        self.history["ea_norm"].append(np.linalg.norm(ea))
        self.history["ef_norm"].append(np.linalg.norm(ef))
        self.history["eomega_norm"].append(np.linalg.norm(eomega))

        return {"omega_hat":self.omega_hat.copy(), "g_hat":self.g_hat.copy(), "a_hat":self.a_hat.copy(),
                "ea":ea.copy(), "ef":ef.copy(), "eomega":eomega.copy()}

    def run(self, omega_traj, f_traj):
        """
        Run the observer over provided trajectories.
        omega_traj: (N,3) angular velocity measurements
        f_traj: (N,3) accelerometer specific force measurements
        """
        N = len(omega_traj)
        out = []
        for i in range(N):
            out.append(self.step(omega_traj[i], f_traj[i]))
        return out


def list_sessions_by_subject_map(processed_root: Path, subject: str, map_id: str) -> List[Path]:
    """根据subject和map_id查找所有匹配的session目录"""
    base = processed_root / subject
    if not base.exists():
        return []
    sessions = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        parts = p.name.split("_")
        if len(parts) >= 3 and parts[2] == map_id:
            if (p / "_modalities").exists():
                sessions.append(p)
    return sessions


def list_all_sessions(processed_root: Path) -> List[Path]:
    """遍历所有被试的所有会话"""
    sessions = []
    for subject_dir in sorted(processed_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        for session_dir in sorted(subject_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            modalities_dir = session_dir / "_modalities"
            if modalities_dir.exists() and (modalities_dir / "imu.npy").exists():
                sessions.append(session_dir)
    return sessions


def load_imu_data(session_dir: Path) -> Tuple[np.ndarray, Dict]:
    """加载IMU数据和通道信息"""
    modalities_dir = session_dir / "_modalities"
    imu_path = modalities_dir / "imu.npy"
    json_path = modalities_dir / "modalities.json"
    
    if not imu_path.exists():
        raise FileNotFoundError(f"未找到IMU数据文件: {imu_path}")
    
    imu_data = np.load(imu_path)  # shape: (9, N)
    
    # 加载通道信息
    channel_info = {}
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            modalities = json.load(f)
            if 'imu' in modalities:
                channel_info = {
                    'channel_names': modalities['imu'].get('channel_names', []),
                    'channel_indices': modalities['imu'].get('channel_indices', [])
                }
    
    return imu_data, channel_info


def extract_imu_channels(imu_data: np.ndarray, channel_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    从IMU数据中提取陀螺仪和加速度计数据
    返回: (omega_traj, f_traj)
    - omega_traj: (N, 3) 角速度 [GYR-X, GYR-Y, GYR-Z]
    - f_traj: (N, 3) 比力 [ACC-X, ACC-Y, ACC-Z]
    """
    # 通道顺序：GYR-X, GYR-Y, GYR-Z, ACC-X, ACC-Y, ACC-Z, MAG-X, MAG-Y, MAG-Z
    if len(channel_names) == 9:
        # 根据通道名称找到索引
        gyr_x_idx = channel_names.index('GYR-X') if 'GYR-X' in channel_names else 0
        gyr_y_idx = channel_names.index('GYR-Y') if 'GYR-Y' in channel_names else 1
        gyr_z_idx = channel_names.index('GYR-Z') if 'GYR-Z' in channel_names else 2
        acc_x_idx = channel_names.index('ACC-X') if 'ACC-X' in channel_names else 3
        acc_y_idx = channel_names.index('ACC-Y') if 'ACC-Y' in channel_names else 4
        acc_z_idx = channel_names.index('ACC-Z') if 'ACC-Z' in channel_names else 5
    else:
        # 默认顺序
        gyr_x_idx, gyr_y_idx, gyr_z_idx = 0, 1, 2
        acc_x_idx, acc_y_idx, acc_z_idx = 3, 4, 5
    
    # 提取数据并转置为 (N, 3)
    omega_traj = np.array([
        imu_data[gyr_x_idx, :],
        imu_data[gyr_y_idx, :],
        imu_data[gyr_z_idx, :]
    ]).T  # (N, 3)
    
    f_traj = np.array([
        imu_data[acc_x_idx, :],
        imu_data[acc_y_idx, :],
        imu_data[acc_z_idx, :]
    ]).T  # (N, 3)
    
    return omega_traj, f_traj


def get_sampling_rate(session_dir: Path) -> int:
    """从modalities.json或npz文件获取IMU采样率"""
    # 优先从modalities.json读取IMU的采样率
    modalities_path = session_dir / "_modalities" / "modalities.json"
    if modalities_path.exists():
        with open(modalities_path, 'r', encoding='utf-8') as f:
            modalities = json.load(f)
            if 'imu' in modalities and 'srate' in modalities['imu']:
                imu_srate = int(modalities['imu']['srate'])
                print(f"✅ 从modalities.json读取IMU采样率: {imu_srate} Hz")
                return imu_srate
    
    # 回退到npz文件
    npz_path = session_dir / f"{session_dir.name}.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        if 'srate' in data:
            srate = int(data['srate'])
            # 如果是1000Hz，IMU可能是100Hz（下采样后）
            if srate == 1000:
                print(f"⚠️ 检测到1000Hz采样率，IMU数据可能已下采样到100Hz")
                return 100  # IMU数据应该是100Hz
            return srate
    
    return 100  # 默认IMU采样率（因为原始数据是100Hz）


def process_session(session_dir: Path, output_dir: Optional[Path] = None) -> bool:
    """处理单个会话的前庭模型模拟"""
    session_name = session_dir.name
    print(f"\n{'='*80}")
    print(f"处理会话: {session_name}")
    print(f"{'='*80}")
    
    try:
        # 加载IMU数据
        imu_data, channel_info = load_imu_data(session_dir)
        print(f"✅ IMU数据形状: {imu_data.shape}")
        
        # 提取通道名称
        channel_names = channel_info.get('channel_names', [])
        if len(channel_names) == 0:
            # 使用默认通道名称
            channel_names = ['GYR-X', 'GYR-Y', 'GYR-Z', 'ACC-X', 'ACC-Y', 'ACC-Z', 'MAG-X', 'MAG-Y', 'MAG-Z']
            print("⚠️ 未找到通道名称信息，使用默认顺序")
        else:
            print(f"✅ 通道名称: {channel_names}")
        
        # 提取陀螺仪和加速度计数据
        omega_traj, f_traj = extract_imu_channels(imu_data, channel_names)
        print(f"✅ 角速度轨迹形状: {omega_traj.shape}")
        print(f"✅ 比力轨迹形状: {f_traj.shape}")
        
        # 获取采样率
        srate = get_sampling_rate(session_dir)
        dt = 1.0 / srate
        print(f"✅ 采样率: {srate} Hz, 时间步长: {dt:.6f} s")
        
        # 创建前庭观察者模型
        observer = VestibularObserver(dt=dt)
        observer.reset()
        
        # 运行模型
        print("🔄 运行前庭模型...")
        results = observer.run(omega_traj, f_traj)
        
        # 提取感觉冲突信号
        # ea: 耳石冲突 (3维), ef: 旋转误差 (3维), eomega: SCC冲突 (3维)
        # 总共9维，但用户要求6维，我们保存ea和eomega（各3维）
        N = len(results)
        imu_errors = np.zeros((6, N))  # (6, N)
        
        for i, result in enumerate(results):
            imu_errors[0:3, i] = result['ea']      # 耳石冲突 (3维)
            imu_errors[3:6, i] = result['eomega']  # SCC冲突 (3维)
        
        print(f"✅ 感觉冲突数据形状: {imu_errors.shape}")
        
        # 保存结果
        if output_dir is None:
            output_dir = session_dir / "_modalities"
        else:
            output_dir = output_dir / session_dir.name / "_modalities"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "imu_err.npy"
        np.save(output_path, imu_errors)
        print(f"✅ 感觉冲突数据已保存: {output_path}")
        
        # 绘制时序图
        plot_conflict_timeseries(imu_errors, session_name, srate, output_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ 处理会话 {session_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def plot_conflict_timeseries(imu_errors: np.ndarray, session_name: str, srate: int, output_dir: Path):
    """绘制感觉冲突信号的时序图（6个维度）"""
    n_channels, n_samples = imu_errors.shape
    duration = n_samples / srate
    time_axis = np.arange(n_samples) / srate
    
    # 通道名称
    channel_names = [
        'ea_X (耳石冲突X)', 'ea_Y (耳石冲突Y)', 'ea_Z (耳石冲突Z)',
        'eomega_X (SCC冲突X)', 'eomega_Y (SCC冲突Y)', 'eomega_Z (SCC冲突Z)'
    ]
    
    # 创建子图：2行3列
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'前庭感觉冲突信号时序图 - {session_name}', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i in range(n_channels):
        ax = axes[i]
        channel_data = imu_errors[i, :]
        
        # 绘制时间序列
        ax.plot(time_axis, channel_data, color=colors[i], linewidth=0.5, alpha=0.7)
        ax.set_xlabel('时间 (秒)', fontsize=10)
        ax.set_ylabel(f'{channel_names[i]}', fontsize=10)
        ax.set_title(f'{channel_names[i]} 时间序列', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=1, alpha=0.7, 
                  label=f'均值: {mean_val:.4f}')
        ax.axhline(y=mean_val + std_val, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        ax.axhline(y=mean_val - std_val, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / f'{session_name}_imu_conflict_timeseries.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 时序图已保存: {output_path}")
    plt.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="前庭观察者模型模拟 - 从IMU数据生成感觉冲突特征")
    parser.add_argument("--subject", type=str, default=None, help="被试姓名（如 zzh），如果提供则只处理该被试")
    parser.add_argument("--map", dest="map_id", type=str, default=None, help="地图编号（如 02），如果提供则只处理该地图")
    parser.add_argument("--batch-all", action="store_true", help="批量处理所有会话（忽略--subject和--map）")
    parser.add_argument("--output", type=str, default=None, help="输出目录（默认：processed/{subject}/_modalities/）")
    args = parser.parse_args()
    
    # 确定路径
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    processed_root = project_root / "processed"
    
    if not processed_root.exists():
        raise FileNotFoundError(f"未找到processed目录: {processed_root}")
    
    # 确定输出目录
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
    
    # 查找会话
    if args.batch_all:
        sessions = list_all_sessions(processed_root)
        print(f"[信息] 批量处理模式：找到 {len(sessions)} 个会话")
    else:
        if args.subject is None or args.map_id is None:
            raise ValueError("单会话处理模式需要提供 --subject 和 --map 参数，或使用 --batch-all 进行批量处理")
        sessions = list_sessions_by_subject_map(processed_root, args.subject, args.map_id)
        if len(sessions) == 0:
            print(f"[错误] 未找到匹配的会话（subject={args.subject}, map={args.map_id}）")
            return
        print(f"[信息] 找到 {len(sessions)} 个匹配的会话")
    
    # 处理每个会话
    success_count = 0
    for session_dir in sessions:
        if process_session(session_dir, output_dir):
            success_count += 1
    
    print(f"\n{'='*80}")
    print(f"✅ 处理完成！成功处理 {success_count}/{len(sessions)} 个会话")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
