import numpy as np

# =========================
# 基础常量 & 小工具
# =========================

G0 = 9.80665  # 用不到也保留一下，方便以后从 m/s^2 转 G

def normalize_vec(v, eps=1e-8):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v), 0.0
    return v / n, n


# ---------- 四元数工具（body rates → 姿态） ----------

def quat_mul(q, r):
    """四元数乘法，q,r 都是 [w,x,y,z]"""
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_normalize(q, eps=1e-12):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def integrate_quaternion(q, omega_rad, dt):
    """
    对应 Simulink 'Quaternion Integrator'
    omega_rad: 3-vector, rad/s, body frame
    """
    omega = np.asarray(omega_rad, dtype=float)
    w_quat = np.concatenate([[0.0], omega])          # 纯虚四元数
    dq = 0.5 * quat_mul(q, w_quat)
    q_new = q + dq * dt
    return quat_normalize(q_new)

def quat_to_dcm(q):
    """
    四元数 → 方向余弦矩阵 (world -> head)
    对应 'Quaternion to Direction Cosine'
    """
    w, x, y, z = quat_normalize(q)
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2),   2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
    ])
    # v_head = R @ v_world
    return R

def quat_to_euler_zyx(q):
    """仅调试用：四元数 → ZYX 欧拉角 (yaw, pitch, roll)"""
    w, x, y, z = quat_normalize(q)
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    s = 2*(w*y - z*x)
    s = np.clip(s, -1.0, 1.0)
    pitch = np.arcsin(s)
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    return np.array([yaw, pitch, roll])


# =========================
# 通用连续 ZPK → 离散 IIR（双线性变换）
# =========================

class ContinuousZPKFilter:
    """
    用连续域 zeros / poles / gain 定义传递函数，
    通过双线性变换 (Tustin) 离散化，再用 DF2T 结构实现单输入单输出滤波。
    """

    def __init__(self, zeros_s, poles_s, gain_s, dt):
        """
        zeros_s, poles_s: list-like, 连续域零极点（单位 rad/s）
        gain_s: 连续域增益
        dt: 采样周期
        """
        self.dt = float(dt)
        self.zeros_s = list(zeros_s)
        self.poles_s = list(poles_s)
        self.gain_s = float(gain_s)

        T = self.dt
        eps_bilinear = 1e-10  # 防止除零的小值

        # 1) s → z 双线性变换: z = (2 + T s) / (2 - T s)
        # 防止除零：如果 2 - T*s 接近0，使用小的epsilon
        zeros_z = []
        for z in self.zeros_s:
            denominator = 2.0 - T*z
            if abs(denominator) < eps_bilinear:
                # 如果分母接近0，使用近似值避免除零
                denominator = np.sign(denominator) * eps_bilinear if denominator != 0 else eps_bilinear
            zeros_z.append((2.0 + T*z) / denominator)
        poles_z = []
        for p in self.poles_s:
            denominator = 2.0 - T*p
            if abs(denominator) < eps_bilinear:
                # 如果分母接近0，使用近似值避免除零
                denominator = np.sign(denominator) * eps_bilinear if denominator != 0 else eps_bilinear
            poles_z.append((2.0 + T*p) / denominator)

        # 2) 数字域增益
        # 防止除零：如果因子接近0，使用小的epsilon
        num_factor = np.prod([max(abs(2.0 - T*p), eps_bilinear) for p in self.poles_s]) if self.poles_s else 1.0
        den_factor = np.prod([max(abs(2.0 - T*z), eps_bilinear) for z in self.zeros_s]) if self.zeros_s else 1.0
        if abs(den_factor) < eps_bilinear:
            den_factor = np.sign(den_factor) * eps_bilinear if den_factor != 0 else eps_bilinear
        gain_z = self.gain_s * num_factor / den_factor

        # 3) 从根构造多项式
        # 使用np.poly，但需要处理可能的数值问题
        try:
            if zeros_z:
                # 清理NaN/Inf值
                zeros_z_clean = [z for z in zeros_z if np.isfinite(z)]
                if len(zeros_z_clean) > 0:
                    num = gain_z * np.poly(zeros_z_clean)
                else:
                    num = gain_z * np.array([1.0])
            else:
                num = gain_z * np.array([1.0])

            if poles_z:
                # 清理NaN/Inf值
                poles_z_clean = [p for p in poles_z if np.isfinite(p)]
                if len(poles_z_clean) > 0:
                    den = np.poly(poles_z_clean)
                else:
                    den = np.array([1.0])
            else:
                den = np.array([1.0])
        except (np.linalg.LinAlgError, ValueError) as e:
            # 如果np.poly失败（可能由于数值问题），使用备用方法
            print(f"警告: np.poly失败 ({e}), 使用备用方法...")
            if zeros_z:
                num = gain_z * np.array([1.0, -sum(zeros_z)]) if len(zeros_z) == 1 else gain_z * np.array([1.0])
            else:
                num = gain_z * np.array([1.0])
            if poles_z:
                den = np.array([1.0, -sum(poles_z)]) if len(poles_z) == 1 else np.array([1.0])
        else:
            den = np.array([1.0])

        num = np.asarray(num, dtype=float)
        den = np.asarray(den, dtype=float)

        # 归一化: a0 = 1
        # 防止除零：如果den[0]接近0，使用小的epsilon
        if abs(den[0]) < 1e-12:
            # 如果分母的首项系数接近0，这是一个严重问题，但我们可以尝试修复
            print(f"警告: 滤波器分母首项系数接近0 ({den[0]}), 尝试修复...")
            den[0] = 1.0
            # 重新归一化其他系数
            if len(den) > 1:
                scale = den[1] if abs(den[1]) > 1e-12 else 1.0
                den = den / scale
                num = num / scale
        
        if abs(den[0]) > 1e-12 and den[0] != 1.0:
            num /= den[0]
            den /= den[0]

        # 4) padding: 保证长度一致
        L = max(len(num), len(den))
        if len(num) < L:
            num = np.pad(num, (L - len(num), 0))
        if len(den) < L:
            den = np.pad(den, (L - len(den), 0))

        self.b = num  # 分子
        self.a = den  # 分母
        self.order = L - 1
        self.state = np.zeros(self.order)

    def reset(self):
        self.state[:] = 0.0

    def step(self, x):
        """
        单点更新，Direct Form II Transposed
        """
        x = float(x)
        b, a = self.b, self.a
        L = self.order
        z = self.state

        if L == 0:
            # 零阶系统：纯增益
            y = b[0] * x
            return y

        # DF2T
        y = b[0] * x + z[0]
        for i in range(L - 1):
            z[i] = b[i + 1] * x + z[i + 1] - a[i + 1] * y
        z[L - 1] = b[L] * x - a[L] * y

        self.state = z
        return y


class MultiAxisZPKFilter:
    """三轴版本，X/Y/Z 共用一套 ZPK 参数"""

    def __init__(self, zeros_s, poles_s, gain_s, dt):
        self.filters = [
            ContinuousZPKFilter(zeros_s, poles_s, gain_s, dt),
            ContinuousZPKFilter(zeros_s, poles_s, gain_s, dt),
            ContinuousZPKFilter(zeros_s, poles_s, gain_s, dt),
        ]

    def reset(self):
        for f in self.filters:
            f.reset()

    def step(self, v3):
        v3 = np.asarray(v3, dtype=float)
        return np.array([f.step(v) for f, v in zip(self.filters, v3)])


# =========================
# 物理模块
# =========================

# ---------- TH(w)：线加速度坐标系变换 ----------

def th_block(a_world_G, D_cos):
    """
    模块 TH(w):
    世界/车辆坐标系加速度 (G) -> 头坐标系 (G)
    """
    a_world_G = np.asarray(a_world_G, dtype=float)
    D_cos = np.asarray(D_cos, dtype=float).reshape(3, 3)
    return D_cos @ a_world_G


# ---------- G(w) / Gest(w)：由角速度积分得到姿态 & 重力 ----------

class OrientationFromOmega:
    """
    对应 G(w) 或 Gest(w) 中的：
    Quaternion Integrator + Quaternion to Direction Cosine + Rotation (g)
    """

    def __init__(self, dt, use_g_unit=True):
        self.dt = float(dt)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.D_cos = np.eye(3)
        self.use_g_unit = use_g_unit

    def reset(self):
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.D_cos = np.eye(3)

    def step(self, omega_rad):
        self.q = integrate_quaternion(self.q, omega_rad, self.dt)
        self.D_cos = quat_to_dcm(self.q)

        if self.use_g_unit:
            g_world = np.array([0.0, 0.0, -1.0])
        else:
            g_world = np.array([0.0, 0.0, -G0])

        g_head = self.D_cos @ g_world
        euler = quat_to_euler_zyx(self.q)
        return g_head, self.D_cos, euler


# ---------- OTO / <OTO>：耳石低通 ----------

class OtolithZPK3:
    """
    用 ZPK 一阶低通实现耳石模块:
    H(s) = 1 / (tau*s + 1), tau = 1/(2*pi*f_oto)
    """

    def __init__(self, f_oto, dt):
        tau = 1.0 / (2.0 * np.pi * f_oto)
        pole = -1.0 / tau              # = -2*pi*f_oto
        zeros_s = []
        poles_s = [pole]
        gain_s = 1.0
        self.filt = MultiAxisZPKFilter(zeros_s, poles_s, gain_s, dt)

    def reset(self):
        self.filt.reset()

    def step(self, f_head):
        return self.filt.step(f_head)


# ---------- SCC / <SCC>：半规管动力学 ----------

class SCC3External:
    """
    外周 SCC: zeros = [0,0], poles = [-1/tau_scc, -1/tau_a, -2*pi*f_scc],
    gain = 2*pi*f_scc
    """

    def __init__(self, tau_scc, tau_a, f_scc, dt):
        zeros_s = [0.0, 0.0]
        poles_s = [-1.0 / tau_scc, -1.0 / tau_a, -2.0 * np.pi * f_scc]
        gain_s = 2.0 * np.pi * f_scc
        self.filt = MultiAxisZPKFilter(zeros_s, poles_s, gain_s, dt)

    def reset(self):
        self.filt.reset()

    def step(self, omega_rad_3):
        return self.filt.step(omega_rad_3)


class SCC3Internal:
    """
    内部 SCC: zeros = [0], poles = [-1/tau_scc, -2*pi*f_scc],
    gain = 2*pi*f_scc
    """

    def __init__(self, tau_scc, f_scc, dt):
        zeros_s = [0.0]
        poles_s = [-1.0 / tau_scc, -2.0 * np.pi * f_scc]
        gain_s = 2.0 * np.pi * f_scc
        self.filt = MultiAxisZPKFilter(zeros_s, poles_s, gain_s, dt)

    def reset(self):
        self.filt.reset()

    def step(self, omega_est_3):
        return self.filt.step(omega_est_3)


# ---------- 重力冲突“大黑块”：e_v = θ * d ----------

def gravity_conflict(u, v, eps=1e-8):
    """
    u, v: 两个 3D 向量 (如 g_head_true, g_head_est)
    输出: e_v = θ * d，θ 为夹角，d 为单位旋转轴方向
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    
    # 清理NaN/Inf输入
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    u_n, _ = normalize_vec(u, eps)
    v_n, _ = normalize_vec(v, eps)

    dot = np.clip(np.dot(u_n, v_n), -1.0, 1.0)
    theta = np.arccos(dot)
    
    # 检查theta是否为NaN
    if np.isnan(theta) or np.isinf(theta):
        return np.zeros(3)

    cross = np.cross(u_n, v_n)
    cross_n, n_c = normalize_vec(cross, eps)
    if n_c < eps:
        return np.zeros(3)
    
    result = theta * cross_n
    # 最终清理
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


# ---------- K 模块：冲突加权 ----------

def K_block(loop_in, gravity_switch, kill_value, kwg, kww, kwvw):
    """
    loop_in: 冲突向量（omega_est，3维向量）
    gravity_switch: Gravity Switch 的当前值
    kill_value: 与 gravity_switch 相等时走 top 增益
    kwg, kww, kwvw: 与 Simulink 中一致
    """
    # 清理输入
    loop_in = np.asarray(loop_in, dtype=float)
    loop_in = np.nan_to_num(loop_in, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 防止除零
    if abs(kww) < 1e-8:
        kww = 1e-8
    if abs(kww + kwvw) < 1e-8:
        kwvw = max(0.0, kwvw)
    
    if gravity_switch == kill_value:
        # top 增益: kwg * ((kww + 1)/kww)
        k_gain = kwg * ((kww + 1.0) / (kww + 1e-8))
    else:
        # bottom 增益: kwg * ((kww + kwvw + 1)/(kww + kwvw))
        k_gain = kwg * ((kww + kwvw + 1.0) / (kww + kwvw + 1e-8))

    result = k_gain * loop_in
    # 清理输出
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result


# =========================
# 前庭 + 内部模型总成
# =========================

class VestibularModel:
    """
    尽量贴合你那张 Simulink 图的 Python 实现：

    - 外周：TH(w) → G(w) → OTO & SCC
    - 内部模型：<OTO>, <SCC>, Gest(w)
    - 冲突计算：重力方向冲突 e_v
    - 症状：K 模块 + MSI 积分

    参数使用你给的：
    f_oto=2, tau_scc=5.7, tau_a=80, f_scc=2,
    kaa=-4, kfg=4, kfw=8, kww=8, kwg=1, kwvw=10
    """

    def __init__(self,
                 dt=0.01,
                 f_oto=2.0,
                 tau_scc=5.7,
                 tau_a=80.0,
                 f_scc=2.0,
                 kaa=-4.0,
                 kfg=4.0,
                 kfw=8.0,
                 kww=8.0,
                 kwg=1.0,
                 kwvw=10.0,
                 kill_value=0):

        self.dt = float(dt)

        # ---- 外部姿态/重力 G(w) ----
        self.ori_true = OrientationFromOmega(dt, use_g_unit=True)

        # ---- 外部耳石 OTO ----
        self.oto = OtolithZPK3(f_oto, dt)

        # ---- 外部 SCC ----
        self.scc = SCC3External(tau_scc, tau_a, f_scc, dt)

        # ---- 内部姿态/重力 Gest(w) ----
        self.ori_est = OrientationFromOmega(dt, use_g_unit=True)

        # ---- 内部耳石 <OTO> ----
        self.oto_int = OtolithZPK3(f_oto, dt)

        # ---- 内部 SCC <SCC> ----
        self.scc_int = SCC3Internal(tau_scc, f_scc, dt)

        # ---- 内部状态：估计角速度 & GIA ----
        self.omega_est = np.zeros(3)
        self.f_hat = np.zeros(3)

        # ---- 反馈增益 ----
        self.kaa = kaa
        self.kfg = kfg
        self.kfw = kfw
        self.kww = kww
        self.kwg = kwg
        self.kwvw = kwvw
        self.kill_value = kill_value

        # 简单 MSI（症状积分）状态
        self.msi = 0.0
        
        # 用于ramp的时间计数器
        self.t = 0.0
        self.ramp_duration = 1.0  # 前1秒逐步开启反馈

    def reset(self):
        self.ori_true.reset()
        self.ori_est.reset()
        self.oto.reset()
        self.oto_int.reset()
        self.scc.reset()
        self.scc_int.reset()
        # 初始化内部状态为零（将在第一次step时用外部输出warm start）
        self.omega_est[:] = 0.0
        self.f_hat[:] = 0.0
        self.msi = 0.0
        # 重置时间计数器
        self.t = 0.0

    def step(self, acc_world_G, gyro_head_dps, gravity_switch=0):
        """
        acc_world_G: 车辆/世界坐标系线加速度 (G)，shape (3,)
        gyro_head_dps: 头坐标系角速度 (deg/s)，shape (3,)
        gravity_switch: 对应 Simulink 里的 Gravity Switch 值（0/1）

        返回一个 dict，包含全部中间变量，方便和 Simulink 对照。
        """

        acc_world_G = np.asarray(acc_world_G, dtype=float)
        gyro_head_dps = np.asarray(gyro_head_dps, dtype=float)

        # 1) deg/s → rad/s
        gyro_head_rad = gyro_head_dps * np.pi / 180.0

        # 2) 外部姿态/重力：G(w)
        g_head_true, D_cos_true, euler_true = self.ori_true.step(gyro_head_rad)

        # 3) TH(w)：将线加速度转到头坐标系
        a_head_G = th_block(acc_world_G, D_cos_true)

        # 4) GIA：线加速度 + 重力
        # f_head = a_head_G + g_head_true
        f_head = g_head_true - a_head_G

        # 5) 外部耳石 / SCC 输出
        oto_out = self.oto.step(f_head)
        scc_out = self.scc.step(gyro_head_rad)
        
        # 清理外部模型输出，防止NaN传播
        oto_out = np.nan_to_num(oto_out, nan=0.0, posinf=0.0, neginf=0.0)
        scc_out = np.nan_to_num(scc_out, nan=0.0, posinf=0.0, neginf=0.0)

        # ========================
        # 内部模型部分
        # ========================

        # 5.1 先用"上一时刻"的内部状态跑 <SCC>, <OTO>, Gest(w)
        # 清理输入，防止NaN传播
        omega_est_clean = np.nan_to_num(self.omega_est, nan=0.0, posinf=0.0, neginf=0.0)
        f_hat_clean = np.nan_to_num(self.f_hat, nan=0.0, posinf=0.0, neginf=0.0)
        
        scc_int_out = self.scc_int.step(omega_est_clean)
        oto_int_out = self.oto_int.step(f_hat_clean)
        
        # 清理滤波器输出，防止NaN传播
        scc_int_out = np.nan_to_num(scc_int_out, nan=0.0, posinf=0.0, neginf=0.0)
        oto_int_out = np.nan_to_num(oto_int_out, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 5.2 误差信号
        e_scc = scc_out - scc_int_out          # 半规管误差
        e_oto = oto_out - oto_int_out          # 耳石误差
        e_v = gravity_conflict(oto_out, oto_int_out)  # 重力方向冲突

        # 清理误差信号
        e_scc = np.nan_to_num(e_scc, nan=0.0, posinf=0.0, neginf=0.0)
        e_oto = np.nan_to_num(e_oto, nan=0.0, posinf=0.0, neginf=0.0)
        e_v = np.nan_to_num(e_v, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 限幅保护：防止异常大的误差值
        def clip_vec(v, limit):
            """限制向量幅度"""
            n = np.linalg.norm(v)
            if n > limit:
                return v * (limit / n)
            return v
        
        e_scc = clip_vec(e_scc, 50.0)   # SCC误差超过50 rad/s不物理
        e_oto = clip_vec(e_oto, 5.0)    # 耳石误差一般更小
        e_v = clip_vec(e_v, 1.0)        # 重力方向冲突通常 < 1 rad

        # 5.3 内部反馈更新（改为动态更新形式，带dt）
        # 计算ramp因子：前ramp_duration秒逐步开启反馈
        ramp = min(1.0, self.t / self.ramp_duration) if self.ramp_duration > 0 else 1.0
        
        # Warm start：第一次step时，用外部输出初始化内部状态
        if self.t == 0.0:
            self.omega_est = gyro_head_rad.copy()
            self.f_hat = f_head.copy()
        
        # 动态更新：omega_est[k+1] = omega_est[k] + dt * (kww*e_scc + kfw*e_v)
        self.omega_est = self.omega_est + self.dt * ramp * (
            self.kww * e_scc +
            self.kfw * e_v
        )
        self.omega_est = np.nan_to_num(self.omega_est, nan=0.0, posinf=0.0, neginf=0.0)
        
        k_out = K_block(self.omega_est,
                        gravity_switch,
                        self.kill_value,
                        self.kwg,
                        self.kww,
                        self.kwvw)
        k_out = np.nan_to_num(k_out, nan=0.0, posinf=0.0, neginf=0.0)
        
        ori_input = k_out + self.kfg * e_v
        ori_input = np.nan_to_num(ori_input, nan=0.0, posinf=0.0, neginf=0.0)
        
        g_head_est, D_cos_est, euler_est = self.ori_est.step(ori_input)
        
        # 清理姿态估计输出
        g_head_est = np.nan_to_num(g_head_est, nan=0.0, posinf=0.0, neginf=0.0)
        D_cos_est = np.nan_to_num(D_cos_est, nan=0.0, posinf=0.0, neginf=0.0)
        euler_est = np.nan_to_num(euler_est, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 动态更新：f_hat[k+1] = f_hat[k] + dt * ramp * (kfg*e_v - kaa*e_oto)
        # 注意：原公式是 f_hat = g_head_est - kaa * e_oto
        # 但为了保持动态更新的一致性，我们使用误差驱动的更新
        # 同时考虑g_head_est的影响，使用目标值平滑更新
        f_hat_target = g_head_est - self.kaa * e_oto
        # 使用ramp平滑过渡到目标值
        self.f_hat = self.f_hat + self.dt * ramp * (f_hat_target - self.f_hat)
        self.f_hat = np.nan_to_num(self.f_hat, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 更新时间计数器
        self.t += self.dt

        # # ========================
        # # 冲突 → 症状：K 模块 + MSI
        # # ========================

        # conflict_mag = np.linalg.norm(e_v)

        # # 这里简单把正的 k_out 积分成 MSI，你可以换成论文里的真实公式
        # self.msi += max(0.0, k_out) * self.dt

        # 打包所有你可能会用到的量
        return {
            # 输入
            "acc_world_G": acc_world_G,
            "gyro_head_dps": gyro_head_dps,
            "gyro_head_rad": gyro_head_rad,

            # 外部姿态 / 感受器
            "D_cos_true": D_cos_true,
            "g_head_true": g_head_true,
            "euler_true": euler_true,
            "a_head_G": a_head_G,
            "f_head": f_head,
            "oto_out": oto_out,
            "scc_out": scc_out,

            # 内部模型
            "omega_est": self.omega_est.copy(),
            "f_hat": self.f_hat.copy(),
            "scc_int_out": scc_int_out,
            "oto_int_out": oto_int_out,
            "D_cos_est": D_cos_est,
            "g_head_est": g_head_est,
            "euler_est": euler_est,

            # 冲突 & 症状
            "e_scc": e_scc,
            "e_oto": e_oto,
            "e_v": e_v,
            # "conflict_mag": conflict_mag,
            "k_out": k_out,
            # "msi": self.msi,
        }
