# 前庭观察者模型 - Simulink框图与代码实现对应关系分析

## 模块对应关系

### 1. 耳石（Otolith）路径模块

#### 图中的应用：
- **`OTO`**: 耳石传感器模块，处理比力（specific force）测量
- **`G(w)`**: 滤波器，处理输入信号
- **`<OTO>`**: 内部模型的耳石预测
- **`Gest(w)`**: 滤波器，处理重力估计
- **`kaa`**: 增益块（对应 `K_au` 和 `K_au_perp`）
- **`kfg`**: 增益块（对应 `K_f`）

#### 代码实现：
```python
# _otolith_afference() - 对应图中的应用
def _otolith_afference(self, specific_force):
    """
    对应图中的 OTO 模块
    直接返回比力（假设IMU加速度计测量即为比力 f = g - a）
    """
    return specific_force
```

**对应关系：**
- ✅ `_otolith_afference()` ↔ `OTO` 模块
- ❌ **缺失 `G(w)` 滤波器** - 代码中没有对输入比力进行滤波处理
- ✅ `<OTO>` 预测 ↔ `f_hat = self.g_hat - self.a_hat`（内部模型预测的比力）
- ❌ **缺失 `Gest(w)` 滤波器** - 代码中没有对重力估计进行滤波
- ✅ `kaa` ↔ `K["K_au"]` 和 `K["K_au_perp"]`（在更新 `a_hat` 时使用）
- ✅ `kfg` ↔ `K["K_f"]`（在更新 `g_hat` 时使用）

### 2. 半规管（SCC）路径模块

#### 图中的应用：
- **`SCC`**: 半规管传感器模块（高通滤波器）
- **`<SCC>`**: 内部模型的SCC预测
- **`kww`**: 增益块（对应 `K_omega`）
- **`kfw`**: 增益块（对应 `K_fomega`）

#### 代码实现：
```python
# _scc_afference() - 对应图中的 SCC 模块
def _scc_afference(self, omega):
    """
    对应图中的 SCC 模块
    使用高通滤波器近似：HPF(omega) = omega - LowPass(omega)
    """
    self.lp_scc_input = self._lowpass_step(self.lp_scc_input, omega, self.tau_lp)
    aff = omega - self.lp_scc_input
    return aff
```

**对应关系：**
- ✅ `_scc_afference()` ↔ `SCC` 模块（高通滤波器实现正确）
- ✅ `<SCC>` 预测 ↔ `alpha_scc_hat = self._scc_afference(self.omega_hat)`（内部模型预测）
- ✅ `kww` ↔ `K["K_omega"]`（在更新 `omega_hat` 时使用）
- ✅ `kfw` ↔ `K["K_fomega"]`（在更新 `omega_hat` 时使用）

### 3. 反馈路径和状态估计

#### 图中的应用：
- **未标记的阴影矩形块**: 内部状态估计（可能是积分器）
- **`K`**: 控制器/积分器块，整合所有误差信号

#### 代码实现：
```python
# 状态更新（对应图中的阴影块和K块）
d_omega = K["K_omega"] * eomega + K["K_fomega"] * ef
self.omega_hat = self.omega_hat + d_omega * self.dt  # 积分器

self.g_hat = self.g_hat + (K["K_f"] * np.cross(ef, self.g_hat)) * self.dt  # 积分器
self.a_hat = self.a_hat + d_a * self.dt  # 积分器
```

**对应关系：**
- ✅ 阴影块 + `K` ↔ `omega_hat`, `g_hat`, `a_hat` 的状态更新（使用积分器）
- ✅ 反馈路径正确实现

### 4. 感觉冲突信号

#### 图中的应用：
- **顶部红色虚线** (`ea`): 耳石冲突 = OTO输出 - 内部估计
- **底部红色虚线** (`eomega`): SCC冲突 = SCC输出 - <SCC>输出

#### 代码实现：
```python
ea = alpha_oto - f_hat          # 耳石冲突
eomega = alpha_scc - alpha_scc_hat  # SCC冲突
ef = self._compute_ef_rotation_error(alpha_oto, f_hat)  # 旋转误差
```

**对应关系：**
- ✅ `ea` ↔ 顶部红色虚线（耳石冲突）
- ✅ `eomega` ↔ 底部红色虚线（SCC冲突）
- ⚠️ **额外的 `ef` 信号** - 图中可能未明确标注，但代码中实现了旋转误差

### 5. 投影和分解

#### 代码中的实现（图中可能隐含）：
```python
ea_utr = np.array([ea[0], ea[1], 0.0])    # 耳石平面的投影
ea_perp = np.array([0.0, 0.0, ea[2]])     # 垂直于耳石平面的分量
```

**对应关系：**
- ✅ 代码实现了耳石平面的投影分解，用于不同的增益（`K_au` vs `K_au_perp`）
- ⚠️ 图中可能隐含在 `kaa` 增益块中，但未明确显示

## 发现的遗漏和差异

### ❌ 缺失的模块：

1. **`G(w)` 滤波器**
   - **位置**：在 `TH(w)` 输入和 `OTO` 之间的滤波器
   - **作用**：可能用于预处理输入信号或建模耳石传感器的动态特性
   - **影响**：当前代码直接使用原始比力，没有滤波预处理

2. **`Gest(w)` 滤波器**
   - **位置**：在重力估计 `g_hat` 输出路径上的滤波器
   - **作用**：可能用于平滑重力估计或建模重力感知的动态特性
   - **影响**：当前代码直接使用重力估计，没有滤波后处理

3. **`pi/180` 转换因子**
   - **位置**：SCC输入路径
   - **作用**：可能是度转弧度的转换
   - **影响**：如果输入数据已经是弧度，则不需要；但如果输入是度数，则需要转换

### ⚠️ 可能的实现差异：

1. **`ef` 旋转误差的计算**
   - **代码中**：使用旋转向量方法计算 `ef`
   - **图中**：可能隐含在反馈路径中，但未明确标注
   - **说明**：`ef` 是用于对齐比力向量的旋转误差，代码实现是正确的

2. **状态更新的顺序**
   - **代码中**：先更新 `omega_hat`，然后 `g_hat`，最后 `a_hat`
   - **图中**：可能是一个统一的 `K` 块处理所有更新
   - **说明**：功能上等价，但实现方式可能略有不同

## 建议的改进

1. **添加 `G(w)` 滤波器**：
   ```python
   def _otolith_afference(self, specific_force):
       # 可以添加低通滤波器预处理
       # self.lp_oto = self._lowpass_step(self.lp_oto, specific_force, tau_oto)
       # return self.lp_oto
       return specific_force
   ```

2. **添加 `Gest(w)` 滤波器**：
   ```python
   # 在更新 g_hat 后添加滤波
   g_hat_filtered = self._lowpass_step(self.lp_g_hat, self.g_hat, tau_g)
   ```

3. **验证单位转换**：
   - 确保角速度输入是弧度/秒（rad/s），而不是度/秒
   - 如果需要，添加 `pi/180` 转换

## 总结

**已正确实现的模块：**
- ✅ SCC模块（高通滤波器）
- ✅ 内部模型预测（<SCC> 和 <OTO>）
- ✅ 感觉冲突计算（`ea` 和 `eomega`）
- ✅ 反馈增益（`kww`, `kfw`, `kaa`, `kfg`）
- ✅ 状态更新（积分器）

**缺失的模块：**
- ❌ `G(w)` 滤波器（耳石输入预处理）
- ❌ `Gest(w)` 滤波器（重力估计后处理）
- ❌ `pi/180` 转换（如果需要）

**代码质量：**
- 核心功能已正确实现
- 反馈结构和状态估计正确
- 缺失的滤波器可能影响动态响应特性，但不影响基本功能

