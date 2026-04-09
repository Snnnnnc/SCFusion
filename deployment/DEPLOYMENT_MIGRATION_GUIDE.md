### 舒适度评分系统迁移到另一台服务器：文件清单、环境要求与启动说明

本文档面向“把当前 **Web 可视化 + 舒适度预测（回放/实时）** 整体迁移到另一台服务器/工作站”的场景，给出**需要迁移的内容**、**环境依赖**、**启动/验证步骤**与**常见问题**。

---

## 1. 你要迁移的“系统边界”是什么？

当前系统由 3 层组成：

- **Web 入口（固定链接）**：`main.py --web` 启动 `FastAPI/Uvicorn`，浏览器访问 `http://localhost:<port>/` 打开 UI（由 `deployment/app.py` 提供 `preview_ui.html`）。
- **预测服务（后端）**：`deployment/comfort_predictor.py` 负责采集/回放 → 缓冲 → 预处理 → 推理 → 通过 WebSocket 推送预测与进度。
- **模型与数据**：
  - **模型 checkpoint**：`results/.../checkpoint_best.pkl`
  - **归一化统计量（可选）**：`data/.../*.pkl`
  - **原始数据回放（可选）**：`data/**/data.bdf` + `evt.bdf`（BDF 读取依赖）

---

## 2. 必迁移文件/目录清单（按“最小可运行”到“完整功能”）

### 2.1 最小可运行（只保证 Web + 预测链路能跑）

- **代码**
  - `main.py`
  - `deployment/`（整个目录）
  - `models/`（模型结构定义，例如 `models/comfort_model.py`、`models/vestibular_model.py` 等）
  - `preview_ui.html`

- **模型权重**
  - 你要部署的 checkpoint 文件，例如：
    - `results/MotionSickness_AllMixClassificationModel_.../checkpoint_best.pkl`

### 2.2 启用“原始 BDF 回放”所需额外内容

- **原始数据目录**
  - `data/` 下对应被试/场次目录（至少包含 `data.bdf`，通常还需要 `evt.bdf`）

- **BDF 读取依赖代码**
  - `EEG_trigger/`（里面包含 `neuracle_lib/readbdfdata.py` 以及依赖）

> 说明：当前实现会使用 `neuracle_lib.readbdfdata` 读取 BDF，并且内部会用 `mne` 读 `evt.bdf` 做 annotation。

### 2.3（可选）迁移归一化统计量

- `data/training_dataset_random/mean_std_info.pkl`（如果该文件是占位符 0/1，则系统会自动走自归一化）
- 或你自己计算的真实统计量文件（例如 `data/real_normalization_stats.pkl`）

---

## 3. 服务器环境要求

### 3.1 推荐运行环境

- **操作系统**：
  - macOS / Linux（推荐 Ubuntu 20.04/22.04）
  - Windows 未做充分验证（主要是端口检测与 `lsof`、以及 BDF 工具链差异）

- **Python**：
  - 推荐 Python 3.9（你当前环境为 `py39`）

- **CPU/GPU**：
  - CPU-only 可运行（推理耗时可能更高）
  - GPU 可选（需匹配 PyTorch + CUDA 版本）

### 3.2 Python 依赖（核心）

最低需要：

- `numpy`
- `torch`
- `scipy`（重采样/滤波用；缺失会降级但不推荐）
- `fastapi`
- `uvicorn`
- `pydantic`

原始 BDF 回放建议安装：

- `mne`（用于读取 BDF/annotation；`neuracle_lib` 内部会用到）

### 3.3 依赖安装建议

在新服务器上建议创建虚拟环境（conda/venv 均可），然后：

```bash
pip install -r requirements.txt
pip install fastapi uvicorn
```

> 如果 `requirements.txt` 不包含 `fastapi/uvicorn/mne`，需要额外安装。

---

## 4. 在新服务器上的启动步骤（推荐流程）

### 4.1 拷贝代码与资源

将本仓库拷贝到新机器（示例路径用 `/opt/motion_sickness_classification`）：

- 代码：整个仓库（或至少第 2 节列出的目录/文件）
- 模型：`checkpoint_best.pkl`
- 数据（如需回放）：对应 `data/**` 与 `EEG_trigger/`

### 4.2 配置端口（固定链接）

当前 Web 入口支持：

- 固定端口（推荐 5100）：`--web_port 5100`
- 已有服务则复用；强制重启：`--web_restart`

启动命令：

```bash
python main.py --web --web_port 5100
```

强制重启同端口：

```bash
python main.py --web --web_port 5100 --web_restart
```

浏览器访问：

- `http://localhost:5100/`

> 服务器若是远程机器，请把 `localhost` 换成服务器 IP/域名，并确保防火墙/安全组放行端口。

---

## 5. 运行验证（必做）

### 5.1 服务健康检查

打开：

- `http://<host>:<port>/api/health`

应返回：

```json
{"service":"motion_sickness_ui","ok":true}
```

### 5.2 UI 联动验证

在浏览器打开首页后：

- 选择模型 `.pkl`
- 回放模式选择 `data.bdf`
- 点击“开始回放”

后端终端应看到周期性输出：

- `预测结果: score=..., conf=..., proc=...ms, progress=...%`

前端：

- 仪表盘随分数变化
- “回放中”按钮填充进度
- 回放结束弹 toast，仪表盘归零

---

## 6. 常见问题与排查

### 6.1 回放选择了 `data.bdf`，为什么还会读取 `evt.bdf`？

这是正常现象：`neuracle_lib/readbdfdata.py` 会同时读取 `data.bdf`（信号）和 `evt.bdf`（事件/annotation）。

### 6.2 “端口被占用”但看起来是 Python 进程

若端口上的服务是我们自己启动的：

- 直接再次运行：`python main.py --web --web_port <port>` 会提示复用（刷新即可）
- 需要同端口重启：加 `--web_restart`

若端口被系统/其他服务占用（例如 macOS 的 `ControlCe` 占用 5000），请换一个端口（如 5100/8000/9000）。

### 6.3 前端不更新，但后端在出预测

确认：

- 浏览器访问的是 `http://<host>:<port>/`（同端口提供 UI 与 API）
- 控制台（Console）是否有 WebSocket 连接错误
- 后端 WebSocket 是否打印 “推送数据失败”

### 6.4 运行很慢/回放不对齐

建议：

- 减少终端输出（Web 模式默认已降噪）
- 确认采样率对齐逻辑：原始 EEG/ECG=1000Hz、IMU=100Hz，进入模型前重采样到 250Hz

---

## 7. 迁移前自检清单（建议你逐项打勾）

- [ ] 新服务器 Python 版本 ≥ 3.9
- [ ] `torch` 可用（GPU/CPU 匹配）
- [ ] 安装了 `fastapi`、`uvicorn`
- [ ] 如需 BDF 回放：安装 `mne` 且迁移 `EEG_trigger/`
- [ ] 模型 checkpoint 路径可访问
- [ ] 数据目录 `data/**` 可访问且包含 `data.bdf`
- [ ] 端口放行（本地/远程）

