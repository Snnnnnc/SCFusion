### 与车端（CARLA 服务器）联动：不迁移模型，仅做数据/评分传输

目标：车端脚本 `auto_record_drive_comfort.py` **常开**运行在车端服务器上；本机舒适度系统作为**评分服务端**，实现：

1. 前端点击“开始回放”→ 服务端下发 start 信号 → 车端开始仿真  
2. 服务端与车端按同一个 `start_at` 对齐，并用倒计时保证“同时起跑”（考虑 CARLA 渲染延迟）  
3. 服务端每 10 秒产出舒适度评分，车端每 10 秒拉取并按规则切换驾驶模式

---

## 1. 通信方式（当前实现：车端轮询 HTTP）

这是最稳、最少依赖的方式，不需要在两端维护长连接：

- **Start 信号**：车端轮询 `GET /api/start`
- **舒适度评分**：车端每 10 秒请求 `GET /api/comfort`

优点：实现简单、网络抖动下更稳定；缺点：严格意义不是 push，但效果等价于 10 秒一次推送。

---

## 2. 服务端（本机舒适度系统）对外接口

假设本机运行在 `http://<SERVER_HOST>:5100/`，则：

- **健康检查**：`GET http://<SERVER_HOST>:5100/api/health`
- **start 状态（车端轮询）**：`GET http://<SERVER_HOST>:5100/api/start`
  - 返回示例：
    ```json
    {"start": true, "start_at": 173..., "countdown_sec": 5}
    ```
- **前端触发 start**：`POST http://<SERVER_HOST>:5100/api/start`
  - body：
    ```json
    {"countdown_sec": 5}
    ```
  - 逻辑：服务端生成 `start_at = now + countdown_sec`，同时写入预测服务配置，用于对齐。
- **最新评分（车端拉取）**：`GET http://<SERVER_HOST>:5100/api/comfort`
  - 返回示例：
    ```json
    {"comfort": 3, "score": 3, "confidence": 0.98, "timestamp": 173..., "start_at": 173..., "started": true}
    ```

> 注意：`start_at` 是 Unix epoch seconds。要保证两台机器时间同步（NTP/chrony），才能做到“同一时刻起跑”。

---

## 3. 前端触发逻辑（你需要做什么）

你在前端页面点击“开始回放”后，系统会：

1) `POST /api/configure`：加载模型与数据  
2) `POST /api/start`：下发 start 信号（默认倒计时 5s）  
3) 建立 WebSocket：用于 UI 仪表盘显示预测与进度

---

## 4. 车端（CARLA 服务器）如何接入

在车端服务器上运行：

```bash
python auto_record_drive_comfort.py \
  --host <CARLA_HOST> --port 2000 \
  --agent Behavior --behavior aggressive \
  --comfort-url http://<SERVER_HOST>:5100/api/comfort \
  --start-signal-url http://<SERVER_HOST>:5100/api/start \
  --countdown 5
```

### 4.1 start_at 对齐（已在脚本内支持）

当前已增强 `auto_record_drive_comfort.py`：

- 若 `/api/start` 返回 `start_at`，车端会计算：
  - `countdown = ceil(start_at - time.time())`
- 若没有 `start_at`，则回退使用命令行 `--countdown`

---

## 4.2 倒计时在哪里设置？如何保证服务端和车端同时开始？

**倒计时设置位置：**

| 位置 | 说明 |
|------|------|
| **前端 UI** | 数据回放模式下有「起跑倒计时 (秒)」输入框，默认 5，可填 0～60。点击「开始回放」时会把该值通过 `POST /api/start` 的 body `{ "countdown_sec": N }` 发给服务端。 |
| **服务端默认** | `deployment/app.py` 里 `START_STATE["countdown_sec"] = 5`，以及 `StartRequest.countdown_sec = 5`。若前端不传则用 5。 |
| **车端命令行** | `auto_record_drive_comfort.py --countdown 5`。仅当服务端未返回 `start_at` 时，车端才用该值；**有 start_at 时以服务端为准**。 |

**如何保证同时开始：**

1. **服务端**：在 BDF/模型就绪后，计算 `start_at = time.time() + countdown_sec`（同一台机上的「当前时间 + 倒计时」），并写入 `START_STATE` 和预测器配置；预测回放以 `start_at` 为墙钟起点（`collector.start_time = start_at`）。
2. **车端**：轮询到 `GET /api/start` 返回 `start: true` 和 `start_at` 后，用 `countdown_sec = ceil(start_at - time.time())` 在本机等待，到点再启动仿真。
3. 两边都以**同一个绝对时间戳 `start_at`** 为起跑时刻，因此只要两台机器**时钟同步**（NTP/chrony），就会在同一秒内起跑。倒计时只是「从当前到起跑」的秒数，由服务端统一算好再下发给车端，避免两边各自数秒产生偏差。

---

## 5. 网络与端口要求

- 本机（舒适度服务端）需要对车端开放 `--web_port`（例如 5100）
- 车端需要能访问 `http://<SERVER_HOST>:5100/api/start` 与 `/api/comfort`
- 若跨网段/公网，建议使用内网 VPN 或反向代理（并考虑鉴权）

---

## 6. 下一步（可选增强）

- **真正 push**：改为车端 WebSocket 订阅评分（服务端推送），减少轮询
- **鉴权**：为 `/api/start` 与 `/api/comfort` 添加 token
- **更强时间同步**：服务端提供 `/api/time`（NTP-like），车端估计偏移量后再按 start_at 排程

