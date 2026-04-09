import os
import sys
import json
import time
import asyncio
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path

# 确保项目根目录在路径中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 使用绝对导入
from deployment.comfort_predictor import ComfortPredictor
from deployment.config import DeploymentConfig

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor_instance = None
predictor_thread = None
predictor_bg_thread = None
predictor_running = False

# 车端对接：start 信号与最近一次预测缓存（供车端轮询拉取）
# subject / map 从当前回放数据路径解析，随 start、comfort 一并下发给车端
START_STATE = {
    "start": False,
    "start_at": None,
    "countdown_sec": 5,
    "subject": None,
    "map": None,
}
LAST_PRED = {"score": 0, "confidence": 0.0, "timestamp": 0.0, "datetime": None, "proc_time_ms": 0.0}


def _parse_subject_map_from_bdf_path(bdf_path: Optional[str]) -> tuple:
    """
    从回放 BDF 文件路径解析 subject 和 map。
    约定：路径中某级目录名为 时间戳_subject_map_* 形式（如 20251108173247_wyx_05_01_wyx），
    则取第二段为 subject、第三段为 map。不要求路径真实存在。
    """
    if not bdf_path or not bdf_path.strip():
        return None, None
    p = Path(bdf_path)
    # 先看直接父目录名
    session_name = p.parent.name
    # 若只有一层（如 wyx），再往上一级找 时间戳_subject_map_* 形式
    if "_" not in session_name:
        parent_parent = p.parent.parent
        session_name = parent_parent.name if parent_parent.name else session_name
    parts = session_name.split("_")
    if len(parts) >= 3:
        return parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1]
    if len(parts) == 1 and session_name:
        return session_name, None
    return None, None


def _ensure_predictor_running():
    """
    关键：预测循环不能只依赖 WebSocket，否则车端仅 HTTP 轮询时永远不会触发预测。
    这里在后台线程启动 predictor.start()，保证 /api/comfort 能拿到真实评分。
    确保只启动一次（防止 WebSocket 和后台线程同时启动导致重复）。
    """
    global predictor_bg_thread, predictor_running, predictor_instance
    if predictor_instance is None:
        return
    # 如果已经在运行，直接返回（防止重复启动）
    if predictor_running:
        return
    # 如果线程还在运行，也返回
    if predictor_bg_thread is not None and predictor_bg_thread.is_alive():
        return

    def _runner():
        global predictor_running
        predictor_running = True
        try:
            predictor_instance.start()
        except Exception as e:
            print(f"预测线程异常: {e}")
        finally:
            predictor_running = False

    predictor_bg_thread = threading.Thread(target=_runner, daemon=True)
    predictor_bg_thread.start()


@app.get("/api/status")
async def get_status():
    """前端/车端可选：查看预测服务是否在跑，以及当前回放进度。"""
    progress = 0.0
    try:
        if predictor_instance and hasattr(predictor_instance, "collector") and hasattr(predictor_instance.collector, "pointers"):
            col = predictor_instance.collector
            if col.data.get("eeg") is not None and col.data["eeg"].shape[1] > 0:
                progress = float(col.pointers["eeg"]) / float(col.data["eeg"].shape[1])
    except Exception:
        pass
    return JSONResponse(
        {
            "running": bool(predictor_running),
            "ready": bool(getattr(predictor_instance, "ready_event", None).is_set()) if predictor_instance else False,
            "progress": float(progress),
            "start": bool(START_STATE.get("start", False)),
            "start_at": START_STATE.get("start_at"),
            "subject": START_STATE.get("subject"),
            "map": START_STATE.get("map"),
        }
    )


@app.get("/api/health")
async def health():
    """用于前端/脚本探测：确认这是我们的服务"""
    return JSONResponse(
        {
            "service": "motion_sickness_ui",
            "ok": True,
        }
    )


@app.get("/")
async def serve_ui():
    """
    直接从同一个端口提供 UI 页面（固定链接：http://localhost:<port>/）。
    注意：preview_ui.html 位于项目根目录。
    """
    ui_path = Path(project_root) / "preview_ui.html"
    if not ui_path.exists():
        raise HTTPException(status_code=404, detail=f"未找到 UI 文件: {ui_path}")
    # FileResponse 会自动设置 Content-Type: text/html
    return FileResponse(str(ui_path))

class DeployRequest(BaseModel):
    checkpoint_path: str
    normalization_stats_path: Optional[str] = None
    use_original_data: bool = False
    subject_id: Optional[str] = None
    map_id: Optional[str] = None
    bdf_path: Optional[str] = None


class StartRequest(BaseModel):
    countdown_sec: int = 5
    # 可选：由前端直接指定 start_at（epoch seconds）。不提供则由服务端生成 now + countdown_sec
    start_at: Optional[float] = None

@app.get("/api/list-files")
async def list_files(path: str = ".", ext: str = ""):
    """浏览本地目录"""
    try:
        # 这里的路径是相对于项目根目录的
        root = Path(os.getcwd())
        target = (root / path).resolve()
        
        # 安全检查：防止越权访问根目录以外
        if not str(target).startswith(str(root)):
            target = root
            
        items = []
        for item in target.iterdir():
            if item.name.startswith('.'): continue
            
            is_dir = item.is_dir()
            # 如果指定了扩展名，则只显示目录或匹配的文件
            if not is_dir and ext and not item.name.endswith(ext):
                continue
                
            items.append({
                "name": item.name,
                "is_dir": is_dir,
                "path": str(item.relative_to(root))
            })
            
        return {
            "current_path": str(target.relative_to(root)) if target != root else ".",
            "items": sorted(items, key=lambda x: (not x['is_dir'], x['name']))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/configure")
async def configure_predictor(req: DeployRequest):
    global predictor_instance

    # 每次重新配置视为新会话：重置 start 状态，便于本次配置后再次「开始回放」时 start API 正常
    START_STATE["start"] = False
    START_STATE["start_at"] = None

    # 从回放数据路径解析 subject / map，供车端与 start、comfort 一起拉取
    parsed_subject, parsed_map = _parse_subject_map_from_bdf_path(req.bdf_path)
    if parsed_subject is not None or parsed_map is not None:
        START_STATE["subject"] = parsed_subject
        START_STATE["map"] = parsed_map
    else:
        # 未指定 bdf_path 或解析失败时，用请求里的 subject_id / map_id
        START_STATE["subject"] = req.subject_id
        START_STATE["map"] = req.map_id

    # 转换为绝对路径
    abs_ckpt = os.path.abspath(req.checkpoint_path)
    if not os.path.exists(abs_ckpt):
        raise HTTPException(status_code=400, detail=f"找不到模型文件: {abs_ckpt}")

    print(f"正在配置预测引擎: {abs_ckpt}")

    config = DeploymentConfig(
        checkpoint_path=abs_ckpt,
        normalization_stats_path=req.normalization_stats_path,
        use_original_data=req.use_original_data,
        subject_filter=req.subject_id or parsed_subject,
        map_filter=req.map_id or parsed_map,
        bdf_path=req.bdf_path,
        start_at=START_STATE.get("start_at"),
        verbose=False  # 默认降噪；需要调试再开
    )

    try:
        if predictor_instance:
            predictor_instance.stop()
        predictor_instance = ComfortPredictor(config)
        # 如果 start 已经下发，直接后台启动预测（不依赖 WS）
        if START_STATE.get("start") is True:
            _ensure_predictor_running()
        return {"status": "success", "message": "配置成功"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/start")
async def get_start():
    """
    车端轮询：是否开始；包含 start_at（epoch seconds）与 countdown_sec。
    关键：只有服务端 ready（BDF/模型已初始化）后，才返回 start: true 和 start_at。
    否则返回 start: false，车端继续等待。
    """
    global predictor_instance
    
    # 如果前端还没请求开始，直接返回 start: false
    if not START_STATE.get("start", False):
        return JSONResponse({
            "start": False,
            "start_at": None,
            "countdown_sec": 5,
            "subject": START_STATE.get("subject"),
            "map": START_STATE.get("map"),
        })
    
    # 确保预测后台启动
    _ensure_predictor_running()
    
    # 检查服务端是否 ready（BDF 已读完、模型已加载）
    is_ready = False
    if predictor_instance is not None and hasattr(predictor_instance, "ready_event"):
        is_ready = predictor_instance.ready_event.is_set()
    
    if not is_ready:
        # 服务端还在初始化，车端继续等待
        return JSONResponse({
            "start": False,
            "start_at": None,
            "countdown_sec": START_STATE.get("countdown_sec", 5),
            "subject": START_STATE.get("subject"),
            "map": START_STATE.get("map"),
        })
    
    # 服务端已 ready，生成 start_at（now + countdown_sec）
    if START_STATE.get("start_at") is None:
        wall_now = time.time()
        countdown = START_STATE.get("countdown_sec", 5)
        start_at = wall_now + countdown
        START_STATE["start_at"] = start_at
        
        # 更新 predictor 配置
        if predictor_instance is not None and hasattr(predictor_instance, "config"):
            predictor_instance.config.start_at = start_at
            # 若 collector 已初始化，也允许把 start_time 推到未来（实现"统一起跑"）
            if hasattr(predictor_instance, "collector") and hasattr(predictor_instance.collector, "start_time"):
                predictor_instance.collector.start_time = start_at
        
        print(f"[START] service ready, start_at={start_at:.3f}, countdown={countdown}s")
    
    return JSONResponse(START_STATE)


def _set_start_at_when_ready():
    """
    Web 单独使用时无人轮询 GET /api/start，预测线程会一直等 start_at。
    本函数在后台等 ready 后自动设置 start_at，让回放能开始。
    """
    global predictor_instance
    if predictor_instance is None:
        return
    countdown = START_STATE.get("countdown_sec", 5)
    # 等待就绪（BDF/模型加载完成），最多等 60 秒
    for _ in range(1200):  # 0.05 * 1200 = 60s
        if getattr(predictor_instance, "ready_event", None) and predictor_instance.ready_event.is_set():
            break
        time.sleep(0.05)
    if not getattr(predictor_instance, "ready_event", None) or not predictor_instance.ready_event.is_set():
        return
    # 若尚未设置 start_at（例如没有车端轮询 GET /api/start），则由我们设置
    if START_STATE.get("start_at") is None and START_STATE.get("start") is True:
        start_at = time.time() + countdown
        START_STATE["start_at"] = start_at
        if hasattr(predictor_instance, "config"):
            predictor_instance.config.start_at = start_at
        if hasattr(predictor_instance, "collector") and hasattr(predictor_instance.collector, "start_time"):
            predictor_instance.collector.start_time = start_at
        print(f"[START] 服务已就绪，已自动设置 start_at（{countdown}s 后开始），回放将开始")


@app.post("/api/start")
async def set_start(req: StartRequest, request: Request):
    """
    前端触发：标记"开始请求"，立即返回（不等待 ready）。
    车端轮询时由 GET /api/start 设置 start_at；仅 Web 时由后台线程在 ready 后自动设置。
    """
    global predictor_instance
    # 立即标记"前端已请求开始"，不阻塞
    START_STATE["start"] = True
    START_STATE["countdown_sec"] = int(req.countdown_sec)
    START_STATE["start_at"] = None  # 先设为 None，等 ready 后再生成

    # 确保预测后台启动（但不等待 ready）
    _ensure_predictor_running()

    # 仅 Web 使用时无人调 GET /api/start，需在 ready 后自动设置 start_at，否则预测线程会一直卡在等待
    t = threading.Thread(target=_set_start_at_when_ready, daemon=True)
    t.start()

    try:
        client_host = request.client.host if request.client else "unknown"
    except Exception:
        client_host = "unknown"
    print(f"[START] start request from {client_host}, waiting for service ready...")

    return JSONResponse({"status": "ok", "message": "等待服务端初始化完成..."})


@app.post("/api/stop")
async def stop_all():
    """可选：停止回放并清除 start 状态，车端轮询将得到 start: false。"""
    global predictor_instance
    START_STATE["start"] = False
    START_STATE["start_at"] = None
    if predictor_instance:
        predictor_instance.stop()
    return JSONResponse({"status": "ok"})


@app.get("/api/comfort")
async def get_comfort():
    """车端每 10s 拉取：返回最近一次舒适度评分。"""
    # 车端只轮询时，确保预测线程被拉起
    if START_STATE.get("start") is True:
        _ensure_predictor_running()
    return JSONResponse(
        {
            "comfort": LAST_PRED.get("score", 0),
            "score": LAST_PRED.get("score", 0),
            "confidence": LAST_PRED.get("confidence", 0.0),
            "timestamp": LAST_PRED.get("timestamp", 0.0),
            "datetime": LAST_PRED.get("datetime"),
            "proc_time_ms": LAST_PRED.get("proc_time_ms", 0.0),
            "start_at": START_STATE.get("start_at"),
            "started": START_STATE.get("start", False),
            "running": bool(predictor_running),
            "subject": START_STATE.get("subject"),
            "map": START_STATE.get("map"),
        }
    )

@app.websocket("/ws/comfort")
async def comfort_websocket(websocket: WebSocket):
    await websocket.accept()
    global predictor_instance
    
    if predictor_instance is None:
        await websocket.send_json({"error": "未配置预测引擎"})
        await websocket.close()
        return

    # 回调函数发送数据给前端
    # ✅ 关键修复：保存当前 websocket handler 的事件循环，用于跨线程安全推送
    ws_loop = asyncio.get_running_loop()

    def on_prediction(result):
        try:
            event_type = result.get("event", "prediction")
            # ⚠️ 关键修复：progress 事件不包含 score，避免覆盖仪表盘数据
            payload = {
                "event": event_type,
                "progress": float(result.get("progress", 0.0)),
                "timestamp": float(result.get("timestamp", 0.0)),
                "datetime": result.get("datetime"),
                "proc_time_ms": float(result.get("proc_time_ms", 0.0)),
            }
            # 只有 prediction 事件才包含 score 和 confidence
            if event_type == "prediction":
                payload["score"] = int(result.get("score", 0))
                payload["confidence"] = float(result.get("confidence", 0.0))
                # 缓存最近一次评分给车端轮询接口
                LAST_PRED.update(
                    {
                        "score": payload["score"],
                        "confidence": payload["confidence"],
                        "timestamp": payload["timestamp"],
                        "datetime": payload["datetime"],
                        "proc_time_ms": payload["proc_time_ms"],
                    }
                )
            elif event_type == "finished":
                # finished 事件也需要 score（用于归零）
                payload["score"] = int(result.get("score", 0))
                payload["confidence"] = float(result.get("confidence", 0.0))
                # 回放自然结束时清除 start 状态，避免车端轮询到旧缓存继续仿真
                START_STATE["start"] = False
                START_STATE["start_at"] = None

            asyncio.run_coroutine_threadsafe(
                websocket.send_json(payload),
                ws_loop
            )
        except Exception as e:
            print(f"推送数据失败: {e}")

    predictor_instance.set_prediction_callback(on_prediction)
    
    # ✅ 关键修复：WebSocket 只设置 callback，不直接启动 predictor
    # predictor 由 _ensure_predictor_running() 统一管理，避免重复启动
    _ensure_predictor_running()
    
    try:
        print("WebSocket 链路已打通，等待预测数据...")
        # WebSocket 保持连接，等待客户端断开或异常
        # 使用无限循环等待，直到连接断开
        while True:
            try:
                # 等待客户端消息（客户端断开时会抛出异常）
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                # 超时继续循环（保持连接）
                continue
            except WebSocketDisconnect:
                # 客户端主动断开
                break
            except Exception:
                # 其他异常（连接已断开）
                break
    except Exception as e:
        print(f"WebSocket 运行异常: {e}")
    finally:
        # ✅ 兜底：如果是原始数据回放，确保前端一定能收到 finished（防止服务结束过快导致消息丢失）
        try:
            if predictor_instance and getattr(predictor_instance.config, "use_original_data", False):
                await websocket.send_json(
                    {
                        "event": "finished",
                        "score": 0,
                        "confidence": 0.0,
                        "timestamp": 0.0,
                        "datetime": None,
                        "proc_time_ms": 0.0,
                        "progress": 1.0,
                    }
                )
        except Exception:
            pass
        # ⚠️ 重要：WebSocket 关闭时不要 stop predictor，因为后台线程可能还在使用
        # predictor 的生命周期由 _ensure_predictor_running() 和 /api/stop 管理
        print("WebSocket 连接已关闭")

# 注意：请通过根目录的 main.py --web 来启动服务，不要直接运行此文件
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
