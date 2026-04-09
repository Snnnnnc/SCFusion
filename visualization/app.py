import os
import sys
import time
import threading
import asyncio
from typing import Dict, Optional
from nicegui import ui, app
import plotly.graph_objects as go

# 添加项目根目录到路径，以便导入 deployment 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from deployment.config import DeploymentConfig
from deployment.comfort_predictor import ComfortPredictor

# --- 样式定义 ---
DARK_BG = '#0b0e14'
ACCENT_COLOR = '#00f2ff'  # 科技感青色
CARD_BG = '#1a1f26'

# --- 全局状态 ---
class UIState:
    def __init__(self):
        self.monitoring_active = False
        self.mode = 'realtime'  # 'realtime' or 'playback'
        self.model_path = ''
        self.data_path = ''
        self.comfort_score = 0.0
        self.is_deployed = False
        self.predictor: Optional[ComfortPredictor] = None
        self.log_messages = []

state = UIState()

# --- 辅助函数 ---
def update_meter(score: float):
    """更新舒适度仪表盘 (Plotly 实现)"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Comfort Score", 'font': {'size': 24, 'color': ACCENT_COLOR}},
        gauge = {
            'axis': {'range': [0, 4], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': ACCENT_COLOR},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 1], 'color': 'rgba(0, 255, 0, 0.3)'},
                {'range': [1, 2], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [2, 3], 'color': 'rgba(255, 165, 0, 0.3)'},
                {'range': [3, 4], 'color': 'rgba(255, 0, 0, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 3.5
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"},
        margin=dict(l=20, r=20, t=50, b=20),
        height=400
    )
    return fig

# --- 回调函数 ---
def on_prediction(result: Dict):
    """当模型输出预测结果时的回调"""
    state.comfort_score = result.get('score', 0.0)
    # 触发 UI 更新
    meter_display.update_figure(update_meter(state.comfort_score))
    log_area.push(f"[{result.get('datetime')}] Predicted Score: {state.comfort_score:.2f}")

async def start_deployment():
    """启动部署/回放"""
    if not state.model_path:
        ui.notify('请先选择模型文件', type='warning')
        return
    
    if state.mode == 'playback' and not state.data_path:
        ui.notify('回放模式需要选择数据文件', type='warning')
        return

    try:
        # 构建配置
        config = DeploymentConfig(
            checkpoint_path=state.model_path,
            use_original_data=(state.mode == 'playback'),
            # 这里可以根据 UI 选择进一步完善参数
            subject_id="zzh" if state.mode == 'playback' else None,
            map_id="01" if state.mode == 'playback' else None,
        )
        
        state.predictor = ComfortPredictor(config)
        state.predictor.set_prediction_callback(on_prediction)
        
        # 在后台线程启动预测服务
        threading.Thread(target=state.predictor.start, kwargs={'collect_interval': 0.02}, daemon=True).start()
        
        state.is_deployed = True
        deploy_btn.disable()
        ui.notify('部署成功，正在监测...', type='positive')
        
    except Exception as e:
        ui.notify(f'部署失败: {e}', type='negative')

def toggle_comfort_monitoring():
    """切换舒适度监测板块"""
    state.monitoring_active = not state.monitoring_active
    if state.monitoring_active:
        left_initial.set_visibility(False)
        left_meter.set_visibility(True)
        right_options.set_visibility(True)
    else:
        left_initial.set_visibility(True)
        left_meter.set_visibility(False)
        right_options.set_visibility(False)

# --- UI 布局 ---
ui.query('body').style(f'background-color: {DARK_BG}; color: white; font-family: "Inter", sans-serif;')

with ui.header().classes('items-center justify-between bg-transparent p-4'):
    ui.label('驾驶员状态监测系统').classes('text-2xl font-bold tracking-wider').style(f'color: {ACCENT_COLOR}')
    ui.icon('settings', size='24px').classes('cursor-pointer')

with ui.row().classes('w-full h-screen no-wrap p-6 gap-6'):
    # --- 左侧显示区 ---
    with ui.column().classes('w-2/3 h-full bg-[#1a1f26] rounded-xl p-8 items-center justify-center shadow-2xl border border-gray-800'):
        # 初始界面：座舱图
        left_initial = ui.column().classes('items-center justify-center w-full h-full')
        with left_initial:
            ui.image('https://images.unsplash.com/photo-1552519507-da3b142c6e3d?auto=format&fit=crop&q=80&w=1000').classes('w-4/5 rounded-lg opacity-80')
            ui.label('等待功能选择...').classes('mt-4 text-gray-400 italic')
        
        # 监测界面：仪表盘
        left_meter = ui.column().classes('w-full items-center justify-center').set_visibility(False)
        with left_meter:
            meter_display = ui.plotly(update_meter(0.0)).classes('w-full')
            ui.label('实时舒适度指数').classes('text-xl font-light tracking-widest mt-4')

    # --- 右侧控制面板 ---
    with ui.column().classes('w-1/3 h-full gap-6'):
        # 标题与主开关
        with ui.card().classes('w-full p-6 bg-[#1a1f26] border border-gray-800 shadow-xl'):
            ui.label('驾驶员状态监测').classes('text-lg font-semibold mb-4 text-gray-300')
            with ui.row().classes('items-center justify-between w-full'):
                ui.label('舒适度监测').classes('text-md')
                ui.switch(on_change=toggle_comfort_monitoring).props('color=cyan')

        # 功能选项区 (选中舒适度后显示)
        right_options = ui.column().classes('w-full gap-6').set_visibility(False)
        with right_options:
            with ui.card().classes('w-full p-6 bg-[#1a1f26] border border-gray-800'):
                ui.label('监测模式').classes('text-sm text-gray-400 mb-2')
                mode_selection = ui.radio(['实时监测', '数据回放'], value='实时监测', on_change=lambda e: setattr(state, 'mode', 'realtime' if e.value == '实时监测' else 'playback')).props('inline color=cyan')
                
                ui.separator().classes('my-4 bg-gray-700')
                
                # 模型加载
                ui.label('模型加载').classes('text-sm text-gray-400 mb-1')
                with ui.row().classes('w-full items-center gap-2'):
                    model_input = ui.input(placeholder='选择本地模型 (.pkl)...').classes('flex-grow').props('dark outlined dense color=cyan')
                    ui.button(icon='folder_open', on_click=lambda: ui.notify('此处应弹出文件选择器 (Web环境受限，请手动输入路径)')).props('flat color=cyan')
                
                # 数据加载 (仅回放模式)
                data_section = ui.column().classes('w-full mt-4')
                with data_section:
                    ui.label('数据加载').classes('text-sm text-gray-400 mb-1')
                    with ui.row().classes('w-full items-center gap-2'):
                        data_input = ui.input(placeholder='选择原始数据 (BDF/Folder)...').classes('flex-grow').props('dark outlined dense color=cyan')
                        ui.button(icon='folder_open').props('flat color=cyan')
                
                # 根据模式切换数据加载部分的可见性
                ui.bind_visibility_from(data_section, mode_selection, 'value', backward=lambda v: v == '数据回放')

            # 确认/应用按钮
            with ui.row().classes('w-full mt-auto'):
                deploy_btn = ui.button('确认部署', on_click=start_deployment).classes('w-full h-12 text-lg font-bold').props('color=cyan rounded')
                # 绑定按钮文本
                ui.bind_text_from(deploy_btn, mode_selection, 'value', backward=lambda v: '开始回放' if v == '数据回放' else '确认部署')
                
                # 监听输入变化，恢复按钮
                def on_input_change():
                    state.model_path = model_input.value
                    state.data_path = data_input.value
                    if state.is_deployed:
                        state.is_deployed = False
                        deploy_btn.enable()
                
                model_input.on('change', on_input_change)
                data_input.on('change', on_input_change)

            # 日志区
            with ui.card().classes('w-full p-4 bg-black/30 border border-gray-800 h-48 overflow-hidden'):
                ui.label('系统日志').classes('text-xs text-gray-500 mb-2')
                log_area = ui.log().classes('w-full h-full text-xs font-mono text-green-400')

ui.run(title='Comfort Sim Dashboard', port=8080, dark=True)
