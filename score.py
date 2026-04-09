#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PsychoPy 实验脚本：周期性晕动/舒适度评分采集
---------------------------------------------------
功能：
- 每隔固定时间弹出评分滑条 (0–10)
- 被试在5秒内选择当前不适程度
- 自动记录时间戳、分值、反应时
- 输出CSV结果文件

依赖：
pip install psychopy pandas
"""

from psychopy import visual, core, event, gui
import pandas as pd
import time
import os

# ==========================
# 参数设置
# ==========================
PROMPT_INTERVAL = 30      # 周期性弹窗间隔 (秒)
RATING_DURATION = 5       # 每次评分窗口时长 (秒)
TOTAL_DURATION = 300      # 实验总时长 (秒)
OUTPUT_FILE = "./comfort_feedback_log.csv"
EXPERIMENT_NAME = "AutoDriving_ComfortFeedback"

# ==========================
# 被试信息输入框
# ==========================
info = {"被试编号": "", "性别": ["男", "女"], "年龄": ""}
dlg = gui.DlgFromDict(dictionary=info, title=EXPERIMENT_NAME)
if not dlg.OK:
    core.quit()

# ==========================
# 创建窗口
# ==========================
win = visual.Window(
    size=(1280, 800),
    fullscr=True,
    color="black",
    units="pix"
)

# ==========================
# 文字与滑条组件
# ==========================
prompt_text = visual.TextStim(
    win,
    text="请在 0–10 滑条上选择您当前的不适程度：\n(0 = 完全舒服，10 = 非常不适)",
    color="white",
    height=30,
    pos=(0, 150)
)

rating_scale = visual.Slider(
    win,
    ticks=list(range(11)),
    labels=["0", "10"],
    granularity=1,
    style='rating',
    size=(800, 20),
    pos=(0, 0),
    color='white'
)

confirm_text = visual.TextStim(
    win,
    text="按 [空格键] 确认提交",
    color="gray",
    height=24,
    pos=(0, -200)
)

countdown_text = visual.TextStim(
    win,
    text="",
    color="yellow",
    height=30,
    pos=(0, -100)
)

# ==========================
# 函数定义
# ==========================
def show_rating_prompt(prompt_id):
    """显示滑条评分界面"""
    rating_scale.reset()
    start_time = time.time()
    timer = core.Clock()
    response_value = None

    while timer.getTime() < RATING_DURATION:
        remaining = RATING_DURATION - timer.getTime()
        countdown_text.text = f"剩余时间: {remaining:.1f}s"

        prompt_text.draw()
        rating_scale.draw()
        confirm_text.draw()
        countdown_text.draw()
        win.flip()

        keys = event.getKeys()
        if 'space' in keys:
            response_value = rating_scale.getRating()
            rt = timer.getTime()
            break

    # 自动提交（超时）
    if response_value is None:
        response_value = rating_scale.getRating()
        rt = timer.getTime()

    record = {
        "prompt_id": prompt_id,
        "timestamp": time.time(),
        "response_value": response_value,
        "response_time": rt
    }
    return record

# ==========================
# 实验流程
# ==========================
results = []
global_clock = core.Clock()

# 开始提示
visual.TextStim(win, text="实验即将开始，请放松并保持坐姿。", color="white", height=36).draw()
win.flip()
core.wait(3)

next_prompt_time = PROMPT_INTERVAL
prompt_id = 1

while global_clock.getTime() < TOTAL_DURATION:
    t_now = global_clock.getTime()

    if t_now >= next_prompt_time:
        record = show_rating_prompt(prompt_id)
        record["sim_time"] = t_now
        results.append(record)
        print(f"✅ 收到评分 #{prompt_id}: {record['response_value']} (t={t_now:.1f}s)")
        prompt_id += 1
        next_prompt_time += PROMPT_INTERVAL

    # 可在此处插入CARLA/平台同步接口
    core.wait(0.1)

# 结束提示
visual.TextStim(win, text="实验结束，感谢您的参与！", color="white", height=36).draw()
win.flip()
core.wait(3)

win.close()

# ==========================
# 保存数据
# ==========================
df = pd.DataFrame(results)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False, float_format="%.3f")
print(f"\n✅ 实验结束，结果已保存至 {OUTPUT_FILE}")
print(df.head())
