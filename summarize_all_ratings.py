import numpy as np
import os
from collections import Counter

root_path = "/Users/shennc/Desktop/THU/25 SPR/comfort sim/motion_sickness_classification/data/processed"

def get_distribution():
    results = {}
    
    # 遍历每个被试文件夹
    for subject in sorted(os.listdir(root_path)):
        subj_path = os.path.join(root_path, subject)
        if not os.path.isdir(subj_path) or subject.startswith('.'):
            continue
            
        results[subject] = {}
        
        # 遍历该被试的每个实验 session (地图)
        for session in sorted(os.listdir(subj_path)):
            sess_path = os.path.join(subj_path, session)
            if not os.path.isdir(sess_path):
                continue
                
            event_file = os.path.join(sess_path, "events.npy")
            if os.path.exists(event_file):
                try:
                    # 读取事件数据
                    data = np.load(event_file, allow_pickle=True)
                    
                    # 格式是 (1, time_steps) 或 (time_steps,)
                    # 每一位都是一个评分值
                    ratings = data.flatten()
                    
                    # 统计分布
                    dist = Counter(ratings)
                    # 记录评分数值
                    results[subject][session] = dict(sorted(dist.items()))
                except Exception as e:
                    results[subject][session] = f"Error: {str(e)}"
    
    # 打印结果
    print(f"{'Subject':<10} | {'Session/Map':<40} | {'Distribution (Rating: Count)'}")
    print("-" * 100)
    for subj, sessions in results.items():
        for sess, dist in sessions.items():
            print(f"{subj:<10} | {sess:<40} | {dist}")

if __name__ == "__main__":
    get_distribution()
