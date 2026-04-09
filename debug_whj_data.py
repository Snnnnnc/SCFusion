import numpy as np
path = "/Users/shennc/Desktop/THU/25 SPR/comfort sim/motion_sickness_classification/data/processed/whj/20251109195509_whj_04_whj/events.npy"
data = np.load(path, allow_pickle=True)
print(f"Full data for whj_04_whj events.npy:")
print(f"Shape: {data.shape}")
print(f"Content:\n{data}")

# 顺便看一下同一个目录下的 imu.npy 长度，对比一下
imu_path = "/Users/shennc/Desktop/THU/25 SPR/comfort sim/motion_sickness_classification/data/processed/whj/20251109195509_whj_04_whj/imu.npy"
imu_data = np.load(imu_path, mmap_mode='r')
print(f"\nIMU data shape: {imu_data.shape}")
