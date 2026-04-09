import numpy as np
import os

path = "/Users/shennc/Desktop/THU/25 SPR/comfort sim/motion_sickness_classification/data/processed/cyz/20251108182115_cyz_01_cyz/events.npy"
try:
    data = np.load(path, allow_pickle=True)
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"First 10 elements:\n{data[:10]}")
    if len(data.shape) > 1 and data.shape[1] >= 3:
        print(f"Unique ratings: {np.unique(data[:, 2])}")
except Exception as e:
    print(f"Error: {e}")
