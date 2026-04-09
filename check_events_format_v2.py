import numpy as np
import os

def check_file(path):
    print(f"\nChecking: {path}")
    try:
        # Use mmap_mode to avoid loading large files into memory
        data = np.load(path, allow_pickle=True, mmap_mode='r')
        print(f"  Shape: {data.shape}")
        # print(f"  Dtype: {data.dtype}")
        if len(data.shape) == 2 and data.shape[1] >= 3:
            ratings = data[:, 2]
            unique, counts = np.unique(ratings, return_counts=True)
            print("  Rating Distribution:")
            for u, c in zip(unique, counts):
                print(f"    {u}: {c}")
        else:
            print(f"  Unexpected shape: {data.shape}")
            print(f"  Sample: {data[:5]}")
    except Exception as e:
        print(f"  Error: {e}")

# Check one smaller session if possible
check_file("/Users/shennc/Desktop/THU/25 SPR/comfort sim/motion_sickness_classification/data/processed/whj/20251109195509_whj_04_whj/events.npy")
check_file("/Users/shennc/Desktop/THU/25 SPR/comfort sim/motion_sickness_classification/data/processed/cyz/20251108182115_cyz_01_cyz/events.npy")
