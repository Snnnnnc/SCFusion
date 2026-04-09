import numpy as np
import pickle
import os

def compute_and_save_stats():
    data_dir = "data/training_dataset_random"
    output_file = "data/real_normalization_stats.pkl"
    
    modalities = ['imu', 'eeg', 'ecg']
    stats = {}
    
    for mod in modalities:
        file_path = os.path.join(data_dir, f"{mod}_windows.npy")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        print(f"Computing stats for {mod}...")
        # Load data (using mmap to save memory if needed, but these files are few GBs max)
        data = np.load(file_path) # (num_windows, channels, window_length)
        
        # Compute mean and std across windows and time steps
        # data shape is (N, C, T)
        # We want mean and std per channel (C)
        mean = np.mean(data, axis=(0, 2))
        std = np.std(data, axis=(0, 2))
        
        # Handle zero std
        std = np.where(std < 1e-8, 1.0, std)
        
        stats[mod] = {
            'mean': mean,
            'std': std
        }
        print(f"  {mod} mean shape: {mean.shape}, std shape: {std.shape}")
        print(f"  {mod} first 3 channels mean: {mean[:3]}")

    # Save to pkl
    with open(output_file, 'wb') as f:
        pickle.dump(stats, f)
    
    print(f"\n✓ Real statistics saved to: {output_file}")
    print("Please use this file with --normalization_stats_path")

if __name__ == "__main__":
    compute_and_save_stats()
