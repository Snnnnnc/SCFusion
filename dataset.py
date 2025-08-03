import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import pickle
from scipy import signal
from sklearn.preprocessing import StandardScaler, LabelEncoder


class PhysiologicalDataset(Dataset):
    """
    Dataset class for physiological signals (EEG and ECG)
    """
    
    def __init__(self, data_dict, modality, window_length=3000, 
                 hop_length=1500, normalize=True, apply_filter=True,
                 eeg_sampling_rate=1000, ecg_sampling_rate=500):
        super().__init__()
        
        self.data_dict = data_dict
        self.modality = modality
        self.window_length = window_length
        self.hop_length = hop_length
        self.normalize = normalize
        self.apply_filter = apply_filter
        self.eeg_sampling_rate = eeg_sampling_rate
        self.ecg_sampling_rate = ecg_sampling_rate
        
        # Extract data
        self.eeg_data = data_dict.get('eeg', None)
        self.ecg_data = data_dict.get('ecg', None)
        self.labels = data_dict.get('labels', None)
        
        # Preprocess data
        self.processed_data = self.preprocess_data()
        
        # Create segments
        self.segments = self.create_segments()
        
    def preprocess_data(self):
        """
        Preprocess physiological signals
        """
        processed = {}
        
        # Process EEG data
        if self.eeg_data is not None:
            processed['eeg'] = self.preprocess_eeg(self.eeg_data)
        
        # Process ECG data
        if self.ecg_data is not None:
            processed['ecg'] = self.preprocess_ecg(self.ecg_data)
        
        return processed
    
    def preprocess_eeg(self, eeg_data):
        """
        Preprocess EEG signals
        """
        processed_eeg = []
        
        for subject_eeg in eeg_data:
            # Apply bandpass filter (0.5-50 Hz)
            if self.apply_filter:
                subject_eeg = self.apply_eeg_filter(subject_eeg)
            
            # Remove artifacts (simple approach)
            subject_eeg = self.remove_eeg_artifacts(subject_eeg)
            
            # Normalize
            if self.normalize:
                subject_eeg = self.normalize_signal(subject_eeg)
            
            processed_eeg.append(subject_eeg)
        
        return np.array(processed_eeg)
    
    def preprocess_ecg(self, ecg_data):
        """
        Preprocess ECG signals
        """
        processed_ecg = []
        
        for subject_ecg in ecg_data:
            # Apply bandpass filter (0.5-40 Hz)
            if self.apply_filter:
                subject_ecg = self.apply_ecg_filter(subject_ecg)
            
            # Normalize
            if self.normalize:
                subject_ecg = self.normalize_signal(subject_ecg)
            
            processed_ecg.append(subject_ecg)
        
        return np.array(processed_ecg)
    
    def apply_eeg_filter(self, eeg_signal):
        """
        Apply bandpass filter to EEG signal
        """
        # Design bandpass filter
        nyquist = self.eeg_sampling_rate / 2
        low = 0.5 / nyquist
        high = 50.0 / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_signal = signal.filtfilt(b, a, eeg_signal, axis=1)
        
        return filtered_signal
    
    def apply_ecg_filter(self, ecg_signal):
        """
        Apply bandpass filter to ECG signal
        """
        # Design bandpass filter
        nyquist = self.ecg_sampling_rate / 2
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_signal = signal.filtfilt(b, a, ecg_signal, axis=1)
        
        return filtered_signal
    
    def remove_eeg_artifacts(self, eeg_signal):
        """
        Remove EEG artifacts using simple thresholding
        """
        # Simple artifact removal using thresholding
        threshold = 3 * np.std(eeg_signal)
        eeg_signal = np.clip(eeg_signal, -threshold, threshold)
        
        return eeg_signal
    
    def normalize_signal(self, signal_data):
        """
        Normalize signal using z-score normalization
        """
        if len(signal_data.shape) == 1:
            # Single channel
            mean_val = np.mean(signal_data)
            std_val = np.std(signal_data)
            if std_val > 0:
                signal_data = (signal_data - mean_val) / std_val
        else:
            # Multiple channels
            for ch in range(signal_data.shape[0]):
                mean_val = np.mean(signal_data[ch])
                std_val = np.std(signal_data[ch])
                if std_val > 0:
                    signal_data[ch] = (signal_data[ch] - mean_val) / std_val
        
        return signal_data
    
    def create_segments(self):
        """
        Create overlapping segments from the signals
        """
        segments = []
        
        # Get the minimum length across all subjects
        min_length = float('inf')
        
        if self.eeg_data is not None:
            min_length = min(min_length, min([eeg.shape[1] for eeg in self.eeg_data]))
        
        if self.ecg_data is not None:
            min_length = min(min_length, min([ecg.shape[1] for ecg in self.ecg_data]))
        
        # Create segments
        for start in range(0, min_length - self.window_length + 1, self.hop_length):
            end = start + self.window_length
            segments.append((start, end))
        
        return segments
    
    def __len__(self):
        """Return the number of segments"""
        return len(self.segments) * len(self.data_dict.get('labels', [0]))
    
    def __getitem__(self, idx):
        """
        Get a segment of data
        """
        # Calculate subject and segment indices
        num_subjects = len(self.data_dict.get('labels', [0]))
        subject_idx = idx // len(self.segments)
        segment_idx = idx % len(self.segments)
        
        start, end = self.segments[segment_idx]
        
        # Extract segment data
        sample = {}
        
        if 'eeg' in self.modality and self.eeg_data is not None:
            sample['eeg'] = torch.FloatTensor(
                self.processed_data['eeg'][subject_idx, :, start:end]
            )
        
        if 'ecg' in self.modality and self.ecg_data is not None:
            sample['ecg'] = torch.FloatTensor(
                self.processed_data['ecg'][subject_idx, :, start:end]
            )
        
        # Add label
        if self.labels is not None:
            sample['label'] = torch.LongTensor([self.labels[subject_idx]])
        
        return sample


class DataArranger:
    """
    Data arrangement and preprocessing utility
    """
    
    def __init__(self, dataset_info, dataset_path, debug=0):
        self.dataset_info = dataset_info
        self.dataset_path = dataset_path
        self.debug = debug
        
    def load_data(self):
        """
        Load EEG and ECG data from files
        """
        data = {}
        
        # Load EEG data
        eeg_path = os.path.join(self.dataset_path, 'raw', 'eeg')
        if os.path.exists(eeg_path):
            data['eeg'] = self.load_eeg_data(eeg_path)
        
        # Load ECG data
        ecg_path = os.path.join(self.dataset_path, 'raw', 'ecg')
        if os.path.exists(ecg_path):
            data['ecg'] = self.load_ecg_data(ecg_path)
        
        # Load labels
        labels_path = os.path.join(self.dataset_path, 'raw', 'labels', 'motion_sickness_scores.csv')
        if os.path.exists(labels_path):
            data['labels'] = self.load_labels(labels_path)
        
        return data
    
    def load_eeg_data(self, eeg_path):
        """
        Load EEG data from files
        """
        eeg_files = sorted([f for f in os.listdir(eeg_path) if f.endswith('.mat') or f.endswith('.npy')])
        
        if self.debug > 0:
            eeg_files = eeg_files[:self.debug]
        
        eeg_data = []
        for file in eeg_files:
            file_path = os.path.join(eeg_path, file)
            
            if file.endswith('.mat'):
                # Load .mat file
                from scipy.io import loadmat
                mat_data = loadmat(file_path)
                # Assume the main variable is named 'eeg' or 'data'
                if 'eeg' in mat_data:
                    eeg_data.append(mat_data['eeg'])
                elif 'data' in mat_data:
                    eeg_data.append(mat_data['data'])
                else:
                    # Use the first variable that's not metadata
                    for key in mat_data.keys():
                        if not key.startswith('__') and mat_data[key].ndim >= 2:
                            eeg_data.append(mat_data[key])
                            break
            
            elif file.endswith('.npy'):
                # Load .npy file
                eeg_data.append(np.load(file_path))
        
        return eeg_data
    
    def load_ecg_data(self, ecg_path):
        """
        Load ECG data from files
        """
        ecg_files = sorted([f for f in os.listdir(ecg_path) if f.endswith('.mat') or f.endswith('.npy')])
        
        if self.debug > 0:
            ecg_files = ecg_files[:self.debug]
        
        ecg_data = []
        for file in ecg_files:
            file_path = os.path.join(ecg_path, file)
            
            if file.endswith('.mat'):
                # Load .mat file
                from scipy.io import loadmat
                mat_data = loadmat(file_path)
                # Assume the main variable is named 'ecg' or 'data'
                if 'ecg' in mat_data:
                    ecg_data.append(mat_data['ecg'])
                elif 'data' in mat_data:
                    ecg_data.append(mat_data['data'])
                else:
                    # Use the first variable that's not metadata
                    for key in mat_data.keys():
                        if not key.startswith('__') and mat_data[key].ndim >= 2:
                            ecg_data.append(mat_data[key])
                            break
            
            elif file.endswith('.npy'):
                # Load .npy file
                ecg_data.append(np.load(file_path))
        
        return ecg_data
    
    def load_labels(self, labels_path):
        """
        Load motion sickness labels
        """
        if labels_path.endswith('.csv'):
            df = pd.read_csv(labels_path)
            # Assume the label column is named 'score' or 'motion_sickness_score'
            if 'score' in df.columns:
                labels = df['score'].values
            elif 'motion_sickness_score' in df.columns:
                labels = df['motion_sickness_score'].values
            else:
                # Use the first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    labels = df[numeric_cols[0]].values
                else:
                    raise ValueError("No numeric label column found")
        else:
            labels = np.load(labels_path)
        
        return labels 