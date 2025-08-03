# Configuration file for Motion Sickness Classification

# Data dimensions
EEG_EMBEDDING_DIM = 512
ECG_EMBEDDING_DIM = 256
EEG_TEMPORAL_DIM = 128
ECG_TEMPORAL_DIM = 64
BERT_DIM = 768  # If using text features

# Sampling rates
EEG_SAMPLING_RATE = 1000
ECG_SAMPLING_RATE = 500

# Channel configurations
EEG_CHANNELS = 64
ECG_CHANNELS = 3

config = {
    "frequency": {
        "eeg": None,
        "ecg": None,
        "bert": None
    },

    "multiplier": {
        "eeg": 1,
        "ecg": 1,
        "bert": 1,
    },

    "feature_dimension": {
        "eeg": (EEG_CHANNELS,),
        "ecg": (ECG_CHANNELS,),
        "bert": (BERT_DIM,)
    },

    "tcn": {
        "embedding_dim": EEG_EMBEDDING_DIM,
        "channels": {
            'eeg': [EEG_EMBEDDING_DIM, EEG_EMBEDDING_DIM//2, EEG_EMBEDDING_DIM//4, EEG_TEMPORAL_DIM],
            'ecg': [ECG_EMBEDDING_DIM, ECG_EMBEDDING_DIM//2, ECG_EMBEDDING_DIM//4, ECG_TEMPORAL_DIM],
            'bert': [BERT_DIM, BERT_DIM//2, BERT_DIM//4, 128]
        },
        "kernel_size": 5,
        "dropout": 0.1,
        "attention": 0,
    },

    "tcn_settings": {
        "eeg": {
            "input_dim": EEG_EMBEDDING_DIM,
            "channel": [256, 256, 128, 128],
            "kernel_size": 5
        },
        "ecg": {
            "input_dim": ECG_EMBEDDING_DIM,
            "channel": [128, 128, 64, 64],
            "kernel_size": 5
        },
        "bert": {
            "input_dim": BERT_DIM,
            "channel": [256, 256, 128, 128],
            "kernel_size": 5
        }
    },

    "backbone_settings": {
        "eeg_state_dict": "eeg_backbone_pretrained",
        "ecg_state_dict": "ecg_backbone_pretrained"
    },

    "fusion_settings": {
        "modal_dim": 64,
        "num_heads": 4,
        "dropout": 0.1,
        "fusion_type": "cross_attention"  # cross_attention, dense_coattention, transformer
    },

    "classification_settings": {
        "num_classes": 11,  # 0-10 motion sickness scores
        "hidden_dims": [512, 256, 128],
        "dropout": 0.3
    },

    "data_preprocessing": {
        "eeg_filter_low": 0.5,
        "eeg_filter_high": 50.0,
        "ecg_filter_low": 0.5,
        "ecg_filter_high": 40.0,
        "normalize": True,
        "segment_length": 3000,  # 3 seconds at 1000Hz
        "overlap": 0.5
    },

    "time_delay": 0,
    "metrics": ["accuracy", "precision", "recall", "f1", "confusion_matrix"],
    "save_plot": 1,

    "backbone": {
        "eeg": {
            "state_dict": "eeg_backbone_pretrained",
            "architecture": "cnn_1d",
            "input_channels": EEG_CHANNELS,
            "output_dim": EEG_EMBEDDING_DIM
        },
        "ecg": {
            "state_dict": "ecg_backbone_pretrained", 
            "architecture": "cnn_1d",
            "input_channels": ECG_CHANNELS,
            "output_dim": ECG_EMBEDDING_DIM
        }
    },
} 