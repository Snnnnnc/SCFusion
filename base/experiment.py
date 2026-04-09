import os
import torch
import numpy as np
from abc import ABC, abstractmethod


class GenericExperiment(ABC):
    """Generic experiment base class"""
    
    def __init__(self, args):
        self.args = args
        
        # Basic experiment parameters
        self.experiment_name = getattr(args, 'experiment_name', 'motion_sickness_classification')
        self.model_name = getattr(args, 'model_name', 'PhysioFusionNet')
        self.stamp = getattr(args, 'stamp', 'default')
        self.dataset_path = getattr(args, 'dataset_path', './data/processed')
        self.save_path = getattr(args, 'save_path', './results')
        self.load_path = getattr(args, 'load_path', './models')
        self.debug = getattr(args, 'debug', False)
        self.resume = getattr(args, 'resume', False)
        self.seed = getattr(args, 'seed', 42)
        
        # Training parameters
        self.learning_rate = getattr(args, 'learning_rate', 0.001)
        self.min_learning_rate = getattr(args, 'min_learning_rate', 1e-6)
        self.batch_size = getattr(args, 'batch_size', 32)
        self.patience = getattr(args, 'patience', 10)
        self.factor = getattr(args, 'factor', 0.5)
        self.scheduler = getattr(args, 'scheduler', 'cosine')
        
        # Cross-validation parameters
        self.num_folds = getattr(args, 'num_folds', 5)
        self.folds_to_run = getattr(args, 'folds_to_run', range(1, self.num_folds + 1))
        
        # Data processing parameters
        self.calc_mean_std = getattr(args, 'calc_mean_std', True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize randomness
        self.init_randomness()
    
    def init_randomness(self):
        """Initialize random seeds for reproducibility"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
    
    def get_feature_dimension(self, config):
        """Get feature dimension from config"""
        return getattr(config, 'feature_dimension', 128)
    
    def get_multiplier(self, config):
        """Get multiplier from config"""
        return getattr(config, 'multiplier', 1.0)
    
    def get_time_delay(self, config):
        """Get time delay from config"""
        return getattr(config, 'time_delay', 0)
    
    def calc_mean_std_fn(self):
        """Calculate mean and std for normalization"""
        # This would be implemented based on your specific data processing needs
        pass
    
    @abstractmethod
    def get_modality(self):
        """Get modality information - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_config(self):
        """Get configuration - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_selected_continuous_label_dim(self):
        """Get selected continuous label dimensions - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def init_data_arranger(self):
        """Initialize data arranger - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def init_dataset(self, data, continuous_label_dim, mode, fold):
        """Initialize dataset - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def init_model(self):
        """Initialize model - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def init_dataloader(self, fold):
        """Initialize dataloaders - must be implemented by subclasses"""
        pass