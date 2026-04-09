import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class GenericVideoTrainer(ABC):
    """Generic trainer base class"""
    
    def __init__(self, device, model_name, models, save_path, fold, 
                 min_epoch, max_epoch, early_stopping, scheduler, 
                 learning_rate, min_learning_rate, patience, batch_size,
                 criterion, factor, verbose, milestone, metrics,
                 load_best_at_each_epoch, save_plot, **kwargs):
        
        self.device = device
        self.model_name = model_name
        self.model = models
        self.save_path = save_path
        self.fold = fold
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.early_stopping = early_stopping
        self.scheduler_type = scheduler
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.patience = patience
        self.batch_size = batch_size
        self.criterion = criterion
        self.factor = factor
        self.verbose = verbose
        self.milestone = milestone
        self.metrics = metrics
        self.load_best_at_each_epoch = load_best_at_each_epoch
        self.save_plot = save_plot
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize training state
        self.fit_finished = False
        self.optimizer = None
        self.scheduler = None
    
    def get_parameters(self):
        """Get trainable parameters"""
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def init_optimizer_and_scheduler(self, epoch=0):
        """Initialize optimizer and scheduler - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def fit(self, dataloader_dict, checkpoint_controller, parameter_controller):
        """Fit the model - must be implemented by subclasses"""
        pass
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                   for k, v in data.items()}
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Store predictions and labels for metrics
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate_epoch(self, dataloader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in dataloader:
                data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in data.items()}
                labels = labels.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='macro', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @abstractmethod
    def test(self, checkpoint_controller, predict_only=1, **kwargs):
        """Test the model - must be implemented by subclasses"""
        pass