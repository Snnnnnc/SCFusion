import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, ReduceLROnPlateau
import numpy as np


class GradualWarmupScheduler(_LRScheduler):
    """Gradual warmup scheduler"""
    
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


class MyWarmupScheduler(_LRScheduler):
    """Custom warmup scheduler with plateau reduction"""
    
    def __init__(self, optimizer, lr, min_lr, best, mode="max", patience=10, factor=0.5, 
                 num_warmup_epoch=10, init_epoch=0):
        self.lr = lr
        self.min_lr = min_lr
        self.best = best
        self.mode = mode
        self.patience = patience
        self.factor = factor
        self.num_warmup_epoch = num_warmup_epoch
        self.wait = 0
        self.warmup_finished = False  # Initialize before super().__init__
        
        # Set initial_lr for all param groups if not set
        for param_group in optimizer.param_groups:
            if 'initial_lr' not in param_group:
                param_group['initial_lr'] = lr
        
        # Initialize base lrs
        super().__init__(optimizer, last_epoch=init_epoch-1)
        
        # Initialize plateau scheduler
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, patience=patience, factor=factor, min_lr=min_lr
        )
        
        # Warmup scheduler
        self.warmup_scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1.0, total_epoch=num_warmup_epoch
        )
    
    @property
    def relative_epoch(self):
        """Get relative epoch (epochs since warmup finished)"""
        if self.warmup_finished:
            return max(0, self.last_epoch - self.num_warmup_epoch)
        return 0
    
    def step(self, metrics=None, epoch=None):
        if epoch is None:
            epoch = self.last_epoch
        
        # 检查warmup_scheduler是否已初始化（可能在初始化时被调用）
        if not hasattr(self, 'warmup_scheduler'):
            # 如果还没有初始化，直接返回（在super().__init__时会被调用）
            return
        
        # Warmup phase
        if not self.warmup_finished and epoch < self.num_warmup_epoch:
            self.warmup_scheduler.step()
            if epoch == self.num_warmup_epoch - 1:
                self.warmup_finished = True
        else:
            # Use plateau scheduler after warmup
            if metrics is not None:
                self.plateau_scheduler.step(metrics, epoch)
            else:
                self.plateau_scheduler.step()
        
        self.last_epoch = epoch
    
    def get_lr(self):
        if not self.warmup_finished:
            return self.warmup_scheduler.get_lr()
        else:
            return self.plateau_scheduler.get_last_lr()