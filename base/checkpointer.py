import os
import pickle
import torch
import pandas as pd
from datetime import datetime


class Checkpointer:
    """Checkpoint management class"""
    
    def __init__(self, checkpoint_filename, trainer, parameter_controller, resume=False):
        self.checkpoint_filename = checkpoint_filename
        self.trainer = trainer
        self.parameter_controller = parameter_controller
        self.resume = resume
        self.log_filename = None  # 将在 init_csv_logger 中设置
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_filename), exist_ok=True)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint_data = {
            'epoch': epoch,
            'trainer_state': {
                'model_state_dict': self.trainer.model.state_dict(),
                'optimizer_state_dict': self.trainer.optimizer.state_dict() if self.trainer.optimizer else None,
                'scheduler_state_dict': self.trainer.scheduler.state_dict() if self.trainer.scheduler else None,
                'best_epoch_info': self.trainer.best_epoch_info,
                'fit_finished': self.trainer.fit_finished,
                # 保存训练历史以便恢复
                'train_losses': getattr(self.trainer, 'train_losses', []),
                'validate_losses': getattr(self.trainer, 'validate_losses', []),
                'early_stopping_counter': getattr(self.trainer, 'early_stopping_counter', 0),
            },
            'parameter_controller_state': {
                'current_stage': getattr(self.parameter_controller, 'current_stage', 0),
                'release_count': getattr(self.parameter_controller, 'release_count', 0),
            },
            'timestamp': datetime.now().isoformat(),
            'is_best': is_best
        }
        
        with open(self.checkpoint_filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        if is_best:
            best_filename = self.checkpoint_filename.replace('.pkl', '_best.pkl')
            with open(best_filename, 'wb') as f:
                pickle.dump(checkpoint_data, f)
    
    def load_checkpoint(self, load_best=False):
        """
        Load checkpoint
        
        Args:
            load_best: If True, load the best model checkpoint instead of the latest one
        
        Returns:
            trainer, parameter_controller: Loaded trainer and parameter controller
        """
        # 确定要加载的checkpoint文件
        if load_best:
            checkpoint_file = self.checkpoint_filename.replace('.pkl', '_best.pkl')
            if not os.path.exists(checkpoint_file):
                print(f"⚠️  最佳模型文件 {checkpoint_file} 不存在，尝试加载最新模型...")
                checkpoint_file = self.checkpoint_filename
        else:
            checkpoint_file = self.checkpoint_filename
        
        if not os.path.exists(checkpoint_file):
            print(f"Checkpoint file {checkpoint_file} not found. Starting from scratch.")
            return self.trainer, self.parameter_controller
        
        print(f"正在加载checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Load trainer state
        trainer_state = checkpoint_data['trainer_state']
        
        # 加载模型权重
        self.trainer.model.load_state_dict(trainer_state['model_state_dict'])
        print("  ✓ 模型权重已加载")
        
        # 加载optimizer状态
        # 如果optimizer未初始化，先初始化它
        if self.trainer.optimizer is None:
            # 需要先初始化optimizer才能加载状态
            # 这里先初始化，状态会在fit方法中加载
            saved_epoch = checkpoint_data['epoch']
            if hasattr(self.trainer, 'init_optimizer_and_scheduler'):
                self.trainer.init_optimizer_and_scheduler(epoch=saved_epoch)
                print("  ✓ Optimizer和Scheduler已初始化（准备加载状态）")
        
        # 现在加载optimizer状态
        if self.trainer.optimizer is not None:
            if trainer_state['optimizer_state_dict'] is not None:
                try:
                    self.trainer.optimizer.load_state_dict(trainer_state['optimizer_state_dict'])
                    print("  ✓ Optimizer状态已加载")
                except Exception as e:
                    print(f"  ⚠️  加载Optimizer状态失败: {e}，将使用新的optimizer")
            else:
                print("  ⚠️  Checkpoint中没有optimizer状态")
            # Resume 时以当前 run 的 learning_rate 为准（命令行 -learning_rate），覆盖 checkpoint 里的 lr
            if hasattr(self.trainer, 'learning_rate'):
                for g in self.trainer.optimizer.param_groups:
                    g['lr'] = self.trainer.learning_rate
                    if 'initial_lr' in g:
                        g['initial_lr'] = self.trainer.learning_rate
                print(f"  ✓ 学习率已覆盖为当前设置: {self.trainer.learning_rate}")

        # 加载scheduler状态
        if self.trainer.scheduler is not None:
            if trainer_state['scheduler_state_dict'] is not None:
                try:
                    self.trainer.scheduler.load_state_dict(trainer_state['scheduler_state_dict'])
                    print("  ✓ Scheduler状态已加载")
                except Exception as e:
                    print(f"  ⚠️  加载Scheduler状态失败: {e}，将使用新的scheduler")
            else:
                print("  ⚠️  Checkpoint中没有scheduler状态")
            # 同步 scheduler 的 base_lrs，使后续 step() 使用当前 run 的 lr
            if hasattr(self.trainer, 'learning_rate') and hasattr(self.trainer.scheduler, 'base_lrs'):
                n = len(self.trainer.optimizer.param_groups)
                self.trainer.scheduler.base_lrs = [self.trainer.learning_rate] * n
        
        # 加载最佳模型信息
        self.trainer.best_epoch_info = trainer_state['best_epoch_info']
        print(f"  ✓ 最佳模型信息已加载 (epoch {trainer_state['best_epoch_info']['epoch']}, acc {trainer_state['best_epoch_info']['accuracy']:.4f})")
        
        # 加载训练完成状态
        self.trainer.fit_finished = trainer_state['fit_finished']
        if self.trainer.fit_finished:
            print("  ⚠️  检测到训练已完成，将不会继续训练")
        
        # 恢复训练历史（如果存在）
        if 'train_losses' in trainer_state:
            self.trainer.train_losses = trainer_state['train_losses']
            print(f"  ✓ 训练损失历史已恢复 ({len(self.trainer.train_losses)} 个epoch)")
        if 'validate_losses' in trainer_state:
            self.trainer.validate_losses = trainer_state['validate_losses']
            print(f"  ✓ 验证损失历史已恢复 ({len(self.trainer.validate_losses)} 个epoch)")
        if 'early_stopping_counter' in trainer_state:
            self.trainer.early_stopping_counter = trainer_state['early_stopping_counter']
        
        # Set start_epoch for resuming training
        if hasattr(self.trainer, 'start_epoch'):
            saved_epoch = checkpoint_data['epoch']
            self.trainer.start_epoch = saved_epoch + 1
            print(f"  ✓ 将从epoch {self.trainer.start_epoch} 继续训练（已训练到epoch {saved_epoch}）")
        
        # Load parameter controller state
        param_state = checkpoint_data['parameter_controller_state']
        if hasattr(self.parameter_controller, 'current_stage'):
            self.parameter_controller.current_stage = param_state['current_stage']
        if hasattr(self.parameter_controller, 'release_count'):
            self.parameter_controller.release_count = param_state['release_count']
        
        print(f"✓ Checkpoint加载完成 (epoch {checkpoint_data['epoch']})")
        return self.trainer, self.parameter_controller
    
    def init_csv_logger(self, args, config):
        """Initialize CSV logger for tracking results"""
        self.log_filename = self.checkpoint_filename.replace('.pkl', '_log.csv')
        
        # Create log headers
        headers = [
            'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 
            'train_f1', 'val_f1', 'learning_rate', 'timestamp'
        ]
        
        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(self.log_filename):
            df = pd.DataFrame(columns=headers)
            df.to_csv(self.log_filename, index=False)
        
        return self.log_filename
    
    def log_epoch(self, epoch, train_loss, val_loss, train_acc, val_acc, 
                  train_f1=None, val_f1=None, learning_rate=None):
        """Log epoch metrics to CSV file"""
        if self.log_filename is None:
            # 如果没有初始化日志文件，尝试创建
            self.log_filename = self.checkpoint_filename.replace('.pkl', '_log.csv')
            if not os.path.exists(self.log_filename):
                self.init_csv_logger(None, None)
        
        # 读取现有数据
        if os.path.exists(self.log_filename):
            df = pd.read_csv(self.log_filename)
        else:
            df = pd.DataFrame(columns=[
                'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 
                'train_f1', 'val_f1', 'learning_rate', 'timestamp'
            ])
        
        # 获取当前学习率
        if learning_rate is None and hasattr(self.trainer, 'optimizer') and self.trainer.optimizer is not None:
            learning_rate = self.trainer.optimizer.param_groups[0]['lr']
        
        # 添加新行
        new_row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_f1': train_f1 if train_f1 is not None else '',
            'val_f1': val_f1 if val_f1 is not None else '',
            'learning_rate': learning_rate if learning_rate is not None else '',
            'timestamp': datetime.now().isoformat()
        }
        
        # 追加到 DataFrame
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # 保存到 CSV
        df.to_csv(self.log_filename, index=False)