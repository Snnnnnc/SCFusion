from base.trainer import GenericVideoTrainer
from base.scheduler import GradualWarmupScheduler, MyWarmupScheduler

from torch import optim
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import pandas as pd
import time
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class Trainer(GenericVideoTrainer):
    def __init__(self, save_model=1, grad_clip=1.0, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize save_model attribute (default to True if not provided)
        self.save_model = bool(save_model) if save_model is not None else True
        
        # Initialize grad_clip attribute
        self.grad_clip = grad_clip

        self.best_epoch_info = {
            'model_weights': copy.deepcopy(self.model.state_dict()),
            'loss': 1e10,
            'accuracy': -1e10,
            'f1': -1e10,
            'epoch': 0,
            'metrics': {
                'train_loss': -1,
                'val_loss': -1,
                'train_acc': -1,
                'val_acc': -1,
            }
        }
        
        # Initialize training state
        self.start_epoch = 0
        self.train_losses = []
        self.validate_losses = []
        self.early_stopping_counter = 0

    def init_optimizer_and_scheduler(self, epoch=0):
        """Initialize optimizer and scheduler"""
        self.optimizer = optim.Adam(
            self.get_parameters(),
            lr=self.learning_rate,
            weight_decay=0.001
        )

        self.scheduler = MyWarmupScheduler(
            optimizer=self.optimizer,
            lr=self.learning_rate,
            min_lr=self.min_learning_rate,
            best=self.best_epoch_info['accuracy'],
            mode="max",
            patience=self.patience,
            factor=self.factor,
            num_warmup_epoch=self.min_epoch,
            init_epoch=epoch
        )

    def fit(self, dataloader_dict, checkpoint_controller, parameter_controller):
        """Main training loop"""
        if self.verbose:
            print("------")
            print("Starting training, on device:", self.device)

        self.time_fit_start = time.time()
        start_epoch = self.start_epoch

        # Initialize optimizer and scheduler if not already initialized
        if self.optimizer is None:
            self.init_optimizer_and_scheduler(epoch=start_epoch)

        if self.best_epoch_info is None:
            self.best_epoch_info = {
                'model_weights': copy.deepcopy(self.model.state_dict()),
                'loss': 1e10,
                'accuracy': -1e10
            }

        # 添加epoch进度条
        epoch_pbar = tqdm(range(start_epoch, self.max_epoch), desc='Training Progress', unit='epoch')
        for epoch in epoch_pbar:
            if self.fit_finished:
                if self.verbose:
                    print("\nEarly Stop!\n")
                epoch_pbar.close()
                break

            improvement = False

            if epoch in self.milestone or (hasattr(parameter_controller, 'get_current_lr') and parameter_controller.get_current_lr() < self.min_learning_rate and epoch >= self.min_epoch and hasattr(self.scheduler, 'relative_epoch') and self.scheduler.relative_epoch > self.min_epoch):
                # Only release parameters if model has spatial attribute (for PhysioFusionNet)
                if hasattr(self.model, 'spatial'):
                    parameter_controller.release_param(self.model.spatial, epoch)
                if hasattr(parameter_controller, 'early_stop') and parameter_controller.early_stop:
                    break

                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            time_epoch_start = time.time()

            if self.verbose:
                print("There are {} layers to update.".format(len(self.optimizer.param_groups[0]['params'])))

            # Training
            train_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            train_loss, train_record_dict = self.train(**train_kwargs)

            # Validation
            validate_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            validate_loss, validate_record_dict = self.validate(**validate_kwargs)

            if validate_loss < 0:
                raise ValueError('validate loss negative')

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)

            validate_accuracy = validate_record_dict['overall']['accuracy']

            self.scheduler.best = self.best_epoch_info['accuracy']

            # Check for improvement
            if validate_accuracy > self.best_epoch_info['accuracy']:
                improvement = True
                self.best_epoch_info['accuracy'] = validate_accuracy
                self.best_epoch_info['loss'] = validate_loss
                self.best_epoch_info['epoch'] = epoch
                self.best_epoch_info['model_weights'] = copy.deepcopy(self.model.state_dict())

                # Save best model
                if self.save_model:
                    checkpoint_controller.save_checkpoint(epoch, is_best=True)

            # Update scheduler
            self.scheduler.step(validate_accuracy)

            # 更新epoch进度条
            time_epoch = time.time() - time_epoch_start
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{validate_loss:.4f}',
                'val_acc': f'{validate_accuracy:.4f}',
                'time': f'{time_epoch:.1f}s'
            })
            
            # Logging
            if self.verbose:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {validate_loss:.4f}, "
                      f"Val Acc: {validate_accuracy:.4f}, Time: {time_epoch:.2f}s")
            
            # 写入 CSV 日志
            train_accuracy = train_record_dict.get('overall', {}).get('accuracy', 0.0)
            train_f1 = train_record_dict.get('overall', {}).get('f1', None)
            val_f1 = validate_record_dict.get('overall', {}).get('f1', None)
            current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer else None
            
            checkpoint_controller.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=validate_loss,
                train_acc=train_accuracy,
                val_acc=validate_accuracy,
                train_f1=train_f1,
                val_f1=val_f1,
                learning_rate=current_lr
            )

            # Early stopping
            if not improvement and epoch >= self.min_epoch:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping:
                    if self.verbose:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs\n")
                    epoch_pbar.close()
                    break
            else:
                self.early_stopping_counter = 0
        
        epoch_pbar.close()

    def train(self, dataloader_dict, epoch):
        """Training step"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []

        # 添加进度条
        pbar = tqdm(dataloader_dict['train'], desc=f'Epoch {epoch} [Train]', leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            batch = self.move_batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Prepare input data
            # 检查是否是ComfortClassificationModel、IMUClassificationModel或MixClassificationModel
            if hasattr(self.model, 'num_patches') and hasattr(self.model, 'encoding_dim'):
                # 检查是否是AllMixClassificationModel（有imu_encoder、eeg_encoder和ecg_encoder，且有final_cross_attention）
                if (hasattr(self.model, 'imu_encoder') and hasattr(self.model, 'eeg_encoder') and 
                    hasattr(self.model, 'ecg_encoder') and hasattr(self.model, 'final_cross_attention')):
                    # AllMixClassificationModel: 传imu_patches、eeg_patches和ecg_patches（三模态特征级融合）
                    imu_patches = batch.get('imu', None)
                    eeg_patches = batch.get('eeg', None)
                    ecg_patches = batch.get('ecg', None)
                    if imu_patches is not None and eeg_patches is not None and ecg_patches is not None:
                        # 确保形状正确: (batch, num_patches, channels, patch_length)
                        if len(imu_patches.shape) == 3:
                            imu_patches = imu_patches.unsqueeze(0)
                        if len(eeg_patches.shape) == 3:
                            eeg_patches = eeg_patches.unsqueeze(0)
                        if len(ecg_patches.shape) == 3:
                            ecg_patches = ecg_patches.unsqueeze(0)
                        outputs = self.model(imu_patches, eeg_patches, ecg_patches)
                    else:
                        raise ValueError("AllMixClassificationModel requires 'imu', 'eeg' and 'ecg' in batch")
                # 检查是否是MixClassificationModel（同时有imu_encoder、eeg_encoder和ecg_encoder，且有kalman_fusion）
                elif (hasattr(self.model, 'imu_encoder') and hasattr(self.model, 'eeg_encoder') and 
                      hasattr(self.model, 'ecg_encoder') and hasattr(self.model, 'kalman_fusion')):
                    # MixClassificationModel: 传imu_patches、eeg_patches和ecg_patches（决策级融合）
                    imu_patches = batch.get('imu', None)
                    eeg_patches = batch.get('eeg', None)
                    ecg_patches = batch.get('ecg', None)
                    if imu_patches is not None and eeg_patches is not None and ecg_patches is not None:
                        # 确保形状正确: (batch, num_patches, channels, patch_length)
                        if len(imu_patches.shape) == 3:
                            imu_patches = imu_patches.unsqueeze(0)
                        if len(eeg_patches.shape) == 3:
                            eeg_patches = eeg_patches.unsqueeze(0)
                        if len(ecg_patches.shape) == 3:
                            ecg_patches = ecg_patches.unsqueeze(0)
                        outputs = self.model(imu_patches, eeg_patches, ecg_patches)
                    else:
                        raise ValueError("MixClassificationModel requires 'imu', 'eeg' and 'ecg' in batch")
                # 检查是否是NewMixClassificationModel（有imu_encoder和ecg_encoder，但没有eeg_encoder，且有cross_attention）
                elif (hasattr(self.model, 'imu_encoder') and hasattr(self.model, 'ecg_encoder') and 
                      not hasattr(self.model, 'eeg_encoder') and hasattr(self.model, 'cross_attention')):
                    # NewMixClassificationModel: 传imu_patches和ecg_patches（特征级融合）
                    imu_patches = batch.get('imu', None)
                    ecg_patches = batch.get('ecg', None)
                    if imu_patches is not None and ecg_patches is not None:
                        # 确保形状正确: (batch, num_patches, channels, patch_length)
                        if len(imu_patches.shape) == 3:
                            imu_patches = imu_patches.unsqueeze(0)
                        if len(ecg_patches.shape) == 3:
                            ecg_patches = ecg_patches.unsqueeze(0)
                        outputs = self.model(imu_patches, ecg_patches)
                    else:
                        raise ValueError("NewMixClassificationModel requires 'imu' and 'ecg' in batch")
                # 检查是否是SimpleMixClassificationModel（有imu_encoder和ecg_encoder，但没有eeg_encoder，且有kalman_fusion）
                elif (hasattr(self.model, 'imu_encoder') and hasattr(self.model, 'ecg_encoder') and 
                      not hasattr(self.model, 'eeg_encoder') and hasattr(self.model, 'kalman_fusion')):
                    # SimpleMixClassificationModel: 传imu_patches和ecg_patches（决策级融合）
                    imu_patches = batch.get('imu', None)
                    ecg_patches = batch.get('ecg', None)
                    if imu_patches is not None and ecg_patches is not None:
                        # 确保形状正确: (batch, num_patches, channels, patch_length)
                        if len(imu_patches.shape) == 3:
                            imu_patches = imu_patches.unsqueeze(0)
                        if len(ecg_patches.shape) == 3:
                            ecg_patches = ecg_patches.unsqueeze(0)
                        outputs = self.model(imu_patches, ecg_patches)
                    else:
                        raise ValueError("SimpleMixClassificationModel requires 'imu' and 'ecg' in batch")
                # 检查是否是SingleModalPhysioModel（仅 EEG 或仅 ECG）
                elif hasattr(self.model, 'physio_encoder'):
                    key = getattr(self.model, 'modal', 'ecg')
                    patches = batch.get(key, None)
                    if patches is not None:
                        if len(patches.shape) == 3:
                            patches = patches.unsqueeze(0)
                        outputs = self.model(patches)
                    else:
                        raise ValueError(f"SingleModalPhysioModel requires '{key}' in batch")
                # 检查是否是IMUClassificationModel（只有imu_encoder）
                elif hasattr(self.model, 'imu_encoder'):
                    # IMUClassificationModel: 只传imu_patches
                    imu_patches = batch.get('imu', None)
                    if imu_patches is not None:
                        # 确保形状正确: (batch, num_patches, channels, patch_length)
                        # 如果输入是 (num_patches, channels, patch_length)，需要添加batch维度
                        if len(imu_patches.shape) == 3:
                            imu_patches = imu_patches.unsqueeze(0)
                        outputs = self.model(imu_patches)
                    else:
                        raise ValueError("IMUClassificationModel requires 'imu' in batch")
                # elif hasattr(self.model, 'imu_encoder'):
                #     ecg_patches = batch.get('ecg', None)
                #     if ecg_patches is not None:
                #         # 确保形状正确: (batch, num_patches, channels, patch_length)
                #         # 如果输入是 (num_patches, channels, patch_length)，需要添加batch维度
                #         if len(ecg_patches.shape) == 3:
                #             ecg_patches = ecg_patches.unsqueeze(0)
                #         outputs = self.model(ecg_patches)

                else:
                    # ComfortClassificationModel: 直接传eeg_patches和ecg_patches
                    eeg_patches = batch.get('eeg', None)
                    ecg_patches = batch.get('ecg', None)
                    if eeg_patches is not None and ecg_patches is not None:
                        # 确保形状正确: (batch, num_patches, channels, patch_length)
                        # 如果输入是 (num_patches, channels, patch_length)，需要添加batch维度
                        if len(eeg_patches.shape) == 3:
                            eeg_patches = eeg_patches.unsqueeze(0)
                        if len(ecg_patches.shape) == 3:
                            ecg_patches = ecg_patches.unsqueeze(0)
                        outputs = self.model(eeg_patches, ecg_patches)
                    else:
                        raise ValueError("ComfortClassificationModel requires both 'eeg' and 'ecg' in batch")
            else:
                # 其他模型：使用字典格式
                input_data = {}
                for modality in self.modality:
                    if modality in batch:
                        input_data[modality] = batch[modality]
                outputs = self.model(input_data)
            targets = batch['label'].squeeze()
            
            # Get sample weights (if available)
            sample_weights = None
            if 'weight' in batch:
                sample_weights = batch['weight'].squeeze()
            
            # 检查模型输出
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print(f"警告: 模型输出包含NaN或Inf")
                print(f"  outputs范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
                print(f"  outputs包含NaN: {torch.isnan(outputs).any()}")
                print(f"  outputs包含Inf: {torch.isinf(outputs).any()}")
                # 使用梯度裁剪或跳过这个batch
                continue
            
            # Compute loss (with sample weights if available)
            loss = self.criterion(outputs, targets, sample_weights=sample_weights)
            
            # 检查loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"警告: loss是NaN或Inf，跳过这个batch")
                print(f"  loss值: {loss}")
                print(f"  outputs范围: [{outputs.min():.4f}, {outputs.max():.4f}]")
                print(f"  targets: {targets}")
                if sample_weights is not None:
                    print(f"  sample_weights范围: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
                continue
            
            # Backward pass
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            if hasattr(self, 'grad_clip') and self.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                if grad_norm > self.grad_clip * 0.9:  # 如果梯度接近裁剪阈值，打印警告
                    if batch_idx % 10 == 0:  # 每10个batch打印一次，避免输出过多
                        print(f"  警告: 梯度范数较大: {grad_norm:.4f} (裁剪阈值: {self.grad_clip})")
            
            self.optimizer.step()
            
            # Collect predictions and targets
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        pbar.close()
        # Compute metrics
        avg_loss = total_loss / len(dataloader_dict['train'])
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='macro', zero_division=0)

        record_dict = {
            'overall': {
                'loss': avg_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }

        return avg_loss, record_dict

    def validate(self, dataloader_dict, epoch):
        """Validation step"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        # 添加进度条
        pbar = tqdm(dataloader_dict['val'], desc=f'Epoch {epoch} [Val]', leave=False)
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                batch = self.move_batch_to_device(batch)
                
                # Prepare input data
                # 检查是否是ComfortClassificationModel、IMUClassificationModel或MixClassificationModel
                if hasattr(self.model, 'num_patches') and hasattr(self.model, 'encoding_dim'):
                    # 检查是否是MixClassificationModel（同时有imu_encoder、eeg_encoder和ecg_encoder）
                    if hasattr(self.model, 'imu_encoder') and hasattr(self.model, 'eeg_encoder') and hasattr(self.model, 'ecg_encoder'):
                        # MixClassificationModel: 传imu_patches、eeg_patches和ecg_patches
                        imu_patches = batch.get('imu', None)
                        eeg_patches = batch.get('eeg', None)
                        ecg_patches = batch.get('ecg', None)
                        if imu_patches is not None and eeg_patches is not None and ecg_patches is not None:
                            if len(imu_patches.shape) == 3:
                                imu_patches = imu_patches.unsqueeze(0)
                            if len(eeg_patches.shape) == 3:
                                eeg_patches = eeg_patches.unsqueeze(0)
                            if len(ecg_patches.shape) == 3:
                                ecg_patches = ecg_patches.unsqueeze(0)
                            outputs = self.model(imu_patches, eeg_patches, ecg_patches)
                        else:
                            raise ValueError("MixClassificationModel requires 'imu', 'eeg' and 'ecg' in batch")
                    # 检查是否是SimpleMixClassificationModel（有imu_encoder和ecg_encoder，但没有eeg_encoder）
                    elif hasattr(self.model, 'imu_encoder') and hasattr(self.model, 'ecg_encoder') and not hasattr(self.model, 'eeg_encoder'):
                        # SimpleMixClassificationModel: 传imu_patches和ecg_patches（不传eeg_patches）
                        imu_patches = batch.get('imu', None)
                        ecg_patches = batch.get('ecg', None)
                        if imu_patches is not None and ecg_patches is not None:
                            if len(imu_patches.shape) == 3:
                                imu_patches = imu_patches.unsqueeze(0)
                            if len(ecg_patches.shape) == 3:
                                ecg_patches = ecg_patches.unsqueeze(0)
                            outputs = self.model(imu_patches, ecg_patches)
                        else:
                            raise ValueError("SimpleMixClassificationModel requires 'imu' and 'ecg' in batch")
                    # 检查是否是SingleModalPhysioModel（仅 EEG 或仅 ECG）
                    elif hasattr(self.model, 'physio_encoder'):
                        key = getattr(self.model, 'modal', 'ecg')
                        patches = batch.get(key, None)
                        if patches is not None:
                            if len(patches.shape) == 3:
                                patches = patches.unsqueeze(0)
                            outputs = self.model(patches)
                        else:
                            raise ValueError(f"SingleModalPhysioModel requires '{key}' in batch")
                    # 检查是否是IMUClassificationModel（只有imu_encoder）
                    elif hasattr(self.model, 'imu_encoder'):
                        # IMUClassificationModel: 只传imu_patches
                        imu_patches = batch.get('imu', None)
                        if imu_patches is not None:
                            # 确保形状正确: (batch, num_patches, channels, patch_length)
                            if len(imu_patches.shape) == 3:
                                imu_patches = imu_patches.unsqueeze(0)
                            outputs = self.model(imu_patches)
                        else:
                            raise ValueError("IMUClassificationModel requires 'imu' in batch")
                    else:
                        # ComfortClassificationModel: 直接传eeg_patches和ecg_patches
                        eeg_patches = batch.get('eeg', None)
                        ecg_patches = batch.get('ecg', None)
                        if eeg_patches is not None and ecg_patches is not None:
                            # 确保形状正确: (batch, num_patches, channels, patch_length)
                            if len(eeg_patches.shape) == 3:
                                eeg_patches = eeg_patches.unsqueeze(0)
                            if len(ecg_patches.shape) == 3:
                                ecg_patches = ecg_patches.unsqueeze(0)
                            outputs = self.model(eeg_patches, ecg_patches)
                        else:
                            raise ValueError("ComfortClassificationModel requires both 'eeg' and 'ecg' in batch")
                else:
                    # 其他模型：使用字典格式
                    input_data = {}
                    for modality in self.modality:
                        if modality in batch:
                            input_data[modality] = batch[modality]
                    outputs = self.model(input_data)
                targets = batch['label'].squeeze()
                
                # Get sample weights (if available)
                sample_weights = None
                if 'weight' in batch:
                    sample_weights = batch['weight'].squeeze()
                
                # Compute loss (with sample weights if available)
                loss = self.criterion(outputs, targets, sample_weights=sample_weights)
                
                # Collect predictions and targets
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        pbar.close()
        
        # 检查验证集是否为空
        if len(all_targets) == 0 or len(all_predictions) == 0:
            print("  ⚠️  警告: 验证集为空，无法计算指标")
            return 0.0, {
                'overall': {
                    'loss': 0.0,
                    'accuracy': float('nan'),
                    'precision': float('nan'),
                    'recall': float('nan'),
                    'f1': float('nan')
                }
            }
        
        # Compute metrics
        avg_loss = total_loss / len(dataloader_dict['val']) if len(dataloader_dict['val']) > 0 else 0.0
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='macro', zero_division=0)

        record_dict = {
            'overall': {
                'loss': avg_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }

        return avg_loss, record_dict

    def test(self, checkpoint_controller, predict_only=0, **kwargs):
        """Test step"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_subject_ids = []

        # 创建测试进度条
        test_loader = kwargs['dataloader_dict']['test']
        pbar = tqdm(test_loader, desc='Testing', unit='batch')

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                batch = self.move_batch_to_device(batch)
                
                # Prepare input data
                # 检查是否是ComfortClassificationModel、IMUClassificationModel或MixClassificationModel
                if hasattr(self.model, 'num_patches') and hasattr(self.model, 'encoding_dim'):
                    # 检查是否是MixClassificationModel（同时有imu_encoder、eeg_encoder和ecg_encoder）
                    if hasattr(self.model, 'imu_encoder') and hasattr(self.model, 'eeg_encoder') and hasattr(self.model, 'ecg_encoder'):
                        # MixClassificationModel: 传imu_patches、eeg_patches和ecg_patches
                        imu_patches = batch.get('imu', None)
                        eeg_patches = batch.get('eeg', None)
                        ecg_patches = batch.get('ecg', None)
                        if imu_patches is not None and eeg_patches is not None and ecg_patches is not None:
                            if len(imu_patches.shape) == 3:
                                imu_patches = imu_patches.unsqueeze(0)
                            if len(eeg_patches.shape) == 3:
                                eeg_patches = eeg_patches.unsqueeze(0)
                            if len(ecg_patches.shape) == 3:
                                ecg_patches = ecg_patches.unsqueeze(0)
                            outputs = self.model(imu_patches, eeg_patches, ecg_patches)
                        else:
                            raise ValueError("MixClassificationModel requires 'imu', 'eeg' and 'ecg' in batch")
                    # 检查是否是SimpleMixClassificationModel（有imu_encoder和ecg_encoder，但没有eeg_encoder）
                    elif hasattr(self.model, 'imu_encoder') and hasattr(self.model, 'ecg_encoder') and not hasattr(self.model, 'eeg_encoder'):
                        # SimpleMixClassificationModel: 传imu_patches和ecg_patches（不传eeg_patches）
                        imu_patches = batch.get('imu', None)
                        ecg_patches = batch.get('ecg', None)
                        if imu_patches is not None and ecg_patches is not None:
                            if len(imu_patches.shape) == 3:
                                imu_patches = imu_patches.unsqueeze(0)
                            if len(ecg_patches.shape) == 3:
                                ecg_patches = ecg_patches.unsqueeze(0)
                            outputs = self.model(imu_patches, ecg_patches)
                        else:
                            raise ValueError("SimpleMixClassificationModel requires 'imu' and 'ecg' in batch")
                    # 检查是否是SingleModalPhysioModel（仅 EEG 或仅 ECG）
                    elif hasattr(self.model, 'physio_encoder'):
                        key = getattr(self.model, 'modal', 'ecg')
                        patches = batch.get(key, None)
                        if patches is not None:
                            if len(patches.shape) == 3:
                                patches = patches.unsqueeze(0)
                            outputs = self.model(patches)
                        else:
                            raise ValueError(f"SingleModalPhysioModel requires '{key}' in batch")
                    # 检查是否是IMUClassificationModel（只有imu_encoder）
                    elif hasattr(self.model, 'imu_encoder'):
                        # IMUClassificationModel: 只传imu_patches
                        imu_patches = batch.get('imu', None)
                        if imu_patches is not None:
                            # 确保形状正确: (batch, num_patches, channels, patch_length)
                            if len(imu_patches.shape) == 3:
                                imu_patches = imu_patches.unsqueeze(0)
                            outputs = self.model(imu_patches)
                        else:
                            raise ValueError("IMUClassificationModel requires 'imu' in batch")
                    else:
                        # ComfortClassificationModel: 直接传eeg_patches和ecg_patches
                        eeg_patches = batch.get('eeg', None)
                        ecg_patches = batch.get('ecg', None)
                        if eeg_patches is not None and ecg_patches is not None:
                            # 确保形状正确: (batch, num_patches, channels, patch_length)
                            if len(eeg_patches.shape) == 3:
                                eeg_patches = eeg_patches.unsqueeze(0)
                            if len(ecg_patches.shape) == 3:
                                ecg_patches = ecg_patches.unsqueeze(0)
                            outputs = self.model(eeg_patches, ecg_patches)
                        else:
                            raise ValueError("ComfortClassificationModel requires both 'eeg' and 'ecg' in batch")
                else:
                    # 其他模型：使用字典格式
                    input_data = {}
                    for modality in self.modality:
                        if modality in batch:
                            input_data[modality] = batch[modality]
                    outputs = self.model(input_data)
                targets = batch['label'].squeeze()
                
                # Get sample weights (if available)
                sample_weights = None
                if 'weight' in batch:
                    sample_weights = batch['weight'].squeeze()
                
                # Compute loss (with sample weights if available)
                loss = self.criterion(outputs, targets, sample_weights=sample_weights)
                
                # Collect predictions, probabilities and targets
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Collect subject_ids (if available)
                if 'subject_id' in batch:
                    subject_ids = batch['subject_id'].cpu().numpy()
                    # 确保是一维数组（可能是 (batch_size, 1) 的形状）
                    subject_ids = subject_ids.flatten() if subject_ids.ndim > 1 else subject_ids
                    all_subject_ids.extend(subject_ids.tolist() if isinstance(subject_ids, np.ndarray) else subject_ids)
                
                total_loss += loss.item()
                
                # 更新进度条
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        pbar.close()
        
        # 检查是否有测试数据
        if len(all_targets) == 0 or len(all_predictions) == 0:
            print("  ⚠️  警告: 测试集为空，无法计算指标")
            record_dict = {
                'overall': {
                    'loss': 0.0,
                    'accuracy': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                },
                'confusion_matrix': np.array([]),
                'predictions': np.array([]),
                'targets': np.array([]),
                'probabilities': np.array([]),
                'subject_ids': np.array([])
            }
            if self.save_plot:
                # self.save_test_results(record_dict, kwargs.get('save_path', './results'))
                self.save_test_results(record_dict, self.save_path)
            return 0.0, record_dict
        
        # Compute metrics
        avg_loss = total_loss / len(kwargs['dataloader_dict']['test']) if len(kwargs['dataloader_dict']['test']) > 0 else 0.0
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='macro', zero_division=0)

        # Create confusion matrix（固定 0-4 五类，保证矩阵为 5x5）
        try:
            num_classes = 5
            cm = confusion_matrix(all_targets, all_predictions, labels=list(range(num_classes)))
        except Exception as e:
            print(f"  ⚠️  警告: 计算混淆矩阵时出错: {e}")
            print(f"  all_targets长度: {len(all_targets)}, all_predictions长度: {len(all_predictions)}")
            cm = np.array([])

        record_dict = {
            'overall': {
                'loss': avg_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'confusion_matrix': cm,
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probabilities),
            'subject_ids': np.array(all_subject_ids) if len(all_subject_ids) > 0 else np.array([])
        }

        # Save results
        if self.save_plot:
            # self.save_test_results(record_dict, kwargs.get('save_path', './results'))
            self.save_test_results(record_dict, self.save_path)

        return avg_loss, record_dict

    def move_batch_to_device(self, batch):
        """Move batch data to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def save_test_results(self, record_dict, save_path):
        """Save test results and plots"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save confusion matrix
        cm = record_dict['confusion_matrix']
        
        # 检查混淆矩阵是否为空
        if cm.size == 0 or (isinstance(cm, np.ndarray) and cm.shape[0] == 0):
            print("  ⚠️  警告: 混淆矩阵为空，跳过保存混淆矩阵图")
        else:
            try:
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                confusion_matrix_path = os.path.join(save_path, 'confusion_matrix.png')
                plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  ✓ 混淆矩阵图已保存到 {confusion_matrix_path}")
            except Exception as e:
                print(f"  ⚠️  警告: 保存混淆矩阵图时出错: {e}")
                print(f"  混淆矩阵形状: {cm.shape if hasattr(cm, 'shape') else 'N/A'}")
                print(f"  混淆矩阵内容: {cm}")
                # 即使保存图片失败，也保存混淆矩阵数据
                try:
                    np.save(os.path.join(save_path, 'confusion_matrix.npy'), cm)
                    print(f"  ✓ 已保存混淆矩阵数据到 confusion_matrix.npy")
                except Exception as e2:
                    print(f"  ⚠️  保存混淆矩阵数据也失败: {e2}")
        
        # Save metrics
        metrics = record_dict['overall']
        with open(os.path.join(save_path, 'test_metrics.txt'), 'w') as f:
            f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Test Precision: {metrics['precision']:.4f}\n")
            f.write(f"Test Recall: {metrics['recall']:.4f}\n")
            f.write(f"Test F1: {metrics['f1']:.4f}\n")
            f.write(f"Test Loss: {metrics['loss']:.4f}\n")
        
        # Save predictions
        np.save(os.path.join(save_path, 'predictions.npy'), record_dict['predictions'])
        np.save(os.path.join(save_path, 'targets.npy'), record_dict['targets'])
        np.save(os.path.join(save_path, 'probabilities.npy'), record_dict['probabilities']) 
        
        # Save subject_ids (if available)
        if 'subject_ids' in record_dict and len(record_dict['subject_ids']) > 0:
            subject_ids = record_dict['subject_ids']
            # 确保 subject_ids 是一维数组
            subject_ids = np.asarray(subject_ids).flatten()
            np.save(os.path.join(save_path, 'subject_ids.npy'), subject_ids)
            
            # 按subject统计准确率
            predictions = record_dict['predictions']
            targets = record_dict['targets']
            # 确保 predictions 和 targets 是一维数组
            predictions = np.asarray(predictions).flatten()
            targets = np.asarray(targets).flatten()
            
            # 确保长度一致
            min_len = min(len(subject_ids), len(predictions), len(targets))
            subject_ids = subject_ids[:min_len]
            predictions = predictions[:min_len]
            targets = targets[:min_len]
            
            # 计算每个subject的准确率
            unique_subjects = np.unique(subject_ids)
            subject_metrics = {}
            
            for subject_id in unique_subjects:
                mask = subject_ids == subject_id
                subject_predictions = predictions[mask]
                subject_targets = targets[mask]
                
                if len(subject_predictions) > 0:
                    subject_acc = accuracy_score(subject_targets, subject_predictions)
                    subject_precision, subject_recall, subject_f1, _ = precision_recall_fscore_support(
                        subject_targets, subject_predictions, average='macro', zero_division=0
                    )
                    subject_metrics[int(subject_id)] = {
                        'accuracy': subject_acc,
                        'precision': subject_precision,
                        'recall': subject_recall,
                        'f1': subject_f1,
                        'num_samples': len(subject_predictions)
                    }
            
            # 保存按subject的统计结果
            subject_metrics_file = os.path.join(save_path, 'subject_metrics.txt')
            with open(subject_metrics_file, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("按被试统计的测试结果\n")
                f.write("=" * 60 + "\n\n")
                
                # 按subject_id排序
                sorted_subjects = sorted(subject_metrics.keys())
                
                for subject_id in sorted_subjects:
                    metrics = subject_metrics[subject_id]
                    f.write(f"Subject {subject_id}:\n")
                    f.write(f"  样本数: {metrics['num_samples']}\n")
                    f.write(f"  准确率: {metrics['accuracy']:.4f}\n")
                    f.write(f"  精确率: {metrics['precision']:.4f}\n")
                    f.write(f"  召回率: {metrics['recall']:.4f}\n")
                    f.write(f"  F1分数: {metrics['f1']:.4f}\n")
                    f.write("\n")
                
                # 计算平均指标
                avg_acc = np.mean([subject_metrics[sid]['accuracy'] for sid in sorted_subjects])
                avg_precision = np.mean([subject_metrics[sid]['precision'] for sid in sorted_subjects])
                avg_recall = np.mean([subject_metrics[sid]['recall'] for sid in sorted_subjects])
                avg_f1 = np.mean([subject_metrics[sid]['f1'] for sid in sorted_subjects])
                
                f.write("=" * 60 + "\n")
                f.write("平均指标（按被试平均）:\n")
                f.write("=" * 60 + "\n")
                f.write(f"平均准确率: {avg_acc:.4f}\n")
                f.write(f"平均精确率: {avg_precision:.4f}\n")
                f.write(f"平均召回率: {avg_recall:.4f}\n")
                f.write(f"平均F1分数: {avg_f1:.4f}\n")
            
            print(f"  ✓ 已保存按被试统计结果到 {subject_metrics_file}")
            
            # 同时保存为CSV格式，方便后续分析
            subject_metrics_csv = os.path.join(save_path, 'subject_metrics.csv')
            df_subject_metrics = pd.DataFrame([
                {
                    'subject_id': sid,
                    'accuracy': subject_metrics[sid]['accuracy'],
                    'precision': subject_metrics[sid]['precision'],
                    'recall': subject_metrics[sid]['recall'],
                    'f1': subject_metrics[sid]['f1'],
                    'num_samples': subject_metrics[sid]['num_samples']
                }
                for sid in sorted_subjects
            ])
            df_subject_metrics.to_csv(subject_metrics_csv, index=False, encoding='utf-8')
            print(f"  ✓ 已保存按被试统计结果（CSV格式）到 {subject_metrics_csv}")
        else:
            print("  ⚠️  未找到subject_id信息，跳过按被试统计") 