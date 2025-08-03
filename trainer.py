from base.trainer import GenericVideoTrainer
from base.scheduler import GradualWarmupScheduler, MyWarmupScheduler

from torch import optim
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import time
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns


class Trainer(GenericVideoTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

        if self.best_epoch_info is None:
            self.best_epoch_info = {
                'model_weights': copy.deepcopy(self.model.state_dict()),
                'loss': 1e10,
                'accuracy': -1e10
            }

        for epoch in np.arange(start_epoch, self.max_epoch):
            if self.fit_finished:
                if self.verbose:
                    print("\nEarly Stop!\n")
                break

            improvement = False

            if epoch in self.milestone or (parameter_controller.get_current_lr() < self.min_learning_rate and epoch >= self.min_epoch and self.scheduler.relative_epoch > self.min_epoch):
                parameter_controller.release_param(self.model.spatial, epoch)
                if parameter_controller.early_stop:
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
                    checkpoint_controller.save_checkpoint()

            # Update scheduler
            self.scheduler.step(validate_accuracy)

            # Logging
            if self.verbose:
                time_epoch = time.time() - time_epoch_start
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {validate_loss:.4f}, "
                      f"Val Acc: {validate_accuracy:.4f}, Time: {time_epoch:.2f}s")

            # Early stopping
            if not improvement and epoch >= self.min_epoch:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.early_stopping:
                    if self.verbose:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs\n")
                    break
            else:
                self.early_stopping_counter = 0

    def train(self, dataloader_dict, epoch):
        """Training step"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []

        for batch_idx, batch in enumerate(dataloader_dict['train']):
            # Move data to device
            batch = self.move_batch_to_device(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Prepare input data
            input_data = {}
            for modality in self.modality:
                if modality in batch:
                    input_data[modality] = batch[modality]
            
            # Forward pass
            outputs = self.model(input_data)
            targets = batch['label'].squeeze()
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Collect predictions and targets
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            total_loss += loss.item()

        # Compute metrics
        avg_loss = total_loss / len(dataloader_dict['train'])
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')

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

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader_dict['val']):
                # Move data to device
                batch = self.move_batch_to_device(batch)
                
                # Prepare input data
                input_data = {}
                for modality in self.modality:
                    if modality in batch:
                        input_data[modality] = batch[modality]
                
                # Forward pass
                outputs = self.model(input_data)
                targets = batch['label'].squeeze()
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Collect predictions and targets
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                total_loss += loss.item()

        # Compute metrics
        avg_loss = total_loss / len(dataloader_dict['val'])
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')

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

        with torch.no_grad():
            for batch_idx, batch in enumerate(kwargs['dataloader_dict']['test']):
                # Move data to device
                batch = self.move_batch_to_device(batch)
                
                # Prepare input data
                input_data = {}
                for modality in self.modality:
                    if modality in batch:
                        input_data[modality] = batch[modality]
                
                # Forward pass
                outputs = self.model(input_data)
                targets = batch['label'].squeeze()
                
                # Compute loss
                loss = self.criterion(outputs, targets)
                
                # Collect predictions, probabilities and targets
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                total_loss += loss.item()

        # Compute metrics
        avg_loss = total_loss / len(kwargs['dataloader_dict']['test'])
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='weighted')

        # Create confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)

        record_dict = {
            'overall': {
                'loss': avg_loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }

        # Save results
        if self.save_plot:
            self.save_test_results(record_dict, kwargs.get('save_path', './results'))

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
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
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