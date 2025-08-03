from base.experiment import GenericExperiment
from base.utils import load_pickle
from base.loss_function import ClassificationLoss, CombinedLoss
from trainer import Trainer

from dataset import DataArranger, PhysiologicalDataset
from base.checkpointer import Checkpointer
from models.model import PhysioFusionNet, CAN

from base.parameter_control import ResnetParamControl

import os
import torch
from torch.utils.data import DataLoader
import numpy as np


class Experiment(GenericExperiment):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.release_count = args.release_count
        self.gradual_release = args.gradual_release
        self.milestone = args.milestone
        self.min_num_epochs = args.min_num_epochs
        self.num_epochs = args.num_epochs
        self.early_stopping = args.early_stopping
        self.load_best_at_each_epoch = args.load_best_at_each_epoch

        self.num_heads = args.num_heads
        self.modal_dim = args.modal_dim
        self.tcn_kernel_size = args.tcn_kernel_size
        self.num_classes = args.num_classes

    def prepare(self):
        """Prepare experiment"""
        self.config = self.get_config()

        self.feature_dimension = self.get_feature_dimension(self.config)
        self.multiplier = self.get_multiplier(self.config)
        self.time_delay = self.get_time_delay(self.config)

        self.get_modality()
        self.continuous_label_dim = self.get_selected_continuous_label_dim()

        # Load dataset info
        dataset_info_path = os.path.join(self.dataset_path, "dataset_info.pkl")
        if os.path.exists(dataset_info_path):
            self.dataset_info = load_pickle(dataset_info_path)
        else:
            self.dataset_info = self.create_dataset_info()
        
        self.data_arranger = self.init_data_arranger()
        
        if self.calc_mean_std:
            self.calc_mean_std_fn()
        
        # Load mean/std info
        mean_std_path = os.path.join(self.dataset_path, "mean_std_info.pkl")
        if os.path.exists(mean_std_path):
            self.mean_std_dict = load_pickle(mean_std_path)
        else:
            self.mean_std_dict = self.create_mean_std_info()

    def create_dataset_info(self):
        """Create dataset info if not exists"""
        # This would be populated with actual dataset information
        dataset_info = {
            'num_subjects': 100,  # Example
            'eeg_channels': self.args.eeg_channels,
            'ecg_channels': self.args.ecg_channels,
            'eeg_sampling_rate': self.args.eeg_sampling_rate,
            'ecg_sampling_rate': self.args.ecg_sampling_rate,
            'signal_length': 30000,  # 30 seconds at 1000Hz
        }
        
        # Save dataset info
        os.makedirs(self.dataset_path, exist_ok=True)
        import pickle
        with open(os.path.join(self.dataset_path, "dataset_info.pkl"), 'wb') as f:
            pickle.dump(dataset_info, f)
        
        return dataset_info

    def create_mean_std_info(self):
        """Create mean/std info if not exists"""
        # This would be calculated from actual data
        mean_std_dict = {}
        for fold in range(1, self.args.num_folds + 1):
            mean_std_dict[fold] = {
                'train': {'eeg': {'mean': 0.0, 'std': 1.0}, 'ecg': {'mean': 0.0, 'std': 1.0}},
                'val': {'eeg': {'mean': 0.0, 'std': 1.0}, 'ecg': {'mean': 0.0, 'std': 1.0}},
                'test': {'eeg': {'mean': 0.0, 'std': 1.0}, 'ecg': {'mean': 0.0, 'std': 1.0}}
            }
        
        # Save mean/std info
        os.makedirs(self.dataset_path, exist_ok=True)
        import pickle
        with open(os.path.join(self.dataset_path, "mean_std_info.pkl"), 'wb') as f:
            pickle.dump(mean_std_dict, f)
        
        return mean_std_dict

    def init_data_arranger(self):
        """Initialize data arranger"""
        arranger = DataArranger(self.dataset_info, self.dataset_path, self.debug)
        return arranger

    def run(self):
        """Run the experiment"""
        # Initialize loss function
        if self.args.class_weights is not None:
            class_weights = torch.tensor([float(w) for w in self.args.class_weights.split(',')])
        else:
            class_weights = None
        
        criterion = ClassificationLoss(
            loss_type='cross_entropy',
            num_classes=self.num_classes,
            class_weights=class_weights
        )

        for fold in iter(self.folds_to_run):
            # Create save path
            save_path = os.path.join(
                self.save_path,
                f"{self.experiment_name}_{self.model_name}_{self.stamp}_fold{fold}_seed{self.seed}"
            )
            os.makedirs(save_path, exist_ok=True)

            checkpoint_filename = os.path.join(save_path, "checkpoint.pkl")

            # Initialize model
            model = self.init_model()

            # Initialize dataloaders
            dataloaders = self.init_dataloader(fold)

            # Initialize trainer
            trainer_kwargs = {
                'device': self.device,
                'model_name': self.model_name,
                'models': model,
                'save_path': save_path,
                'fold': fold,
                'min_epoch': self.min_num_epochs,
                'max_epoch': self.num_epochs,
                'early_stopping': self.early_stopping,
                'scheduler': self.scheduler,
                'learning_rate': self.learning_rate,
                'min_learning_rate': self.min_learning_rate,
                'patience': self.patience,
                'batch_size': self.batch_size,
                'criterion': criterion,
                'factor': self.factor,
                'verbose': True,
                'milestone': self.milestone,
                'metrics': self.config['metrics'],
                'load_best_at_each_epoch': self.load_best_at_each_epoch,
                'save_plot': self.config['save_plot']
            }

            trainer = Trainer(**trainer_kwargs)

            # Initialize parameter controller
            parameter_controller = ResnetParamControl(
                trainer,
                gradual_release=self.gradual_release,
                release_count=self.release_count,
                backbone_mode=["eeg", "ecg"]
            )

            # Initialize checkpoint controller
            checkpoint_controller = Checkpointer(
                checkpoint_filename,
                trainer,
                parameter_controller,
                resume=self.resume
            )

            if self.resume:
                trainer, parameter_controller = checkpoint_controller.load_checkpoint()
            else:
                checkpoint_controller.init_csv_logger(self.args, self.config)

            # Train model
            if not trainer.fit_finished:
                trainer.fit(
                    dataloaders,
                    parameter_controller=parameter_controller,
                    checkpoint_controller=checkpoint_controller
                )

            # Test model
            test_kwargs = {
                'dataloader_dict': dataloaders,
                'epoch': None,
                'partition': 'test'
            }
            trainer.test(checkpoint_controller, predict_only=1, **test_kwargs)

    def init_dataset(self, data, continuous_label_dim, mode, fold):
        """Initialize dataset"""
        dataset = PhysiologicalDataset(
            data=data,
            modality=self.modality,
            window_length=self.args.window_length,
            hop_length=self.args.hop_length,
            normalize=self.args.normalize_data,
            apply_filter=self.args.apply_filter,
            eeg_sampling_rate=self.args.eeg_sampling_rate,
            ecg_sampling_rate=self.args.ecg_sampling_rate
        )
        return dataset

    def init_model(self):
        """Initialize model"""
        self.init_randomness()
        modality = [modal for modal in self.modality if "continuous_label" not in modal]

        if self.model_name == "PhysioFusionNet":
            model = PhysioFusionNet(
                backbone_settings=self.config['backbone_settings'],
                modality=modality,
                example_length=self.args.window_length,
                kernel_size=self.tcn_kernel_size,
                tcn_channel=self.config['tcn']['channels'],
                modal_dim=self.modal_dim,
                num_heads=self.num_heads,
                num_classes=self.num_classes,
                root_dir=self.load_path,
                device=self.device
            )
            model.init()
        elif self.model_name == "CAN":
            model = CAN(
                modalities=modality,
                tcn_settings=self.config['tcn_settings'],
                backbone_settings=self.config['backbone_settings'],
                output_dim=self.num_classes,
                root_dir=self.load_path,
                device=self.device
            )

        return model

    def get_modality(self):
        """Get modality information"""
        pass

    def get_config(self):
        """Get configuration"""
        from configs import config
        return config

    def get_selected_continuous_label_dim(self):
        """Get selected continuous label dimensions"""
        # For classification, we use all classes
        return list(range(self.num_classes))

    def init_dataloader(self, fold):
        """Initialize dataloaders for a specific fold"""
        # Load data
        data = self.data_arranger.load_data()
        
        # Create data splits (simplified version)
        # In practice, you would load pre-computed splits
        n_samples = len(data.get('labels', [0]))
        indices = np.arange(n_samples)
        
        # Simple split for demonstration
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create datasets
        train_data = self.subset_data(data, train_indices)
        val_data = self.subset_data(data, val_indices)
        test_data = self.subset_data(data, test_indices)
        
        # Initialize datasets
        train_dataset = self.init_dataset(train_data, self.continuous_label_dim, 'train', fold)
        val_dataset = self.init_dataset(val_data, self.continuous_label_dim, 'val', fold)
        test_dataset = self.init_dataset(test_data, self.continuous_label_dim, 'test', fold)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

    def subset_data(self, data, indices):
        """Create subset of data based on indices"""
        subset_data = {}
        
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                subset_data[key] = [value[i] for i in indices]
            elif isinstance(value, np.ndarray):
                subset_data[key] = value[indices]
            else:
                subset_data[key] = value
        
        return subset_data 