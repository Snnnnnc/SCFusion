from base.experiment import GenericExperiment
from base.utils import load_pickle
from base.loss_function import ClassificationLoss, CombinedLoss
from trainer import Trainer

from dataset import DataArranger, PhysiologicalDataset, IMUDataset, MixDataset, RawIMUDataset
from base.checkpointer import Checkpointer
from models.model import PhysioFusionNet, CAN
from models.comfort_model import ComfortClassificationModel, SingleModalPhysioModel, IMUClassificationModel, MixClassificationModel, SimpleMixClassificationModel, NewMixClassificationModel, AllMixClassificationModel

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
        self.mode = getattr(args, 'mode', 'physio')  # 'physio', 'imu' 或 'mix'

    def prepare(self):
        """Prepare experiment"""
        from tqdm import tqdm
        
        print("=" * 60)
        print("准备实验环境...")
        print("=" * 60)
        
        with tqdm(total=6, desc='Preparing', unit='step') as pbar:
            pbar.set_description('加载配置')
            self.config = self.get_config()
            self.feature_dimension = self.get_feature_dimension(self.config)
            self.multiplier = self.get_multiplier(self.config)
            self.time_delay = self.get_time_delay(self.config)
            pbar.update(1)
            
            pbar.set_description('设置模态')
            self.get_modality()
            print(f"  模式: {self.mode}")
            print(f"  模态: {self.modality}")
            self.continuous_label_dim = self.get_selected_continuous_label_dim()
            pbar.update(1)
            
            pbar.set_description('加载数据集信息')
            dataset_info_path = os.path.join(self.dataset_path, "dataset_info.pkl")
            if os.path.exists(dataset_info_path):
                self.dataset_info = load_pickle(dataset_info_path)
            else:
                self.dataset_info = self.create_dataset_info()
            pbar.update(1)
            
            pbar.set_description('初始化数据加载器')
            self.data_arranger = self.init_data_arranger()
            pbar.update(1)
            
            if self.calc_mean_std:
                pbar.set_description('计算均值和标准差')
                self.calc_mean_std_fn()
            else:
                pbar.set_description('跳过均值/标准差计算')
            pbar.update(1)
            
            pbar.set_description('加载均值/标准差信息')
            mean_std_path = os.path.join(self.dataset_path, "mean_std_info.pkl")
            if os.path.exists(mean_std_path):
                self.mean_std_dict = load_pickle(mean_std_path)
            else:
                self.mean_std_dict = self.create_mean_std_info()
            pbar.update(1)
        
        print("✓ 实验环境准备完成！\n")

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
        from tqdm import tqdm
        
        print("=" * 60)
        print("初始化损失函数...")
        
        # 计算或使用类别权重
        class_weights = self.compute_class_weights()
        
        criterion = ClassificationLoss(
            loss_type='cross_entropy',
            num_classes=self.num_classes,
            class_weights=class_weights
        )
        
        if class_weights is not None:
            print(f"✓ 损失函数初始化完成（使用类别权重）")
            print(f"  类别权重: {class_weights.tolist()}")
        else:
            print("✓ 损失函数初始化完成（未使用类别权重）")
        print()

        # 添加fold进度条
        folds_list = list(self.folds_to_run) if isinstance(self.folds_to_run, range) else self.folds_to_run
        fold_pbar = tqdm(folds_list, desc='Cross-Validation Folds', unit='fold')
        for fold in fold_pbar:
            fold_pbar.set_description(f'Fold {fold}/{len(folds_list)}')
            # Create save path (根据mode调整路径)
            mode_suffix = f"_{self.mode}" if self.mode != 'physio' else "_physio"
            save_path = os.path.join(
                self.save_path,
                f"{self.experiment_name}_{self.model_name}_{self.stamp}{mode_suffix}_fold{fold}_seed{self.seed}"
            )
            os.makedirs(save_path, exist_ok=True)

            checkpoint_filename = os.path.join(save_path, "checkpoint.pkl")

            # Initialize model
            print(f"\n初始化模型 (mode={self.mode}, model_name={self.model_name})...")
            model = self.init_model()
            print(f"✓ 模型初始化完成")

            # Initialize dataloaders
            print(f"\n初始化数据加载器 (mode={self.mode})...")
            dataloaders = self.init_dataloader(fold)
            print(f"✓ 数据加载器初始化完成")

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
                'save_plot': self.config['save_plot'],
                'save_model': getattr(self.args, 'save_model', 1),  # 传递 save_model 参数
                'grad_clip': getattr(self.args, 'grad_clip', 1.0),  # 传递 grad_clip 参数
            }

            trainer = Trainer(**trainer_kwargs)

            # Initialize parameter controller
            if self.mode == 'imu':
                backbone_mode = ["imu"]
            elif self.mode == 'mix':
                backbone_mode = ["imu", "eeg", "ecg"]
            else:
                backbone_mode = ["eeg", "ecg"]
            
            parameter_controller = ResnetParamControl(
                trainer,
                gradual_release=self.gradual_release,
                release_count=self.release_count,
                backbone_mode=backbone_mode
            )

            # Initialize checkpoint controller
            checkpoint_controller = Checkpointer(
                checkpoint_filename,
                trainer,
                parameter_controller,
                resume=self.resume
            )

            if self.resume:
                # 支持选择加载最佳模型或最新模型
                load_best = getattr(self.args, 'resume_from_best', False)
                trainer, parameter_controller = checkpoint_controller.load_checkpoint(load_best=load_best)
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
            # 在测试前加载最佳模型
            print(f"\n加载最佳模型进行测试...")
            if hasattr(trainer, 'best_epoch_info') and trainer.best_epoch_info.get('model_weights') is not None:
                trainer.model.load_state_dict(trainer.best_epoch_info['model_weights'])
                print(f"✓ 已加载最佳模型 (epoch {trainer.best_epoch_info['epoch']}, acc {trainer.best_epoch_info['accuracy']:.4f})")
            else:
                # 如果best_epoch_info中没有模型权重，尝试从checkpoint文件加载
                print("  从checkpoint文件加载最佳模型...")
                trainer, parameter_controller = checkpoint_controller.load_checkpoint(load_best=True)
                print(f"✓ 已从checkpoint文件加载最佳模型")
            
            test_kwargs = {
                'dataloader_dict': dataloaders,
                'epoch': None,
                'partition': 'test'
            }
            trainer.test(checkpoint_controller, predict_only=1, **test_kwargs)
            
            fold_pbar.set_postfix({'status': 'completed'})
        
        fold_pbar.close()
        print("\n" + "=" * 60)
        print("所有实验完成！")
        print("=" * 60)

    def init_dataset(self, data, continuous_label_dim, mode, fold, normalization_stats=None):
        """Initialize dataset"""
        if self.mode == 'imu':
            # IMU数据集
            dataset = IMUDataset(
                data_dict=data,
                window_length=self.args.window_length,
                hop_length=self.args.hop_length,
                normalize=self.args.normalize_data,
                apply_filter=self.args.apply_filter,
                imu_sampling_rate=getattr(self.args, 'imu_sampling_rate', 250),
                normalization_stats=normalization_stats
            )
        elif self.mode == 'rawimu':
            # Raw IMU数据集（只使用前6维：加速度和角速度）
            dataset = RawIMUDataset(
                data_dict=data,
                window_length=self.args.window_length,
                hop_length=self.args.hop_length,
                normalize=self.args.normalize_data,
                apply_filter=self.args.apply_filter,
                imu_sampling_rate=getattr(self.args, 'imu_sampling_rate', 250),
                normalization_stats=normalization_stats
            )
        elif self.mode == 'mix':
            # Mix模式：同时加载IMU和Physio数据
            dataset = MixDataset(
                data_dict=data,
                window_length=self.args.window_length,
                hop_length=self.args.hop_length,
                normalize=self.args.normalize_data,
                apply_filter=self.args.apply_filter,
                imu_sampling_rate=getattr(self.args, 'imu_sampling_rate', 250),
                eeg_sampling_rate=self.args.eeg_sampling_rate,
                ecg_sampling_rate=self.args.ecg_sampling_rate,
                normalization_stats=normalization_stats
            )
        elif self.mode == 'simplemix':
            # SimpleMix模式：同时加载IMU和ECG数据（不加载EEG）
            dataset = MixDataset(
                data_dict=data,
                window_length=self.args.window_length,
                hop_length=self.args.hop_length,
                normalize=self.args.normalize_data,
                apply_filter=self.args.apply_filter,
                imu_sampling_rate=getattr(self.args, 'imu_sampling_rate', 250),
                eeg_sampling_rate=self.args.eeg_sampling_rate,
                ecg_sampling_rate=self.args.ecg_sampling_rate,
                normalization_stats=normalization_stats
            )
        elif self.mode == 'newmix':
            # NewMix模式：同时加载IMU和ECG数据（不加载EEG），使用特征级融合
            dataset = MixDataset(
                data_dict=data,
                window_length=self.args.window_length,
                hop_length=self.args.hop_length,
                normalize=self.args.normalize_data,
                apply_filter=self.args.apply_filter,
                imu_sampling_rate=getattr(self.args, 'imu_sampling_rate', 250),
                eeg_sampling_rate=self.args.eeg_sampling_rate,
                ecg_sampling_rate=self.args.ecg_sampling_rate,
                normalization_stats=normalization_stats
            )
        elif self.mode == 'allmix':
            # AllMix模式：同时加载IMU、EEG和ECG数据，使用三模态特征级融合
            dataset = MixDataset(
                data_dict=data,
                window_length=self.args.window_length,
                hop_length=self.args.hop_length,
                normalize=self.args.normalize_data,
                apply_filter=self.args.apply_filter,
                imu_sampling_rate=getattr(self.args, 'imu_sampling_rate', 250),
                eeg_sampling_rate=self.args.eeg_sampling_rate,
                ecg_sampling_rate=self.args.ecg_sampling_rate,
                normalization_stats=normalization_stats
            )
        else:
            # 生理信号数据集
            dataset = PhysiologicalDataset(
                data_dict=data,  # 修正参数名：data -> data_dict
                modality=self.modality,
                window_length=self.args.window_length,
                hop_length=self.args.hop_length,
                normalize=self.args.normalize_data,
                apply_filter=self.args.apply_filter,
                eeg_sampling_rate=self.args.eeg_sampling_rate,
                ecg_sampling_rate=self.args.ecg_sampling_rate,
                normalization_stats=normalization_stats  # 传递归一化统计量
            )
        return dataset
    
    def compute_class_weights(self):
        """
        计算类别权重
        
        优先级：
        1. 如果用户手动指定了 class_weights，使用用户指定的
        2. 如果 auto_compute_class_weights=True，自动从训练数据计算
        3. 否则不使用类别权重
        
        Returns:
            torch.Tensor: 类别权重，如果为None则不使用权重
        """
        # 如果用户手动指定了类别权重，直接使用
        if self.args.class_weights is not None:
            weights = [float(w) for w in self.args.class_weights.split(',')]
            if len(weights) != self.num_classes:
                print(f"⚠️  警告: 指定的类别权重数量 ({len(weights)}) 与类别数 ({self.num_classes}) 不匹配")
                print(f"   将使用自动计算的类别权重")
            else:
                return torch.tensor(weights, device=self.device)
        
        # 检查是否启用自动计算
        auto_compute = getattr(self.args, 'auto_compute_class_weights', True)
        if not auto_compute:
            return None
        
        # 从训练数据计算类别权重
        try:
            # 只加载labels，避免加载大量数据
            labels = self.data_arranger.load_labels_only()
            
            if labels is None:
                print("⚠️  警告: 无法获取标签数据，将不使用类别权重")
                return None
            
            # 确保是numpy数组
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)
            
            # 统计每个类别的数量（labels 已由 load_labels_only 映射为 0-4 五分类）
            from collections import Counter
            counter = Counter(labels)
            # 固定使用 0-4 五个类别，保证权重与 num_classes=5 一致
            classes = list(range(self.num_classes))
            class_counts = [counter.get(cls, 0) for cls in classes]
            num_classes = self.num_classes
            total_samples = sum(class_counts)
            if total_samples == 0:
                print("⚠️  警告: 训练集标签为空，将不使用类别权重")
                return None
            
            # 使用平衡权重方法计算（避免 count_i=0 导致除零，用 max(count, 1)）
            # 公式: weight_i = total_samples / (num_classes * count_i)
            weights = [total_samples / (num_classes * max(count, 1)) for count in class_counts]
            
            # 归一化到最小权重为1（可选，但通常推荐）
            min_weight = min(weights)
            weights = [w / min_weight for w in weights]
            
            print(f"✓ 自动计算类别权重:")
            print(f"  类别分布: {dict(zip(classes, class_counts))}")
            print(f"  计算权重: {[f'{w:.4f}' for w in weights]}")
            
            return torch.tensor(weights, device=self.device)
            
        except Exception as e:
            print(f"⚠️  警告: 计算类别权重时出错: {e}")
            print(f"   将不使用类别权重")
            return None
    
    def compute_normalization_stats(self, train_data, val_data=None, test_data=None):
        """
        计算每个subject的归一化统计量（mean和std）
        对每个subject，分别使用该subject在各个数据集中的数据计算统计量
        这样可以避免信息泄漏，并且符合subject-wise的要求
        
        返回格式: {subject_id: {modality: {mean, std}}}
        """
        from tqdm import tqdm
        print(f"\n{'='*60}")
        print(f"计算Per-Subject归一化统计量 (mode={self.mode})...")
        print(f"{'='*60}")
        
        # 收集所有数据集的数据和subject_ids
        all_data_dicts = [('train', train_data)]
        if val_data is not None:
            all_data_dicts.append(('val', val_data))
        if test_data is not None:
            all_data_dicts.append(('test', test_data))
        
        # 获取所有唯一的subject_id
        all_subject_ids = set()
        for name, data_dict in all_data_dicts:
            subject_ids = data_dict.get('subject_ids', None)
            if subject_ids is not None:
                if isinstance(subject_ids, np.ndarray):
                    all_subject_ids.update(subject_ids.tolist())
                else:
                    all_subject_ids.update(subject_ids)
        
        if len(all_subject_ids) == 0:
            print("  ⚠️  警告: 未找到subject_ids，将回退到全局归一化")
            return self._compute_global_normalization_stats(train_data)
        
        all_subject_ids = sorted(all_subject_ids)
        print(f"  找到 {len(all_subject_ids)} 个唯一的subject")
        
        # 按subject计算统计量
        # 格式: {subject_id: {modality: {mean, std}}}
        stats = {}
        
        # 确定需要处理的模态
        modalities = []
        if self.mode == 'imu':
            modalities = ['imu']
        elif self.mode == 'mix':
            modalities = ['imu', 'eeg', 'ecg']
        else:  # physio
            modalities = ['eeg', 'ecg']
        
        # 对每个subject计算统计量
        for subject_id in tqdm(all_subject_ids, desc="计算各subject统计量"):
            stats[subject_id] = {}
            
            for modality in modalities:
                # 收集该subject在所有数据集中的数据
                subject_data_list = []
                
                for name, data_dict in all_data_dicts:
                    modality_data = data_dict.get(modality, None)
                    subject_ids = data_dict.get('subject_ids', None)
                    indices = data_dict.get('_indices', None)  # 检查是否有索引映射
                    
                    if modality_data is None or subject_ids is None:
                        continue
                    
                    # 处理_indices映射：如果有_indices，需要先获取实际的subject_ids
                    if indices is not None:
                        # 有索引映射：通过_indices获取实际的subject_ids
                        if isinstance(subject_ids, np.ndarray):
                            actual_subject_ids = subject_ids[indices]
                        else:
                            actual_subject_ids = [subject_ids[i] for i in indices]
                        
                        # 获取该subject在当前数据集中的索引（相对于_indices的位置）
                        if isinstance(actual_subject_ids, np.ndarray):
                            subject_mask = (actual_subject_ids == subject_id)
                            local_indices = np.where(subject_mask)[0]  # 在_indices中的位置
                        else:
                            local_indices = [i for i, sid in enumerate(actual_subject_ids) if sid == subject_id]
                        
                        if len(local_indices) == 0:
                            continue
                        
                        # 通过local_indices获取实际的原始数据索引
                        if isinstance(indices, np.ndarray):
                            actual_indices = indices[local_indices]
                        else:
                            actual_indices = [indices[i] for i in local_indices]
                    else:
                        # 没有索引映射：直接使用原始索引
                        if isinstance(subject_ids, np.ndarray):
                            subject_mask = (subject_ids == subject_id)
                            actual_indices = np.where(subject_mask)[0]
                        else:
                            actual_indices = [i for i, sid in enumerate(subject_ids) if sid == subject_id]
                    
                    if len(actual_indices) == 0:
                        continue
                    # 可选：每 subject 最多用 N 个样本算统计量以加速
                    max_per_subject = getattr(self.args, 'norm_stats_max_samples', 0)
                    if max_per_subject > 0 and len(actual_indices) > max_per_subject:
                        rng = np.random.RandomState(self.seed + int(subject_id))
                        actual_indices = rng.choice(actual_indices, size=max_per_subject, replace=False)
                    
                    # 获取该subject的数据
                    if isinstance(modality_data, np.ndarray):
                        # numpy数组格式：通过索引访问
                        if len(modality_data.shape) == 4:  # (num_windows, num_patches, channels, patch_samples)
                            subject_modality_data = modality_data[actual_indices]  # (num_samples_in_split, num_patches, channels, patch_samples)
                            subject_data_list.append(subject_modality_data)
                        elif len(modality_data.shape) == 3:  # (num_patches, channels, patch_samples) - 单个样本
                            # 这种情况应该不会出现，但为了安全起见
                            for idx in actual_indices:
                                if idx < len(modality_data):
                                    subject_data_list.append(modality_data[idx:idx+1])
                    elif isinstance(modality_data, list):
                        # 列表格式：每个元素是 (num_patches, channels, patch_samples)
                        for idx in actual_indices:
                            if idx < len(modality_data):
                                subject_data_list.append(modality_data[idx])
                
                if len(subject_data_list) == 0:
                    continue
                
                # 合并该subject的所有数据
                # subject_data_list中的每个元素可能是：
                # - 4维数组: (num_samples_in_split, num_patches, channels, patch_samples) 来自numpy数组
                # - 3维数组: (num_patches, channels, patch_samples) 来自列表
                # 需要统一处理：先将所有元素转换为4维，然后concatenate
                normalized_list = []
                for data_item in subject_data_list:
                    if len(data_item.shape) == 4:
                        # 已经是4维，直接添加
                        normalized_list.append(data_item)
                    elif len(data_item.shape) == 3:
                        # 3维，添加一个维度变成4维
                        normalized_list.append(data_item[np.newaxis, :, :, :])  # (1, num_patches, channels, patch_samples)
                    else:
                        raise ValueError(f"意外的数据形状: {data_item.shape}, 期望3维或4维")
                
                # 现在所有元素都是4维，使用concatenate在第一个维度上合并
                if len(normalized_list) == 1:
                    all_subject_data = normalized_list[0]
                else:
                    all_subject_data = np.concatenate(normalized_list, axis=0)  # (total_samples, num_patches, channels, patch_samples)
                
                # 确保是4维数组
                if len(all_subject_data.shape) != 4:
                    raise ValueError(f"合并后的数据形状不正确: {all_subject_data.shape}, 期望4维 (num_samples, num_patches, channels, patch_samples)")
                
                # 检查NaN和Inf
                nan_count = np.isnan(all_subject_data).sum()
                inf_count = np.isinf(all_subject_data).sum()
                if nan_count > 0 or inf_count > 0:
                    all_subject_data = np.nan_to_num(all_subject_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 计算该subject的统计量
                num_samples, num_patches, num_channels, patch_length = all_subject_data.shape
                
                # 检查是否有数据
                if num_samples == 0:
                    print(f"    ⚠️  警告: Subject {subject_id} 的 {modality} 数据为空，跳过")
                    continue
                
                # 检查数据是否全为0或NaN
                if np.all(all_subject_data == 0) or np.all(np.isnan(all_subject_data)):
                    print(f"    ⚠️  警告: Subject {subject_id} 的 {modality} 数据全为0或NaN，跳过")
                    continue
                
                reshaped = all_subject_data.reshape(-1, num_channels, patch_length)  # (N, C, T)
                
                # 计算每个通道的mean和std
                subject_mean = reshaped.mean(axis=(0, 2))  # (num_channels,)
                subject_std = reshaped.std(axis=(0, 2))  # (num_channels,)
                
                # 处理std为0或NaN的情况
                zero_std_mask = subject_std < 1e-8
                if zero_std_mask.any():
                    subject_std[zero_std_mask] = 1.0
                
                subject_mean = np.nan_to_num(subject_mean, nan=0.0)
                subject_std = np.nan_to_num(subject_std, nan=1.0)
                
                stats[subject_id][modality] = {
                    'mean': subject_mean,
                    'std': subject_std
                }
        
        print(f"\n✓ Per-Subject归一化统计量计算完成")
        print(f"  共计算了 {len(stats)} 个subject的统计量")
        
        # 检查是否有subject没有统计量
        missing_subjects = set(all_subject_ids) - set(stats.keys())
        if missing_subjects:
            print(f"  ⚠️  警告: 以下 {len(missing_subjects)} 个subject没有统计量: {sorted(missing_subjects)}")
            print(f"  这些subject的数据可能为空，在归一化时将使用样本内归一化")
        
        # 显示每个subject的统计量情况
        print(f"  各subject的统计量情况:")
        for subject_id in sorted(all_subject_ids):
            if subject_id in stats:
                modalities_with_stats = list(stats[subject_id].keys())
                print(f"    Subject {subject_id}: {modalities_with_stats}")
            else:
                print(f"    Subject {subject_id}: 无统计量")
        
        print(f"{'='*60}\n")
        return stats
    
    def _extract_data_for_global_normalization(self, modality_data, indices, modality_name):
        """
        从train_data中提取数据用于全局归一化统计量计算
        处理_indices映射和不同数据格式
        """
        if modality_data is None:
            return None
        
        # 处理numpy数组格式（可能有_indices映射）
        if isinstance(modality_data, np.ndarray):
            if len(modality_data.shape) == 4:  # (num_windows, num_patches, channels, patch_samples)
                # 如果有_indices，只使用训练集的样本
                if indices is not None:
                    return modality_data[indices]  # (num_train_samples, num_patches, channels, patch_samples)
                else:
                    return modality_data  # 使用全部数据
            else:
                print(f"  ⚠️  警告: {modality_name}数据形状不正确: {modality_data.shape}")
                return None
        # 处理列表格式
        elif isinstance(modality_data, list) and len(modality_data) > 0:
            first_item = modality_data[0]
            if isinstance(first_item, np.ndarray) and len(first_item.shape) == 3:
                # 如果有_indices，只使用训练集的样本
                if indices is not None:
                    return np.stack([modality_data[i] for i in indices])
                else:
                    return np.stack(modality_data)  # (num_samples, num_patches, channels, patch_samples)
            else:
                print(f"  ⚠️  警告: {modality_name}数据格式不正确")
                return None
        else:
            print(f"  ⚠️  警告: {modality_name}数据为空或格式不正确")
            return None
    
    def _compute_global_normalization_stats(self, train_data):
        """
        计算全局归一化统计量（Random划分时使用，只使用训练集）
        注意：需要处理_indices映射的情况
        若 args.norm_stats_max_samples > 0，仅用该数量的训练样本估计 mean/std，以加速。
        """
        print("  使用全局归一化（Random划分，只使用训练集）...")
        stats = {}
        
        # 检查是否有_indices映射
        indices = train_data.get('_indices', None)
        # 可选：只用部分训练样本计算统计量以加速（大数据集时有效）
        max_samples = getattr(self.args, 'norm_stats_max_samples', 0)
        if max_samples > 0 and indices is not None and len(indices) > max_samples:
            np.random.seed(self.seed)
            indices = np.random.choice(indices, size=max_samples, replace=False)
            print(f"  归一化统计量仅使用 {max_samples} 个训练样本（加速）")
        elif max_samples > 0 and indices is None:
            n = len(train_data.get('labels', []))
            if n > max_samples:
                np.random.seed(self.seed)
                indices = np.random.choice(n, size=max_samples, replace=False)
                print(f"  归一化统计量仅使用 {max_samples} 个训练样本（加速）")
        
        if self.mode == 'imu' or self.mode == 'rawimu':
            # 处理IMU数据
            if 'imu' in train_data and train_data['imu'] is not None:
                imu_data = train_data['imu']
                all_imu = self._extract_data_for_global_normalization(imu_data, indices, 'imu')
                if all_imu is not None and all_imu.size > 0:
                    num_samples, num_patches, num_channels, patch_length = all_imu.shape
                    if num_samples > 0:
                        # rawimu模式：只使用前6维
                        if self.mode == 'rawimu':
                            all_imu = all_imu[:, :, :6, :]  # 只保留前6维
                            num_channels = 6
                        
                        # 检查原始数据是否包含NaN或Inf
                        nan_count = np.isnan(all_imu).sum()
                        inf_count = np.isinf(all_imu).sum()
                        if nan_count > 0 or inf_count > 0:
                            print(f"⚠️  警告: 训练数据中包含NaN或Inf")
                            print(f"  NaN数量: {nan_count}")
                            print(f"  Inf数量: {inf_count}")
                            # 替换NaN和Inf为0
                            all_imu = np.nan_to_num(all_imu, nan=0.0, posinf=0.0, neginf=0.0)
                            print(f"  已替换为0")
                        
                        reshaped = all_imu.reshape(-1, num_channels, patch_length)  # (N, C, T)
                        # 计算每个通道的mean和std（跨所有样本和patches）
                        imu_mean = reshaped.mean(axis=(0, 2))  # (num_channels,)
                        imu_std = reshaped.std(axis=(0, 2))  # (num_channels,)
                        
                        # 检查std是否为0或NaN
                        zero_std_mask = imu_std < 1e-8
                        if zero_std_mask.any():
                            print(f"⚠️  警告: 某些通道的std为0或接近0")
                            print(f"  通道索引: {np.where(zero_std_mask)[0]}")
                            print(f"  这些通道的std值: {imu_std[zero_std_mask]}")
                            # 将std为0的通道设置为1，避免除零
                            imu_std[zero_std_mask] = 1.0
                        
                        # 检查mean和std是否包含NaN
                        if np.isnan(imu_mean).any() or np.isnan(imu_std).any():
                            print(f"⚠️  警告: 归一化统计量包含NaN")
                            print(f"  mean包含NaN: {np.isnan(imu_mean).any()}")
                            print(f"  std包含NaN: {np.isnan(imu_std).any()}")
                            # 替换NaN为0
                            imu_mean = np.nan_to_num(imu_mean, nan=0.0)
                            imu_std = np.nan_to_num(imu_std, nan=1.0)
                        
                        stats['imu'] = {'mean': imu_mean, 'std': imu_std}
                        mode_name = "Raw IMU" if self.mode == 'rawimu' else "IMU"
                        print(f"\n✓ {mode_name}归一化统计量计算完成")
                        print(f"  通道数: {num_channels}")
                        print(f"  mean范围: [{imu_mean.min():.6f}, {imu_mean.max():.6f}]")
                        print(f"  std范围: [{imu_std.min():.6f}, {imu_std.max():.6f}]")
                        print(f"{'='*60}\n")
        
        if self.mode == 'mix':
            # Mix模式：同时计算IMU和Physio的归一化统计量
            # IMU统计量
            if 'imu' in train_data and train_data['imu'] is not None:
                imu_data = train_data['imu']
                all_imu = self._extract_data_for_global_normalization(imu_data, indices, 'imu')
                if all_imu is not None and all_imu.size > 0:
                    num_samples, num_patches, num_channels, patch_length = all_imu.shape
                    if num_samples > 0:
                        all_imu = np.nan_to_num(all_imu, nan=0.0, posinf=0.0, neginf=0.0)
                        reshaped = all_imu.reshape(-1, num_channels, patch_length)
                        imu_mean = reshaped.mean(axis=(0, 2))
                        imu_std = reshaped.std(axis=(0, 2))
                        zero_std_mask = imu_std < 1e-8
                        if zero_std_mask.any():
                            imu_std[zero_std_mask] = 1.0
                        imu_mean = np.nan_to_num(imu_mean, nan=0.0)
                        imu_std = np.nan_to_num(imu_std, nan=1.0)
                        stats['imu'] = {'mean': imu_mean, 'std': imu_std}
            
            # EEG统计量
            if 'eeg' in train_data and train_data['eeg'] is not None:
                eeg_data = train_data['eeg']
                all_eeg = self._extract_data_for_global_normalization(eeg_data, indices, 'eeg')
                if all_eeg is not None and all_eeg.size > 0:
                    num_samples, num_patches, num_channels, patch_length = all_eeg.shape
                    if num_samples > 0:
                        all_eeg = np.nan_to_num(all_eeg, nan=0.0, posinf=0.0, neginf=0.0)
                        reshaped = all_eeg.reshape(-1, num_channels, patch_length)
                        eeg_mean = reshaped.mean(axis=(0, 2))
                        eeg_std = reshaped.std(axis=(0, 2))
                        stats['eeg'] = {'mean': eeg_mean, 'std': eeg_std}
            
            # ECG统计量
            if 'ecg' in train_data and train_data['ecg'] is not None:
                ecg_data = train_data['ecg']
                all_ecg = self._extract_data_for_global_normalization(ecg_data, indices, 'ecg')
                if all_ecg is not None and all_ecg.size > 0:
                    num_samples, num_patches, num_channels, patch_length = all_ecg.shape
                    if num_samples > 0:
                        all_ecg = np.nan_to_num(all_ecg, nan=0.0, posinf=0.0, neginf=0.0)
                        reshaped = all_ecg.reshape(-1, num_channels, patch_length)
                        ecg_mean = reshaped.mean(axis=(0, 2))
                        ecg_std = reshaped.std(axis=(0, 2))
                        stats['ecg'] = {'mean': ecg_mean, 'std': ecg_std}
        elif self.mode != 'imu':
            # 处理EEG数据
            if 'eeg' in train_data and train_data['eeg'] is not None:
                eeg_data = train_data['eeg']
                all_eeg = self._extract_data_for_global_normalization(eeg_data, indices, 'eeg')
                if all_eeg is not None and all_eeg.size > 0:
                    num_samples, num_patches, num_channels, patch_length = all_eeg.shape
                    if num_samples > 0:
                        all_eeg = np.nan_to_num(all_eeg, nan=0.0, posinf=0.0, neginf=0.0)
                        reshaped = all_eeg.reshape(-1, num_channels, patch_length)  # (N, C, T)
                        # 计算每个通道的mean和std（跨所有样本和patches）
                        eeg_mean = reshaped.mean(axis=(0, 2))  # (num_channels,)
                        eeg_std = reshaped.std(axis=(0, 2))  # (num_channels,)
                        stats['eeg'] = {'mean': eeg_mean, 'std': eeg_std}
            
            # 处理ECG数据
            if 'ecg' in train_data and train_data['ecg'] is not None:
                ecg_data = train_data['ecg']
                all_ecg = self._extract_data_for_global_normalization(ecg_data, indices, 'ecg')
                if all_ecg is not None and all_ecg.size > 0:
                    num_samples, num_patches, num_channels, patch_length = all_ecg.shape
                    if num_samples > 0:
                        all_ecg = np.nan_to_num(all_ecg, nan=0.0, posinf=0.0, neginf=0.0)
                        reshaped = all_ecg.reshape(-1, num_channels, patch_length)
                        ecg_mean = reshaped.mean(axis=(0, 2))  # (num_channels,)
                        ecg_std = reshaped.std(axis=(0, 2))  # (num_channels,)
                        stats['ecg'] = {'mean': ecg_mean, 'std': ecg_std}
        
        print(f"\n✓ 归一化统计量计算完成")
        print(f"{'='*60}\n")
        return stats

    def init_model(self):
        """Initialize model"""
        self.init_randomness()
        modality = [modal for modal in self.modality if "continuous_label" not in modal]

        if self.mode == 'imu':
            # IMU模式：自动使用IMUClassificationModel
            # 处理hidden_dims参数（可能是字符串或列表）
            hidden_dims = getattr(self.args, 'hidden_dims', None)
            if hidden_dims is None:
                hidden_dims = [512, 256, 128]
            elif isinstance(hidden_dims, str):
                # 如果是字符串，解析为列表（例如 "512,256,128"）
                hidden_dims = [int(x.strip()) for x in hidden_dims.split(',')]
            elif not isinstance(hidden_dims, list):
                hidden_dims = [512, 256, 128]
            
            model = IMUClassificationModel(
                imu_channels=getattr(self.args, 'imu_channels', 18),
                patch_length=getattr(self.args, 'patch_length', 250),  # 1秒 @ 250Hz
                num_patches=getattr(self.args, 'num_patches', 10),  # 10个patches
                encoding_dim=getattr(self.args, 'encoding_dim', 256),
                num_heads=getattr(self.args, 'num_heads', 8),
                num_classes=self.num_classes,
                attention_output_mode=getattr(self.args, 'attention_output_mode', 'global'),
                hidden_dims=hidden_dims,
                dropout=getattr(self.args, 'dropout', 0.1),
            )
        elif self.mode == 'rawimu':
            # Raw IMU模式：只使用前6维（加速度和角速度），使用IMUClassificationModel
            # 处理hidden_dims参数（可能是字符串或列表）
            hidden_dims = getattr(self.args, 'hidden_dims', None)
            if hidden_dims is None:
                hidden_dims = [512, 256, 128]
            elif isinstance(hidden_dims, str):
                # 如果是字符串，解析为列表（例如 "512,256,128"）
                hidden_dims = [int(x.strip()) for x in hidden_dims.split(',')]
            elif not isinstance(hidden_dims, list):
                hidden_dims = [512, 256, 128]
            
            model = IMUClassificationModel(
                imu_channels=6,  # 只使用前6维：加速度和角速度
                patch_length=getattr(self.args, 'patch_length', 250),  # 1秒 @ 250Hz
                num_patches=getattr(self.args, 'num_patches', 10),  # 10个patches
                encoding_dim=getattr(self.args, 'encoding_dim', 256),
                num_heads=getattr(self.args, 'num_heads', 8),
                num_classes=self.num_classes,
                attention_output_mode=getattr(self.args, 'attention_output_mode', 'global'),
                hidden_dims=hidden_dims,
                dropout=getattr(self.args, 'dropout', 0.1),
            )
        elif self.mode == 'mix':
            # Mix模式：使用MixClassificationModel
            # 处理hidden_dims参数（可能是字符串或列表）
            hidden_dims = getattr(self.args, 'hidden_dims', None)
            if hidden_dims is None:
                hidden_dims = [128, 64]  # Mix模式的patch分类器使用较小的网络
            elif isinstance(hidden_dims, str):
                hidden_dims = [int(x.strip()) for x in hidden_dims.split(',')]
            elif not isinstance(hidden_dims, list):
                hidden_dims = [128, 64]
            
            model = MixClassificationModel(
                imu_channels=getattr(self.args, 'imu_channels', 18),
                eeg_channels=getattr(self.args, 'eeg_channels', 59),
                ecg_channels=getattr(self.args, 'ecg_channels', 1),
                patch_length=getattr(self.args, 'patch_length', 250),  # 1秒 @ 250Hz
                num_patches=getattr(self.args, 'num_patches', 10),  # 10个patches
                encoding_dim=getattr(self.args, 'encoding_dim', 256),
                num_heads=getattr(self.args, 'num_heads', 8),
                num_classes=self.num_classes,
                attention_output_mode='patch_wise',  # Mix模式必须使用patch_wise
                hidden_dims=hidden_dims,
                dropout=getattr(self.args, 'dropout', 0.1),
                state_predictor_hidden=getattr(self.args, 'state_predictor_hidden', 64),
                use_gru=getattr(self.args, 'use_gru', True),
                gain_net_hidden_dims=getattr(self.args, 'gain_net_hidden_dims', [64, 128]),
            )
        elif self.mode == 'simplemix':
            # SimpleMix模式：使用SimpleMixClassificationModel
            # 处理hidden_dims参数（可能是字符串或列表）
            hidden_dims = getattr(self.args, 'hidden_dims', None)
            if hidden_dims is None:
                hidden_dims = [128, 64]  # SimpleMix模式的patch分类器使用较小的网络
            elif isinstance(hidden_dims, str):
                hidden_dims = [int(x.strip()) for x in hidden_dims.split(',')]
            elif not isinstance(hidden_dims, list):
                hidden_dims = [128, 64]
            
            model = SimpleMixClassificationModel(
                imu_channels=getattr(self.args, 'imu_channels', 18),
                ecg_channels=getattr(self.args, 'ecg_channels', 1),
                patch_length=getattr(self.args, 'patch_length', 250),  # 1秒 @ 250Hz
                num_patches=getattr(self.args, 'num_patches', 10),  # 10个patches
                encoding_dim=getattr(self.args, 'encoding_dim', 256),
                num_heads=getattr(self.args, 'num_heads', 8),
                num_classes=self.num_classes,
                attention_output_mode='patch_wise',  # SimpleMix模式必须使用patch_wise
                hidden_dims=hidden_dims,
                dropout=getattr(self.args, 'dropout', 0.1),
                state_predictor_hidden=getattr(self.args, 'state_predictor_hidden', 64),
                use_gru=getattr(self.args, 'use_gru', True),
                gain_net_hidden_dims=getattr(self.args, 'gain_net_hidden_dims', [64, 128]),
            )
        elif self.mode == 'newmix':
            # NewMix模式：使用NewMixClassificationModel（特征级融合，结构与physio一致）
            # 处理hidden_dims参数（可能是字符串或列表）
            hidden_dims = getattr(self.args, 'hidden_dims', None)
            if hidden_dims is None:
                hidden_dims = [512, 256, 128]  # NewMix模式使用和physio相同的网络结构
            elif isinstance(hidden_dims, str):
                hidden_dims = [int(x.strip()) for x in hidden_dims.split(',')]
            elif not isinstance(hidden_dims, list):
                hidden_dims = [512, 256, 128]
            
            model = NewMixClassificationModel(
                imu_channels=getattr(self.args, 'imu_channels', 18),
                ecg_channels=getattr(self.args, 'ecg_channels', 1),
                patch_length=getattr(self.args, 'patch_length', 250),  # 1秒 @ 250Hz
                num_patches=getattr(self.args, 'num_patches', 10),  # 10个patches
                encoding_dim=getattr(self.args, 'encoding_dim', 256),
                num_heads=getattr(self.args, 'num_heads', 8),
                num_classes=self.num_classes,
                attention_output_mode=getattr(self.args, 'attention_output_mode', 'global'),
                hidden_dims=hidden_dims,
                dropout=getattr(self.args, 'dropout', 0.1),
            )
        elif self.mode == 'allmix':
            # AllMix模式：使用AllMixClassificationModel（三模态特征级融合）
            # 处理hidden_dims参数（可能是字符串或列表）
            hidden_dims = getattr(self.args, 'hidden_dims', None)
            if hidden_dims is None:
                hidden_dims = [512, 256, 128]  # AllMix模式使用和physio相同的网络结构
            elif isinstance(hidden_dims, str):
                hidden_dims = [int(x.strip()) for x in hidden_dims.split(',')]
            elif not isinstance(hidden_dims, list):
                hidden_dims = [512, 256, 128]
            
            model = AllMixClassificationModel(
                imu_channels=getattr(self.args, 'imu_channels', 18),
                eeg_channels=getattr(self.args, 'eeg_channels', 59),
                ecg_channels=getattr(self.args, 'ecg_channels', 1),
                patch_length=getattr(self.args, 'patch_length', 250),  # 1秒 @ 250Hz
                num_patches=getattr(self.args, 'num_patches', 10),  # 10个patches
                encoding_dim=getattr(self.args, 'encoding_dim', 256),
                num_heads=getattr(self.args, 'num_heads', 8),
                num_classes=self.num_classes,
                attention_output_mode=getattr(self.args, 'attention_output_mode', 'global'),
                hidden_dims=hidden_dims,
                dropout=getattr(self.args, 'dropout', 0.1),
            )
        elif self.mode == 'eeg' or self.mode == 'ecg':
            # 单模态生理信号：仅 EEG 或仅 ECG，使用 SingleModalPhysioModel
            hidden_dims = getattr(self.args, 'hidden_dims', None)
            if hidden_dims is None:
                hidden_dims = [512, 256, 128]
            elif isinstance(hidden_dims, str):
                hidden_dims = [int(x.strip()) for x in hidden_dims.split(',')]
            elif not isinstance(hidden_dims, list):
                hidden_dims = [512, 256, 128]
            channels = getattr(self.args, 'eeg_channels', 59) if self.mode == 'eeg' else getattr(self.args, 'ecg_channels', 1)
            model = SingleModalPhysioModel(
                channels=channels,
                patch_length=getattr(self.args, 'patch_length', 250),
                num_patches=getattr(self.args, 'num_patches', 10),
                encoding_dim=getattr(self.args, 'encoding_dim', 256),
                num_heads=getattr(self.args, 'num_heads', 8),
                num_classes=self.num_classes,
                attention_output_mode=getattr(self.args, 'attention_output_mode', 'global'),
                hidden_dims=hidden_dims,
                dropout=getattr(self.args, 'dropout', 0.1),
                modal=self.mode,
            )
        else:
            # physio模式：根据model_name选择模型
            # PhysioClassificationModel是ComfortClassificationModel的别名
            if self.model_name == "PhysioClassificationModel" or self.model_name == "ComfortClassificationModel" or self.model_name == "ComfortModel" or self.model_name is None:
                # 使用ComfortClassificationModel
                # 处理hidden_dims参数（可能是字符串或列表）
                hidden_dims = getattr(self.args, 'hidden_dims', None)
                if hidden_dims is None:
                    hidden_dims = [512, 256, 128]
                elif isinstance(hidden_dims, str):
                    # 如果是字符串，解析为列表（例如 "512,256,128"）
                    hidden_dims = [int(x.strip()) for x in hidden_dims.split(',')]
                elif not isinstance(hidden_dims, list):
                    hidden_dims = [512, 256, 128]
                
                model = ComfortClassificationModel(
                    eeg_channels=getattr(self.args, 'eeg_channels', 59),
                    ecg_channels=getattr(self.args, 'ecg_channels', 1),
                    patch_length=getattr(self.args, 'patch_length', 250),  # 1秒 @ 250Hz
                    num_patches=getattr(self.args, 'num_patches', 10),  # 10个patches
                    encoding_dim=getattr(self.args, 'encoding_dim', 256),
                    num_heads=getattr(self.args, 'num_heads', 8),
                    num_classes=self.num_classes,
                    attention_output_mode=getattr(self.args, 'attention_output_mode', 'global'),
                    hidden_dims=hidden_dims,
                    dropout=getattr(self.args, 'dropout', 0.1),
                )
            elif self.model_name == "PhysioFusionNet":
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
            else:
                raise ValueError(f"physio模式下不支持的模型: {self.model_name}")

        return model

    def get_modality(self):
        """Get modality information"""
        if self.mode == 'imu' or self.mode == 'rawimu':
            # IMU模式或RawIMU模式：只使用IMU模态
            self.modality = ['imu']
        elif self.mode == 'mix':
            # Mix模式：同时使用IMU、EEG和ECG模态
            self.modality = ['imu', 'eeg', 'ecg']
        elif self.mode == 'simplemix':
            self.modality = ['imu', 'ecg']
        elif self.mode == 'newmix':
            # NewMix模式：使用IMU和ECG模态（特征级融合）
            self.modality = ['imu', 'ecg']
        elif self.mode == 'allmix':
            # AllMix模式：使用IMU、EEG和ECG三个模态（三模态特征级融合）
            self.modality = ['imu', 'eeg', 'ecg']
        elif self.mode == 'eeg':
            self.modality = ['eeg']
        elif self.mode == 'ecg':
            self.modality = ['ecg']
        else:
            # physio模式：使用EEG和ECG
            self.modality = self.args.modality if hasattr(self.args, 'modality') else ['eeg', 'ecg']

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
        from tqdm import tqdm
        
        # Load data
        print("  加载数据...")
        data = self.data_arranger.load_data(mode=self.mode)
        
        # 检查是否使用subject-wise划分
        use_subject_wise = getattr(self.args, 'subject_wise_split', 1)
        subject_ids = data.get('subject_ids', None)
        
        # 记录实际使用的划分方式
        is_subject_wise_split = False
        
        if use_subject_wise and subject_ids is not None:
            is_subject_wise_split = True
            # Subject-wise划分：按subject分组，确保同一subject的数据不会同时出现在训练集和测试集
            print("  使用Subject-wise划分...")
            n_samples = len(data.get('labels', [0]))
            
            # 获取所有唯一的subject
            unique_subjects = np.unique(subject_ids)
            n_subjects = len(unique_subjects)
            print(f"  总样本数: {n_samples}, 总subject数: {n_subjects}")
            
            # 按subject分组
            subject_to_indices = {}
            for idx, subj_id in enumerate(subject_ids):
                if subj_id not in subject_to_indices:
                    subject_to_indices[subj_id] = []
                subject_to_indices[subj_id].append(idx)
            
            # 将subject列表打乱（使用固定seed保证可复现）
            np.random.seed(self.seed + fold)
            shuffled_subjects = unique_subjects.copy()
            np.random.shuffle(shuffled_subjects)
            
            # 按比例划分subject
            train_subject_size = int(5/9 * n_subjects)
            val_subject_size = int(2/9 * n_subjects)
            
            train_subjects = shuffled_subjects[:train_subject_size]
            val_subjects = shuffled_subjects[train_subject_size:train_subject_size + val_subject_size]
            test_subjects = shuffled_subjects[train_subject_size + val_subject_size:]
            
            # 根据subject分配样本索引
            train_indices = []
            val_indices = []
            test_indices = []
            
            for subj_id in train_subjects:
                train_indices.extend(subject_to_indices[subj_id])
            for subj_id in val_subjects:
                val_indices.extend(subject_to_indices[subj_id])
            for subj_id in test_subjects:
                test_indices.extend(subject_to_indices[subj_id])
            
            train_indices = np.array(train_indices, dtype=np.int64)
            val_indices = np.array(val_indices, dtype=np.int64)
            test_indices = np.array(test_indices, dtype=np.int64)
            
            print(f"  训练集: {len(train_subjects)} subjects, {len(train_indices)} samples")
            print(f"  验证集: {len(val_subjects)} subjects, {len(val_indices)} samples")
            print(f"  测试集: {len(test_subjects)} subjects, {len(test_indices)} samples")
            
            # 检查数据集是否为空
            if len(train_indices) == 0:
                raise ValueError("训练集为空！请检查数据划分逻辑")
            if len(val_indices) == 0:
                print("  ⚠️  警告: 验证集为空！")
            if len(test_indices) == 0:
                print("  ⚠️  警告: 测试集为空！")
        else:
            # 随机划分（原有逻辑）
            is_subject_wise_split = False
            if use_subject_wise and subject_ids is None:
                print("  ⚠️  警告: 请求使用subject-wise划分，但未找到subject_ids，将使用随机划分")
            
            print("  使用随机划分...")
            n_samples = len(data.get('labels', [0]))
            indices = np.arange(n_samples)
            
            # 随机打乱（使用固定seed保证可复现）
            np.random.seed(self.seed + fold)
            np.random.shuffle(indices)
            
            # Simple split for demonstration
            train_size = int(0.7 * n_samples)
            val_size = int(0.15 * n_samples)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            print(f"  数据拆分: 训练集 {len(train_indices)}, 验证集 {len(val_indices)}, 测试集 {len(test_indices)}")
            
            # 检查数据集是否为空
            if len(train_indices) == 0:
                raise ValueError("训练集为空！请检查数据划分逻辑")
            if len(val_indices) == 0:
                print("  ⚠️  警告: 验证集为空！")
            if len(test_indices) == 0:
                print("  ⚠️  警告: 测试集为空！")
        
        # Create datasets
        print("  创建数据子集...")
        train_data = self.subset_data(data, train_indices)
        print("  ✓ 训练集子集创建完成")
        val_data = self.subset_data(data, val_indices)
        print("  ✓ 验证集子集创建完成")
        test_data = self.subset_data(data, test_indices)
        print("  ✓ 测试集子集创建完成")
        
        # 根据划分方式选择归一化方法（如果启用归一化）
        normalization_stats = None
        if self.args.normalize_data:
            if is_subject_wise_split:
                # Subject-wise划分：使用Per-Subject归一化
                print("  计算Per-Subject归一化统计量（Subject-wise划分）...")
                normalization_stats = self.compute_normalization_stats(train_data, val_data, test_data)
                print("  ✓ Per-Subject归一化统计量计算完成")
            else:
                # Random划分：使用全局归一化（只使用训练集）
                print("  计算全局归一化统计量（Random划分，只使用训练集）...")
                normalization_stats = self._compute_global_normalization_stats(train_data)
                print("  ✓ 全局归一化统计量计算完成")
        
        # Initialize datasets
        print("  初始化数据集（这可能需要一些时间进行预处理）...")
        print("    - 初始化训练集...")
        train_dataset = self.init_dataset(train_data, self.continuous_label_dim, 'train', fold, 
                                         normalization_stats=normalization_stats)
        train_size = len(train_dataset)
        print(f"    ✓ 训练集初始化完成 (样本数: {train_size})")
        
        print("    - 初始化验证集...")
        val_dataset = self.init_dataset(val_data, self.continuous_label_dim, 'val', fold,
                                       normalization_stats=normalization_stats)
        val_size = len(val_dataset)
        print(f"    ✓ 验证集初始化完成 (样本数: {val_size})")
        
        print("    - 初始化测试集...")
        test_dataset = self.init_dataset(test_data, self.continuous_label_dim, 'test', fold,
                                        normalization_stats=normalization_stats)
        test_size = len(test_dataset)
        print(f"    ✓ 测试集初始化完成 (样本数: {test_size})")
        
        # 再次检查数据集是否为空
        if train_size == 0:
            raise ValueError("训练集为空！无法进行训练")
        if val_size == 0:
            print("  ⚠️  警告: 验证集为空！将无法进行验证")
        if test_size == 0:
            print("  ⚠️  警告: 测试集为空！将无法进行测试")
        
        # Create dataloaders
        # 在 macOS 上，num_workers > 0 可能导致问题，使用 0 或 1
        import platform
        if platform.system() == 'Darwin':  # macOS
            num_workers = 0
            pin_memory = False
        else:
            num_workers = 4
            pin_memory = True
        
        print(f"  创建 DataLoader (num_workers={num_workers}, pin_memory={pin_memory})...")
        print("    - 创建训练集 DataLoader...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        print("    ✓ 训练集 DataLoader 创建完成")
        print("    - 创建验证集 DataLoader...")
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        print("    ✓ 验证集 DataLoader 创建完成")
        print("    - 创建测试集 DataLoader...")
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        print("    ✓ 测试集 DataLoader 创建完成")
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }

    def subset_data(self, data, indices):
        """
        Create subset of data based on indices
        优化：对于numpy数组，不复制数据，只保存索引映射，避免复制大量数据
        """
        subset_data = {}
        
        # 保存索引映射，而不是复制数据（转换为numpy数组以便快速索引）
        if isinstance(indices, np.ndarray):
            subset_data['_indices'] = indices
        else:
            subset_data['_indices'] = np.asarray(indices, dtype=np.int64)
        
        for key, value in data.items():
            if key == '_indices':
                continue  # 跳过已添加的索引
            if isinstance(value, list) and len(value) > 0:
                # 列表格式：需要复制（因为需要保持列表结构）
                subset_data[key] = [value[i] for i in indices]
            elif isinstance(value, np.ndarray):
                # numpy数组格式：不复制，直接保存原始数据和索引映射
                # Dataset会通过索引映射访问原始数据
                subset_data[key] = value  # 保存原始数据引用
            else:
                # 其他类型（如标量）：直接复制
                subset_data[key] = value
        
        return subset_data 