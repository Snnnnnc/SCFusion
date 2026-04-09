import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassificationLoss(nn.Module):
    """
    Classification loss for motion sickness classification
    Supports multiple loss functions
    """
    
    def __init__(self, loss_type='cross_entropy', num_classes=11, class_weights=None):
        super().__init__()
        
        self.loss_type = loss_type
        self.num_classes = num_classes
        
        if loss_type == 'cross_entropy':
            if class_weights is not None:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        elif loss_type == 'focal':
            self.criterion = FocalLoss(alpha=1, gamma=2, num_classes=num_classes)
        elif loss_type == 'label_smoothing':
            self.criterion = LabelSmoothingLoss(num_classes=num_classes, smoothing=0.1)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def forward(self, predictions, targets, sample_weights=None):
        """
        Forward pass
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            sample_weights: Sample weights [batch_size] (optional)
            
        Returns:
            torch.Tensor: Loss value
        """
        if sample_weights is not None:
            # 使用样本权重：先计算每个样本的loss，然后加权平均
            if self.loss_type == 'cross_entropy':
                # 使用reduction='none'计算每个样本的loss
                loss_per_sample = F.cross_entropy(
                    predictions, 
                    targets, 
                    reduction='none',
                    weight=self.criterion.weight if hasattr(self.criterion, 'weight') and self.criterion.weight is not None else None
                )
                
                # 检查是否有NaN或Inf
                if torch.isnan(loss_per_sample).any() or torch.isinf(loss_per_sample).any():
                    print(f"警告: loss_per_sample包含NaN或Inf")
                    print(f"  predictions范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
                    print(f"  predictions包含NaN: {torch.isnan(predictions).any()}")
                    print(f"  predictions包含Inf: {torch.isinf(predictions).any()}")
                    print(f"  targets范围: [{targets.min()}, {targets.max()}]")
                    print(f"  loss_per_sample包含NaN: {torch.isnan(loss_per_sample).sum()}个")
                    print(f"  loss_per_sample包含Inf: {torch.isinf(loss_per_sample).sum()}个")
                
                # 检查sample_weights
                if torch.isnan(sample_weights).any() or torch.isinf(sample_weights).any():
                    print(f"警告: sample_weights包含NaN或Inf")
                    sample_weights = torch.clamp(sample_weights, min=0.0, max=1e6)  # 限制范围
                    sample_weights = torch.where(torch.isnan(sample_weights) | torch.isinf(sample_weights), 
                                                torch.zeros_like(sample_weights), sample_weights)
                
                # 加权平均，避免除零
                weights_sum = sample_weights.sum()
                if weights_sum < 1e-8:
                    print(f"警告: sample_weights.sum()={weights_sum:.6f}，使用平均loss")
                    return loss_per_sample.mean()
                
                weighted_loss = (loss_per_sample * sample_weights).sum() / weights_sum
                
                # 检查最终loss
                if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                    print(f"警告: weighted_loss是NaN或Inf，使用平均loss")
                    return loss_per_sample.mean()
                
                return weighted_loss
            else:
                # 对于其他loss类型，也使用类似的方法
                # 这里需要根据具体的loss类型来实现
                loss_per_sample = self.criterion(predictions, targets)
                if isinstance(loss_per_sample, torch.Tensor) and loss_per_sample.dim() == 0:
                    # 如果loss是标量，说明不支持reduction='none'
                    # 回退到不使用样本权重
                    return loss_per_sample
                else:
                    weighted_loss = (loss_per_sample * sample_weights).sum() / sample_weights.sum()
                    return weighted_loss
        else:
            # 没有样本权重，使用标准loss
            return self.criterion(predictions, targets)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    
    def __init__(self, alpha=1, gamma=2, num_classes=11):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, inputs, targets):
        """
        Forward pass
        
        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Ground truth [batch_size]
            
        Returns:
            torch.Tensor: Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss for regularization
    """
    
    def __init__(self, num_classes=11, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        """
        Forward pass
        
        Args:
            inputs: Predictions [batch_size, num_classes]
            targets: Ground truth [batch_size]
            
        Returns:
            torch.Tensor: Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, self.num_classes).float()
        targets_one_hot = targets_one_hot * (1 - self.smoothing) + self.smoothing / self.num_classes
        loss = -(targets_one_hot * log_probs).sum(dim=1).mean()
        return loss


class MultimodalFusionLoss(nn.Module):
    """
    Loss function for multimodal fusion
    Combines classification loss with fusion consistency loss
    """
    
    def __init__(self, classification_loss, fusion_weight=0.1):
        super().__init__()
        
        self.classification_loss = classification_loss
        self.fusion_weight = fusion_weight
    
    def forward(self, predictions, targets, modality_features=None):
        """
        Forward pass
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            modality_features: Dictionary of modality features
            
        Returns:
            torch.Tensor: Combined loss
        """
        # Classification loss
        cls_loss = self.classification_loss(predictions, targets)
        
        # Fusion consistency loss (if modality features provided)
        fusion_loss = 0
        if modality_features is not None and len(modality_features) > 1:
            fusion_loss = self.compute_fusion_consistency(modality_features)
        
        # Combined loss
        total_loss = cls_loss + self.fusion_weight * fusion_loss
        
        return total_loss
    
    def compute_fusion_consistency(self, modality_features):
        """
        Compute fusion consistency loss
        
        Args:
            modality_features: Dictionary of modality features
            
        Returns:
            torch.Tensor: Fusion consistency loss
        """
        modalities = list(modality_features.keys())
        if len(modalities) < 2:
            return 0
        
        # Compute cosine similarity between modality features
        features = []
        for modal in modalities:
            # Global average pooling
            if len(modality_features[modal].shape) == 3:
                feature = torch.mean(modality_features[modal], dim=1)  # [batch, features]
            else:
                feature = modality_features[modal]
            features.append(feature)
        
        # Normalize features
        features = [F.normalize(f, p=2, dim=1) for f in features]
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                sim = torch.sum(features[i] * features[j], dim=1)  # [batch]
                similarities.append(sim)
        
        # Encourage high similarity (consistency)
        consistency_loss = -torch.mean(torch.stack(similarities))
        
        return consistency_loss


class CCCLoss(nn.Module):
    """
    Concordance Correlation Coefficient Loss
    Adapted for classification tasks
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        """
        Forward pass
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            
        Returns:
            torch.Tensor: CCC loss
        """
        # Convert to regression-like format
        pred_probs = F.softmax(predictions, dim=1)
        target_probs = F.one_hot(targets, predictions.size(1)).float()
        
        # Compute CCC for each class
        ccc_losses = []
        for i in range(predictions.size(1)):
            pred_class = pred_probs[:, i]
            target_class = target_probs[:, i]
            
            ccc = self.compute_ccc(pred_class, target_class)
            ccc_losses.append(1 - ccc)  # Convert to loss
        
        return torch.mean(torch.stack(ccc_losses))
    
    def compute_ccc(self, pred, target):
        """
        Compute Concordance Correlation Coefficient
        
        Args:
            pred: Predictions [batch_size]
            target: Targets [batch_size]
            
        Returns:
            torch.Tensor: CCC value
        """
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        
        pred_var = torch.var(pred)
        target_var = torch.var(target)
        
        covariance = torch.mean((pred - pred_mean) * (target - target_mean))
        
        denominator = pred_var + target_var + (pred_mean - target_mean) ** 2
        
        if denominator == 0:
            return torch.tensor(0.0, device=pred.device)
        
        ccc = 2 * covariance / denominator
        return ccc


class CombinedLoss(nn.Module):
    """
    Combined loss function for motion sickness classification
    """
    
    def __init__(self, loss_weights={'classification': 1.0, 'focal': 0.5, 'consistency': 0.1}):
        super().__init__()
        
        self.loss_weights = loss_weights
        
        # Initialize different loss functions
        self.classification_loss = ClassificationLoss('cross_entropy')
        self.focal_loss = FocalLoss()
        self.consistency_loss = MultimodalFusionLoss(self.classification_loss)
    
    def forward(self, predictions, targets, modality_features=None):
        """
        Forward pass
        
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            modality_features: Dictionary of modality features
            
        Returns:
            torch.Tensor: Combined loss
        """
        total_loss = 0
        
        # Classification loss
        if 'classification' in self.loss_weights:
            cls_loss = self.classification_loss(predictions, targets)
            total_loss += self.loss_weights['classification'] * cls_loss
        
        # Focal loss
        if 'focal' in self.loss_weights:
            focal_loss = self.focal_loss(predictions, targets)
            total_loss += self.loss_weights['focal'] * focal_loss
        
        # Consistency loss
        if 'consistency' in self.loss_weights and modality_features is not None:
            consistency_loss = self.consistency_loss.compute_fusion_consistency(modality_features)
            total_loss += self.loss_weights['consistency'] * consistency_loss
        
        return total_loss 