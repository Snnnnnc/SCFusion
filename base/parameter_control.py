import torch
import torch.nn as nn


class ResnetParamControl:
    """Parameter controller for gradual parameter release"""
    
    def __init__(self, trainer, gradual_release=True, release_count=2, backbone_mode=["eeg", "ecg"]):
        self.trainer = trainer
        self.gradual_release = gradual_release
        self.release_count = release_count
        self.backbone_mode = backbone_mode
        self.current_stage = 0
        
        if gradual_release:
            self.setup_parameter_groups()
    
    def setup_parameter_groups(self):
        """Setup parameter groups for gradual release"""
        # This is a simplified version - you would implement based on your specific model structure
        self.parameter_groups = []
        
        # Get all parameters
        all_params = list(self.trainer.model.parameters())
        
        # Create groups based on backbone_mode
        for i, mode in enumerate(self.backbone_mode):
            if i < len(all_params):
                self.parameter_groups.append(all_params[i:i+1])
        
        # Set initial requires_grad
        self.update_parameter_gradients()
    
    def update_parameter_gradients(self):
        """Update which parameters require gradients based on current stage"""
        if not self.gradual_release:
            return
        
        # Enable gradients for parameters up to current stage
        for i, group in enumerate(self.parameter_groups):
            for param in group:
                param.requires_grad = (i <= self.current_stage)
    
    def should_release_parameters(self, epoch, milestone):
        """Check if parameters should be released at this epoch"""
        if not self.gradual_release:
            return False
        
        if self.current_stage >= self.release_count - 1:
            return False
        
        # Check if milestone reached
        return epoch >= milestone
    
    def release_next_parameters(self, epoch, milestone):
        """Release next set of parameters"""
        if self.should_release_parameters(epoch, milestone):
            self.current_stage += 1
            self.update_parameter_gradients()
            print(f"Released parameters for stage {self.current_stage} at epoch {epoch}")
            return True
        return False
    
    def release_param(self, module, epoch):
        """Release parameters - compatibility method"""
        # For ComfortClassificationModel, gradual release may not be applicable
        # This is a compatibility method for the trainer
        if not self.gradual_release:
            return
        # For now, just call release_next_parameters if needed
        # The actual implementation depends on the model structure
        pass
    
    def get_current_lr(self):
        """Get current learning rate from trainer"""
        if hasattr(self.trainer, 'optimizer') and self.trainer.optimizer is not None:
            return self.trainer.optimizer.param_groups[0]['lr']
        return 0.0
    
    @property
    def early_stop(self):
        """Check if early stopping should occur"""
        # This can be implemented based on your needs
        return False