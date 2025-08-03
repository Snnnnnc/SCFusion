import argparse
import torch
import numpy as np
import os
import sys
from models.model import PhysioFusionNet
from dataset import PhysiologicalDataset
from torch.utils.data import DataLoader
from base.utils import compute_metrics
import matplotlib.pyplot as plt
import seaborn as sns


def test_model(checkpoint_path, test_data_path, config_path, output_path):
    """
    Test a trained model
    
    Args:
        checkpoint_path: Path to model checkpoint
        test_data_path: Path to test data
        config_path: Path to config file
        output_path: Path to save results
    """
    
    # Load configuration
    sys.path.append(os.path.dirname(config_path))
    from configs import config
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = PhysioFusionNet(
        backbone_settings=config['backbone_settings'],
        modality=['eeg', 'ecg'],
        example_length=3000,
        kernel_size=5,
        tcn_channel=config['tcn']['channels'],
        modal_dim=64,
        num_heads=4,
        num_classes=11,
        root_dir='',
        device=device
    )
    model.init()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load test data
    test_data = np.load(test_data_path, allow_pickle=True).item()
    
    # Create dataset
    test_dataset = PhysiologicalDataset(
        data=test_data,
        modality=['eeg', 'ecg'],
        window_length=3000,
        hop_length=1500,
        normalize=True,
        apply_filter=True
    )
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # Test model
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Prepare input data
            input_data = {}
            for modality in ['eeg', 'ecg']:
                if modality in batch:
                    input_data[modality] = batch[modality]
            
            # Forward pass
            outputs = model(input_data)
            targets = batch['label'].squeeze()
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_path, 'test_results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Test Precision: {metrics['precision']:.4f}\n")
        f.write(f"Test Recall: {metrics['recall']:.4f}\n")
        f.write(f"Test F1: {metrics['f1']:.4f}\n")
    
    # Save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save predictions
    np.save(os.path.join(output_path, 'predictions.npy'), all_predictions)
    np.save(os.path.join(output_path, 'targets.npy'), all_targets)
    np.save(os.path.join(output_path, 'probabilities.npy'), all_probabilities)
    
    print(f"Test completed. Results saved to {output_path}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Motion Sickness Classification Model')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_data_path', type=str, required=True,
                        help='Path to test data')
    parser.add_argument('--config_path', type=str, default='configs.py',
                        help='Path to config file')
    parser.add_argument('--output_path', type=str, default='./test_results',
                        help='Path to save results')
    
    args = parser.parse_args()
    
    test_model(
        checkpoint_path=args.checkpoint_path,
        test_data_path=args.test_data_path,
        config_path=args.config_path,
        output_path=args.output_path
    ) 