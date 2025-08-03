import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import argparse


def generate_results_summary(results_dir, output_dir):
    """
    Generate comprehensive results summary
    
    Args:
        results_dir: Directory containing experiment results
        output_dir: Directory to save summary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all results
    all_results = []
    
    for fold_dir in os.listdir(results_dir):
        fold_path = os.path.join(results_dir, fold_dir)
        if os.path.isdir(fold_path):
            # Look for test results
            test_metrics_path = os.path.join(fold_path, 'test_metrics.txt')
            if os.path.exists(test_metrics_path):
                # Parse metrics
                metrics = parse_metrics_file(test_metrics_path)
                metrics['fold'] = fold_dir
                all_results.append(metrics)
    
    if not all_results:
        print("No results found!")
        return
    
    # Create summary dataframe
    df = pd.DataFrame(all_results)
    
    # Save summary
    df.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)
    
    # Generate summary statistics
    summary_stats = {
        'mean_accuracy': df['accuracy'].mean(),
        'std_accuracy': df['accuracy'].std(),
        'mean_f1': df['f1'].mean(),
        'std_f1': df['f1'].std(),
        'mean_precision': df['precision'].mean(),
        'std_precision': df['precision'].std(),
        'mean_recall': df['recall'].mean(),
        'std_recall': df['recall'].std(),
    }
    
    # Save summary statistics
    with open(os.path.join(output_dir, 'summary_statistics.txt'), 'w') as f:
        f.write("Summary Statistics:\n")
        f.write("=" * 50 + "\n")
        for metric, value in summary_stats.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Create visualization
    create_results_visualization(df, output_dir)
    
    print(f"Results summary generated in {output_dir}")


def parse_metrics_file(metrics_path):
    """Parse metrics from text file"""
    metrics = {}
    with open(metrics_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(': ')
                metrics[key.lower().replace(' ', '_')] = float(value)
    return metrics


def create_results_visualization(df, output_dir):
    """Create visualization of results"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Motion Sickness Classification Results', fontsize=16, fontweight='bold')
    
    # 1. Accuracy across folds
    axes[0, 0].bar(range(len(df)), df['accuracy'], color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Accuracy Across Folds')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(range(len(df)))
    axes[0, 0].set_xticklabels(df['fold'])
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. F1 Score across folds
    axes[0, 1].bar(range(len(df)), df['f1'], color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('F1 Score Across Folds')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_xticks(range(len(df)))
    axes[0, 1].set_xticklabels(df['fold'])
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision vs Recall
    axes[1, 0].scatter(df['precision'], df['recall'], s=100, alpha=0.7, c='orange')
    axes[1, 0].set_title('Precision vs Recall')
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_ylabel('Recall')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add fold labels
    for i, fold in enumerate(df['fold']):
        axes[1, 0].annotate(fold, (df['precision'].iloc[i], df['recall'].iloc[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Metrics comparison
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    means = [df[metric].mean() for metric in metrics]
    stds = [df[metric].std() for metric in metrics]
    
    x_pos = np.arange(len(metrics))
    axes[1, 1].bar(x_pos, means, yerr=stds, capsize=5, color='lightcoral', alpha=0.7)
    axes[1, 1].set_title('Average Metrics with Standard Deviation')
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'results_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create confusion matrix heatmap (if available)
    create_confusion_matrix_heatmap(df, output_dir)


def create_confusion_matrix_heatmap(df, output_dir):
    """Create confusion matrix heatmap"""
    # Look for confusion matrix files
    cm_files = []
    for fold_dir in os.listdir(os.path.dirname(output_dir)):
        fold_path = os.path.join(os.path.dirname(output_dir), fold_dir)
        if os.path.isdir(fold_path):
            cm_path = os.path.join(fold_path, 'confusion_matrix.png')
            if os.path.exists(cm_path):
                cm_files.append((fold_dir, cm_path))
    
    if cm_files:
        # Create a grid of confusion matrices
        n_files = len(cm_files)
        cols = min(3, n_files)
        rows = (n_files + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (fold, cm_path) in enumerate(cm_files):
            row = i // cols
            col = i % cols
            
            # Load and display confusion matrix
            cm_img = plt.imread(cm_path)
            axes[row, col].imshow(cm_img)
            axes[row, col].set_title(f'Confusion Matrix - {fold}')
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(n_files, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrices_grid.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()


def generate_classification_report(results_dir, output_dir):
    """Generate detailed classification report"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all predictions and targets
    all_predictions = []
    all_targets = []
    
    for fold_dir in os.listdir(results_dir):
        fold_path = os.path.join(results_dir, fold_dir)
        if os.path.isdir(fold_path):
            predictions_path = os.path.join(fold_path, 'predictions.npy')
            targets_path = os.path.join(fold_path, 'targets.npy')
            
            if os.path.exists(predictions_path) and os.path.exists(targets_path):
                predictions = np.load(predictions_path)
                targets = np.load(targets_path)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
    
    if all_predictions and all_targets:
        # Generate classification report
        report = classification_report(all_targets, all_predictions, 
                                     target_names=[f'Class_{i}' for i in range(11)],
                                     output_dict=True)
        
        # Save detailed report
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write("Detailed Classification Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(classification_report(all_targets, all_predictions, 
                                        target_names=[f'Class_{i}' for i in range(11)]))
        
        # Create per-class performance plot
        create_per_class_performance_plot(report, output_dir)


def create_per_class_performance_plot(report, output_dir):
    """Create per-class performance visualization"""
    # Extract per-class metrics
    classes = []
    precision = []
    recall = []
    f1 = []
    
    for i in range(11):
        class_name = f'Class_{i}'
        if class_name in report:
            classes.append(f'Class {i}')
            precision.append(report[class_name]['precision'])
            recall.append(report[class_name]['recall'])
            f1.append(report[class_name]['f1-score'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(classes))
    width = 0.25
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_performance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Motion Sickness Classification Results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./results_summary',
                        help='Directory to save summary')
    
    args = parser.parse_args()
    
    # Generate results summary
    generate_results_summary(args.results_dir, args.output_dir)
    
    # Generate detailed classification report
    generate_classification_report(args.results_dir, args.output_dir)
    
    print(f"Results generation completed. Output saved to {args.output_dir}") 