import sys
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Motion Sickness Classification with Multimodal Physiological Signals')

    # 1. Experiment Setting
    # 1.1. Server
    parser.add_argument('-gpu', default=0, type=int, help='Which gpu to use?')
    parser.add_argument('-cpu', default=4, type=int, help='How many threads are allowed?')
    parser.add_argument('-high_performance_cluster', default=0, type=int, 
                        help='On high-performance server or not?')

    # 1.2. Paths
    parser.add_argument('-dataset_path', default='./data', type=str,
                        help='The root directory of the dataset.')
    parser.add_argument('-load_path', default='./checkpoints', type=str,
                        help='The path to load the trained models.')
    parser.add_argument('-save_path', default='./results', type=str,
                        help='The path to save the trained models and results.')
    parser.add_argument('-python_package_path', default='./', type=str,
                        help='The path to the entire repository.')

    # 1.3. Experiment name and stamp
    parser.add_argument('-experiment_name', default="MotionSickness", help='The experiment name.')
    parser.add_argument('-stamp', default='PhysioFusionNet_v1', type=str, 
                        help='To indicate different experiment instances')

    # 1.4. Load checkpoint or not?
    parser.add_argument('-resume', default=0, type=int, help='Resume from checkpoint?')

    # 1.5. Debug or not?
    parser.add_argument('-debug', default=0, type=int, 
                        help='The number of trials to load for debugging. Set to 0 for non-debugging execution.')

    # 1.6. What modality to use?
    parser.add_argument('-modality', default=['eeg', 'ecg'], nargs="*",
                        help='Modalities to use: eeg, ecg')
    parser.add_argument('-calc_mean_std', default=0, type=int,
                        help='Calculate the mean and std and save to a pickle file')

    # 1.7. Classification settings
    parser.add_argument('-num_classes', default=11, type=int,
                        help='Number of classes (0-10 for motion sickness scores)')
    parser.add_argument('-class_weights', default=None, type=str,
                        help='Class weights for imbalanced data (comma-separated)')

    # 1.8. Whether to save the models?
    parser.add_argument('-save_model', default=1, type=int, help='Whether to save the models?')

    # 2. Training settings.
    parser.add_argument('-num_heads', default=4, type=int, help='Number of attention heads')
    parser.add_argument('-modal_dim', default=64, type=int, help='Modal embedding dimension')
    parser.add_argument('-tcn_kernel_size', default=5, type=int,
                        help='The size of the 1D kernel for temporal convolutional networks.')

    # 2.1. Overall settings
    parser.add_argument('-model_name', default="PhysioFusionNet", help='Model name: PhysioFusionNet, CAN')
    parser.add_argument('-cross_validation', default=1, type=int)
    parser.add_argument('-num_folds', default=5, type=int)
    parser.add_argument('-folds_to_run', default=[1], nargs="+", type=int, 
                        help='Which fold(s) to run?')

    # 2.2. Epochs and data
    parser.add_argument('-num_epochs', default=100, type=int, help='The total of epochs to run during training.')
    parser.add_argument('-min_num_epochs', default=10, type=int, help='The minimum epoch to run at least.')
    parser.add_argument('-early_stopping', default=20, type=int,
                        help='If no improvement, the number of epoch to run before halting the training')
    parser.add_argument('-window_length', default=3000, type=int, 
                        help='The length in point number to windowing the data.')
    parser.add_argument('-hop_length', default=1500, type=int, 
                        help='The step size or stride to move the window.')
    parser.add_argument('-batch_size', default=32, type=int)

    # 2.3. Scheduler and Parameter Control
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-scheduler', default='cosine', type=str, help='plateau, cosine, step')
    parser.add_argument('-learning_rate', default=1e-4, type=float, help='The initial learning rate.')
    parser.add_argument('-min_learning_rate', default=1e-7, type=float, help='The minimum learning rate.')
    parser.add_argument('-patience', default=10, type=int, help='Patience for learning rate changes.')
    parser.add_argument('-factor', default=0.5, type=float, help='The multiplier to decrease the learning rate.')
    parser.add_argument('-gradual_release', default=1, type=int, help='Whether to gradually release some layers?')
    parser.add_argument('-release_count', default=3, type=int, help='How many layer groups to release?')
    parser.add_argument('-milestone', default=[0], nargs="+", type=int, help='The specific epochs to do something.')
    parser.add_argument('-load_best_at_each_epoch', default=1, type=int,
                        help='Whether to load the best models state at the end of each epoch?')

    # 2.4. Data preprocessing settings
    parser.add_argument('-eeg_sampling_rate', default=1000, type=int, help='EEG sampling rate')
    parser.add_argument('-ecg_sampling_rate', default=500, type=int, help='ECG sampling rate')
    parser.add_argument('-eeg_channels', default=64, type=int, help='Number of EEG channels')
    parser.add_argument('-ecg_channels', default=3, type=int, help='Number of ECG channels')
    parser.add_argument('-normalize_data', default=1, type=int, help='Whether to normalize the data')
    parser.add_argument('-apply_filter', default=1, type=int, help='Whether to apply bandpass filter')

    # 2.5. Evaluation settings
    parser.add_argument('-metrics', default=["accuracy", "precision", "recall", "f1"], nargs="*", 
                        help='The evaluation metrics.')
    parser.add_argument('-save_plot', default=1, type=int,
                        help='Whether to plot the confusion matrix and results or not?')

    args = parser.parse_args()
    sys.path.insert(0, args.python_package_path)

    # Create necessary directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.load_path, exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./data/processed', exist_ok=True)
    os.makedirs('./data/splits', exist_ok=True)

    from experiment import Experiment

    exp = Experiment(args)
    exp.prepare()
    exp.run() 