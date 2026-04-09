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
    parser.add_argument('-dataset_path', default='./data/training_dataset', type=str,
                        help='The root directory of the dataset.')
    parser.add_argument('-load_path', default='./checkpoints', type=str,
                        help='The path to load the trained models.')
    parser.add_argument('-save_path', default='./results_new', type=str,
                        help='The path to save the trained models and results.')
    parser.add_argument('-python_package_path', default='./', type=str,
                        help='The path to the entire repository.')

    # 1.3. Experiment name and stamp
    parser.add_argument('-experiment_name', default="MotionSickness", help='The experiment name.')
    parser.add_argument('-stamp', default='PhysioFusionNet_v1', type=str, 
                        help='To indicate different experiment instances')

    # 1.4. Load checkpoint or not?
    parser.add_argument('-resume', default=0, type=int, help='Resume from checkpoint? (1=resume, 0=start from scratch)')
    parser.add_argument('-resume_from_best', default=0, type=int, help='If resume=1, load best model instead of latest? (1=load best, 0=load latest)')
    parser.add_argument('-subject_wise_split', default=0, type=int, help='Use subject-wise split? (1=subject-wise, 0=random split)')

    # 1.5. Debug or not?
    parser.add_argument('-debug', default=0, type=int, 
                        help='The number of trials to load for debugging. Set to 0 for non-debugging execution.')

    # 1.6. What modality to use?
    parser.add_argument('-modality', default=['eeg', 'ecg'], nargs="*",
                        help='Modalities to use: eeg, ecg')
    parser.add_argument('-mode', default='physio', type=str, choices=['physio', 'imu', 'mix', 'rawimu', 'eeg', 'ecg', 'simplemix', 'newmix', 'allmix'],
                        help='Training mode: physio (physiological signals), imu (IMU data), mix (IMU+Physio fusion), rawimu (raw IMU: only 6D acc+gyro, no conflicts), simplemix (IMU+ECG decision-level fusion), newmix (IMU+ECG feature-level fusion), allmix (IMU+EEG+ECG three-modal feature-level fusion)')
    parser.add_argument('-calc_mean_std', default=0, type=int,
                        help='Calculate the mean and std and save to a pickle file')

    # 1.7. Classification settings
    parser.add_argument('-num_classes', default=5, type=int,
                        help='Number of classes (0-4 for motion sickness scores, default: 5)')
    parser.add_argument('-class_weights', default=None, type=str,
                        help='Class weights for imbalanced data (comma-separated, e.g., "1.0,2.0,3.0,5.0,5.0"). If not specified, will auto-compute from training data.')
    parser.add_argument('-auto_compute_class_weights', default=1, type=int,
                        help='Auto-compute class weights from training data (default: 1). Set to 0 to disable.')

    # 1.8. Whether to save the models?
    parser.add_argument('-save_model', default=1, type=int, help='Whether to save the models?')

    # 2. Training settings.
    parser.add_argument('-num_heads', default=8, type=int, help='Number of attention heads (default: 8 for ComfortClassificationModel)')
    parser.add_argument('-modal_dim', default=64, type=int, help='Modal embedding dimension (for PhysioFusionNet)')
    parser.add_argument('-tcn_kernel_size', default=5, type=int,
                        help='The size of the 1D kernel for temporal convolutional networks.')

    # 2.1. Overall settings
    parser.add_argument('-model_name', default=None, help='Model name (optional, will be auto-selected based on mode): ComfortClassificationModel, IMUClassificationModel, MixClassificationModel, SimpleMixClassificationModel, NewMixClassificationModel, AllMixClassificationModel, PhysioFusionNet, CAN')
    parser.add_argument('-cross_validation', default=1, type=int)
    parser.add_argument('-num_folds', default=5, type=int)
    parser.add_argument('-folds_to_run', default=[1], nargs="+", type=int, 
                        help='Which fold(s) to run?')

    # 2.2. Epochs and data
    parser.add_argument('-num_epochs', default=100, type=int, help='The total of epochs to run during training.')
    parser.add_argument('-min_num_epochs', default=10, type=int, help='The minimum epoch to run at least.')
    parser.add_argument('-early_stopping', default=50, type=int,
                        help='If no improvement, the number of epoch to run before halting the training')
    parser.add_argument('-window_length', default=2500, type=int, 
                        help='The length in point number to windowing the data (10s @ 250Hz = 2500).')
    parser.add_argument('-hop_length', default=750, type=int, 
                        help='The step size or stride to move the window (3s @ 250Hz = 750).')
    parser.add_argument('-batch_size', default=32, type=int)
    
    # 2.2.1. Patch-based model settings (for ComfortClassificationModel)
    parser.add_argument('-patch_length', default=250, type=int, 
                        help='Patch length in time points (1s @ 250Hz = 250)')
    parser.add_argument('-num_patches', default=10, type=int, 
                        help='Number of patches per window (10 patches for 10s window)')
    parser.add_argument('-encoding_dim', default=256, type=int, 
                        help='Encoding dimension for patch encoder (default: 256)')
    parser.add_argument('-attention_output_mode', default='global', type=str, 
                        help='Attention output mode: global or patch_wise (default: global)')
    parser.add_argument('-dropout', default=0.1, type=float, 
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('-hidden_dims', default='512,256,128', type=str, 
                        help='Hidden dimensions for classifier (comma-separated, default: 512,256,128)')

    # 2.3. Scheduler and Parameter Control
    parser.add_argument('-seed', default=42, type=int)
    parser.add_argument('-scheduler', default='cosine', type=str, help='plateau, cosine, step')
    parser.add_argument('-learning_rate', default=5e-5, type=float, help='The initial learning rate.')
    parser.add_argument('-grad_clip', default=1.0, type=float, help='Gradient clipping max norm (default: 1.0, set to 0 to disable)')
    parser.add_argument('-min_learning_rate', default=1e-7, type=float, help='The minimum learning rate.')
    parser.add_argument('-patience', default=10, type=int, help='Patience for learning rate changes.')
    parser.add_argument('-factor', default=0.5, type=float, help='The multiplier to decrease the learning rate.')
    parser.add_argument('-gradual_release', default=1, type=int, help='Whether to gradually release some layers?')
    parser.add_argument('-release_count', default=3, type=int, help='How many layer groups to release?')
    parser.add_argument('-milestone', default=[0], nargs="+", type=int, help='The specific epochs to do something.')
    parser.add_argument('-load_best_at_each_epoch', default=1, type=int,
                        help='Whether to load the best models state at the end of each epoch?')

    # 2.4. Data preprocessing settings
    parser.add_argument('-eeg_sampling_rate', default=250, type=int, help='EEG sampling rate (default: 250Hz)')
    parser.add_argument('-ecg_sampling_rate', default=250, type=int, help='ECG sampling rate (default: 250Hz)')
    parser.add_argument('-imu_sampling_rate', default=250, type=int, help='IMU sampling rate (default: 250Hz)')
    parser.add_argument('-eeg_channels', default=59, type=int, help='Number of EEG channels (default: 59)')
    parser.add_argument('-ecg_channels', default=1, type=int, help='Number of ECG channels (default: 1)')
    parser.add_argument('-imu_channels', default=18, type=int, help='Number of IMU channels (default: 18)')
    parser.add_argument('-normalize_data', default=1, type=int, help='Whether to normalize the data')
    parser.add_argument('-norm_stats_max_samples', default=0, type=int,
                        help='Global norm: max train samples for mean/std (0=all). Per-subject: max samples per subject. Set e.g. 10000 to speed up.')
    parser.add_argument('-apply_filter', default=0, type=int, help='Whether to apply bandpass filter (default: 0, data already preprocessed)')

    # 2.5. Evaluation settings
    parser.add_argument('-metrics', default=["accuracy", "precision", "recall", "f1"], nargs="*", 
                        help='The evaluation metrics.')
    parser.add_argument('-save_plot', default=1, type=int,
                        help='Whether to plot the confusion matrix and results or not?')

    parser.add_argument("--web", action="store_true", help="启动Web可视化界面")
    parser.add_argument("--web_port", type=int, default=5100, help="Web 可视化端口（默认 5100，建议避开 5000）")
    parser.add_argument("--web_restart", action="store_true", help="若端口上已存在我方服务，则强制重启（默认：复用并退出）")
    args = parser.parse_args()

    # 如果是Web模式，启动 API 服务
    if args.web:
        import uvicorn
        import socket
        import time
        import subprocess
        import signal
        import json
        import urllib.request
        import urllib.error
        from deployment.app import app
        
        preferred_port = args.web_port

        def port_in_use(p: int) -> bool:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('127.0.0.1', p)) == 0

        def get_listen_pids(p: int):
            """返回监听指定端口的 pid 列表"""
            try:
                out = subprocess.check_output(
                    ["lsof", "-nP", f"-iTCP:{p}", "-sTCP:LISTEN", "-t"],
                    text=True
                ).strip()
                if not out:
                    return []
                return [int(x) for x in out.splitlines() if x.strip().isdigit()]
            except Exception:
                return []

        def pid_command(pid: int) -> str:
            try:
                return subprocess.check_output(["ps", "-p", str(pid), "-o", "command="], text=True).strip()
            except Exception:
                return ""

        def is_our_service_running(p: int) -> bool:
            """
            如果 127.0.0.1:p 上已经是我们的 FastAPI 服务在跑，则直接复用该固定链接，
            不再换端口、不再杀进程，用户刷新即可重连。
            """
            # 1) health check 做重试，避免“服务在启动/半启动”被误判
            url = f"http://127.0.0.1:{p}/api/health"
            for _ in range(5):
                try:
                    req = urllib.request.Request(url, headers={"Accept": "application/json"})
                    with urllib.request.urlopen(req, timeout=0.5) as resp:
                        if resp.status != 200:
                            time.sleep(0.1)
                            continue
                        data = json.loads(resp.read().decode("utf-8"))
                        if data.get("service") == "motion_sickness_ui" and data.get("ok") is True:
                            return True
                except (urllib.error.URLError, urllib.error.HTTPError, ValueError, socket.timeout, TimeoutError):
                    time.sleep(0.1)
                except Exception:
                    time.sleep(0.1)

            # 2) health 不通时，用 pid/命令行兜底识别“是否我方”
            for pid in get_listen_pids(p):
                cmd = pid_command(pid)
                is_pythonish = ("python" in cmd) or ("uvicorn" in cmd)
                looks_like_ours = (
                    ("motion_sickness_classification" in cmd)
                    or ("deployment.app" in cmd)
                    or ("deployment/app.py" in cmd)
                    or ("main.py" in cmd)
                )
                if is_pythonish and looks_like_ours:
                    return True

            return False

        def try_kill_if_ours(p: int) -> bool:
            """
            如果端口被占用，尝试识别是否为我们自己启动的 uvicorn/python 服务。
            是则自动关闭（SIGTERM -> SIGKILL），避免让用户手动 kill。
            返回：是否进行了 kill 动作（不保证成功释放端口）。
            """
            try:
                # macOS: 找到监听端口的 PID
                out = subprocess.check_output(["lsof", "-nP", f"-iTCP:{p}", "-sTCP:LISTEN", "-t"], text=True).strip()
                if not out:
                    return False
                pids = [int(x) for x in out.splitlines() if x.strip().isdigit()]
            except Exception:
                return False

            killed_any = False
            for pid in pids:
                try:
                    cmd = subprocess.check_output(["ps", "-p", str(pid), "-o", "command="], text=True).strip()
                    # 只对明显是 python/uvicorn 且包含我们仓库关键词的进程动手
                    is_pythonish = ("python" in cmd) or ("uvicorn" in cmd)
                    looks_like_ours = (
                        ("motion_sickness_classification" in cmd)
                        or ("deployment.app" in cmd)
                        or ("deployment/app.py" in cmd)
                        or ("main.py" in cmd)
                    )
                    if is_pythonish and looks_like_ours:
                        os.kill(pid, signal.SIGTERM)
                        killed_any = True
                except Exception:
                    continue

            if not killed_any:
                return False

            # 等待端口释放，不行再 SIGKILL
            for _ in range(20):
                if not port_in_use(p):
                    return True
                time.sleep(0.1)

            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass

            for _ in range(20):
                if not port_in_use(p):
                    return True
                time.sleep(0.1)

            return True

        # 端口策略（固定链接）：
        # - 永远使用 preferred_port
        # - 若端口上已存在我方服务：默认“复用并退出”（刷新即可重连）；需要重启则 --web_restart
        # - 若端口被非我方程序占用：打印 lsof 并退出（否则无法固定链接）
        port = preferred_port

        if port_in_use(port):
            if is_our_service_running(port):
                if not args.web_restart:
                    print("\n" + "="*50)
                    print("检测到服务已在运行（固定链接不变，刷新即可重连）")
                    print(f"打开链接即可使用: http://localhost:{port}/")
                    print(f"健康检查: http://localhost:{port}/api/health")
                    print(f"如需强制重启：python main.py --web --web_port {port} --web_restart")
                    print("="*50 + "\n")
                    sys.exit(0)

                # 强制重启我方服务（固定端口不变）
                try_kill_if_ours(port)
                for _ in range(50):
                    if not port_in_use(port):
                        break
                    time.sleep(0.1)

                if port_in_use(port):
                    print("\n" + "="*50)
                    print(f"❌ 检测到我方服务占用端口 {port}，但无法释放端口完成重启。")
                    print("建议：等待 1-2 秒后重试，或手动结束占用进程。")
                    print(f"固定链接仍然是: http://localhost:{port}/")
                    print("="*50 + "\n")
                    sys.exit(1)
            else:
                # 被其他软件占用：输出占用信息并退出
                detail = ""
                try:
                    detail = subprocess.check_output(
                        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"],
                        text=True
                    )
                except Exception:
                    pass
                print("\n" + "="*50)
                print(f"❌ 端口 {port} 被其他程序占用，无法使用固定链接。")
                if detail:
                    print("占用详情（lsof）：")
                    print(detail.strip())
                print(f"解决方案：换一个固定端口运行，例如：python main.py --web --web_port 5100")
                print(f"（保持固定链接）：http://localhost:<web_port>/")
                print("="*50 + "\n")
                sys.exit(1)
        
        def guess_lan_ip() -> str:
            """
            尝试推断本机局域网 IP（用于让“车端服务器”访问本机服务端）。
            优先使用 UDP 探测拿到出口网卡 IP；失败则回退到 getaddrinfo。
            """
            # 1) UDP 探测（不需要真的发包）
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("8.8.8.8", 80))
                    ip = s.getsockname()[0]
                    if ip and not ip.startswith("127."):
                        return ip
            except Exception:
                pass

            # 2) 回退：枚举本机地址
            try:
                host = socket.gethostname()
                infos = socket.getaddrinfo(host, None, socket.AF_INET)
                for info in infos:
                    ip = info[4][0]
                    if ip and not ip.startswith("127."):
                        return ip
            except Exception:
                pass

            return ""

        lan_ip = guess_lan_ip()

        print("\n" + "="*60)
        print("正在启动舒适度预测可视化系统...")
        print(f"本机访问（浏览器打开）: http://localhost:{port}/")
        if lan_ip:
            print(f"车端访问（局域网/IP）: http://{lan_ip}:{port}/")
            print(f"车端接口（comfort） : http://{lan_ip}:{port}/api/comfort")
            print(f"车端接口（start）   : http://{lan_ip}:{port}/api/start")
        else:
            print("车端访问（局域网/IP）: 未能自动检测到局域网 IP")
            print("  - 你可以手动用 `ifconfig`(mac/linux) 或 `ipconfig`(win) 查到本机 IP 后替换 <SERVER_HOST>")
        print(f"健康检查            : http://localhost:{port}/api/health")
        print("="*60 + "\n")
        
        uvicorn.run(app, host="0.0.0.0", port=port)
        sys.exit(0)
    sys.path.insert(0, args.python_package_path)

    # Create necessary directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.load_path, exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    os.makedirs('./data/processed', exist_ok=True)
    os.makedirs('./data/splits', exist_ok=True)

    # 添加调试信息以定位导入问题
    print("=" * 60)
    print("开始导入模块...")
    print("=" * 60)
    
    # 逐步导入，以便定位问题
    print("步骤 1: 导入基础模块...")
    import os
    import sys
    print("  ✓ os, sys 导入成功")
    
    print("步骤 2: 导入 torch...")
    try:
        import torch
        print(f"  ✓ torch 导入成功 (版本: {torch.__version__})")
        print(f"  - CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA 设备数量: {torch.cuda.device_count()}")
    except Exception as e:
        print(f"  ✗ torch 导入失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("步骤 3: 导入 experiment 模块...")
    try:
        import experiment
        print("  ✓ experiment 模块导入成功")
    except ImportError as e:
        print(f"  ✗ experiment 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ experiment 模块导入时发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("步骤 4: 从 experiment 导入 Experiment 类...")
    try:
        from experiment import Experiment
        # from experiment_test import Experiment
        print("  ✓ Experiment 类导入成功")
    except ImportError as e:
        print(f"  ✗ Experiment 类导入失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ Experiment 类导入时发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("=" * 60)
    print("所有模块导入完成！")
    print("=" * 60)
    
    # 根据mode自动设置model_name（如果未指定）
    if args.model_name is None:
        if args.mode == 'imu' or args.mode == 'rawimu':
            args.model_name = 'IMUClassificationModel'
        elif args.mode == 'mix':
            args.model_name = 'MixClassificationModel'
        elif args.mode == 'simplemix':
            args.model_name = 'SimpleMixClassificationModel'
        elif args.mode == 'newmix':
            args.model_name = 'NewMixClassificationModel'
        elif args.mode == 'allmix':
            args.model_name = 'AllMixClassificationModel'
        elif args.mode == 'eeg' or args.mode == 'ecg':
            # 单模态生理信号：仅 EEG 或仅 ECG
            args.model_name = 'SingleModalPhysioModel'
        else:
            args.model_name = 'PhysioClassificationModel'
        print(f"根据mode={args.mode}自动设置model_name={args.model_name}")
    
    # 调试信息
    print("\n" + "=" * 60)
    print("实验配置信息")
    print("=" * 60)
    print(f"模式 (mode): {args.mode}")
    print(f"模型名称 (model_name): {args.model_name}")
    print(f"数据集路径 (dataset_path): {args.dataset_path}")
    print(f"保存路径 (save_path): {args.save_path}")
    if args.mode == 'imu':
        print(f"IMU通道数 (imu_channels): {args.imu_channels}")
        print(f"IMU采样率 (imu_sampling_rate): {args.imu_sampling_rate}")
    elif args.mode == 'rawimu':
        print(f"IMU通道数 (imu_channels): 6 (仅使用原始加速度和角速度)")
        print(f"IMU采样率 (imu_sampling_rate): {args.imu_sampling_rate}")
    elif args.mode == 'mix':
        print(f"IMU通道数 (imu_channels): {args.imu_channels}")
        print(f"EEG通道数 (eeg_channels): {args.eeg_channels}")
        print(f"ECG通道数 (ecg_channels): {args.ecg_channels}")
        print(f"IMU采样率 (imu_sampling_rate): {args.imu_sampling_rate}")
        print(f"EEG采样率 (eeg_sampling_rate): {args.eeg_sampling_rate}")
        print(f"ECG采样率 (ecg_sampling_rate): {args.ecg_sampling_rate}")
    elif args.mode == 'physio':
        print(f"EEG通道数 (eeg_channels): {args.eeg_channels}")
        print(f"ECG通道数 (ecg_channels): {args.ecg_channels}")
    elif args.mode == 'eeg':
        print(f"EEG通道数 (eeg_channels): {args.eeg_channels}")
        print(f"EEG采样率 (eeg_sampling_rate): {args.eeg_sampling_rate}")
    elif args.mode == 'ecg':
        print(f"ECG通道数 (ecg_channels): {args.ecg_channels}")
        print(f"ECG采样率 (ecg_sampling_rate): {args.ecg_sampling_rate}")
    print(f"类别数 (num_classes): {args.num_classes}")
    print(f"批次大小 (batch_size): {args.batch_size}")
    print("=" * 60 + "\n")

    exp = Experiment(args)
    exp.prepare()
    exp.run() 