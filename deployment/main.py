"""
舒适度预测部署主程序入口
"""

import argparse
import os
import sys
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from deployment.config import DeploymentConfig
from deployment.comfort_predictor import ComfortPredictor
from deployment.utils import setup_logging, save_prediction_result, format_prediction_output


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="舒适度预测闭环系统部署")
    
    # 路径配置
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="模型checkpoint文件路径 (.pkl)")
    parser.add_argument("--normalization_stats_path", type=str,
                        help="归一化统计量文件路径 (.pkl)")
    
    # 模型配置
    parser.add_argument("--mode", type=str, choices=['allmix', 'newmix', 'simplemix', 'mix', 'imu', 'physio', 'rawimu'],
                        help="模型模式（若不指定，将从路径推断）")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="计算设备 (默认: cuda:0)")
    
    # 数据流配置
    parser.add_argument("--simulated", action="store_true", default=False,
                        help="使用模拟随机数据生成器 (默认: False)")
    parser.add_argument("--original", action="store_true", default=False,
                        help="使用 data/ 目录下的原始 BDF 数据回放 (默认: False)")
    parser.add_argument("--subject", type=str, help="指定被试姓名 (original 模式必填)")
    parser.add_argument("--map", type=str, help="指定地图编号 (original 模式必填)")
    parser.add_argument("--data_root", type=str, default="data",
                        help="原始数据根目录 (默认: data)")
    
    parser.add_argument("--collect_interval", type=float, default=0.1,
                        help="数据采集间隔/休眠时间，单位秒 (默认: 0.1)")
    
    # 窗口参数
    parser.add_argument("--window_length", type=int, default=2500,
                        help="预测窗口长度，采样点数 (默认: 2500, 即10秒@250Hz)")
    parser.add_argument("--target_srate", type=int, default=250,
                        help="目标采样率，Hz (默认: 250)")
    
    # 日志输出
    parser.add_argument("--log_dir", type=str, default="./deployment_logs",
                        help="日志输出目录")
    parser.add_argument("--output_dir", type=str, default="./deployment_output",
                        help="预测结果输出目录")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="是否打印详细运行信息")
    
    return parser.parse_args()


def main():
    """主程序"""
    args = parse_args()
    
    # 1. 设置日志
    logger = setup_logging(args.log_dir)
    
    logger.info("="*60)
    logger.info("舒适度预测闭环系统 - 启动中...")
    logger.info("="*60)
    
    try:
        # 2. 检查 original 模式参数
        if args.original and (not args.subject or not args.map):
            logger.error("--original 模式需要提供 --subject 和 --map")
            sys.exit(1)

        # 3. 创建配置对象
        config = DeploymentConfig(
            checkpoint_path=args.checkpoint_path,
            normalization_stats_path=args.normalization_stats_path,
            mode=args.mode,
            device=args.device,
            window_length=args.window_length,
            target_sampling_rate=args.target_srate,
            use_simulated_data=args.simulated,
            use_original_data=args.original,
            subject_id=args.subject,
            subject_filter=args.subject,
            map_filter=args.map,
            data_root=args.data_root,
            log_dir=args.log_dir,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        # 4. 实例化预测服务
        predictor = ComfortPredictor(config)
        
        # 4. 设置回调函数（闭环控制核心入口）
        def on_prediction(result):
            # 格式化输出
            output_str = format_prediction_output(result)
            if args.verbose:
                print(output_str)
            logger.info(output_str)
            
            # 保存结果
            save_prediction_result(result, args.output_dir)
            
            # --- 闭环控制逻辑 ---
            score = result['score']
            confidence = result['confidence']
            
            if score >= 3:
                logger.warning(f"检测到高不舒适感: {score}分 (置信度: {confidence:.3f})")
                # TODO: 在此处集成 CARLA 驾驶模式切换策略
                # 例如: if score >= 3: carla_controller.switch_to_relaxed_mode()
        
        predictor.set_prediction_callback(on_prediction)
        
        # 5. 启动服务
        predictor.start(collect_interval=args.collect_interval)
        
    except FileNotFoundError as e:
        logger.error(f"文件未找到: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("程序被用户手动停止")
    except Exception as e:
        logger.error(f"运行过程中发生未知错误: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
