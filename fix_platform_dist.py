#!/usr/bin/env python3
"""
修复CARLA安装时的platform.dist问题
适用于Python 3.12+版本
"""

import platform
import sys
import subprocess

def fix_platform_dist():
    """修复platform.dist在Python 3.12+中的问题"""
    if not hasattr(platform, 'dist'):
        def dist():
            """替代platform.dist函数"""
            try:
                # 尝试从distro包获取信息
                import distro
                return distro.linux_distribution()
            except ImportError:
                # 如果没有distro包，返回空值
                return ('', '', '')
        
        # 将替代函数添加到platform模块
        platform.dist = dist
        print("✅ 已修复platform.dist问题")

def install_distro():
    """安装distro包作为platform.dist的替代"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'distro'])
        print("✅ 已安装distro包")
    except subprocess.CalledProcessError:
        print("⚠️  安装distro包失败，但不影响修复")

def install_carla():
    """安装CARLA"""
    try:
        # 先应用修复
        fix_platform_dist()
        
        # 安装distro包
        install_distro()
        
        # 重新应用修复（确保distro可用）
        fix_platform_dist()
        
        # 安装CARLA
        print("🚀 开始安装CARLA...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'carla'])
        print("✅ CARLA安装成功！")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ CARLA安装失败: {e}")
        print("💡 请尝试手动下载CARLA包安装")

if __name__ == "__main__":
    install_carla()



