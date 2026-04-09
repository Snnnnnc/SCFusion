#!/usr/bin/env python
"""
检查系统中可用的COM端口
"""

import serial.tools.list_ports

print("="*70)
print("系统中可用的串口设备：")
print("="*70)

ports = list(serial.tools.list_ports.comports())

if not ports:
    print("\n⚠️  未检测到任何串口设备！")
    print("\n可能原因：")
    print("  1. EEG设备未连接")
    print("  2. USB线缆连接不良")
    print("  3. 设备驱动未安装")
    print("\n请检查：")
    print("  - 设备是否已通过USB连接到电脑")
    print("  - 设备是否已开机/通电")
    print("  - 设备管理器中是否显示设备")
else:
    print(f"\n找到 {len(ports)} 个串口设备：\n")
    for idx, port in enumerate(ports, 1):
        print(f"[{idx}] {port.device}")
        print(f"    描述: {port.description}")
        print(f"    硬件ID: {port.hwid}")
        print()
    
    print("="*70)
    print("使用建议：")
    print("="*70)
    print("\n在 eeg_config.py 或测试脚本中使用以上端口号")
    print("例如：")
    print(f'  EEG_COM_PORT = "{ports[0].device}"')

print()

