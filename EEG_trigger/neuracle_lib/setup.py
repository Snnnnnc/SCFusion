from setuptools import setup, find_packages

setup(
    name="neuracle_lib",            # 你的库名
    version="0.1",
    packages=find_packages(), # 自动查找含 __init__.py 的包
    install_requires=[],      # 如果有依赖可以加这里
    author="Neuracle",
    description="A custom Python library for data handling and triggers.",
)
