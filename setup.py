from setuptools import find_packages, setup

setup(
    name="hidiffusion",
    version="0.1.8",
    author="Shen Zhang, Zhaowei Chen, Zhenyu Zhao, Yuhao Chen, Yao Tang, Jiajun Liang",
    url="",
    description="HiDiffusion: A training-free method to increase the resolution and speed of diffusion models.",
    packages=find_packages(),
    include_package_data=True,
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
)