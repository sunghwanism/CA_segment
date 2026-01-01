# setup.py
from setuptools import setup, find_packages

setup(
    name="CA-Segmentation",
    version="0.1",
    packages=find_packages(), # __init__.py가 있는 모든 폴더를 패키지로 인식
)