#!/usr/bin/env python3
"""
Setup script for OCR Image Analyzer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ocr-analyzer",
    version="1.0.0",
    author="OCR Analyzer Team",
    description="OCR Image Analyzer with MLX-based Summarization and Sentiment Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pillow>=10.0.0",
        "pytesseract>=0.3.10",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "pathlib",
        "typing-extensions",
    ],
    extras_require={
        "mlx": [
            "mlx>=0.12.0",
            "mlx-lm>=0.8.0",
        ],
        "transformers": [
            "transformers>=4.35.0",
            "torch>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ocr-analyzer=ocr_analyzer:main",
        ],
    },
)