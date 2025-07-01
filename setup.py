#!/usr/bin/env python3
"""
Setup script for OCR MLX Analyzer
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ocr-mlx-analyzer",
    version="1.0.0",
    author="OCR MLX Team",
    description="OCR text extraction with MLX model analysis",
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
        "mlx",
        "mlx-lm",
        "Pillow>=9.0.0",
        "pytesseract>=0.3.10",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "transformers>=4.20.0",
        "torch>=1.12.0",
        "click>=8.0.0",
        "pathlib2>=2.3.0",
    ],
    entry_points={
        "console_scripts": [
            "ocr-mlx=ocr_mlx_analyzer:main",
        ],
    },
)