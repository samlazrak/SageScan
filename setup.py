#!/usr/bin/env python3
"""
Setup script for Text Summarizer with OCR and MLX
"""

from setuptools import setup, find_packages
import os

# Read the README file
current_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="text-summarizer-ocr-mlx",
    version="1.0.0",
    author="AI Assistant",
    description="A Python application that combines OCR with MLX language models for text summarization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/text-summarizer-ocr-mlx",
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
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Multimedia :: Graphics :: Capture :: Scanners",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "text-summarizer=text_summarizer:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "isort>=5.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)