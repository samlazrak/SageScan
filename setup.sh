#!/bin/bash

# OCR Text Summarizer Setup Script
# This script installs system dependencies and Python packages

set -e

echo "=== OCR Text Summarizer Setup ==="
echo

# Update system packages
echo "Updating system packages..."
sudo apt-get update

# Install Tesseract OCR and dependencies
echo "Installing Tesseract OCR..."
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev

# Install system dependencies for OpenCV and image processing
echo "Installing system dependencies..."
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0

# Upgrade pip
echo "Upgrading pip..."
pip3 install --upgrade pip

# Install Python dependencies
echo "Installing Python packages..."
pip3 install -r requirements.txt

echo
echo "=== Setup Complete ==="
echo
echo "You can now run the OCR summarizer with:"
echo "  python3 ocr_summarizer.py --text 'Your text here'"
echo "  python3 ocr_summarizer.py --image path/to/image.jpg"
echo
echo "For more usage examples, run:"
echo "  python3 ocr_summarizer.py --help"