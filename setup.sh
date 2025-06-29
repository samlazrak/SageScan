#!/bin/bash

# ScanSage Setup Script
# This script automates the setup of the ScanSage conda environment

set -e  # Exit on any error

echo "🚀 ScanSage Setup Script"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH"
    echo "Please install conda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_success "Conda found: $(conda --version)"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found. Please run this script from the ScanSage project root."
    exit 1
fi

# Check if Tesseract is installed
if ! command -v tesseract &> /dev/null; then
    print_warning "Tesseract OCR not found in PATH"
    echo "Please install Tesseract OCR:"
    echo "  macOS: brew install tesseract"
    echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr"
    echo "  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
    echo ""
    read -p "Continue with setup anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_success "Tesseract found: $(tesseract --version | head -n 1)"
fi

# Create conda environment
ENV_NAME="ScanSage"
print_status "Creating conda environment: $ENV_NAME"

if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment $ENV_NAME already exists"
    read -p "Remove existing environment and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        print_status "Using existing environment"
    fi
fi

if ! conda env list | grep -q "^$ENV_NAME "; then
    print_status "Creating new conda environment with Python 3.8..."
    conda create -n $ENV_NAME python=3.8 -y
fi

print_success "Conda environment created/verified"

# Activate environment and install dependencies
print_status "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install pip if not available
if ! command -v pip &> /dev/null; then
    print_status "Installing pip in conda environment..."
    conda install pip -y
fi

print_success "Pip available: $(pip --version)"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install build tools if needed
print_status "Installing build tools..."
conda install -y setuptools wheel

# Install Python dependencies with better error handling
print_status "Installing Python dependencies..."

# First, install packages that work better with conda
print_status "Installing core packages via conda..."
conda install -y -c conda-forge numpy pandas scikit-learn scipy pillow opencv pytorch transformers nltk spacy python-dotenv requests pydantic

# Then install remaining packages via pip
print_status "Installing remaining packages via pip..."
pip install pytesseract openai langchain langchain-openai fastapi uvicorn python-multipart

print_success "Python dependencies installed"

# Download NLTK data
print_status "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

print_success "NLTK data downloaded"

# Download spaCy model
print_status "Downloading spaCy model..."
python -m spacy download en_core_web_sm

print_success "spaCy model downloaded"

# Set up environment file
if [ ! -f ".env" ]; then
    if [ -f "env.example" ]; then
        print_status "Creating .env file from template..."
        cp env.example .env
        print_success ".env file created"
        print_warning "Please edit .env file with your API keys for LLM features"
    else
        print_warning "No env.example found. You may need to create a .env file manually."
    fi
else
    print_success ".env file already exists"
fi

# Verify installation
print_status "Verifying installation..."
python -c "
import sys
print(f'Python version: {sys.version}')
import pytesseract
print('✓ pytesseract imported successfully')
import PIL
print('✓ Pillow imported successfully')
import cv2
print('✓ OpenCV imported successfully')
import numpy as np
print('✓ NumPy imported successfully')
import pandas as pd
print('✓ Pandas imported successfully')
import transformers
print('✓ Transformers imported successfully')
import torch
print('✓ PyTorch imported successfully')
import sklearn
print('✓ Scikit-learn imported successfully')
import nltk
print('✓ NLTK imported successfully')
import spacy
print('✓ spaCy imported successfully')
import openai
print('✓ OpenAI imported successfully')
import langchain
print('✓ LangChain imported successfully')
print('\\n🎉 All dependencies verified successfully!')
"

print_success "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate $ENV_NAME"
echo "2. Edit .env file with your API keys (optional)"
echo "3. Run the example: python examples/example_usage.py"
echo "4. Check the documentation: README.md and QUICK_START.md"
echo ""
echo "Happy processing! 🚀" 