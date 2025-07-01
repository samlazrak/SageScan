#!/bin/bash

# Text Summarizer with OCR and MLX - Installation Script
# This script installs all necessary dependencies and sets up the application

set -e  # Exit on any error

echo "🔤 Text Summarizer with OCR and MLX - Installation Script"
echo "========================================================"

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' || echo "0.0")
min_version="3.8"

if [ "$(printf '%s\n' "$min_version" "$python_version" | sort -V | head -n1)" != "$min_version" ]; then
    echo "❌ Python 3.8+ is required. Found: $python_version"
    exit 1
fi
echo "✅ Python $python_version found"

# Check if we're on macOS, Linux, or other
OS="$(uname -s)"
case "${OS}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    *)          MACHINE="UNKNOWN:${OS}"
esac
echo "📊 Detected OS: $MACHINE"

# Install system dependencies
echo "📦 Installing system dependencies..."

if [ "$MACHINE" = "Mac" ]; then
    # macOS
    if command -v brew >/dev/null 2>&1; then
        echo "🍺 Installing Tesseract via Homebrew..."
        brew install tesseract
    else
        echo "⚠️ Homebrew not found. Please install Tesseract manually:"
        echo "   Visit: https://github.com/tesseract-ocr/tesseract#installation"
    fi
elif [ "$MACHINE" = "Linux" ]; then
    # Linux - try different package managers
    if command -v apt-get >/dev/null 2>&1; then
        echo "📦 Installing Tesseract via apt..."
        sudo apt-get update
        sudo apt-get install -y tesseract-ocr libtesseract-dev
    elif command -v yum >/dev/null 2>&1; then
        echo "📦 Installing Tesseract via yum..."
        sudo yum install -y tesseract tesseract-devel
    elif command -v dnf >/dev/null 2>&1; then
        echo "📦 Installing Tesseract via dnf..."
        sudo dnf install -y tesseract tesseract-devel
    else
        echo "⚠️ Could not detect package manager. Please install Tesseract manually."
    fi
else
    echo "⚠️ Unsupported OS. Please install Tesseract manually."
fi

# Check if tesseract is available
if command -v tesseract >/dev/null 2>&1; then
    echo "✅ Tesseract OCR installed successfully"
    tesseract --version | head -1
else
    echo "❌ Tesseract installation failed or not in PATH"
    echo "Please install Tesseract manually and ensure it's in your PATH"
    exit 1
fi

# Create virtual environment (optional but recommended)
read -p "🐍 Create a virtual environment? (recommended) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
    echo "💡 To activate later, run: source venv/bin/activate"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip

# Install dependencies from requirements.txt
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Python dependencies installed"
else
    echo "⚠️ requirements.txt not found. Installing core dependencies..."
    pip install mlx mlx-lm opencv-python pytesseract pillow numpy click rich transformers
fi

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import sys
try:
    import cv2
    import pytesseract
    import numpy as np
    from PIL import Image
    import click
    from rich.console import Console
    print('✅ All core dependencies imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)

try:
    import mlx.core as mx
    from mlx_lm import load
    print('✅ MLX dependencies imported successfully')
except ImportError as e:
    print(f'⚠️ MLX import warning: {e}')
    print('MLX may not be optimized for your platform')
"

# Make the main script executable
if [ -f "text_summarizer.py" ]; then
    chmod +x text_summarizer.py
    echo "✅ Made text_summarizer.py executable"
fi

# Test basic functionality
echo "🚀 Testing basic functionality..."
if python3 -c "from text_summarizer import TextSummarizer; print('✅ TextSummarizer class imported successfully')"; then
    echo "✅ Application is ready to use!"
else
    echo "❌ There was an issue with the installation"
    exit 1
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📚 Quick start:"
echo "  # Summarize direct text:"
echo "  python3 text_summarizer.py -t 'Your text here...'"
echo ""
echo "  # Summarize from image:"
echo "  python3 text_summarizer.py -i image.png"
echo ""
echo "  # Run demo:"
echo "  python3 demo.py"
echo ""
echo "  # Get help:"
echo "  python3 text_summarizer.py --help"
echo ""

if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "💡 Remember to activate your virtual environment:"
    echo "   source venv/bin/activate"
    echo ""
fi

echo "📖 For detailed documentation, see README.md"