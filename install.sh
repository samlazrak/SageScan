#!/bin/bash

# OCR Image Analyzer Installation Script
# Supports Ubuntu/Debian and macOS

set -e

echo "🚀 OCR Image Analyzer Installation Script"
echo "========================================="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo "📋 Detected: Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo "📋 Detected: macOS"
else
    echo "❌ Unsupported operating system: $OSTYPE"
    exit 1
fi

echo ""

# Check if Python 3.8+ is available
echo "🐍 Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.8"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo "✓ Python $PYTHON_VERSION (compatible)"
else
    echo "❌ Python $PYTHON_VERSION is too old. Please install Python 3.8 or higher."
    exit 1
fi

# Install system dependencies
echo ""
echo "📦 Installing system dependencies..."

if [[ "$OS" == "linux" ]]; then
    # Ubuntu/Debian
    if command -v apt &> /dev/null; then
        echo "Installing Tesseract OCR..."
        sudo apt update
        sudo apt install -y tesseract-ocr tesseract-ocr-eng
        echo "✓ Tesseract OCR installed"
    else
        echo "❌ This script requires apt package manager (Ubuntu/Debian)"
        echo "Please install Tesseract OCR manually for your distribution"
        exit 1
    fi
elif [[ "$OS" == "macos" ]]; then
    # macOS
    if command -v brew &> /dev/null; then
        echo "Installing Tesseract OCR..."
        brew install tesseract
        echo "✓ Tesseract OCR installed"
    else
        echo "❌ Homebrew is required for macOS installation"
        echo "Please install Homebrew from: https://brew.sh/"
        exit 1
    fi
fi

# Check if pip is available
echo ""
echo "📦 Checking pip..."
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not available. Please install pip for Python 3."
    exit 1
fi
echo "✓ pip3 available"

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

echo "✓ Core dependencies installed"

# Check if running on Apple Silicon for MLX
if [[ "$OS" == "macos" ]]; then
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        echo ""
        echo "🚀 Apple Silicon detected! Installing MLX for optimal performance..."
        pip3 install mlx mlx-lm
        echo "✓ MLX installed"
    else
        echo ""
        echo "💡 Intel Mac detected. MLX not installed (Apple Silicon only)"
        echo "   The application will use fallback analysis methods"
    fi
fi

# Make scripts executable
echo ""
echo "🔧 Setting up scripts..."
chmod +x ocr_analyzer.py
chmod +x test_installation.py

# Test installation
echo ""
echo "🧪 Testing installation..."
python3 test_installation.py

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Quick Start:"
echo "  python3 ocr_analyzer.py --help"
echo "  python3 ocr_analyzer.py your_image.jpg"
echo ""
echo "Example usage:"
echo "  python3 ocr_analyzer.py document.jpg receipt.png"
echo "  python3 ocr_analyzer.py --output results.json *.jpg"
echo ""
echo "For more information, see README.md"