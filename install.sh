#!/bin/bash

# OCR + MLX Text Analyzer Installation Script
# ===========================================

set -e  # Exit on any error

echo "🔧 OCR + MLX Text Analyzer Installation"
echo "======================================="

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

# Check if running on macOS with Apple Silicon
check_system() {
    print_status "Checking system compatibility..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if [[ $(uname -m) == "arm64" ]]; then
            print_success "Apple Silicon Mac detected - MLX supported"
            SYSTEM="macos_arm"
        else
            print_warning "Intel Mac detected - MLX may have limited support"
            SYSTEM="macos_intel"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_warning "Linux detected - MLX may have limited support"
        SYSTEM="linux"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
}

# Check Python version
check_python() {
    print_status "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8 or later."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    REQUIRED_VERSION="3.8"
    
    if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]]; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python $PYTHON_VERSION found, but Python $REQUIRED_VERSION or later is required"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    case $SYSTEM in
        "macos_arm"|"macos_intel")
            if command -v brew &> /dev/null; then
                print_status "Installing Tesseract OCR via Homebrew..."
                brew install tesseract
                print_success "Tesseract OCR installed"
            else
                print_error "Homebrew not found. Please install Homebrew first: https://brew.sh"
                exit 1
            fi
            ;;
        "linux")
            if command -v apt-get &> /dev/null; then
                print_status "Installing Tesseract OCR via apt..."
                sudo apt-get update
                sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
                print_success "Tesseract OCR installed"
            elif command -v yum &> /dev/null; then
                print_status "Installing Tesseract OCR via yum..."
                sudo yum install -y tesseract
                print_success "Tesseract OCR installed"
            elif command -v dnf &> /dev/null; then
                print_status "Installing Tesseract OCR via dnf..."
                sudo dnf install -y tesseract
                print_success "Tesseract OCR installed"
            else
                print_error "Package manager not found. Please install Tesseract OCR manually."
                exit 1
            fi
            ;;
    esac
}

# Create virtual environment
create_venv() {
    print_status "Creating Python virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing old one..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    print_success "Virtual environment created and activated"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Install dependencies from requirements.txt
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed from requirements.txt"
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Check if tesseract is available
    if command -v tesseract &> /dev/null; then
        TESSERACT_VERSION=$(tesseract --version | head -n1)
        print_success "Tesseract found: $TESSERACT_VERSION"
    else
        print_error "Tesseract not found in PATH"
        exit 1
    fi
    
    # Check Python imports
    python3 -c "
import sys
try:
    import cv2
    print('✓ OpenCV imported successfully')
except ImportError as e:
    print('✗ OpenCV import failed:', e)
    sys.exit(1)

try:
    import PIL
    print('✓ Pillow imported successfully')
except ImportError as e:
    print('✗ Pillow import failed:', e)
    sys.exit(1)

try:
    import pytesseract
    print('✓ pytesseract imported successfully')
except ImportError as e:
    print('✗ pytesseract import failed:', e)
    sys.exit(1)

try:
    import mlx.core
    import mlx_lm
    print('✓ MLX imported successfully')
except ImportError as e:
    print('⚠ MLX import failed (may be expected on non-Apple Silicon):', e)

try:
    import click
    print('✓ Click imported successfully')
except ImportError as e:
    print('✗ Click import failed:', e)
    sys.exit(1)
"
    
    if [ $? -eq 0 ]; then
        print_success "All imports verified"
    else
        print_error "Some imports failed"
        exit 1
    fi
}

# Test basic functionality
test_functionality() {
    print_status "Testing basic functionality..."
    
    # Test CLI help
    if python3 ocr_mlx_analyzer.py --help > /dev/null 2>&1; then
        print_success "Main application CLI working"
    else
        print_error "Main application CLI test failed"
        exit 1
    fi
    
    if python3 batch_processor.py --help > /dev/null 2>&1; then
        print_success "Batch processor CLI working"
    else
        print_error "Batch processor CLI test failed"
        exit 1
    fi
}

# Main installation process
main() {
    echo
    print_status "Starting installation process..."
    echo
    
    check_system
    check_python
    install_system_deps
    create_venv
    install_python_deps
    verify_installation
    test_functionality
    
    echo
    print_success "🎉 Installation completed successfully!"
    echo
    echo "📖 Usage Instructions:"
    echo "====================="
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate"
    echo
    echo "2. Process a single image:"
    echo "   python3 ocr_mlx_analyzer.py path/to/image.jpg"
    echo
    echo "3. Process multiple images:"
    echo "   python3 batch_processor.py path/to/images/ --output-dir results/"
    echo
    echo "4. View help for more options:"
    echo "   python3 ocr_mlx_analyzer.py --help"
    echo "   python3 batch_processor.py --help"
    echo
    echo "📚 See README.md for detailed documentation."
    echo
}

# Run installation if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi