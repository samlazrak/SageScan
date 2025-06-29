#!/bin/bash

# ScanSage Fallback Installation Script
# This script uses only conda packages to avoid pip build issues

set -e

echo "🔄 ScanSage Fallback Installation"
echo "=================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found. Please run this script from the ScanSage project root."
    exit 1
fi

ENV_NAME="ScanSage"

# Create environment if it doesn't exist
if ! conda env list | grep -q "^$ENV_NAME "; then
    print_status "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.8 -y
fi

# Activate environment
print_status "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install all packages via conda-forge
print_status "Installing packages via conda-forge (this may take a while)..."
conda install -y -c conda-forge \
    numpy pandas scikit-learn scipy \
    pillow opencv pytorch transformers \
    nltk spacy python-dotenv requests pydantic \
    pytesseract openai fastapi uvicorn python-multipart

# Try to install langchain packages via pip (they're not available in conda)
print_status "Installing LangChain packages via pip..."
pip install langchain langchain-openai

print_success "Fallback installation completed!"

# Download NLTK data
print_status "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download spaCy model
print_status "Downloading spaCy model..."
python -m spacy download en_core_web_sm

print_success "NLP data downloaded"

# Set up environment file
if [ ! -f ".env" ] && [ -f "env.example" ]; then
    print_status "Creating .env file from template..."
    cp env.example .env
    print_warning "Please edit .env file with your API keys for LLM features"
fi

print_success "Fallback installation completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: conda activate $ENV_NAME"
echo "2. Edit .env file with your API keys (optional)"
echo "3. Run the test: python test_installation.py"
echo "4. Run the example: python examples/example_usage.py" 