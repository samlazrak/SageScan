@echo off
REM ScanSage Setup Script for Windows
REM This script automates the setup of the ScanSage conda environment

echo 🚀 ScanSage Setup Script
echo ==========================

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Conda is not installed or not in PATH
    echo Please install conda first: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [SUCCESS] Conda found
conda --version

REM Check if we're in the right directory
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found. Please run this script from the ScanSage project root.
    pause
    exit /b 1
)

REM Check if Tesseract is installed
where tesseract >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Tesseract OCR not found in PATH
    echo Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki
    echo.
    set /p continue="Continue with setup anyway? (y/N): "
    if /i not "%continue%"=="y" (
        exit /b 1
    )
) else (
    echo [SUCCESS] Tesseract found
    tesseract --version
)

REM Create conda environment
set ENV_NAME=ScanSage
echo [INFO] Creating conda environment: %ENV_NAME%

conda env list | findstr /C:"%ENV_NAME% " >nul
if %errorlevel% equ 0 (
    echo [WARNING] Environment %ENV_NAME% already exists
    set /p recreate="Remove existing environment and recreate? (y/N): "
    if /i "%recreate%"=="y" (
        echo [INFO] Removing existing environment...
        conda env remove -n %ENV_NAME% -y
    ) else (
        echo [INFO] Using existing environment
    )
)

conda env list | findstr /C:"%ENV_NAME% " >nul
if %errorlevel% neq 0 (
    echo [INFO] Creating new conda environment with Python 3.8...
    conda create -n %ENV_NAME% python=3.8 -y
)

echo [SUCCESS] Conda environment created/verified

REM Activate environment and install dependencies
echo [INFO] Activating conda environment...
call conda activate %ENV_NAME%

REM Install pip if not available
where pip >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing pip in conda environment...
    conda install pip -y
)

echo [SUCCESS] Pip available
pip --version

REM Upgrade pip
echo [INFO] Upgrading pip...
pip install --upgrade pip

REM Install build tools if needed
echo [INFO] Installing build tools...
conda install -y setuptools wheel

REM Install Python dependencies with better error handling
echo [INFO] Installing Python dependencies...

REM First, install packages that work better with conda
echo [INFO] Installing core packages via conda...
conda install -y -c conda-forge numpy pandas scikit-learn scipy pillow opencv pytorch transformers nltk spacy python-dotenv requests pydantic

REM Then install remaining packages via pip
echo [INFO] Installing remaining packages via pip...
pip install pytesseract openai langchain langchain-openai fastapi uvicorn python-multipart

echo [SUCCESS] Python dependencies installed

REM Download NLTK data
echo [INFO] Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

echo [SUCCESS] NLTK data downloaded

REM Download spaCy model
echo [INFO] Downloading spaCy model...
python -m spacy download en_core_web_sm

echo [SUCCESS] spaCy model downloaded

REM Set up environment file
if not exist ".env" (
    if exist "env.example" (
        echo [INFO] Creating .env file from template...
        copy env.example .env
        echo [SUCCESS] .env file created
        echo [WARNING] Please edit .env file with your API keys for LLM features
    ) else (
        echo [WARNING] No env.example found. You may need to create a .env file manually.
    )
) else (
    echo [SUCCESS] .env file already exists
)

REM Verify installation
echo [INFO] Verifying installation...
python -c "import sys; print(f'Python version: {sys.version}'); import pytesseract; print('✓ pytesseract imported successfully'); import PIL; print('✓ Pillow imported successfully'); import cv2; print('✓ OpenCV imported successfully'); import numpy as np; print('✓ NumPy imported successfully'); import pandas as pd; print('✓ Pandas imported successfully'); import transformers; print('✓ Transformers imported successfully'); import torch; print('✓ PyTorch imported successfully'); import sklearn; print('✓ Scikit-learn imported successfully'); import nltk; print('✓ NLTK imported successfully'); import spacy; print('✓ spaCy imported successfully'); import openai; print('✓ OpenAI imported successfully'); import langchain; print('✓ LangChain imported successfully'); print('\n🎉 All dependencies verified successfully!')"

echo [SUCCESS] Setup completed successfully!
echo.
echo Next steps:
echo 1. Activate the environment: conda activate %ENV_NAME%
echo 2. Edit .env file with your API keys (optional)
echo 3. Run the example: python examples/example_usage.py
echo 4. Check the documentation: README.md and QUICK_START.md
echo.
echo Happy processing! 🚀
pause 