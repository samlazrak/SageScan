#!/usr/bin/env python3
"""
Basic test script for OCR Text Summarizer

This script tests basic functionality without requiring full dependencies.
Useful for verifying the application structure and imports.
"""

import sys
import os

def test_imports():
    """Test if the main module can be imported."""
    print("Testing imports...")
    
    try:
        # Test basic Python imports
        import argparse
        import logging
        from pathlib import Path
        from typing import Optional, List
        print("✓ Basic Python modules imported successfully")
        
        # Test if our main module structure is correct
        # We'll do a lightweight test without actual dependencies
        with open('ocr_summarizer.py', 'r') as f:
            content = f.read()
            
        # Check for key classes and functions
        required_components = [
            'class OCRProcessor',
            'class MLXSummarizer', 
            'class OCRSummarizerApp',
            'def main(',
            'extract_text',
            'summarize',
            'process_image',
            'process_text'
        ]
        
        for component in required_components:
            if component in content:
                print(f"✓ Found {component}")
            else:
                print(f"✗ Missing {component}")
                return False
                
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_file_structure():
    """Test if all required files are present."""
    print("\nTesting file structure...")
    
    required_files = [
        'ocr_summarizer.py',
        'requirements.txt',
        'setup.sh',
        'README.md',
        'example_usage.py'
    ]
    
    all_present = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_present = False
    
    return all_present


def test_requirements():
    """Test requirements.txt content."""
    print("\nTesting requirements.txt...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        required_packages = [
            'mlx',
            'mlx-lm',
            'pillow',
            'pytesseract',
            'opencv-python',
            'numpy'
        ]
        
        all_present = True
        for package in required_packages:
            if package in requirements:
                print(f"✓ {package} in requirements")
            else:
                print(f"✗ {package} missing from requirements")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"✗ Error reading requirements.txt: {e}")
        return False


def test_executable_permissions():
    """Test if executable files have proper permissions."""
    print("\nTesting executable permissions...")
    
    executable_files = ['setup.sh']
    
    all_executable = True
    for file in executable_files:
        if os.path.exists(file):
            if os.access(file, os.X_OK):
                print(f"✓ {file} is executable")
            else:
                print(f"✗ {file} is not executable (run: chmod +x {file})")
                all_executable = False
        else:
            print(f"✗ {file} not found")
            all_executable = False
    
    return all_executable


def main():
    """Run all basic tests."""
    print("OCR Text Summarizer - Basic Tests")
    print("=" * 40)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Requirements", test_requirements),
        ("Imports", test_imports),
        ("Executable Permissions", test_executable_permissions)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    print("=" * 40)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ All basic tests PASSED!")
        print("\nNext steps:")
        print("1. Install dependencies: ./setup.sh")
        print("2. Test with text: python3 ocr_summarizer.py --text 'Test text'")
        print("3. Test with image: python3 ocr_summarizer.py --image your_image.jpg")
    else:
        print("✗ Some tests FAILED!")
        print("\nPlease fix the issues above before proceeding.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)