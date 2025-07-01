#!/usr/bin/env python3
"""
Test script to verify OCR Analyzer installation
"""

import sys
import importlib
from pathlib import Path

def test_import(module_name, friendly_name=None):
    """Test if a module can be imported."""
    if friendly_name is None:
        friendly_name = module_name
    
    try:
        importlib.import_module(module_name)
        print(f"✓ {friendly_name}")
        return True
    except ImportError as e:
        print(f"✗ {friendly_name}: {e}")
        return False

def test_tesseract():
    """Test if Tesseract is available."""
    try:
        import pytesseract
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_image = Image.fromarray(np.ones((100, 300, 3), dtype=np.uint8) * 255)
        
        # Try to run Tesseract
        pytesseract.image_to_string(test_image)
        print("✓ Tesseract OCR")
        return True
    except Exception as e:
        print(f"✗ Tesseract OCR: {e}")
        return False

def main():
    """Run installation tests."""
    print("🔍 Testing OCR Analyzer Installation\n")
    
    print("Core Dependencies:")
    tests = [
        ("PIL", "Pillow (PIL)"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("click", "Click"),
        ("rich", "Rich"),
    ]
    
    core_success = all(test_import(module, name) for module, name in tests)
    
    print("\nOCR Dependencies:")
    ocr_success = all([
        test_import("pytesseract", "PyTesseract"),
        test_tesseract()
    ])
    
    print("\nMLX Dependencies (Apple Silicon only):")
    mlx_available = all([
        test_import("mlx.core", "MLX Core"),
        test_import("mlx_lm", "MLX Language Models"),
    ])
    
    if not mlx_available:
        print("ℹ️  MLX not available - fallback analysis will be used")
    
    print("\nMain Application:")
    app_success = test_import("ocr_analyzer", "OCR Analyzer")
    
    if not Path("ocr_analyzer.py").exists():
        print("✗ ocr_analyzer.py not found in current directory")
        app_success = False
    else:
        print("✓ ocr_analyzer.py found")
    
    print("\n" + "="*50)
    
    if core_success and ocr_success:
        print("🎉 Installation successful!")
        print("\nYou can now run:")
        print("  python ocr_analyzer.py --help")
        print("  python ocr_analyzer.py your_image.jpg")
        
        if mlx_available:
            print("\n🚀 MLX support detected - you'll get the best performance!")
        else:
            print("\n💡 MLX not available - using fallback analysis")
            print("   Install MLX for better performance on Apple Silicon:")
            print("   pip install mlx mlx-lm")
        
        return 0
    else:
        print("❌ Installation incomplete!")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
        
        if not ocr_success:
            print("\nFor Tesseract OCR:")
            print("  macOS: brew install tesseract")
            print("  Ubuntu: sudo apt install tesseract-ocr")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())