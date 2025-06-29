#!/usr/bin/env python3
"""
ScanSage Installation Test Script
Run this script to verify that all dependencies are properly installed.
"""

import sys
import importlib
from typing import List, Tuple, Optional

def test_import(module_name: str, package_name: Optional[str] = None) -> Tuple[bool, str]:
    """Test if a module can be imported successfully."""
    try:
        if package_name:
            module = importlib.import_module(package_name)
        else:
            module = importlib.import_module(module_name)
        return True, f"✓ {module_name} imported successfully"
    except ImportError as e:
        return False, f"✗ {module_name} import failed: {e}"
    except Exception as e:
        return False, f"✗ {module_name} error: {e}"

def test_ocr_functionality():
    """Test OCR functionality."""
    try:
        import pytesseract
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Create a simple test image with text
        
        # Create a white image
        img = Image.new('RGB', (200, 50), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic text if font not available
        try:
            # Try to use a system font
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
        
        # Draw text
        draw.text((10, 10), "Test OCR", fill='black', font=font)
        
        # Convert to numpy array for OpenCV
        img_array = np.array(img)
        
        # Test OCR
        text = pytesseract.image_to_string(img_array)
        
        if "Test" in text or "OCR" in text:
            return True, "✓ OCR functionality working"
        else:
            return False, f"✗ OCR test failed - extracted: '{text.strip()}'"
            
    except Exception as e:
        return False, f"✗ OCR test failed: {e}"

def main():
    print("🧪 ScanSage Installation Test")
    print("=" * 40)
    
    # Test Python version
    print(f"Python version: {sys.version}")
    print()
    
    # List of modules to test
    modules_to_test = [
        ("pytesseract", None),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("sklearn", "Scikit-learn"),
        ("nltk", "NLTK"),
        ("spacy", "spaCy"),
        ("openai", "OpenAI"),
        ("langchain", "LangChain"),
        ("python-dotenv", "python-dotenv"),
        ("requests", "Requests"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
    ]
    
    # Test imports
    print("Testing module imports:")
    failed_imports = []
    
    for module_name, display_name in modules_to_test:
        success, message = test_import(module_name, display_name)
        print(f"  {message}")
        if not success:
            failed_imports.append(module_name)
    
    print()
    
    # Test OCR functionality
    print("Testing OCR functionality:")
    ocr_success, ocr_message = test_ocr_functionality()
    print(f"  {ocr_message}")
    
    print()
    
    # Summary
    if failed_imports:
        print("❌ Installation Issues Found:")
        for module in failed_imports:
            print(f"  - {module} failed to import")
        print("\nPlease check your installation and try running the setup script again.")
        return False
    else:
        print("✅ All tests passed! ScanSage is ready to use.")
        print("\nNext steps:")
        print("1. Activate your environment: conda activate ScanSage")
        print("2. Run an example: python examples/example_usage.py")
        print("3. Check the documentation: README.md and QUICK_START.md")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 