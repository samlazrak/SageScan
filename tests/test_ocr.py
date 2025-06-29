"""
Unit tests for OCR processor.
"""

import unittest
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.ocr_processor import OCRProcessor

class TestOCRProcessor(unittest.TestCase):
    """Test cases for OCR processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ocr_processor = OCRProcessor()
        self.test_image_path = self._create_test_image()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def _create_test_image(self):
        """Create a test image with text for OCR testing."""
        # Create a simple image with text
        img = Image.new('RGB', (400, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some text
        try:
            # Try to use a default font
            font = ImageFont.load_default()
        except:
            # Fallback to basic text
            font = None
        
        draw.text((50, 50), "Hello World", fill='black', font=font)
        draw.text((50, 100), "This is a test note", fill='black', font=font)
        draw.text((50, 150), "For OCR processing", fill='black', font=font)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        img.save(temp_file.name, 'JPEG')
        return temp_file.name
    
    def test_ocr_processor_initialization(self):
        """Test OCR processor initialization."""
        processor = OCRProcessor()
        self.assertIsNotNone(processor)
        self.assertEqual(processor.tesseract_config, '--psm 6')
        self.assertIn('eng', processor.supported_languages)
    
    def test_extract_text(self):
        """Test text extraction from image."""
        result = self.ocr_processor.extract_text(self.test_image_path)
        
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertIn('confidence', result)
        self.assertIn('word_count', result)
        self.assertIn('bounding_boxes', result)
        self.assertIn('image_path', result)
        
        # Check that some text was extracted
        self.assertIsInstance(result['text'], str)
        self.assertIsInstance(result['confidence'], (int, float))
        self.assertIsInstance(result['word_count'], int)
        self.assertIsInstance(result['bounding_boxes'], list)
    
    def test_extract_text_with_confidence_threshold(self):
        """Test text extraction with confidence threshold."""
        result = self.ocr_processor.extract_text_with_confidence_threshold(
            self.test_image_path, 
            min_confidence=0.0
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertIn('confidence', result)
    
    def test_batch_extract(self):
        """Test batch text extraction."""
        image_paths = [self.test_image_path]
        results = self.ocr_processor.batch_extract(image_paths)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], dict)
        self.assertIn('text', results[0])
    
    def test_invalid_image_format(self):
        """Test handling of invalid image format."""
        # Create a text file instead of image
        temp_file = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        temp_file.write(b"This is not an image")
        temp_file.close()
        
        with self.assertRaises(ValueError):
            self.ocr_processor.extract_text(temp_file.name)
        
        os.remove(temp_file.name)
    
    def test_nonexistent_image(self):
        """Test handling of nonexistent image file."""
        with self.assertRaises(Exception):
            self.ocr_processor.extract_text("nonexistent_image.jpg")
    
    def test_image_preprocessing(self):
        """Test image preprocessing functionality."""
        # Test that preprocessing methods exist and are callable
        processor = OCRProcessor()
        
        # Create a simple test image
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test preprocessing
        processed = processor._preprocess_image(test_image)
        self.assertIsInstance(processed, np.ndarray)
    
    def test_bounding_box_extraction(self):
        """Test bounding box extraction."""
        # Create a test result structure
        test_data = {
            'level': [1, 2, 3],
            'conf': ['90', '85', '0'],  # One with confidence 0
            'text': ['Hello', 'World', ''],
            'left': [10, 20, 30],
            'top': [10, 20, 30],
            'width': [50, 60, 70],
            'height': [20, 25, 30]
        }
        
        boxes = self.ocr_processor._extract_bounding_boxes(test_data)
        
        self.assertIsInstance(boxes, list)
        # Should only include boxes with confidence > 0
        self.assertEqual(len(boxes), 2)
        
        for box in boxes:
            self.assertIn('text', box)
            self.assertIn('confidence', box)
            self.assertIn('bbox', box)

if __name__ == '__main__':
    unittest.main() 