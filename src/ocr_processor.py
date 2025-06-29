"""
OCR (Optical Character Recognition) processor for extracting text from scanned images.
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Dict, Any, Optional, Tuple, List
import logging
from .utils import validate_image_file

logger = logging.getLogger(__name__)

class OCRProcessor:
    """Handles OCR text extraction from scanned images."""
    
    def __init__(self, tesseract_config: str = '--psm 6'):
        """
        Initialize OCR processor.
        
        Args:
            tesseract_config: Tesseract configuration string
        """
        self.tesseract_config = tesseract_config
        self.supported_languages = ['eng']  # Can be extended for multi-language support
        
    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from scanned image.
        
        Args:
            image_path: Path to the scanned image
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        if not validate_image_file(image_path):
            raise ValueError(f"Unsupported image format: {image_path}")
        
        try:
            # Load and preprocess image
            image = self._load_image(image_path)
            processed_image = self._preprocess_image(image)
            
            # Extract text using Tesseract
            text_data = pytesseract.image_to_data(
                processed_image, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text and confidence scores
            extracted_text = pytesseract.image_to_string(
                processed_image, 
                config=self.tesseract_config
            )
            
            # Calculate average confidence
            confidences = [int(conf) for conf in text_data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Get bounding boxes for debugging
            bounding_boxes = self._extract_bounding_boxes(text_data)
            
            result = {
                'text': extracted_text.strip(),
                'confidence': avg_confidence,
                'word_count': len(extracted_text.split()),
                'bounding_boxes': bounding_boxes,
                'image_path': image_path
            }
            
            logger.info(f"OCR completed for {image_path}. "
                       f"Confidence: {avg_confidence:.2f}%, "
                       f"Words: {result['word_count']}")
            
            return result
            
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {str(e)}")
            raise
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return image
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Apply thresholding to get binary image
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Apply morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _extract_bounding_boxes(self, text_data: Dict) -> List[Dict]:
        """Extract bounding boxes for detected text regions."""
        boxes = []
        n_boxes = len(text_data['level'])
        
        for i in range(n_boxes):
            if int(text_data['conf'][i]) > 0:  # Only include text with confidence > 0
                box = {
                    'text': text_data['text'][i],
                    'confidence': int(text_data['conf'][i]),
                    'bbox': (
                        text_data['left'][i],
                        text_data['top'][i],
                        text_data['left'][i] + text_data['width'][i],
                        text_data['top'][i] + text_data['height'][i]
                    )
                }
                boxes.append(box)
        
        return boxes
    
    def extract_text_with_confidence_threshold(self, image_path: str, min_confidence: float = 60.0) -> Dict[str, Any]:
        """
        Extract text with confidence threshold filtering.
        
        Args:
            image_path: Path to the scanned image
            min_confidence: Minimum confidence threshold (0-100)
            
        Returns:
            Dictionary containing filtered text and metadata
        """
        result = self.extract_text(image_path)
        
        # Filter text by confidence
        if result['confidence'] < min_confidence:
            logger.warning(f"Low OCR confidence ({result['confidence']:.2f}%) for {image_path}")
        
        return result
    
    def batch_extract(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Extract text from multiple images.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of extraction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.extract_text(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {str(e)}")
                results.append({
                    'text': '',
                    'confidence': 0,
                    'word_count': 0,
                    'bounding_boxes': [],
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results 