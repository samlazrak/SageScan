#!/usr/bin/env python3
"""
OCR + MLX Text Analyzer
========================

A Python application that uses OCR to extract text from images and then
uses MLX models for text summarization and sentiment analysis.
"""

import os
import sys
import click
import cv2
import numpy as np
from PIL import Image
import pytesseract
from pathlib import Path
import logging

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
except ImportError:
    print("MLX not available. Please install mlx and mlx-lm packages.")
    sys.exit(1)


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OCRProcessor:
    """Handles OCR processing of images."""
    
    def __init__(self):
        self.setup_tesseract()
    
    def setup_tesseract(self):
        """Setup tesseract OCR engine."""
        # Common tesseract paths
        tesseract_paths = [
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract'
        ]
        
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
        else:
            logger.warning("Tesseract not found in common paths. Assuming it's in PATH.")
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get image with only black and white
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        return sure_bg
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using OCR."""
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image_path)
            
            # Convert back to PIL Image for pytesseract
            pil_img = Image.fromarray(processed_img)
            
            # Extract text
            text = pytesseract.image_to_string(pil_img, config='--psm 6')
            
            # Clean up text
            text = text.strip()
            text = ' '.join(text.split())  # Remove extra whitespace
            
            logger.info(f"Extracted {len(text)} characters from {image_path}")
            return text
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return ""


class MLXAnalyzer:
    """Handles text analysis using MLX models."""
    
    def __init__(self, model_name: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the MLX model and tokenizer."""
        try:
            logger.info(f"Loading MLX model: {self.model_name}")
            self.model, self.tokenizer = load(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Falling back to smaller model...")
            try:
                self.model_name = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
                self.model, self.tokenizer = load(self.model_name)
                logger.info("Fallback model loaded successfully")
            except Exception as e2:
                logger.error(f"Error loading fallback model: {str(e2)}")
                raise
    
    def summarize_text(self, text: str, max_tokens: int = 150) -> str:
        """Generate a summary of the given text."""
        if not text.strip():
            return "No text to summarize."
        
        prompt = f"""Please provide a concise summary of the following text:

Text: {text}

Summary:"""
        
        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=0.3
            )
            
            # Extract just the summary part
            summary = response.strip()
            if "Summary:" in summary:
                summary = summary.split("Summary:")[-1].strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating summary."
    
    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment of the given text."""
        if not text.strip():
            return {"sentiment": "neutral", "confidence": 0.0, "explanation": "No text to analyze."}
        
        prompt = f"""Analyze the sentiment of the following text. Respond with one of: positive, negative, or neutral, followed by a confidence score (0.0-1.0) and a brief explanation.

Text: {text}

Sentiment Analysis:
Sentiment: """
        
        try:
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=100,
                temp=0.1
            )
            
            # Parse the response
            lines = response.strip().split('\n')
            sentiment = "neutral"
            confidence = 0.5
            explanation = "Analysis completed."
            
            for line in lines:
                line = line.strip().lower()
                if 'positive' in line:
                    sentiment = "positive"
                elif 'negative' in line:
                    sentiment = "negative"
                elif 'neutral' in line:
                    sentiment = "neutral"
                
                # Try to extract confidence if mentioned
                if 'confidence' in line or any(char.isdigit() for char in line):
                    import re
                    numbers = re.findall(r'0\.\d+|\d+\.\d+', line)
                    if numbers:
                        try:
                            confidence = float(numbers[0])
                            if confidence > 1.0:
                                confidence = confidence / 100.0  # Convert percentage
                        except ValueError:
                            pass
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "explanation": response.strip()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {"sentiment": "neutral", "confidence": 0.0, "explanation": "Error analyzing sentiment."}


class OCRMLXApp:
    """Main application class."""
    
    def __init__(self, model_name: str = None):
        self.ocr_processor = OCRProcessor()
        if model_name:
            self.mlx_analyzer = MLXAnalyzer(model_name)
        else:
            self.mlx_analyzer = MLXAnalyzer()
    
    def process_image(self, image_path: str) -> dict:
        """Process an image through the complete pipeline."""
        logger.info(f"Processing image: {image_path}")
        
        # Extract text using OCR
        extracted_text = self.ocr_processor.extract_text(image_path)
        
        if not extracted_text.strip():
            return {
                "image_path": image_path,
                "extracted_text": "",
                "summary": "No text found in image.",
                "sentiment": {"sentiment": "neutral", "confidence": 0.0, "explanation": "No text to analyze."}
            }
        
        # Generate summary
        summary = self.mlx_analyzer.summarize_text(extracted_text)
        
        # Analyze sentiment
        sentiment = self.mlx_analyzer.analyze_sentiment(extracted_text)
        
        return {
            "image_path": image_path,
            "extracted_text": extracted_text,
            "summary": summary,
            "sentiment": sentiment
        }


@click.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--model', '-m', default=None, help='MLX model name to use')
@click.option('--output', '-o', type=click.Path(), help='Output file to save results')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(image_path, model, output, verbose):
    """
    OCR + MLX Text Analyzer
    
    Extract text from images using OCR and analyze with MLX models.
    
    IMAGE_PATH: Path to the image file to process
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize the application
        app = OCRMLXApp(model_name=model)
        
        # Process the image
        results = app.process_image(image_path)
        
        # Display results
        print("\n" + "="*60)
        print(f"OCR + MLX Analysis Results")
        print("="*60)
        print(f"Image: {results['image_path']}")
        print("\nExtracted Text:")
        print("-" * 20)
        print(results['extracted_text'])
        print("\nSummary:")
        print("-" * 10)
        print(results['summary'])
        print("\nSentiment Analysis:")
        print("-" * 20)
        sentiment_data = results['sentiment']
        print(f"Sentiment: {sentiment_data['sentiment'].upper()}")
        print(f"Confidence: {sentiment_data['confidence']:.2f}")
        print(f"Explanation: {sentiment_data['explanation']}")
        print("="*60)
        
        # Save to file if requested
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output}")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()