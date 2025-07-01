#!/usr/bin/env python3
"""
OCR Text Summarizer using MLX Language Models

This application extracts text from images using OCR (Optical Character Recognition)
and then summarizes the extracted text using a small language model in MLX format.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List
import logging

# Image processing and OCR
import cv2
import numpy as np
from PIL import Image
import pytesseract

# MLX for language model inference
import mlx.core as mx
from mlx_lm import load, generate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """Handles optical character recognition from images."""
    
    def __init__(self):
        """Initialize the OCR processor."""
        # Configure tesseract if needed
        # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Adjust path if needed
        
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Apply threshold to get better text contrast
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from image using OCR.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Extracted text as string
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Convert to PIL Image for pytesseract
            pil_image = Image.fromarray(processed_image)
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(pil_image, config='--psm 6')
            
            # Clean up the text
            cleaned_text = self._clean_text(text)
            
            logger.info(f"Extracted {len(cleaned_text)} characters from {image_path}")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned = '\n'.join(lines)
        
        # Remove non-printable characters except newlines and tabs
        cleaned = ''.join(char for char in cleaned if char.isprintable() or char in '\n\t')
        
        return cleaned.strip()


class MLXSummarizer:
    """Handles text summarization using MLX language models."""
    
    def __init__(self, model_name: str = "mlx-community/Llama-3.2-1B-Instruct-4bit"):
        """
        Initialize the MLX summarizer.
        
        Args:
            model_name: Name of the MLX model to use for summarization
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the MLX language model and tokenizer."""
        try:
            logger.info(f"Loading MLX model: {self.model_name}")
            self.model, self.tokenizer = load(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            logger.info("Trying fallback model...")
            # Try a smaller fallback model
            try:
                self.model_name = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
                self.model, self.tokenizer = load(self.model_name)
                logger.info(f"Fallback model {self.model_name} loaded successfully")
            except Exception as fallback_e:
                logger.error(f"Error loading fallback model: {str(fallback_e)}")
                raise
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """
        Summarize the input text using the MLX language model.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary
            
        Returns:
            Summarized text
        """
        if not text.strip():
            return "No text provided for summarization."
        
        # Create a prompt for summarization
        prompt = self._create_summary_prompt(text, max_length)
        
        try:
            logger.info("Generating summary...")
            
            # Generate summary using MLX
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_length + 50,  # Allow some buffer
                temp=0.7,
                verbose=False
            )
            
            # Extract and clean the summary
            summary = self._extract_summary(response, prompt)
            
            logger.info(f"Generated summary of {len(summary)} characters")
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def _create_summary_prompt(self, text: str, max_length: int) -> str:
        """
        Create a prompt for text summarization.
        
        Args:
            text: Input text to summarize
            max_length: Target length for the summary
            
        Returns:
            Formatted prompt
        """
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant that creates concise, accurate summaries of text content. Your summaries should:
- Capture the main points and key information
- Be clear and well-structured
- Be approximately {max_length} words or less
- Maintain the original meaning and context

<|eot_id|><|start_header_id|>user<|end_header_id|>

Please summarize the following text:

{text}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt
    
    def _extract_summary(self, response: str, prompt: str) -> str:
        """
        Extract the summary from the model response.
        
        Args:
            response: Raw response from the model
            prompt: Original prompt used
            
        Returns:
            Cleaned summary text
        """
        # Remove the prompt from the response
        if prompt in response:
            summary = response.replace(prompt, "").strip()
        else:
            summary = response.strip()
        
        # Remove any special tokens or artifacts
        summary = summary.replace("<|eot_id|>", "").strip()
        
        # Take only the first paragraph if multiple paragraphs
        summary = summary.split('\n\n')[0].strip()
        
        return summary if summary else "Unable to generate summary."


class OCRSummarizerApp:
    """Main application class that orchestrates OCR and summarization."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the OCR Summarizer application.
        
        Args:
            model_name: Optional MLX model name to use
        """
        self.ocr_processor = OCRProcessor()
        self.summarizer = MLXSummarizer(model_name) if model_name else MLXSummarizer()
    
    def process_image(self, image_path: str, output_file: Optional[str] = None) -> dict:
        """
        Process an image: extract text and generate summary.
        
        Args:
            image_path: Path to the input image
            output_file: Optional path to save results
            
        Returns:
            Dictionary containing extracted text and summary
        """
        logger.info(f"Processing image: {image_path}")
        
        # Validate input file
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Extract text from image
        extracted_text = self.ocr_processor.extract_text(image_path)
        
        if not extracted_text.strip():
            logger.warning("No text extracted from image")
            return {
                "image_path": image_path,
                "extracted_text": "",
                "summary": "No text found in the image.",
                "status": "no_text"
            }
        
        # Generate summary
        summary = self.summarizer.summarize(extracted_text)
        
        result = {
            "image_path": image_path,
            "extracted_text": extracted_text,
            "summary": summary,
            "status": "success"
        }
        
        # Save results if output file specified
        if output_file:
            self._save_results(result, output_file)
        
        return result
    
    def process_text(self, text: str, output_file: Optional[str] = None) -> dict:
        """
        Process raw text: generate summary directly.
        
        Args:
            text: Input text to summarize
            output_file: Optional path to save results
            
        Returns:
            Dictionary containing text and summary
        """
        logger.info("Processing raw text input")
        
        if not text.strip():
            return {
                "extracted_text": "",
                "summary": "No text provided.",
                "status": "no_text"
            }
        
        # Generate summary
        summary = self.summarizer.summarize(text)
        
        result = {
            "extracted_text": text,
            "summary": summary,
            "status": "success"
        }
        
        # Save results if output file specified
        if output_file:
            self._save_results(result, output_file)
        
        return result
    
    def _save_results(self, result: dict, output_file: str):
        """
        Save processing results to a file.
        
        Args:
            result: Processing results dictionary
            output_file: Path to save the results
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=== OCR & SUMMARIZATION RESULTS ===\n\n")
                
                if "image_path" in result:
                    f.write(f"Source Image: {result['image_path']}\n\n")
                
                f.write("EXTRACTED TEXT:\n")
                f.write("-" * 50 + "\n")
                f.write(result['extracted_text'])
                f.write("\n\n")
                
                f.write("SUMMARY:\n")
                f.write("-" * 50 + "\n")
                f.write(result['summary'])
                f.write("\n")
            
            logger.info(f"Results saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {str(e)}")


def main():
    """Main function to run the OCR Summarizer application."""
    parser = argparse.ArgumentParser(
        description="Extract text from images using OCR and summarize using MLX language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ocr_summarizer.py --image document.jpg
  python ocr_summarizer.py --image scan.png --output results.txt
  python ocr_summarizer.py --text "Your text here" --output summary.txt
  python ocr_summarizer.py --image photo.jpg --model "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image", "-i",
        help="Path to image file for OCR text extraction"
    )
    input_group.add_argument(
        "--text", "-t",
        help="Raw text to summarize (instead of using OCR)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        help="Output file to save results (optional)"
    )
    
    # Model options
    parser.add_argument(
        "--model", "-m",
        default="mlx-community/Llama-3.2-1B-Instruct-4bit",
        help="MLX model name for summarization (default: mlx-community/Llama-3.2-1B-Instruct-4bit)"
    )
    
    # Processing options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize application
        app = OCRSummarizerApp(model_name=args.model)
        
        # Process input
        if args.image:
            result = app.process_image(args.image, args.output)
        else:
            result = app.process_text(args.text, args.output)
        
        # Display results
        print("\n" + "=" * 60)
        print("OCR & SUMMARIZATION RESULTS")
        print("=" * 60)
        
        if result["status"] == "success":
            if args.image:
                print(f"\nSource Image: {args.image}")
            
            print(f"\nExtracted Text ({len(result['extracted_text'])} characters):")
            print("-" * 40)
            print(result['extracted_text'][:500] + ("..." if len(result['extracted_text']) > 500 else ""))
            
            print(f"\nSummary:")
            print("-" * 40)
            print(result['summary'])
            
        else:
            print(f"\nStatus: {result['status']}")
            print(result['summary'])
        
        print("\n" + "=" * 60)
        
        if args.output:
            print(f"Full results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()