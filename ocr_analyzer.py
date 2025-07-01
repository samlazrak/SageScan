#!/usr/bin/env python3
"""
OCR Image Analyzer with MLX-based Summarization and Sentiment Analysis

This application uses OCR to extract text from images and then employs
a small MLX language model to summarize the content and analyze sentiment.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

import cv2
import numpy as np
from PIL import Image
import pytesseract
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load, generate
except ImportError:
    print("MLX not available. Please install mlx and mlx-lm for Apple Silicon Macs.")
    print("For other platforms, this will use a fallback approach.")
    mx = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()


class ImagePreprocessor:
    """Handles image preprocessing for better OCR results."""
    
    @staticmethod
    def preprocess_image(image_path: str) -> np.ndarray:
        """Preprocess image to improve OCR accuracy."""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        return processed


class OCRExtractor:
    """Handles text extraction from images using Tesseract OCR."""
    
    def __init__(self):
        # Configure Tesseract parameters for better accuracy
        self.config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?;:\'"()[]{}@#$%^&*-_=+<>/\\ '
    
    def extract_text(self, image_path: str) -> str:
        """Extract text from an image using OCR."""
        try:
            # Preprocess the image
            preprocessor = ImagePreprocessor()
            processed_img = preprocessor.preprocess_image(image_path)
            
            # Convert back to PIL Image for pytesseract
            pil_img = Image.fromarray(processed_img)
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(pil_img, config=self.config)
            
            # Clean up the text
            cleaned_text = self._clean_text(text)
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {image_path}: {e}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Remove excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        cleaned = ' '.join(lines)
        
        # Remove multiple spaces
        import re
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        return cleaned.strip()


class MLXTextAnalyzer:
    """Handles text analysis using MLX models."""
    
    def __init__(self, model_name: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the MLX model and tokenizer."""
        if mx is None:
            console.print("[yellow]MLX not available. Using fallback analysis.[/yellow]")
            return
        
        try:
            console.print(f"[blue]Loading MLX model: {self.model_name}[/blue]")
            self.model, self.tokenizer = load(self.model_name)
            console.print("[green]✓ Model loaded successfully[/green]")
        except Exception as e:
            logger.error(f"Error loading MLX model: {e}")
            console.print(f"[red]Failed to load model: {e}[/red]")
            self.model = None
            self.tokenizer = None
    
    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Summarize the given text."""
        if not text.strip():
            return "No text to summarize."
        
        if self.model is None or self.tokenizer is None:
            return self._fallback_summarize(text, max_length)
        
        try:
            prompt = f"""Please provide a concise summary of the following text in 1-2 sentences:

Text: {text}

Summary:"""
            
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_length,
                temp=0.3
            )
            
            # Extract just the summary part
            summary = response.split("Summary:")[-1].strip()
            return summary if summary else "Unable to generate summary."
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return self._fallback_summarize(text, max_length)
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """Analyze sentiment of the given text."""
        if not text.strip():
            return {"sentiment": "neutral", "confidence": 0.0, "explanation": "No text to analyze"}
        
        if self.model is None or self.tokenizer is None:
            return self._fallback_sentiment(text)
        
        try:
            prompt = f"""Analyze the sentiment of the following text. Respond with only one word: positive, negative, or neutral.

Text: {text}

Sentiment:"""
            
            response = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=10,
                temp=0.1
            )
            
            sentiment = response.split("Sentiment:")[-1].strip().lower()
            
            # Clean up the response
            if "positive" in sentiment:
                sentiment = "positive"
            elif "negative" in sentiment:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            # Get explanation
            explanation_prompt = f"""Briefly explain why the following text has a {sentiment} sentiment in one sentence:

Text: {text}

Explanation:"""
            
            explanation_response = generate(
                self.model,
                self.tokenizer,
                prompt=explanation_prompt,
                max_tokens=50,
                temp=0.3
            )
            
            explanation = explanation_response.split("Explanation:")[-1].strip()
            
            return {
                "sentiment": sentiment,
                "confidence": 0.85,  # MLX models are generally reliable
                "explanation": explanation
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return self._fallback_sentiment(text)
    
    def _fallback_summarize(self, text: str, max_length: int = 100) -> str:
        """Fallback summarization using simple text processing."""
        sentences = text.split('.')
        if len(sentences) <= 2:
            return text
        
        # Take first and last sentences as a simple summary
        summary = f"{sentences[0].strip()}. {sentences[-2].strip() if len(sentences) > 1 else ''}"
        
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
    
    def _fallback_sentiment(self, text: str) -> Dict[str, any]:
        """Fallback sentiment analysis using keyword-based approach."""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'happy', 'love', 'best', 'awesome', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'sad', 'angry', 'disappointed', 'frustrated']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.8, positive_count / 10)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.8, negative_count / 10)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "explanation": f"Keyword-based analysis found {positive_count} positive and {negative_count} negative indicators."
        }


class OCRAnalyzerApp:
    """Main application class that orchestrates OCR and analysis."""
    
    def __init__(self, model_name: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"):
        self.ocr_extractor = OCRExtractor()
        self.text_analyzer = MLXTextAnalyzer(model_name)
    
    def process_image(self, image_path: str) -> Dict[str, any]:
        """Process a single image through OCR and analysis pipeline."""
        result = {
            "image_path": image_path,
            "extracted_text": "",
            "summary": "",
            "sentiment_analysis": {},
            "success": False
        }
        
        try:
            # Extract text using OCR
            with console.status(f"[bold blue]Extracting text from {Path(image_path).name}..."):
                extracted_text = self.ocr_extractor.extract_text(image_path)
            
            if not extracted_text.strip():
                result["summary"] = "No text found in image."
                result["sentiment_analysis"] = {"sentiment": "neutral", "confidence": 0.0, "explanation": "No text to analyze"}
                return result
            
            result["extracted_text"] = extracted_text
            
            # Summarize text
            with console.status("[bold blue]Generating summary..."):
                summary = self.text_analyzer.summarize_text(extracted_text)
            result["summary"] = summary
            
            # Analyze sentiment
            with console.status("[bold blue]Analyzing sentiment..."):
                sentiment_analysis = self.text_analyzer.analyze_sentiment(extracted_text)
            result["sentiment_analysis"] = sentiment_analysis
            
            result["success"] = True
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            result["error"] = str(e)
        
        return result
    
    def process_images(self, image_paths: List[str]) -> List[Dict[str, any]]:
        """Process multiple images."""
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing images...", total=len(image_paths))
            
            for image_path in image_paths:
                result = self.process_image(image_path)
                results.append(result)
                progress.advance(task)
        
        return results
    
    def display_results(self, results: List[Dict[str, any]]):
        """Display analysis results in a formatted table."""
        table = Table(title="OCR Analysis Results")
        table.add_column("Image", style="cyan", no_wrap=True)
        table.add_column("Text Length", justify="right", style="magenta")
        table.add_column("Summary", style="green")
        table.add_column("Sentiment", justify="center", style="yellow")
        table.add_column("Confidence", justify="center", style="blue")
        
        for result in results:
            if result["success"]:
                image_name = Path(result["image_path"]).name
                text_length = str(len(result["extracted_text"]))
                summary = result["summary"][:50] + "..." if len(result["summary"]) > 50 else result["summary"]
                sentiment = result["sentiment_analysis"]["sentiment"]
                confidence = f"{result['sentiment_analysis']['confidence']:.2f}"
                
                # Color code sentiment
                if sentiment == "positive":
                    sentiment = f"[green]{sentiment}[/green]"
                elif sentiment == "negative":
                    sentiment = f"[red]{sentiment}[/red]"
                else:
                    sentiment = f"[yellow]{sentiment}[/yellow]"
                
                table.add_row(image_name, text_length, summary, sentiment, confidence)
            else:
                image_name = Path(result["image_path"]).name
                table.add_row(image_name, "ERROR", result.get("error", "Unknown error"), "N/A", "N/A")
        
        console.print(table)
        
        # Display detailed results for each image
        for result in results:
            if result["success"] and result["extracted_text"].strip():
                self._display_detailed_result(result)
    
    def _display_detailed_result(self, result: Dict[str, any]):
        """Display detailed results for a single image."""
        image_name = Path(result["image_path"]).name
        
        # Create panels for different sections
        text_panel = Panel(
            result["extracted_text"][:500] + ("..." if len(result["extracted_text"]) > 500 else ""),
            title=f"📖 Extracted Text - {image_name}",
            border_style="blue"
        )
        
        summary_panel = Panel(
            result["summary"],
            title="📝 Summary",
            border_style="green"
        )
        
        sentiment_info = result["sentiment_analysis"]
        sentiment_text = (
            f"Sentiment: {sentiment_info['sentiment'].upper()}\n"
            f"Confidence: {sentiment_info['confidence']:.2f}\n"
            f"Explanation: {sentiment_info['explanation']}"
        )
        
        sentiment_panel = Panel(
            sentiment_text,
            title="💭 Sentiment Analysis",
            border_style="yellow"
        )
        
        console.print(text_panel)
        console.print(summary_panel)
        console.print(sentiment_panel)
        console.print()


def validate_image_files(image_paths: List[str]) -> List[str]:
    """Validate and filter image file paths."""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
    valid_paths = []
    
    for path in image_paths:
        path_obj = Path(path)
        if not path_obj.exists():
            console.print(f"[red]Warning: File does not exist: {path}[/red]")
            continue
        
        if path_obj.suffix.lower() not in valid_extensions:
            console.print(f"[red]Warning: Unsupported file type: {path}[/red]")
            continue
        
        valid_paths.append(str(path_obj))
    
    return valid_paths


@click.command()
@click.argument('images', nargs=-1, required=True, type=click.Path())
@click.option('--model', '-m', default="mlx-community/Qwen2.5-0.5B-Instruct-4bit", 
              help='MLX model to use for text analysis')
@click.option('--output', '-o', type=click.Path(), help='Output file to save results (JSON format)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(images, model, output, verbose):
    """
    OCR Image Analyzer with MLX-based Summarization and Sentiment Analysis
    
    Processes images through OCR to extract text, then uses MLX models
    to summarize the content and analyze sentiment.
    
    IMAGES: One or more image files to process
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Display header
    console.print(Panel.fit(
        "[bold blue]OCR Image Analyzer[/bold blue]\n"
        "[dim]Extract text, summarize, and analyze sentiment from images[/dim]",
        border_style="blue"
    ))
    
    # Validate image files
    valid_images = validate_image_files(images)
    if not valid_images:
        console.print("[red]No valid image files provided![/red]")
        sys.exit(1)
    
    console.print(f"[green]Found {len(valid_images)} valid image(s) to process[/green]")
    
    # Initialize the analyzer
    try:
        analyzer = OCRAnalyzerApp(model_name=model)
    except Exception as e:
        console.print(f"[red]Failed to initialize analyzer: {e}[/red]")
        sys.exit(1)
    
    # Process images
    results = analyzer.process_images(valid_images)
    
    # Display results
    analyzer.display_results(results)
    
    # Save results if output file specified
    if output:
        import json
        try:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]Results saved to {output}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to save results: {e}[/red]")
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    console.print(f"\n[bold green]✓ Successfully processed {successful}/{len(results)} images[/bold green]")


if __name__ == "__main__":
    main()