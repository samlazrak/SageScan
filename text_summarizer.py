#!/usr/bin/env python3
"""
Text Summarizer Application using OCR and MLX Language Models

This application can:
1. Extract text from images using OCR
2. Summarize text using small language models in MLX format
3. Process both direct text input and image files
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union
import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import cv2
import numpy as np
from PIL import Image
import pytesseract

try:
    import mlx.core as mx
    from mlx_lm import load, generate
except ImportError:
    print("MLX not available. Please install mlx and mlx-lm for model functionality.")
    mx = None

console = Console()

class TextSummarizer:
    """Main class for handling OCR and text summarization"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load the MLX language model"""
        if mx is None:
            raise ImportError("MLX is not available. Please install mlx and mlx-lm.")
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading MLX model...", total=None)
            try:
                self.model, self.tokenizer = load(self.model_path)
                progress.update(task, description="✅ Model loaded successfully")
            except Exception as e:
                progress.update(task, description=f"❌ Failed to load model: {e}")
                raise
                
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
        
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        console.print(f"🔍 Processing image: {image_path}")
        
        try:
            # Preprocess the image
            processed_img = self.preprocess_image(image_path)
            
            # Convert numpy array to PIL Image for pytesseract
            pil_img = Image.fromarray(processed_img)
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(pil_img, config='--psm 6')
            
            if not text.strip():
                console.print("⚠️ No text found in image")
                return ""
                
            console.print(f"✅ Extracted {len(text)} characters from image")
            return text.strip()
            
        except Exception as e:
            console.print(f"❌ Error extracting text from image: {e}")
            raise
            
    def summarize_text(self, text: str, max_tokens: int = 150) -> str:
        """Summarize text using MLX language model"""
        if not text.strip():
            return "No text to summarize."
            
        if self.model is None or self.tokenizer is None:
            self.load_model()
            
        # Create a prompt for summarization
        prompt = f"""Please provide a concise summary of the following text:

Text: {text}

Summary:"""
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating summary...", total=None)
            
            try:
                # Generate summary using MLX
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False
                )
                
                # Extract just the summary part
                summary = response.strip()
                if "Summary:" in summary:
                    summary = summary.split("Summary:")[-1].strip()
                    
                progress.update(task, description="✅ Summary generated")
                return summary
                
            except Exception as e:
                progress.update(task, description=f"❌ Failed to generate summary: {e}")
                raise
                
    def process_file(self, file_path: str, max_tokens: int = 150) -> dict:
        """Process a file (either text or image) and return summary"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Check file type
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        text_extensions = {'.txt', '.md', '.rst'}
        
        if file_path.suffix.lower() in image_extensions:
            # Extract text from image
            extracted_text = self.extract_text_from_image(str(file_path))
            source_type = "image"
        elif file_path.suffix.lower() in text_extensions:
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                extracted_text = f.read()
            source_type = "text_file"
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
        # Generate summary
        summary = self.summarize_text(extracted_text, max_tokens)
        
        return {
            "source_type": source_type,
            "original_text": extracted_text,
            "summary": summary,
            "file_path": str(file_path)
        }

@click.command()
@click.option(
    '--input', '-i', 
    type=str, 
    help='Input file path (image or text file) or direct text string'
)
@click.option(
    '--text', '-t',
    type=str,
    help='Direct text input to summarize'
)
@click.option(
    '--model', '-m',
    type=str,
    default="mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit",
    help='MLX model path or identifier'
)
@click.option(
    '--max-tokens',
    type=int,
    default=150,
    help='Maximum tokens for summary generation'
)
@click.option(
    '--output', '-o',
    type=str,
    help='Output file to save the summary'
)
def main(input: Optional[str], text: Optional[str], model: str, max_tokens: int, output: Optional[str]):
    """
    Text Summarizer using OCR and MLX Language Models
    
    Examples:
    \b
    # Summarize text from an image
    python text_summarizer.py -i image.png
    
    # Summarize direct text input
    python text_summarizer.py -t "Your text here..."
    
    # Summarize text file
    python text_summarizer.py -i document.txt
    
    # Use custom model and save output
    python text_summarizer.py -i image.jpg -m custom-model-path -o summary.txt
    """
    
    console.print(Panel.fit("🔤 Text Summarizer with OCR and MLX", style="bold blue"))
    
    # Validate input
    if not input and not text:
        console.print("❌ Please provide either --input (file path) or --text (direct text)")
        sys.exit(1)
        
    # Initialize summarizer
    summarizer = TextSummarizer(model_path=model)
    
    try:
        if text:
            # Process direct text input
            console.print("📝 Processing direct text input...")
            summary = summarizer.summarize_text(text, max_tokens)
            result = {
                "source_type": "direct_text",
                "original_text": text,
                "summary": summary,
                "file_path": None
            }
        else:
            # Process file input
            console.print(f"📁 Processing file: {input}")
            result = summarizer.process_file(input, max_tokens)
            
        # Display results
        console.print("\n" + "="*60)
        console.print(Panel(
            f"[bold]Source:[/bold] {result['source_type']}\n"
            f"[bold]Original length:[/bold] {len(result['original_text'])} characters\n"
            f"[bold]Summary length:[/bold] {len(result['summary'])} characters",
            title="📊 Processing Results",
            style="green"
        ))
        
        console.print(Panel(
            result['summary'],
            title="📋 Summary",
            style="cyan"
        ))
        
        # Save output if requested
        if output:
            output_data = f"""# Text Summarization Result

## Source Information
- **Type:** {result['source_type']}
- **File:** {result['file_path'] or 'Direct input'}
- **Original length:** {len(result['original_text'])} characters
- **Summary length:** {len(result['summary'])} characters

## Original Text
{result['original_text']}

## Summary
{result['summary']}
"""
            with open(output, 'w', encoding='utf-8') as f:
                f.write(output_data)
            console.print(f"💾 Results saved to: {output}")
            
    except Exception as e:
        console.print(f"❌ Error: {e}")
        sys.exit(1)

# Example text for demonstration
EXAMPLE_TEXT = """## Using the think tool

Before taking any action or responding to the user after receiving tool results, use the think tool as a scratchpad to:
- List the specific rules that apply to the current request
- Check if all required information is collected
- Verify that the planned action complies with all policies
- Iterate over tool results for correctness 

Here are some examples of what to iterate over inside the think tool:

User wants to cancel flight ABC123
- Need to verify: user ID, reservation ID, reason
- Check cancellation rules:
  * Is it within 24h of booking?
  * If not, check ticket class and insurance
- Verify no segments flown or are in the past
- Plan: collect missing info, verify rules, get confirmation

User wants to book 3 tickets to NYC with 2 checked bags each
- Need user ID to check:
  * Membership tier for baggage allowance
  * Which payments methods exist in profile
- Baggage calculation:
  * Economy class × 3 passengers
  * If regular member: 1 free bag each → 3 extra bags = $150
  * If silver member: 2 free bags each → 0 extra bags = $0
  * If gold member: 3 free bags each → 0 extra bags = $0
- Payment rules to verify:
  * Max 1 travel certificate, 1 credit card, 3 gift cards
  * All payment methods must be in profile
  * Travel certificate remainder goes to waste
- Plan:
1. Get user ID
2. Verify membership level for bag fees
3. Check which payment methods in profile and if their combination is allowed
4. Calculate total: ticket price + any bag fees
5. Get explicit confirmation for booking"""

if __name__ == "__main__":
    # If no arguments provided, demonstrate with example text
    if len(sys.argv) == 1:
        console.print(Panel.fit("🚀 Demo Mode - Summarizing Example Text", style="bold yellow"))
        summarizer = TextSummarizer()
        try:
            summary = summarizer.summarize_text(EXAMPLE_TEXT)
            console.print(Panel(
                summary,
                title="📋 Example Summary",
                style="cyan"
            ))
        except Exception as e:
            console.print(f"❌ Demo failed: {e}")
            console.print("Run with --help for usage instructions")
    else:
        main()