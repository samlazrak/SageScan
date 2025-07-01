#!/usr/bin/env python3
"""
OCR Image Analyzer Demo

This is a simplified demo version that shows the application structure
and functionality without requiring external dependencies like OpenCV,
Tesseract, or MLX to be installed.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DemoImageProcessor:
    """Demo image processor that simulates OCR functionality."""
    
    def extract_text(self, image_path: str) -> str:
        """Simulate text extraction from an image."""
        image_name = Path(image_path).name.lower()
        
        # Simulate different types of content based on filename
        if 'receipt' in image_name:
            return """GROCERY STORE RECEIPT
Date: 2024-01-15
Items:
- Apples: $4.99
- Bread: $2.50
- Milk: $3.25
Total: $10.74
Thank you for shopping with us!"""
        
        elif 'document' in image_name or 'report' in image_name:
            return """QUARTERLY SALES REPORT
Q4 2023 Performance Summary

Our company has achieved remarkable growth this quarter, with sales increasing by 25% compared to the previous quarter. Customer satisfaction ratings have improved significantly, with positive feedback highlighting our enhanced customer service and product quality. The marketing team's new campaign proved highly effective, resulting in increased brand awareness and customer acquisition. Moving forward, we plan to expand our product line and explore new market opportunities."""
        
        elif 'invoice' in image_name:
            return """INVOICE #INV-2024-001
Date: January 15, 2024
Bill To: ABC Company
Services Rendered:
- Consulting Services: $1,500.00
- Software Development: $3,000.00
- Project Management: $800.00
Subtotal: $5,300.00
Tax: $530.00
Total: $5,830.00
Payment Due: February 15, 2024"""
        
        elif 'review' in image_name or 'feedback' in image_name:
            return """CUSTOMER FEEDBACK
This product exceeded my expectations! The quality is outstanding and the customer service was exceptional. I would definitely recommend this to others and will purchase again. The delivery was fast and the packaging was perfect. Very satisfied with this purchase."""
        
        else:
            return f"""Sample text extracted from {image_name}
This is a demonstration of OCR text extraction functionality.
The actual application would process real images using Tesseract OCR
with advanced image preprocessing for optimal text recognition.
This text contains neutral content for demonstration purposes."""

class DemoTextAnalyzer:
    """Demo text analyzer that simulates MLX model functionality."""
    
    def summarize_text(self, text: str, max_length: int = 100) -> str:
        """Simulate text summarization."""
        if not text.strip():
            return "No text to summarize."
        
        # Simple extractive summarization (first sentence + key info)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) <= 2:
            return text
        
        # Create a simple summary
        summary = f"{sentences[0]}."
        if len(summary) < max_length and len(sentences) > 1:
            summary += f" {sentences[1]}."
        
        return summary if len(summary) <= max_length else summary[:max_length] + "..."
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """Simulate sentiment analysis."""
        if not text.strip():
            return {"sentiment": "neutral", "confidence": 0.0, "explanation": "No text to analyze"}
        
        # Simple keyword-based sentiment analysis
        positive_words = ['excellent', 'outstanding', 'exceptional', 'satisfied', 'exceeded', 'perfect', 
                         'remarkable', 'growth', 'improved', 'effective', 'positive', 'good', 'great']
        negative_words = ['terrible', 'awful', 'disappointing', 'poor', 'bad', 'horrible', 'worst',
                         'failed', 'declined', 'negative', 'frustrated', 'angry']
        neutral_words = ['total', 'date', 'invoice', 'receipt', 'payment', 'services', 'items']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        neutral_count = sum(1 for word in neutral_words if word in text_lower)
        
        if positive_count > negative_count and positive_count > 0:
            sentiment = "positive"
            confidence = min(0.9, (positive_count / (positive_count + negative_count + 1)) * 1.2)
            explanation = f"Text contains {positive_count} positive indicators like 'excellent', 'outstanding', 'growth'."
        elif negative_count > positive_count and negative_count > 0:
            sentiment = "negative" 
            confidence = min(0.9, (negative_count / (positive_count + negative_count + 1)) * 1.2)
            explanation = f"Text contains {negative_count} negative indicators."
        elif neutral_count > 2:
            sentiment = "neutral"
            confidence = 0.7
            explanation = "Text appears to be factual/business content with neutral tone."
        else:
            sentiment = "neutral"
            confidence = 0.5
            explanation = "Mixed or unclear sentiment indicators."
        
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "explanation": explanation
        }

class DemoOCRAnalyzer:
    """Demo OCR analyzer that simulates the full application."""
    
    def __init__(self):
        self.image_processor = DemoImageProcessor()
        self.text_analyzer = DemoTextAnalyzer()
        print("🤖 Demo OCR Analyzer initialized")
        print("📝 This is a demonstration version - no dependencies required!")
        print("")
    
    def process_image(self, image_path: str) -> Dict[str, any]:
        """Process a single image (demo version)."""
        print(f"📷 Processing: {Path(image_path).name}")
        
        result = {
            "image_path": image_path,
            "extracted_text": "",
            "summary": "",
            "sentiment_analysis": {},
            "success": False,
            "demo_mode": True
        }
        
        try:
            # Check if file exists (for real files)
            if not Path(image_path).exists():
                print(f"⚠️  File not found: {image_path} (simulating content)")
            
            # Simulate text extraction
            extracted_text = self.image_processor.extract_text(image_path)
            result["extracted_text"] = extracted_text
            
            # Simulate summarization
            summary = self.text_analyzer.summarize_text(extracted_text)
            result["summary"] = summary
            
            # Simulate sentiment analysis
            sentiment = self.text_analyzer.analyze_sentiment(extracted_text)
            result["sentiment_analysis"] = sentiment
            
            result["success"] = True
            print(f"✅ Completed: {Path(image_path).name}")
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            result["error"] = str(e)
        
        return result
    
    def display_results(self, results: List[Dict[str, any]]):
        """Display results in a formatted way."""
        print("\n" + "="*80)
        print("📊 OCR ANALYSIS RESULTS")
        print("="*80)
        
        for i, result in enumerate(results, 1):
            if result["success"]:
                print(f"\n📄 IMAGE {i}: {Path(result['image_path']).name}")
                print("-" * 60)
                
                # Extracted text (truncated)
                text = result["extracted_text"]
                preview = text[:200] + "..." if len(text) > 200 else text
                print(f"📖 EXTRACTED TEXT ({len(text)} characters):")
                print(f"   {preview}")
                
                # Summary
                print(f"\n📝 SUMMARY:")
                print(f"   {result['summary']}")
                
                # Sentiment
                sentiment_info = result["sentiment_analysis"]
                sentiment_emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
                emoji = sentiment_emoji.get(sentiment_info["sentiment"], "🤔")
                
                print(f"\n💭 SENTIMENT ANALYSIS:")
                print(f"   {emoji} {sentiment_info['sentiment'].upper()} (confidence: {sentiment_info['confidence']:.2f})")
                print(f"   📄 {sentiment_info['explanation']}")
            else:
                print(f"\n❌ FAILED: {Path(result['image_path']).name}")
                print(f"   Error: {result.get('error', 'Unknown error')}")
        
        print("\n" + "="*80)
        successful = sum(1 for r in results if r["success"])
        print(f"✅ Successfully processed {successful}/{len(results)} images")
        print("="*80)

def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="OCR Image Analyzer Demo - No dependencies required!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 demo.py receipt.jpg document.png
  python3 demo.py --output demo_results.json invoice.pdf
  python3 demo.py report.jpg review.png feedback.jpeg

This demo simulates the full OCR analyzer functionality without requiring
external dependencies like OpenCV, Tesseract, or MLX. Perfect for testing
the application structure and understanding the workflow.
        """
    )
    
    parser.add_argument('images', nargs='+', help='Image files to process (can be real or simulated)')
    parser.add_argument('--output', '-o', help='Save results to JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Display header
    print("🚀 OCR IMAGE ANALYZER - DEMO MODE")
    print("=" * 50)
    print("🎯 This demo simulates OCR analysis without requiring dependencies")
    print("📚 Try filenames like: receipt.jpg, document.png, invoice.pdf, review.jpeg")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = DemoOCRAnalyzer()
    
    # Process images
    print(f"🔄 Processing {len(args.images)} image(s)...")
    results = []
    
    for image_path in args.images:
        result = analyzer.process_image(image_path)
        results.append(result)
    
    # Display results
    analyzer.display_results(results)
    
    # Save results if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")
    
    print(f"\n🎉 Demo completed! To use the full version with real OCR:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Install Tesseract OCR for your system")
    print("   3. Run: python3 ocr_analyzer.py your_real_image.jpg")

if __name__ == "__main__":
    main()