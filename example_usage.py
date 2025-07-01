#!/usr/bin/env python3
"""
Example Usage of OCR + MLX Text Analyzer
========================================

This script demonstrates how to use the OCR MLX analyzer programmatically.
"""

import sys
import json
from pathlib import Path
from ocr_mlx_analyzer import OCRMLXApp


def create_sample_text_image():
    """Create a sample text image for testing if PIL is available."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io
        
        # Create a sample image with text
        img = Image.new('RGB', (800, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to default if not available
        try:
            font = ImageFont.truetype("Arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        text = "Hello World! This is a sample text for OCR analysis. The sentiment here is quite positive and optimistic."
        draw.text((50, 50), text, fill='black', font=font)
        
        # Save the image
        img.save('sample_text.png')
        print("✓ Created sample image: sample_text.png")
        return 'sample_text.png'
        
    except ImportError:
        print("⚠ PIL not available for creating sample image")
        return None


def example_single_image_analysis():
    """Example of analyzing a single image."""
    print("\n🔍 Single Image Analysis Example")
    print("=" * 40)
    
    # Create or use existing sample image
    sample_image = create_sample_text_image()
    
    if not sample_image:
        print("No sample image available. Please provide an image file.")
        return
    
    try:
        # Initialize the analyzer
        print("Initializing OCR + MLX analyzer...")
        app = OCRMLXApp()
        
        # Process the image
        print(f"Processing image: {sample_image}")
        results = app.process_image(sample_image)
        
        # Display results
        print("\n📊 Results:")
        print("-" * 20)
        print(f"Extracted Text: {results['extracted_text']}")
        print(f"Summary: {results['summary']}")
        print(f"Sentiment: {results['sentiment']['sentiment'].upper()}")
        print(f"Confidence: {results['sentiment']['confidence']:.2f}")
        
        # Save results
        with open('example_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n💾 Results saved to: example_results.json")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def example_custom_model():
    """Example of using a custom MLX model."""
    print("\n🤖 Custom Model Example")
    print("=" * 30)
    
    try:
        # Initialize with a specific model
        print("Initializing with custom model...")
        app = OCRMLXApp(model_name="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
        
        # Create sample text for analysis
        sample_text = "I absolutely love this new technology! It's amazing how well it works."
        
        # Analyze text directly (bypassing OCR)
        print(f"Analyzing text: {sample_text}")
        summary = app.mlx_analyzer.summarize_text(sample_text)
        sentiment = app.mlx_analyzer.analyze_sentiment(sample_text)
        
        print(f"\n📝 Summary: {summary}")
        print(f"😊 Sentiment: {sentiment['sentiment']} (confidence: {sentiment['confidence']:.2f})")
        
    except Exception as e:
        print(f"❌ Error with custom model: {str(e)}")
        print("💡 Tip: Make sure MLX is properly installed on Apple Silicon")


def example_batch_simulation():
    """Example simulating batch processing."""
    print("\n📚 Batch Processing Simulation")
    print("=" * 35)
    
    # Sample texts that might be extracted from images
    sample_texts = [
        "This product is fantastic! I highly recommend it to everyone.",
        "The service was terrible. Very disappointed with the experience.",
        "It's okay, nothing special but gets the job done.",
        "Amazing quality and fast delivery. Five stars!",
        "Poor quality materials, wouldn't buy again."
    ]
    
    try:
        app = OCRMLXApp()
        results = []
        
        for i, text in enumerate(sample_texts, 1):
            print(f"Processing text {i}/{len(sample_texts)}...")
            
            # Simulate analysis results
            summary = app.mlx_analyzer.summarize_text(text)
            sentiment = app.mlx_analyzer.analyze_sentiment(text)
            
            result = {
                "text_id": i,
                "original_text": text,
                "summary": summary,
                "sentiment": sentiment
            }
            results.append(result)
        
        # Display batch summary
        print("\n📊 Batch Summary:")
        sentiments = [r['sentiment']['sentiment'] for r in results]
        sentiment_counts = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }
        
        for sentiment, count in sentiment_counts.items():
            percentage = count / len(results) * 100
            print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Save batch results
        with open('batch_example_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("\n💾 Batch results saved to: batch_example_results.json")
        
    except Exception as e:
        print(f"❌ Error in batch simulation: {str(e)}")


def cleanup_files():
    """Clean up example files."""
    files_to_remove = ['sample_text.png', 'example_results.json', 'batch_example_results.json']
    
    for file in files_to_remove:
        if Path(file).exists():
            Path(file).unlink()
            print(f"🗑️ Removed: {file}")


def main():
    """Main example function."""
    print("🚀 OCR + MLX Text Analyzer Examples")
    print("=" * 40)
    
    # Check if required modules are available
    try:
        import mlx.core
        print("✓ MLX available")
    except ImportError:
        print("⚠ MLX not available - examples may have limited functionality")
    
    try:
        import pytesseract
        print("✓ Tesseract available")
    except ImportError:
        print("❌ Tesseract not available - OCR will not work")
        return
    
    # Run examples
    example_single_image_analysis()
    example_custom_model()
    example_batch_simulation()
    
    # Ask about cleanup
    print("\n🧹 Cleanup")
    print("=" * 10)
    response = input("Remove example files? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        cleanup_files()
    
    print("\n✅ Examples completed!")
    print("💡 Check the main application with: python3 ocr_mlx_analyzer.py --help")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)