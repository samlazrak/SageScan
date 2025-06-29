#!/usr/bin/env python3
"""
Example usage of the Scanned Notes Processor.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.note_processor import NoteProcessor
from src.utils import setup_logging

def example_single_note():
    """Example: Process a single scanned note."""
    print("=== Example: Single Note Processing ===")
    
    # Initialize processor
    processor = NoteProcessor(
        llm_provider="openai",
        llm_model="gpt-3.5-turbo"
    )
    
    # Process a single note (replace with actual image path)
    image_path = "examples/sample_notes/sample_scan.jpg"
    
    if os.path.exists(image_path):
        result = processor.process_single_note(image_path)
        
        print(f"Filename: {result.filename}")
        print(f"Summary: {result.summary}")
        print(f"Sentiment: {result.sentiment} (score: {result.sentiment_score:.2f})")
        print(f"Key Topics: {', '.join(result.key_topics)}")
        print(f"Action Items: {', '.join(result.action_items)}")
        print(f"Word Count: {result.word_count}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"OCR Confidence: {result.ocr_confidence:.2f}%")
    else:
        print(f"Sample image not found: {image_path}")
        print("Please add a sample image to test this example.")

def example_batch_processing():
    """Example: Process multiple scanned notes."""
    print("\n=== Example: Batch Processing ===")
    
    # Initialize processor
    processor = NoteProcessor(
        llm_provider="openai",
        llm_model="gpt-3.5-turbo"
    )
    
    # Process all notes in a folder
    folder_path = "examples/sample_notes"
    
    if os.path.exists(folder_path):
        results = processor.process_batch(folder_path)
        
        print(f"Processed {len(results)} files:")
        for result in results:
            print(f"\n- {result.filename}:")
            print(f"  Summary: {result.summary}")
            print(f"  Sentiment: {result.sentiment}")
            print(f"  Words: {result.word_count}")
        
        # Get processing statistics
        stats = processor.get_processing_stats(results)
        print(f"\nProcessing Statistics:")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")
        print(f"  Total Words: {stats['total_words']}")
        print(f"  Avg Processing Time: {stats['avg_processing_time']:.2f}s")
    else:
        print(f"Sample folder not found: {folder_path}")
        print("Please add sample images to test this example.")

def example_digital_notes():
    """Example: Process digital notes."""
    print("\n=== Example: Digital Notes Processing ===")
    
    # Initialize processor
    processor = NoteProcessor(
        llm_provider="openai",
        llm_model="gpt-3.5-turbo"
    )
    
    # Process digital notes (PDF, DOCX, TXT)
    folder_path = "examples/sample_notes"
    
    if os.path.exists(folder_path):
        results = processor.process_digital_notes(folder_path)
        
        print(f"Processed {len(results)} digital files:")
        for result in results:
            print(f"\n- {result.filename}:")
            print(f"  Summary: {result.summary}")
            print(f"  Sentiment: {result.sentiment}")
            print(f"  Words: {result.word_count}")
    else:
        print(f"Sample folder not found: {folder_path}")
        print("Please add sample digital files to test this example.")

def example_save_results():
    """Example: Save processing results."""
    print("\n=== Example: Saving Results ===")
    
    # Initialize processor
    processor = NoteProcessor(
        llm_provider="openai",
        llm_model="gpt-3.5-turbo"
    )
    
    # Process a single note
    image_path = "examples/sample_notes/sample_scan.jpg"
    
    if os.path.exists(image_path):
        result = processor.process_single_note(image_path)
        
        # Save results to JSON file
        output_path = "examples/results.json"
        processor.save_results([result], output_path)
        
        print(f"Results saved to: {output_path}")
    else:
        print(f"Sample image not found: {image_path}")

def example_custom_configuration():
    """Example: Custom configuration."""
    print("\n=== Example: Custom Configuration ===")
    
    # Initialize processor with custom settings
    processor = NoteProcessor(
        llm_provider="openai",
        llm_model="gpt-4",  # Use GPT-4 for better analysis
        remove_stopwords=False,  # Keep stopwords
        lemmatize=False  # Don't lemmatize
    )
    
    # Example text for analysis
    sample_text = """
    This is a sample note about machine learning. 
    The project is progressing well and we're seeing positive results.
    Key points to remember: implement the new algorithm, test with larger datasets.
    """
    
    # Analyze the text directly
    from src.llm_processor import LLMProcessor
    llm = LLMProcessor(provider="openai", model="gpt-4")
    result = llm.analyze_text(sample_text)
    
    print(f"Sample Text: {sample_text.strip()}")
    print(f"Summary: {result.summary}")
    print(f"Sentiment: {result.sentiment} (score: {result.sentiment_score:.2f})")
    print(f"Key Topics: {', '.join(result.key_topics)}")
    print(f"Action Items: {', '.join(result.action_items)}")

def main():
    """Run all examples."""
    print("Scanned Notes Processor - Example Usage")
    print("=" * 50)
    
    # Setup logging
    setup_logging("INFO")
    
    # Run examples
    example_single_note()
    example_batch_processing()
    example_digital_notes()
    example_save_results()
    example_custom_configuration()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo run the processor on your own files:")
    print("python main.py --input your_notes/ --output results/ --stats")

if __name__ == "__main__":
    main() 