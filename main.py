#!/usr/bin/env python3
"""
Command-line interface for the Scanned Notes Processor.
"""

import argparse
import sys
import os
from pathlib import Path
import json
import logging

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.note_processor import NoteProcessor
from src.utils import setup_logging, ensure_directory

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Process scanned notes through OCR, preprocessing, and LLM analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single scanned image
  python main.py --input scan.jpg --output result.json

  # Process all images in a folder
  python main.py --input scanned_notes/ --output results/

  # Process digital notes
  python main.py --input digital_notes/ --output results/ --digital

  # Use specific LLM provider
  python main.py --input notes/ --output results/ --llm-provider openai --llm-model gpt-4
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input file or folder path'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output file or folder path'
    )
    
    parser.add_argument(
        '--digital', '-d',
        action='store_true',
        help='Process digital notes (PDF, DOCX, TXT) instead of scanned images'
    )
    
    parser.add_argument(
        '--llm-provider',
        default='local',
        choices=['openai', 'anthropic', 'local', 'ollama', 'llama-cpp'],
        help='LLM provider to use (default: local)'
    )
    
    parser.add_argument(
        '--llm-model',
        default='llama2',
        help='LLM model to use (default: llama2)'
    )
    
    parser.add_argument(
        '--api-key',
        help='API key for LLM provider (not needed for local LLMs)'
    )
    
    parser.add_argument(
        '--server-url',
        help='Server URL for local LLM providers (e.g., http://localhost:11434 for Ollama)'
    )
    
    parser.add_argument(
        '--no-stopwords',
        action='store_true',
        help='Do not remove stopwords during preprocessing'
    )
    
    parser.add_argument(
        '--no-lemmatize',
        action='store_true',
        help='Do not lemmatize words during preprocessing'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show processing statistics'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Validate input
    if not os.path.exists(args.input):
        logger.error(f"Input path does not exist: {args.input}")
        sys.exit(1)
    
    # Ensure output directory exists
    ensure_directory(os.path.dirname(args.output))
    
    try:
        # Initialize processor
        processor = NoteProcessor(
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            api_key=args.api_key,
            server_url=args.server_url,
            remove_stopwords=not args.no_stopwords,
            lemmatize=not args.no_lemmatize
        )
        
        # Process based on input type
        if os.path.isfile(args.input):
            # Single file processing
            logger.info(f"Processing single file: {args.input}")
            
            if args.digital:
                results = [processor._process_digital_file(args.input)]
            else:
                results = [processor.process_single_note(args.input)]
            
            # Save results
            processor.save_results(results, args.output)
            
        elif os.path.isdir(args.input):
            # Batch processing
            logger.info(f"Processing folder: {args.input}")
            
            if args.digital:
                results = processor.process_digital_notes(args.input)
            else:
                results = processor.process_batch(args.input)
            
            # Save results
            processor.save_results(results, args.output)
            
        else:
            logger.error(f"Invalid input path: {args.input}")
            sys.exit(1)
        
        # Show statistics if requested
        if args.stats:
            stats = processor.get_processing_stats(results)
            print("\n" + "="*50)
            print("PROCESSING STATISTICS")
            print("="*50)
            print(f"Total files: {stats['total_files']}")
            print(f"Successful: {stats['successful_files']}")
            print(f"Failed: {stats['failed_files']}")
            print(f"Success rate: {stats['success_rate']:.1f}%")
            print(f"Total words: {stats['total_words']}")
            print(f"Total sentences: {stats['total_sentences']}")
            print(f"Avg words per file: {stats['avg_words_per_file']:.1f}")
            print(f"Avg processing time: {stats['avg_processing_time']:.2f}s")
            print(f"Avg confidence: {stats['avg_confidence']:.2f}")
            print(f"Avg OCR confidence: {stats['avg_ocr_confidence']:.2f}")
            
            if stats['sentiment_distribution']:
                print("\nSentiment Distribution:")
                for sentiment, count in stats['sentiment_distribution'].items():
                    print(f"  {sentiment}: {count}")
        
        logger.info(f"Processing completed. Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 