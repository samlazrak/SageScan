#!/usr/bin/env python3
"""
Batch Processor for OCR + MLX Text Analyzer
============================================

Process multiple images in a directory using the OCR + MLX analyzer.
"""

import os
import sys
import json
import click
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from ocr_mlx_analyzer import OCRMLXApp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp'}


class BatchProcessor:
    """Handles batch processing of multiple images."""
    
    def __init__(self, model_name: str = None, max_workers: int = 2):
        if model_name:
            self.app = OCRMLXApp(model_name=model_name)
        else:
            self.app = OCRMLXApp()
        self.max_workers = max_workers
        self.results = []
    
    def find_images(self, input_dir: str) -> list:
        """Find all supported image files in the directory."""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_dir}")
        
        image_files = []
        for file_path in input_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_files.append(str(file_path))
        
        logger.info(f"Found {len(image_files)} image files in {input_dir}")
        return sorted(image_files)
    
    def process_single_image(self, image_path: str) -> dict:
        """Process a single image and return results."""
        try:
            logger.info(f"Processing: {image_path}")
            result = self.app.process_image(image_path)
            logger.info(f"Completed: {image_path}")
            return result
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return {
                "image_path": image_path,
                "extracted_text": "",
                "summary": f"Error processing image: {str(e)}",
                "sentiment": {"sentiment": "neutral", "confidence": 0.0, "explanation": "Processing failed."}
            }
    
    def process_images(self, image_files: list, use_threading: bool = True) -> list:
        """Process multiple images with optional threading."""
        results = []
        
        if use_threading and len(image_files) > 1:
            logger.info(f"Processing {len(image_files)} images with {self.max_workers} workers")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_image = {
                    executor.submit(self.process_single_image, image_path): image_path
                    for image_path in image_files
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_image):
                    result = future.result()
                    results.append(result)
        else:
            logger.info(f"Processing {len(image_files)} images sequentially")
            for image_path in image_files:
                result = self.process_single_image(image_path)
                results.append(result)
        
        return results
    
    def save_results(self, results: list, output_dir: str, format_type: str = 'json'):
        """Save batch processing results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'json':
            # Save as single JSON file
            output_file = output_path / 'batch_results.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_file}")
            
            # Save individual files for each image
            for result in results:
                if result['extracted_text']:
                    image_name = Path(result['image_path']).stem
                    individual_file = output_path / f'{image_name}_analysis.json'
                    with open(individual_file, 'w', encoding='utf-8') as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)
        
        elif format_type == 'csv':
            # Save as CSV
            import csv
            output_file = output_path / 'batch_results.csv'
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Image Path', 'Extracted Text', 'Summary', 'Sentiment', 'Confidence'])
                
                for result in results:
                    sentiment_data = result['sentiment']
                    writer.writerow([
                        result['image_path'],
                        result['extracted_text'][:200] + '...' if len(result['extracted_text']) > 200 else result['extracted_text'],
                        result['summary'],
                        sentiment_data['sentiment'],
                        sentiment_data['confidence']
                    ])
            logger.info(f"CSV results saved to: {output_file}")
    
    def generate_summary_report(self, results: list) -> dict:
        """Generate a summary report of batch processing."""
        total_images = len(results)
        successful_ocr = sum(1 for r in results if r['extracted_text'].strip())
        
        sentiments = [r['sentiment']['sentiment'] for r in results if r['sentiment']['sentiment']]
        sentiment_counts = {
            'positive': sentiments.count('positive'),
            'negative': sentiments.count('negative'),
            'neutral': sentiments.count('neutral')
        }
        
        avg_confidence = sum(r['sentiment']['confidence'] for r in results) / total_images if total_images > 0 else 0
        
        total_characters = sum(len(r['extracted_text']) for r in results)
        
        return {
            'total_images': total_images,
            'successful_ocr': successful_ocr,
            'ocr_success_rate': successful_ocr / total_images if total_images > 0 else 0,
            'sentiment_distribution': sentiment_counts,
            'average_confidence': avg_confidence,
            'total_characters_extracted': total_characters,
            'average_characters_per_image': total_characters / total_images if total_images > 0 else 0
        }


@click.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--output-dir', '-o', default='./batch_results', help='Output directory for results')
@click.option('--model', '-m', default=None, help='MLX model name to use')
@click.option('--format', '-f', type=click.Choice(['json', 'csv']), default='json', help='Output format')
@click.option('--workers', '-w', type=int, default=2, help='Number of worker threads')
@click.option('--sequential', is_flag=True, help='Process images sequentially (no threading)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_dir, output_dir, model, format, workers, sequential, verbose):
    """
    Batch process images using OCR + MLX Text Analyzer.
    
    INPUT_DIR: Directory containing images to process
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize batch processor
        processor = BatchProcessor(model_name=model, max_workers=workers)
        
        # Find images
        image_files = processor.find_images(input_dir)
        
        if not image_files:
            logger.warning(f"No supported image files found in {input_dir}")
            return
        
        # Process images
        results = processor.process_images(image_files, use_threading=not sequential)
        
        # Save results
        processor.save_results(results, output_dir, format)
        
        # Generate and display summary
        summary = processor.generate_summary_report(results)
        
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)
        print(f"Total Images Processed: {summary['total_images']}")
        print(f"Successful OCR: {summary['successful_ocr']} ({summary['ocr_success_rate']:.1%})")
        print(f"Total Characters Extracted: {summary['total_characters_extracted']:,}")
        print(f"Average Characters per Image: {summary['average_characters_per_image']:.1f}")
        print(f"Average Sentiment Confidence: {summary['average_confidence']:.2f}")
        print("\nSentiment Distribution:")
        for sentiment, count in summary['sentiment_distribution'].items():
            percentage = count / summary['total_images'] * 100 if summary['total_images'] > 0 else 0
            print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        print("="*60)
        
        # Save summary report
        summary_file = Path(output_dir) / 'summary_report.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary report saved to: {summary_file}")
    
    except KeyboardInterrupt:
        print("\nBatch processing cancelled by user.")
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()