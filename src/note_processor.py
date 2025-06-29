"""
Main note processor that orchestrates the entire pipeline.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

from .ocr_processor import OCRProcessor
from .text_preprocessor import TextPreprocessor
from .llm_processor import LLMProcessor, AnalysisResult
from .utils import (
    validate_image_file, 
    validate_digital_file, 
    get_files_from_folder,
    save_results,
    create_output_filename
)

logger = logging.getLogger(__name__)

@dataclass
class NoteResult:
    """Result of note processing."""
    filename: str
    extracted_text: str
    processed_text: str
    summary: str
    sentiment: str
    sentiment_score: float
    key_topics: List[str]
    action_items: List[str]
    confidence: float
    ocr_confidence: float
    word_count: int
    sentence_count: int
    processing_time: float
    model_used: str
    error: Optional[str] = None

class NoteProcessor:
    """Main processor that orchestrates OCR, preprocessing, and LLM analysis."""
    
    def __init__(self, 
                 llm_provider: str = "local",
                 llm_model: str = "llama2",
                 api_key: Optional[str] = None,
                 server_url: Optional[str] = None,
                 remove_stopwords: bool = True,
                 lemmatize: bool = True):
        """
        Initialize note processor.
        
        Args:
            llm_provider: LLM provider to use
            llm_model: LLM model to use
            api_key: API key for LLM provider (not needed for local LLMs)
            server_url: Server URL for local LLM providers
            remove_stopwords: Whether to remove stopwords during preprocessing
            lemmatize: Whether to lemmatize words during preprocessing
        """
        self.ocr_processor = OCRProcessor()
        self.text_preprocessor = TextPreprocessor(
            remove_stopwords=remove_stopwords,
            lemmatize=lemmatize
        )
        self.llm_processor = LLMProcessor(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key,
            base_url=server_url
        )
        
        logger.info(f"NoteProcessor initialized with {llm_provider}/{llm_model}")
    
    def process_single_note(self, image_path: str) -> NoteResult:
        """
        Process a single scanned note.
        
        Args:
            image_path: Path to the scanned image
            
        Returns:
            NoteResult with all processing results
        """
        import time
        start_time = time.time()
        
        try:
            # Step 1: OCR text extraction
            logger.info(f"Starting OCR for: {image_path}")
            ocr_result = self.ocr_processor.extract_text(image_path)
            
            if not ocr_result['text'].strip():
                logger.warning(f"No text extracted from {image_path}")
                return NoteResult(
                    filename=os.path.basename(image_path),
                    extracted_text="",
                    processed_text="",
                    summary="No text found in image",
                    sentiment="neutral",
                    sentiment_score=0.0,
                    key_topics=[],
                    action_items=[],
                    confidence=0.0,
                    ocr_confidence=ocr_result['confidence'],
                    word_count=0,
                    sentence_count=0,
                    processing_time=time.time() - start_time,
                    model_used=self.llm_processor.model,
                    error="No text extracted"
                )
            
            # Step 2: Text preprocessing
            logger.info(f"Preprocessing text for: {image_path}")
            preprocess_result = self.text_preprocessor.preprocess_text(ocr_result['text'])
            
            # Step 3: LLM analysis
            logger.info(f"Analyzing text with LLM for: {image_path}")
            analysis_result = self.llm_processor.analyze_text(preprocess_result['processed_text'])
            
            processing_time = time.time() - start_time
            
            result = NoteResult(
                filename=os.path.basename(image_path),
                extracted_text=ocr_result['text'],
                processed_text=preprocess_result['processed_text'],
                summary=analysis_result.summary,
                sentiment=analysis_result.sentiment,
                sentiment_score=analysis_result.sentiment_score,
                key_topics=analysis_result.key_topics,
                action_items=analysis_result.action_items,
                confidence=analysis_result.confidence,
                ocr_confidence=ocr_result['confidence'],
                word_count=preprocess_result['word_count'],
                sentence_count=preprocess_result['sentence_count'],
                processing_time=processing_time,
                model_used=analysis_result.model_used
            )
            
            logger.info(f"Processing completed for {image_path} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Processing failed for {image_path}: {str(e)}")
            return NoteResult(
                filename=os.path.basename(image_path),
                extracted_text="",
                processed_text="",
                summary="Processing failed",
                sentiment="neutral",
                sentiment_score=0.0,
                key_topics=[],
                action_items=[],
                confidence=0.0,
                ocr_confidence=0.0,
                word_count=0,
                sentence_count=0,
                processing_time=time.time() - start_time,
                model_used="none",
                error=str(e)
            )
    
    def process_batch(self, folder_path: str) -> List[NoteResult]:
        """
        Process multiple scanned notes from a folder.
        
        Args:
            folder_path: Path to folder containing scanned images
            
        Returns:
            List of NoteResult objects
        """
        # Get all valid image files
        image_files = get_files_from_folder(folder_path, validate_image_file)
        
        if not image_files:
            logger.warning(f"No valid image files found in {folder_path}")
            return []
        
        logger.info(f"Processing {len(image_files)} files from {folder_path}")
        
        results = []
        for image_file in image_files:
            try:
                result = self.process_single_note(image_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {str(e)}")
                # Add error result
                results.append(NoteResult(
                    filename=os.path.basename(image_file),
                    extracted_text="",
                    processed_text="",
                    summary="Processing failed",
                    sentiment="neutral",
                    sentiment_score=0.0,
                    key_topics=[],
                    action_items=[],
                    confidence=0.0,
                    ocr_confidence=0.0,
                    word_count=0,
                    sentence_count=0,
                    processing_time=0.0,
                    model_used="none",
                    error=str(e)
                ))
        
        return results
    
    def process_digital_notes(self, folder_path: str) -> List[NoteResult]:
        """
        Process digital notes (PDF, DOCX, TXT) from a folder.
        
        Args:
            folder_path: Path to folder containing digital notes
            
        Returns:
            List of NoteResult objects
        """
        # Get all valid digital files
        digital_files = get_files_from_folder(folder_path, validate_digital_file)
        
        if not digital_files:
            logger.warning(f"No valid digital files found in {folder_path}")
            return []
        
        logger.info(f"Processing {len(digital_files)} digital files from {folder_path}")
        
        results = []
        for digital_file in digital_files:
            try:
                result = self._process_digital_file(digital_file)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {digital_file}: {str(e)}")
                results.append(NoteResult(
                    filename=os.path.basename(digital_file),
                    extracted_text="",
                    processed_text="",
                    summary="Processing failed",
                    sentiment="neutral",
                    sentiment_score=0.0,
                    key_topics=[],
                    action_items=[],
                    confidence=0.0,
                    ocr_confidence=0.0,
                    word_count=0,
                    sentence_count=0,
                    processing_time=0.0,
                    model_used="none",
                    error=str(e)
                ))
        
        return results
    
    def _process_digital_file(self, file_path: str) -> NoteResult:
        """Process a single digital file."""
        import time
        start_time = time.time()
        
        try:
            # Extract text from digital file
            extracted_text = self._extract_text_from_digital_file(file_path)
            
            if not extracted_text.strip():
                logger.warning(f"No text extracted from {file_path}")
                return NoteResult(
                    filename=os.path.basename(file_path),
                    extracted_text="",
                    processed_text="",
                    summary="No text found in file",
                    sentiment="neutral",
                    sentiment_score=0.0,
                    key_topics=[],
                    action_items=[],
                    confidence=0.0,
                    ocr_confidence=100.0,  # Digital files have perfect "OCR"
                    word_count=0,
                    sentence_count=0,
                    processing_time=time.time() - start_time,
                    model_used=self.llm_processor.model,
                    error="No text extracted"
                )
            
            # Preprocess text
            preprocess_result = self.text_preprocessor.preprocess_text(extracted_text)
            
            # Analyze with LLM
            analysis_result = self.llm_processor.analyze_text(preprocess_result['processed_text'])
            
            processing_time = time.time() - start_time
            
            return NoteResult(
                filename=os.path.basename(file_path),
                extracted_text=extracted_text,
                processed_text=preprocess_result['processed_text'],
                summary=analysis_result.summary,
                sentiment=analysis_result.sentiment,
                sentiment_score=analysis_result.sentiment_score,
                key_topics=analysis_result.key_topics,
                action_items=analysis_result.action_items,
                confidence=analysis_result.confidence,
                ocr_confidence=100.0,
                word_count=preprocess_result['word_count'],
                sentence_count=preprocess_result['sentence_count'],
                processing_time=processing_time,
                model_used=analysis_result.model_used
            )
            
        except Exception as e:
            logger.error(f"Digital file processing failed for {file_path}: {str(e)}")
            raise
    
    def _extract_text_from_digital_file(self, file_path: str) -> str:
        """Extract text from various digital file formats."""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_ext == '.pdf':
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    return text
            except ImportError:
                logger.warning("PyPDF2 not installed. Install with: pip install PyPDF2")
                return ""
        
        elif file_ext == '.docx':
            try:
                from docx import Document
                doc = Document(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            except ImportError:
                logger.warning("python-docx not installed. Install with: pip install python-docx")
                return ""
        
        else:
            logger.warning(f"Unsupported file format: {file_ext}")
            return ""
    
    def save_results(self, results: List[NoteResult], output_path: str) -> None:
        """Save processing results to file."""
        # Convert to dictionary format
        results_dict = [asdict(result) for result in results]
        save_results(results_dict, output_path)
    
    def get_processing_stats(self, results: List[NoteResult]) -> Dict[str, Any]:
        """Get statistics about processing results."""
        if not results:
            return {}
        
        total_files = len(results)
        successful_files = len([r for r in results if not r.error])
        failed_files = total_files - successful_files
        
        total_words = sum(r.word_count for r in results)
        total_sentences = sum(r.sentence_count for r in results)
        avg_processing_time = sum(r.processing_time for r in results) / total_files
        
        # Sentiment distribution
        sentiments = [r.sentiment for r in results if r.sentiment]
        sentiment_counts = {}
        for sentiment in sentiments:
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        # Average confidence
        avg_confidence = sum(r.confidence for r in results) / total_files
        avg_ocr_confidence = sum(r.ocr_confidence for r in results) / total_files
        
        return {
            'total_files': total_files,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'success_rate': successful_files / total_files * 100,
            'total_words': total_words,
            'total_sentences': total_sentences,
            'avg_words_per_file': total_words / successful_files if successful_files > 0 else 0,
            'avg_sentences_per_file': total_sentences / successful_files if successful_files > 0 else 0,
            'avg_processing_time': avg_processing_time,
            'sentiment_distribution': sentiment_counts,
            'avg_confidence': avg_confidence,
            'avg_ocr_confidence': avg_ocr_confidence
        } 