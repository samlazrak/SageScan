"""
Text preprocessing module for cleaning and normalizing OCR-extracted text.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Handles text cleaning and preprocessing for OCR-extracted text."""
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        """
        Initialize text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Initialize NLTK components
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            logger.warning("NLTK data not found. Please run: python -c 'import nltk; nltk.download(\"punkt\"); nltk.download(\"stopwords\"); nltk.download(\"wordnet\")'")
            self.lemmatizer = None
            self.stop_words = set()
        
        # Initialize spaCy for advanced NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def preprocess_text(self, text: str) -> Dict[str, Any]:
        """
        Preprocess OCR-extracted text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Dictionary containing processed text and metadata
        """
        if not text or not text.strip():
            return {
                'processed_text': '',
                'original_text': text,
                'word_count': 0,
                'sentence_count': 0,
                'cleaning_stats': {}
            }
        
        original_text = text
        
        # Step 1: Basic cleaning
        cleaned_text = self._basic_cleaning(text)
        
        # Step 2: Fix common OCR errors
        cleaned_text = self._fix_ocr_errors(cleaned_text)
        
        # Step 3: Normalize whitespace
        cleaned_text = self._normalize_whitespace(cleaned_text)
        
        # Step 4: Sentence segmentation
        sentences = self._segment_sentences(cleaned_text)
        
        # Step 5: Tokenization and advanced processing
        processed_sentences = []
        for sentence in sentences:
            processed_sentence = self._process_sentence(sentence)
            if processed_sentence.strip():
                processed_sentences.append(processed_sentence)
        
        processed_text = ' '.join(processed_sentences)
        
        # Calculate statistics
        word_count = len(processed_text.split())
        sentence_count = len(processed_sentences)
        
        # Calculate cleaning statistics
        cleaning_stats = self._calculate_cleaning_stats(original_text, processed_text)
        
        result = {
            'processed_text': processed_text,
            'original_text': original_text,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'cleaning_stats': cleaning_stats,
            'sentences': processed_sentences
        }
        
        logger.info(f"Text preprocessing completed. "
                   f"Original words: {len(original_text.split())}, "
                   f"Processed words: {word_count}")
        
        return result
    
    def _basic_cleaning(self, text: str) -> str:
        """Perform basic text cleaning."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', '', text)
        
        # Fix common punctuation issues
        text = re.sub(r'\.{2,}', '.', text)  # Multiple dots to single dot
        text = re.sub(r'\!{2,}', '!', text)  # Multiple exclamation marks
        text = re.sub(r'\?{2,}', '?', text)  # Multiple question marks
        
        return text.strip()
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors."""
        # Common OCR substitutions
        ocr_fixes = {
            '0': 'o',  # Zero to letter o
            '1': 'l',  # One to letter l (in some contexts)
            '5': 's',  # Five to letter s
            '8': 'B',  # Eight to letter B
            '|': 'I',  # Pipe to letter I
            'l': 'I',  # Lowercase l to uppercase I (in some contexts)
        }
        
        # Apply fixes (be conservative)
        for wrong, correct in ocr_fixes.items():
            # Only replace in context where it makes sense
            if wrong == '0':
                # Replace 0 with o only when surrounded by letters
                text = re.sub(r'([a-zA-Z])0([a-zA-Z])', r'\1o\2', text)
            elif wrong == '1':
                # Replace 1 with l only in word contexts
                text = re.sub(r'([a-zA-Z])1([a-zA-Z])', r'\1l\2', text)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s*', r'\1 ', text)
        
        # Fix spacing around parentheses
        text = re.sub(r'\(\s+', '(', text)
        text = re.sub(r'\s+\)', ')', text)
        
        return text.strip()
    
    def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences."""
        try:
            return sent_tokenize(text)
        except:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _process_sentence(self, sentence: str) -> str:
        """Process individual sentence."""
        # Tokenize
        try:
            tokens = word_tokenize(sentence.lower())
        except:
            tokens = sentence.lower().split()
        
        # Remove stopwords if enabled
        if self.remove_stopwords and self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize if enabled
        if self.lemmatize and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def _calculate_cleaning_stats(self, original_text: str, processed_text: str) -> Dict[str, Any]:
        """Calculate statistics about the cleaning process."""
        original_words = len(original_text.split())
        processed_words = len(processed_text.split())
        
        return {
            'original_word_count': original_words,
            'processed_word_count': processed_words,
            'words_removed': original_words - processed_words,
            'reduction_percentage': ((original_words - processed_words) / original_words * 100) if original_words > 0 else 0
        }
    
    def extract_key_phrases(self, text: str, max_phrases: int = 5) -> List[str]:
        """Extract key phrases from text using spaCy."""
        if not self.nlp:
            logger.warning("spaCy not available for key phrase extraction")
            return []
        
        doc = self.nlp(text)
        
        # Extract noun chunks and named entities
        key_phrases = []
        
        # Add noun chunks
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 4:  # Limit phrase length
                key_phrases.append(chunk.text)
        
        # Add named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
                key_phrases.append(ent.text)
        
        # Remove duplicates and limit
        key_phrases = list(set(key_phrases))[:max_phrases]
        
        return key_phrases
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        if not self.nlp:
            return 'unknown'
        
        doc = self.nlp(text)
        return doc.lang_
    
    def batch_preprocess(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Preprocess multiple texts."""
        results = []
        
        for text in texts:
            try:
                result = self.preprocess_text(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to preprocess text: {str(e)}")
                results.append({
                    'processed_text': '',
                    'original_text': text,
                    'word_count': 0,
                    'sentence_count': 0,
                    'cleaning_stats': {},
                    'error': str(e)
                })
        
        return results 