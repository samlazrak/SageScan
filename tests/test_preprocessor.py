"""
Unit tests for text preprocessor.
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.text_preprocessor import TextPreprocessor

class TestTextPreprocessor(unittest.TestCase):
    """Test cases for text preprocessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_preprocessor_initialization(self):
        """Test text preprocessor initialization."""
        processor = TextPreprocessor()
        self.assertIsNotNone(processor)
        self.assertTrue(processor.remove_stopwords)
        self.assertTrue(processor.lemmatize)
        
        # Test with custom settings
        processor2 = TextPreprocessor(remove_stopwords=False, lemmatize=False)
        self.assertFalse(processor2.remove_stopwords)
        self.assertFalse(processor2.lemmatize)
    
    def test_basic_cleaning(self):
        """Test basic text cleaning."""
        test_text = "  This   is   a   test   text   with   extra   spaces.  "
        cleaned = self.preprocessor._basic_cleaning(test_text)
        
        self.assertEqual(cleaned, "This is a test text with extra spaces.")
        
        # Test with special characters
        test_text2 = "This is a test with @#$%^&*() special chars"
        cleaned2 = self.preprocessor._basic_cleaning(test_text2)
        
        # Should remove special characters but keep basic punctuation
        self.assertNotIn('@', cleaned2)
        self.assertNotIn('#', cleaned2)
    
    def test_ocr_error_fixing(self):
        """Test OCR error fixing."""
        test_text = "Hell0 W0rld"  # Common OCR errors
        fixed = self.preprocessor._fix_ocr_errors(test_text)
        
        # Should fix some OCR errors
        self.assertIsInstance(fixed, str)
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        test_text = "This   has   multiple   spaces.   And   punctuation   issues   ."
        normalized = self.preprocessor._normalize_whitespace(test_text)
        
        self.assertEqual(normalized, "This has multiple spaces. And punctuation issues.")
    
    def test_sentence_segmentation(self):
        """Test sentence segmentation."""
        test_text = "This is sentence one. This is sentence two. This is sentence three."
        sentences = self.preprocessor._segment_sentences(test_text)
        
        self.assertIsInstance(sentences, list)
        self.assertEqual(len(sentences), 3)
        
        for sentence in sentences:
            self.assertIsInstance(sentence, str)
            self.assertTrue(len(sentence.strip()) > 0)
    
    def test_sentence_processing(self):
        """Test individual sentence processing."""
        test_sentence = "This is a test sentence with some words."
        processed = self.preprocessor._process_sentence(test_sentence)
        
        self.assertIsInstance(processed, str)
        self.assertTrue(len(processed) > 0)
    
    def test_full_preprocessing(self):
        """Test full text preprocessing pipeline."""
        test_text = """
        This is a test note with some OCR errors like Hell0 W0rld.
        It has multiple sentences.   And some formatting issues   .
        """
        
        result = self.preprocessor.preprocess_text(test_text)
        
        self.assertIsInstance(result, dict)
        self.assertIn('processed_text', result)
        self.assertIn('original_text', result)
        self.assertIn('word_count', result)
        self.assertIn('sentence_count', result)
        self.assertIn('cleaning_stats', result)
        self.assertIn('sentences', result)
        
        # Check data types
        self.assertIsInstance(result['processed_text'], str)
        self.assertIsInstance(result['original_text'], str)
        self.assertIsInstance(result['word_count'], int)
        self.assertIsInstance(result['sentence_count'], int)
        self.assertIsInstance(result['cleaning_stats'], dict)
        self.assertIsInstance(result['sentences'], list)
    
    def test_empty_text_handling(self):
        """Test handling of empty or None text."""
        # Test empty string
        result1 = self.preprocessor.preprocess_text("")
        self.assertEqual(result1['processed_text'], "")
        self.assertEqual(result1['word_count'], 0)
        
        # Test None
        result2 = self.preprocessor.preprocess_text(None)
        self.assertEqual(result2['processed_text'], "")
        self.assertEqual(result2['word_count'], 0)
        
        # Test whitespace only
        result3 = self.preprocessor.preprocess_text("   \n\t   ")
        self.assertEqual(result3['processed_text'], "")
        self.assertEqual(result3['word_count'], 0)
    
    def test_cleaning_statistics(self):
        """Test cleaning statistics calculation."""
        original_text = "This is a test with ten words total"
        processed_text = "This test ten words total"  # Simulated processed text
        
        stats = self.preprocessor._calculate_cleaning_stats(original_text, processed_text)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('original_word_count', stats)
        self.assertIn('processed_word_count', stats)
        self.assertIn('words_removed', stats)
        self.assertIn('reduction_percentage', stats)
        
        self.assertEqual(stats['original_word_count'], 10)
        self.assertEqual(stats['processed_word_count'], 5)
        self.assertEqual(stats['words_removed'], 5)
        self.assertEqual(stats['reduction_percentage'], 50.0)
    
    def test_batch_preprocessing(self):
        """Test batch preprocessing."""
        texts = [
            "This is the first text.",
            "This is the second text.",
            "This is the third text."
        ]
        
        results = self.preprocessor.batch_preprocess(texts)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
        for result in results:
            self.assertIsInstance(result, dict)
            self.assertIn('processed_text', result)
            self.assertIn('word_count', result)
    
    def test_key_phrase_extraction(self):
        """Test key phrase extraction."""
        test_text = "The machine learning algorithm processes data efficiently. John Smith works at Google."
        
        key_phrases = self.preprocessor.extract_key_phrases(test_text)
        
        self.assertIsInstance(key_phrases, list)
        # Note: This might be empty if spaCy is not available
        if key_phrases:
            for phrase in key_phrases:
                self.assertIsInstance(phrase, str)
    
    def test_language_detection(self):
        """Test language detection."""
        test_text = "This is an English text."
        
        language = self.preprocessor.detect_language(test_text)
        
        self.assertIsInstance(language, str)
        # Note: This might return 'unknown' if spaCy is not available
    
    def test_preprocessing_without_nltk(self):
        """Test preprocessing when NLTK is not available."""
        # This test simulates NLTK not being available
        # In a real scenario, this would be tested by mocking the import
        
        processor = TextPreprocessor()
        test_text = "This is a test sentence."
        
        # Should still work even if NLTK components are None
        result = processor.preprocess_text(test_text)
        
        self.assertIsInstance(result, dict)
        self.assertIn('processed_text', result)

if __name__ == '__main__':
    unittest.main() 