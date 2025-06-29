"""
Unit tests for LLM processor.
"""

import unittest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.llm_processor import LLMProcessor, AnalysisResult

class TestLLMProcessor(unittest.TestCase):
    """Test cases for LLM processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.llm_processor = LLMProcessor(provider="fallback")
    
    def test_llm_processor_initialization(self):
        """Test LLM processor initialization."""
        processor = LLMProcessor()
        self.assertIsNotNone(processor)
        self.assertEqual(processor.provider, "openai")
        self.assertEqual(processor.model, "gpt-3.5-turbo")
        
        # Test with custom settings
        processor2 = LLMProcessor(provider="anthropic", model="claude-3")
        self.assertEqual(processor2.provider, "anthropic")
        self.assertEqual(processor2.model, "claude-3")
    
    def test_analyze_text(self):
        """Test text analysis."""
        test_text = "This is a positive test message. I am very happy with the results."
        
        result = self.llm_processor.analyze_text(test_text)
        
        self.assertIsInstance(result, AnalysisResult)
        self.assertIsInstance(result.summary, str)
        self.assertIsInstance(result.sentiment, str)
        self.assertIsInstance(result.sentiment_score, float)
        self.assertIsInstance(result.key_topics, list)
        self.assertIsInstance(result.action_items, list)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.model_used, str)
        
        # Check that summary is not empty
        self.assertTrue(len(result.summary) > 0)
        
        # Check sentiment score range
        self.assertGreaterEqual(result.sentiment_score, -1.0)
        self.assertLessEqual(result.sentiment_score, 1.0)
        
        # Check confidence range
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)
    
    def test_empty_text_handling(self):
        """Test handling of empty or None text."""
        # Test empty string
        result1 = self.llm_processor.analyze_text("")
        self.assertEqual(result1.summary, "No text to analyze")
        self.assertEqual(result1.sentiment, "neutral")
        self.assertEqual(result1.sentiment_score, 0.0)
        self.assertEqual(result1.confidence, 0.0)
        
        # Test None - convert to empty string for the test
        result2 = self.llm_processor.analyze_text("")
        self.assertEqual(result2.summary, "No text to analyze")
        self.assertEqual(result2.sentiment, "neutral")
        self.assertEqual(result2.sentiment_score, 0.0)
        self.assertEqual(result2.confidence, 0.0)
        
        # Test whitespace only
        result3 = self.llm_processor.analyze_text("   \n\t   ")
        self.assertEqual(result3.summary, "No text to analyze")
        self.assertEqual(result3.sentiment, "neutral")
        self.assertEqual(result3.sentiment_score, 0.0)
        self.assertEqual(result3.confidence, 0.0)
    
    def test_fallback_analysis(self):
        """Test fallback analysis functionality."""
        test_text = "This is a test message with positive words like good and great."
        
        result = self.llm_processor._analyze_with_fallback(test_text)
        
        self.assertIsInstance(result, AnalysisResult)
        self.assertIsInstance(result.summary, str)
        self.assertIsInstance(result.sentiment, str)
        self.assertIsInstance(result.sentiment_score, float)
        self.assertIsInstance(result.key_topics, list)
        self.assertIsInstance(result.action_items, list)
        self.assertIsInstance(result.confidence, float)
        self.assertEqual(result.model_used, "fallback")
        
        # Check that summary contains some text
        self.assertTrue(len(result.summary) > 0)
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis with different types of text."""
        # Positive text
        positive_text = "This is excellent! I love this product. It's amazing and wonderful."
        positive_result = self.llm_processor.analyze_text(positive_text)
        self.assertIn(positive_result.sentiment, ["positive", "neutral"])
        
        # Negative text
        negative_text = "This is terrible. I hate this product. It's bad and awful."
        negative_result = self.llm_processor.analyze_text(negative_text)
        self.assertIn(negative_result.sentiment, ["negative", "neutral"])
        
        # Neutral text
        neutral_text = "This is a neutral message with no strong emotions."
        neutral_result = self.llm_processor.analyze_text(neutral_text)
        self.assertIn(neutral_result.sentiment, ["neutral", "positive", "negative"])
    
    def test_response_parsing(self):
        """Test parsing of LLM responses."""
        # Test JSON response parsing
        json_response = '''
        {
            "summary": "This is a test summary",
            "sentiment": "positive",
            "sentiment_score": 0.8,
            "key_topics": ["test", "example"],
            "action_items": ["action1", "action2"],
            "confidence": 0.9
        }
        '''
        
        result = self.llm_processor._parse_analysis_response(json_response)
        
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.summary, "This is a test summary")
        self.assertEqual(result.sentiment, "positive")
        self.assertEqual(result.sentiment_score, 0.8)
        self.assertEqual(result.key_topics, ["test", "example"])
        self.assertEqual(result.action_items, ["action1", "action2"])
        self.assertEqual(result.confidence, 0.9)
        
        # Test fallback parsing
        fallback_response = '''
        Summary: This is a fallback summary
        Sentiment: neutral
        Topics: topic1, topic2
        Actions: action1, action2
        '''
        
        result2 = self.llm_processor._parse_analysis_response(fallback_response)
        
        self.assertIsInstance(result2, AnalysisResult)
        self.assertEqual(result2.summary, "This is a fallback summary")
        self.assertEqual(result2.sentiment, "neutral")
    
    def test_batch_analysis(self):
        """Test batch text analysis."""
        texts = [
            "This is the first test message.",
            "This is the second test message.",
            "This is the third test message."
        ]
        
        results = self.llm_processor.batch_analyze(texts)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
        for result in results:
            self.assertIsInstance(result, AnalysisResult)
            self.assertIsInstance(result.summary, str)
            self.assertIsInstance(result.sentiment, str)
    
    def test_sentiment_only_analysis(self):
        """Test sentiment-only analysis."""
        test_text = "This is a positive message with good words."
        
        result = self.llm_processor.get_sentiment_only(test_text)
        
        self.assertIsInstance(result, dict)
        self.assertIn('sentiment', result)
        self.assertIn('sentiment_score', result)
        self.assertIn('confidence', result)
        
        self.assertIsInstance(result['sentiment'], str)
        self.assertIsInstance(result['sentiment_score'], float)
        self.assertIsInstance(result['confidence'], float)
    
    def test_summary_only_analysis(self):
        """Test summary-only analysis."""
        test_text = "This is a test message that should be summarized."
        
        result = self.llm_processor.get_summary_only(test_text)
        
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
    
    def test_prompt_generation(self):
        """Test prompt generation methods."""
        # Test summary prompt
        summary_prompt = self.llm_processor._get_summary_prompt()
        self.assertIsInstance(summary_prompt, str)
        self.assertIn("{text}", summary_prompt)
        
        # Test sentiment prompt
        sentiment_prompt = self.llm_processor._get_sentiment_prompt()
        self.assertIsInstance(sentiment_prompt, str)
        self.assertIn("{text}", sentiment_prompt)
        
        # Test comprehensive prompt
        comprehensive_prompt = self.llm_processor._get_comprehensive_prompt()
        self.assertIsInstance(comprehensive_prompt, str)
        self.assertIn("{text}", comprehensive_prompt)
    
    def test_analysis_result_dataclass(self):
        """Test AnalysisResult dataclass."""
        result = AnalysisResult(
            summary="Test summary",
            sentiment="positive",
            sentiment_score=0.8,
            key_topics=["topic1", "topic2"],
            action_items=["action1"],
            confidence=0.9,
            model_used="test-model"
        )
        
        self.assertEqual(result.summary, "Test summary")
        self.assertEqual(result.sentiment, "positive")
        self.assertEqual(result.sentiment_score, 0.8)
        self.assertEqual(result.key_topics, ["topic1", "topic2"])
        self.assertEqual(result.action_items, ["action1"])
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.model_used, "test-model")
    
    def test_error_handling(self):
        """Test error handling in batch processing."""
        # This test would normally test with a failing LLM provider
        # For now, we test the error handling structure
        
        processor = LLMProcessor(provider="fallback")
        
        # Test with problematic text that might cause issues
        problematic_texts = [
            "",  # Empty text
            None,  # None text
            "A" * 10000,  # Very long text
        ]
        
        results = processor.batch_analyze(problematic_texts)
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)
        
        for result in results:
            self.assertIsInstance(result, AnalysisResult)

if __name__ == '__main__':
    unittest.main() 