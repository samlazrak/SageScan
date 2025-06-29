"""
LLM processor for sentiment analysis and summary generation.
"""

import os
import json
import requests
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass

# Try to import LLM libraries
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain.llms import OpenAI as LangchainOpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Result of LLM analysis."""
    summary: str
    sentiment: str
    sentiment_score: float
    key_topics: List[str]
    action_items: List[str]
    confidence: float
    model_used: str

class LocalLLMClient:
    """Client for local LLM servers (Ollama and llama.cpp)."""
    
    def __init__(self, provider: str, base_url: str, model: str):
        self.provider = provider
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        self.session.timeout = 60
    
    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 500) -> str:
        """Generate text using local LLM server."""
        if self.provider == "ollama":
            return self._generate_ollama(prompt, temperature, max_tokens)
        elif self.provider == "llama-cpp":
            return self._generate_llama_cpp(prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported local provider: {self.provider}")
    
    def _generate_ollama(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate text using Ollama API."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            raise
    
    def _generate_llama_cpp(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Generate text using llama.cpp API."""
        try:
            payload = {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stop": ["</s>", "Human:", "Assistant:"],
                "stream": False
            }
            
            response = self.session.post(
                f"{self.base_url}/completion",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("content", "")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"llama.cpp API request failed: {e}")
            raise

class LLMProcessor:
    """Handles LLM-based text analysis including sentiment and summarization."""
    
    def __init__(self, 
                 provider: str = "local",
                 model: str = "llama2",
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        """
        Initialize LLM processor.
        
        Args:
            provider: LLM provider ("openai", "anthropic", "local", "ollama", "llama-cpp")
            model: Model name to use
            api_key: API key for the provider (not needed for local LLMs)
            base_url: Base URL for local LLM servers
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        
        # Initialize client based on provider
        self.client = self._initialize_client(base_url)
        
        # Analysis prompts
        self.summary_prompt = self._get_summary_prompt()
        self.sentiment_prompt = self._get_sentiment_prompt()
        self.comprehensive_prompt = self._get_comprehensive_prompt()
    
    def _initialize_client(self, base_url: Optional[str] = None):
        """Initialize the appropriate LLM client."""
        if self.provider == "openai" and OPENAI_AVAILABLE:
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            return OpenAI(api_key=api_key) if api_key else None
            
        elif self.provider == "openai" and LANGCHAIN_AVAILABLE:
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            return ChatOpenAI(
                model_name=self.model,
                openai_api_key=api_key,
                temperature=0.3
            ) if api_key else None
            
        elif self.provider in ["ollama", "local"]:
            # For "local" provider, default to ollama
            actual_provider = "ollama" if self.provider == "local" else self.provider
            base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            return LocalLLMClient(actual_provider, base_url, self.model)
            
        elif self.provider == "llama-cpp":
            base_url = base_url or os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080")
            return LocalLLMClient(self.provider, base_url, self.model)
            
        else:
            logger.warning(f"Provider {self.provider} not available or not configured")
            return None
    
    def analyze_text(self, text: str) -> AnalysisResult:
        """
        Perform comprehensive text analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            AnalysisResult with summary, sentiment, and other insights
        """
        if not text or not text.strip():
            return AnalysisResult(
                summary="No text to analyze",
                sentiment="neutral",
                sentiment_score=0.0,
                key_topics=[],
                action_items=[],
                confidence=0.0,
                model_used="none"
            )
        
        try:
            if self.provider == "openai" and isinstance(self.client, OpenAI):
                return self._analyze_with_openai(text)
            elif self.provider == "openai" and LANGCHAIN_AVAILABLE:
                return self._analyze_with_langchain(text)
            elif self.provider in ["ollama", "llama-cpp", "local"]:
                return self._analyze_with_local_llm(text)
            else:
                return self._analyze_with_fallback(text)
                
        except Exception as e:
            logger.error(f"LLM analysis failed: {str(e)}")
            return self._analyze_with_fallback(text)
    
    def _analyze_with_local_llm(self, text: str) -> AnalysisResult:
        """Analyze text using local LLM server."""
        try:
            # Create comprehensive analysis prompt
            prompt = self.comprehensive_prompt.format(text=text[:4000])  # Limit text length
            
            # Get temperature and max_tokens from environment
            temperature = float(os.getenv("TEMPERATURE", "0.3"))
            max_tokens = int(os.getenv("MAX_TOKENS", "500"))
            
            # Generate response
            result_text = self.client.generate(prompt, temperature, max_tokens)
            
            # Parse the response
            return self._parse_analysis_response(result_text)
            
        except Exception as e:
            logger.error(f"Local LLM analysis failed: {str(e)}")
            raise
    
    def _analyze_with_openai(self, text: str) -> AnalysisResult:
        """Analyze text using OpenAI API."""
        try:
            # Create comprehensive analysis prompt
            prompt = self.comprehensive_prompt.format(text=text[:4000])  # Limit text length
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes text for sentiment, summary, and key insights."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content
            
            # Parse the response
            return self._parse_analysis_response(result_text)
            
        except Exception as e:
            logger.error(f"OpenAI analysis failed: {str(e)}")
            raise
    
    def _analyze_with_langchain(self, text: str) -> AnalysisResult:
        """Analyze text using LangChain."""
        try:
            prompt = self.comprehensive_prompt.format(text=text[:4000])
            
            messages = [
                SystemMessage(content="You are a helpful assistant that analyzes text for sentiment, summary, and key insights."),
                HumanMessage(content=prompt)
            ]
            
            response = self.client(messages)
            result_text = response.content
            
            return self._parse_analysis_response(result_text)
            
        except Exception as e:
            logger.error(f"LangChain analysis failed: {str(e)}")
            raise
    
    def _analyze_with_fallback(self, text: str) -> AnalysisResult:
        """Fallback analysis using simple heuristics."""
        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'positive', 'happy', 'success']
        negative_words = ['bad', 'terrible', 'negative', 'sad', 'failure', 'problem']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            sentiment_score = 0.7
        elif negative_count > positive_count:
            sentiment = "negative"
            sentiment_score = -0.7
        else:
            sentiment = "neutral"
            sentiment_score = 0.0
        
        # Simple summary (first few sentences)
        sentences = text.split('.')[:3]
        summary = '. '.join(sentences) + '.'
        
        return AnalysisResult(
            summary=summary,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            key_topics=[],
            action_items=[],
            confidence=0.3,
            model_used="fallback"
        )
    
    def _parse_analysis_response(self, response_text: str) -> AnalysisResult:
        """Parse the LLM response into structured format."""
        try:
            # Try to parse as JSON first
            if response_text.strip().startswith('{'):
                data = json.loads(response_text)
                return AnalysisResult(
                    summary=data.get('summary', ''),
                    sentiment=data.get('sentiment', 'neutral'),
                    sentiment_score=data.get('sentiment_score', 0.0),
                    key_topics=data.get('key_topics', []),
                    action_items=data.get('action_items', []),
                    confidence=data.get('confidence', 0.8),
                    model_used=self.model
                )
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing
        lines = response_text.split('\n')
        summary = ""
        sentiment = "neutral"
        sentiment_score = 0.0
        key_topics = []
        action_items = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('Summary:'):
                summary = line.replace('Summary:', '').strip()
            elif line.startswith('Sentiment:'):
                sentiment = line.replace('Sentiment:', '').strip()
            elif line.startswith('Topics:'):
                topics_text = line.replace('Topics:', '').strip()
                key_topics = [t.strip() for t in topics_text.split(',') if t.strip()]
            elif line.startswith('Actions:'):
                actions_text = line.replace('Actions:', '').strip()
                action_items = [a.strip() for a in actions_text.split(',') if a.strip()]
        
        return AnalysisResult(
            summary=summary,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            key_topics=key_topics,
            action_items=action_items,
            confidence=0.7,
            model_used=self.model
        )
    
    def _get_summary_prompt(self) -> str:
        """Get prompt for summary generation."""
        return """
        Please provide a concise summary of the following text in 2-3 sentences:
        
        {text}
        
        Summary:
        """
    
    def _get_sentiment_prompt(self) -> str:
        """Get prompt for sentiment analysis."""
        return """
        Analyze the sentiment of the following text. Provide:
        1. Overall sentiment (positive, negative, neutral)
        2. Sentiment score (-1 to 1, where -1 is very negative, 1 is very positive)
        3. Brief explanation
        
        Text: {text}
        
        Analysis:
        """
    
    def _get_comprehensive_prompt(self) -> str:
        """Get comprehensive analysis prompt."""
        return """
        Analyze the following text and provide a structured response in JSON format:
        
        {text}
        
        Please provide:
        1. A concise summary (2-3 sentences)
        2. Overall sentiment (positive, negative, neutral)
        3. Sentiment score (-1 to 1)
        4. Key topics or themes mentioned
        5. Any action items or important points
        6. Confidence level (0-1)
        
        Respond in this JSON format:
        {{
            "summary": "brief summary here",
            "sentiment": "positive/negative/neutral",
            "sentiment_score": 0.5,
            "key_topics": ["topic1", "topic2"],
            "action_items": ["action1", "action2"],
            "confidence": 0.8
        }}
        """
    
    def batch_analyze(self, texts: List[str]) -> List[AnalysisResult]:
        """Analyze multiple texts."""
        results = []
        
        for text in texts:
            try:
                result = self.analyze_text(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze text: {str(e)}")
                results.append(AnalysisResult(
                    summary="Analysis failed",
                    sentiment="neutral",
                    sentiment_score=0.0,
                    key_topics=[],
                    action_items=[],
                    confidence=0.0,
                    model_used="error"
                ))
        
        return results
    
    def get_sentiment_only(self, text: str) -> Dict[str, Any]:
        """Get only sentiment analysis."""
        result = self.analyze_text(text)
        return {
            'sentiment': result.sentiment,
            'sentiment_score': result.sentiment_score,
            'confidence': result.confidence
        }
    
    def get_summary_only(self, text: str) -> str:
        """Get only summary."""
        result = self.analyze_text(text)
        return result.summary 