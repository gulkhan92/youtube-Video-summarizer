"""
Summarization module for generating video summaries using LLMs
Supports Google Gemini and OpenAI GPT models
"""

import os
from typing import Optional
from abc import ABC, abstractmethod

# Import with error handling
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Gemini summarizer unavailable.")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. OpenAI summarizer unavailable.")


# Custom exceptions
class SummarizerError(Exception):
    """Base exception for summarization errors"""
    pass

class APIKeyError(SummarizerError):
    """API key is missing or invalid"""
    pass

class ModelNotAvailableError(SummarizerError):
    """Requested summarization model is not available"""
    pass


# Prompt templates
SUMMARIZATION_PROMPT = """You are a professional video summarizer. Create a clear, concise summary of the following video transcript.

**Guidelines:**
- Focus on the main topic and key points
- Include important examples or evidence
- Keep the summary between {max_length} and {min_length} words
- Use bullet points for better readability
- Write in clear, professional language

**Transcript:**
{transcript}

**Summary:**
"""

BULLET_POINT_PROMPT = """Based on the transcript below, extract the 5-7 most important points as bullet points.

Transcript:
{transcript}

Key Points:
"""


class BaseSummarizer(ABC):
    """Abstract base class for all summarizers"""
    
    @abstractmethod
    def summarize(self, text: str, max_length: int = 300) -> str:
        """Generate summary of the given text"""
        pass
    
    @abstractmethod
    def get_bullet_points(self, text: str) -> list:
        """Extract key bullet points from text"""
        pass


class GeminiSummarizer(BaseSummarizer):
    """Summarizer using Google's Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        """
        Initialize Gemini summarizer.
        
        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model_name: Gemini model to use
        """
        if not GEMINI_AVAILABLE:
            raise ModelNotAvailableError(
                "Google Generative AI not installed. Run: pip install google-generativeai"
            )
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise APIKeyError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    def summarize(self, text: str, max_length: int = 300, min_length: int = 100) -> str:
        """
        Generate summary using Gemini.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length in words
            min_length: Minimum summary length in words
            
        Returns:
            Generated summary
        """
        if not text or len(text.strip()) < 50:
            return "Transcript too short to summarize meaningfully."
        
        # Truncate if too long (Gemini has 1M token limit, but let's be safe)
        max_chars = 30000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        # Create prompt
        prompt = SUMMARIZATION_PROMPT.format(
            transcript=text,
            max_length=max_length,
            min_length=min_length
        )
        
        try:
            response = self.model.generate_content(prompt)
            summary = response.text.strip()
            
            # Fallback if response is empty
            if not summary:
                summary = "Unable to generate summary. Please try again."
            
            return summary
            
        except Exception as e:
            raise SummarizerError(f"Gemini summarization failed: {str(e)}")
    
    def get_bullet_points(self, text: str) -> list:
        """Extract bullet points using Gemini"""
        if not text:
            return ["No content to extract points from."]
        
        prompt = BULLET_POINT_PROMPT.format(transcript=text[:20000])
        
        try:
            response = self.model.generate_content(prompt)
            bullet_text = response.text.strip()
            
            # Parse bullet points (handle different bullet formats)
            bullets = []
            for line in bullet_text.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                    bullets.append(line.lstrip('-•* ').strip())
                elif line and len(bullets) > 0 and not line[0].isdigit():
                    # Continuation of previous bullet
                    bullets[-1] += " " + line
            
            return bullets if bullets else ["No key points extracted."]
            
        except Exception as e:
            return [f"Failed to extract bullet points: {str(e)}"]


class OpenAISummarizer(BaseSummarizer):
    """Summarizer using OpenAI's GPT API"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI summarizer.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model_name: GPT model to use (gpt-3.5-turbo, gpt-4, etc.)
        """
        if not OPENAI_AVAILABLE:
            raise ModelNotAvailableError(
                "OpenAI not installed. Run: pip install openai"
            )
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise APIKeyError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
    
    def _call_gpt(self, prompt: str, max_tokens: int = 500) -> str:
        """Make API call to GPT"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful video summarizer assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3  # Lower temperature for more consistent summaries
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise SummarizerError(f"OpenAI API call failed: {str(e)}")
    
    def summarize(self, text: str, max_length: int = 300, min_length: int = 100) -> str:
        """Generate summary using GPT"""
        if not text or len(text.strip()) < 50:
            return "Transcript too short to summarize meaningfully."
        
        # Truncate to fit token limits (rough estimate: 4 chars per token)
        max_chars = 12000  # ~3000 tokens
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        prompt = SUMMARIZATION_PROMPT.format(
            transcript=text,
            max_length=max_length,
            min_length=min_length
        )
        
        return self._call_gpt(prompt, max_tokens=400)
    
    def get_bullet_points(self, text: str) -> list:
        """Extract bullet points using GPT"""
        if not text:
            return ["No content to extract points from."]
        
        prompt = BULLET_POINT_PROMPT.format(transcript=text[:10000])
        response = self._call_gpt(prompt, max_tokens=300)
        
        # Parse bullet points
        bullets = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*') or line.startswith('-')):
                bullets.append(line.lstrip('-•* ').strip())
        
        return bullets if bullets else [response[:200]]


class SimpleLocalSummarizer(BaseSummarizer):
    """Fallback summarizer that doesn't require API calls"""
    
    def summarize(self, text: str, max_length: int = 300, min_length: int = 100) -> str:
        """Very simple extractive summarization - takes first few sentences"""
        if not text:
            return "No transcript available."
        
        # Simple: take first 3-5 sentences
        sentences = text.split('. ')
        num_sentences = min(5, len(sentences))
        summary = '. '.join(sentences[:num_sentences])
        
        if len(summary) > max_length * 6:  # Rough word count
            summary = summary[:max_length * 6] + "..."
        
        return summary + "\n\n(Note: This is a basic extractive summary. For better results, add API keys.)"
    
    def get_bullet_points(self, text: str) -> list:
        """Extract first few sentences as bullet points"""
        if not text:
            return ["No content available."]
        
        sentences = text.split('. ')
        bullets = [s.strip() + '.' for s in sentences[:5] if s.strip()]
        return bullets[:5]


def get_summarizer(model_type: str = "gemini") -> BaseSummarizer:
    """
    Factory function to get the appropriate summarizer.
    
    Args:
        model_type: One of 'gemini', 'openai', 'local'
        
    Returns:
        Summarizer instance
        
    Raises:
        ModelNotAvailableError: If requested model is not available
    """
    
    model_type = model_type.lower()
    
    if model_type == "gemini":
        try:
            return GeminiSummarizer()
        except (APIKeyError, ModelNotAvailableError) as e:
            print(f"Gemini not available: {e}. Falling back to local.")
            return SimpleLocalSummarizer()
    
    elif model_type == "openai":
        try:
            return OpenAISummarizer()
        except (APIKeyError, ModelNotAvailableError) as e:
            print(f"OpenAI not available: {e}. Falling back to local.")
            return SimpleLocalSummarizer()
    
    elif model_type == "local":
        return SimpleLocalSummarizer()
    
    else:
        raise ModelNotAvailableError(f"Unknown model type: {model_type}. Use 'gemini', 'openai', or 'local'")


# Simplified function for direct use in app.py
def summarize_text(
    transcript: str, 
    model: str = "gemini", 
    max_length: int = 300
) -> str:
    """
    Convenience function to summarize text.
    
    Args:
        transcript: Text to summarize
        model: Model to use ('gemini', 'openai', 'local')
        max_length: Maximum summary length in words
        
    Returns:
        Summary text
    """
    summarizer = get_summarizer(model)
    return summarizer.summarize(transcript, max_length=max_length)


# Test the module
if __name__ == "__main__":
    print("Testing Summarizer Module...")
    
    test_text = """
    This is a test transcript about Python programming. Python is a high-level programming language 
    that is widely used for web development, data science, and artificial intelligence. 
    It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability 
    with its use of significant indentation. The language has a large standard library and 
    supports multiple programming paradigms including object-oriented and functional programming.
    """
    
    # Test each available summarizer
    for model_name in ["local", "gemini", "openai"]:
        print(f"\n--- Testing {model_name.upper()} ---")
        try:
            summarizer = get_summarizer(model_name)
            summary = summarizer.summarize(test_text, max_length=50)
            print(f"Summary: {summary[:200]}...")
        except Exception as e:
            print(f"Error with {model_name}: {e}")