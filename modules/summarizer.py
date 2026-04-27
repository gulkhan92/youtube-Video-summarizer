"""
Summarization module for generating video summaries using LLMs.
Supports Google Gemini (new genai SDK primary, old fallback), OpenAI GPT, and local fallback.
"""

import os
from typing import Optional
from abc import ABC, abstractmethod

# ─── New Google SDK (matches main.py) ─────────────────────────────────────────
try:
    from google import genai as google_genai
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False

# ─── Old Google SDK (backward compatibility) ──────────────────────────────────
try:
    import google.generativeai as legacy_genai
    LEGACY_GENAI_AVAILABLE = True
except ImportError:
    LEGACY_GENAI_AVAILABLE = False

# ─── OpenAI ───────────────────────────────────────────────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ─── Custom exceptions ────────────────────────────────────────────────────────
class SummarizerError(Exception):
    """Base exception for summarization errors"""
    pass

class APIKeyError(SummarizerError):
    """API key is missing or invalid"""
    pass

class ModelNotAvailableError(SummarizerError):
    """Requested summarization model is not available"""
    pass


# ─── Prompt templates ─────────────────────────────────────────────────────────
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


# ─── Abstract base ────────────────────────────────────────────────────────────
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


# ─── Gemini (new SDK) ─────────────────────────────────────────────────────────
class GeminiNewSDKSummarizer(BaseSummarizer):
    """Summarizer using Google's NEW google-genai SDK (matches main.py)"""

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        if not GOOGLE_GENAI_AVAILABLE:
            raise ModelNotAvailableError(
                "google-genai not installed. Run: pip install google-genai"
            )

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise APIKeyError("GOOGLE_API_KEY not found in environment variables")

        self.client = google_genai.Client(api_key=self.api_key)
        self.model_name = model_name or self.DEFAULT_MODEL

    def summarize(self, text: str, max_length: int = 300, min_length: int = 100) -> str:
        if not text or len(text.strip()) < 50:
            return "Transcript too short to summarize meaningfully."

        max_chars = 30000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        prompt = SUMMARIZATION_PROMPT.format(
            transcript=text, max_length=max_length, min_length=min_length
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            summary = response.text.strip() if response.text else ""
            return summary or "Unable to generate summary. Please try again."
        except Exception as e:
            raise SummarizerError(f"Gemini summarization failed: {e}")

    def get_bullet_points(self, text: str) -> list:
        if not text:
            return ["No content to extract points from."]
        prompt = BULLET_POINT_PROMPT.format(transcript=text[:20000])
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return self._parse_bullets(response.text or "")
        except Exception as e:
            return [f"Failed to extract bullet points: {e}"]

    @staticmethod
    def _parse_bullets(text: str) -> list:
        bullets = []
        for line in text.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line.startswith("*")):
                bullets.append(line.lstrip("-•* ").strip())
            elif line and bullets and not line[0].isdigit():
                bullets[-1] += " " + line
        return bullets if bullets else ["No key points extracted."]


# ─── Gemini (legacy SDK) ──────────────────────────────────────────────────────
class GeminiLegacySummarizer(BaseSummarizer):
    """Summarizer using the OLD google-generativeai SDK"""

    DEFAULT_MODEL = "gemini-1.5-flash"

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        if not LEGACY_GENAI_AVAILABLE:
            raise ModelNotAvailableError(
                "google-generativeai not installed. Run: pip install google-generativeai"
            )

        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise APIKeyError("GOOGLE_API_KEY not found in environment variables")

        legacy_genai.configure(api_key=self.api_key)
        self.model = legacy_genai.GenerativeModel(model_name or self.DEFAULT_MODEL)

    def summarize(self, text: str, max_length: int = 300, min_length: int = 100) -> str:
        if not text or len(text.strip()) < 50:
            return "Transcript too short to summarize meaningfully."

        max_chars = 30000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        prompt = SUMMARIZATION_PROMPT.format(
            transcript=text, max_length=max_length, min_length=min_length
        )

        try:
            response = self.model.generate_content(prompt)
            summary = response.text.strip() if response.text else ""
            return summary or "Unable to generate summary. Please try again."
        except Exception as e:
            raise SummarizerError(f"Gemini summarization failed: {e}")

    def get_bullet_points(self, text: str) -> list:
        if not text:
            return ["No content to extract points from."]
        prompt = BULLET_POINT_PROMPT.format(transcript=text[:20000])
        try:
            response = self.model.generate_content(prompt)
            return self._parse_bullets(response.text or "")
        except Exception as e:
            return [f"Failed to extract bullet points: {e}"]

    @staticmethod
    def _parse_bullets(text: str) -> list:
        bullets = []
        for line in text.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line.startswith("*")):
                bullets.append(line.lstrip("-•* ").strip())
            elif line and bullets and not line[0].isdigit():
                bullets[-1] += " " + line
        return bullets if bullets else ["No key points extracted."]


# ─── OpenAI ───────────────────────────────────────────────────────────────────
class OpenAISummarizer(BaseSummarizer):
    """Summarizer using OpenAI's GPT API"""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-3.5-turbo"):
        if not OPENAI_AVAILABLE:
            raise ModelNotAvailableError(
                "openai not installed. Run: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise APIKeyError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name

    def _call_gpt(self, prompt: str, max_tokens: int = 500) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful video summarizer assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise SummarizerError(f"OpenAI API call failed: {e}")

    def summarize(self, text: str, max_length: int = 300, min_length: int = 100) -> str:
        if not text or len(text.strip()) < 50:
            return "Transcript too short to summarize meaningfully."

        max_chars = 12000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."

        prompt = SUMMARIZATION_PROMPT.format(
            transcript=text, max_length=max_length, min_length=min_length
        )
        return self._call_gpt(prompt, max_tokens=400)

    def get_bullet_points(self, text: str) -> list:
        if not text:
            return ["No content to extract points from."]
        prompt = BULLET_POINT_PROMPT.format(transcript=text[:10000])
        response = self._call_gpt(prompt, max_tokens=300)
        bullets = []
        for line in response.split("\n"):
            line = line.strip()
            if line and (line.startswith("-") or line.startswith("•") or line.startswith("*")):
                bullets.append(line.lstrip("-•* ").strip())
        return bullets if bullets else [response[:200]]


# ─── Local fallback ───────────────────────────────────────────────────────────
class SimpleLocalSummarizer(BaseSummarizer):
    """Fallback summarizer that doesn't require API calls"""

    def summarize(self, text: str, max_length: int = 300, min_length: int = 100) -> str:
        if not text:
            return "No transcript available."
        sentences = text.split(". ")
        num_sentences = min(5, len(sentences))
        summary = ". ".join(sentences[:num_sentences])
        if len(summary) > max_length * 6:
            summary = summary[:max_length * 6] + "..."
        return summary + "\n\n(Note: This is a basic extractive summary. For better results, add API keys.)"

    def get_bullet_points(self, text: str) -> list:
        if not text:
            return ["No content available."]
        sentences = text.split(". ")
        return [s.strip() + "." for s in sentences[:5] if s.strip()]


# ─── Factory ──────────────────────────────────────────────────────────────────
def get_summarizer(model_type: str = "gemini") -> BaseSummarizer:
    """
    Factory function to get the appropriate summarizer.
    Tries new Google SDK first, falls back to legacy SDK, then local.
    """
    model_type = model_type.lower()

    if model_type == "gemini":
        if GOOGLE_GENAI_AVAILABLE:
            try:
                return GeminiNewSDKSummarizer()
            except (APIKeyError, ModelNotAvailableError):
                pass  # Try legacy next
        if LEGACY_GENAI_AVAILABLE:
            try:
                return GeminiLegacySummarizer()
            except (APIKeyError, ModelNotAvailableError):
                pass  # Fall through to local
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
        raise ModelNotAvailableError(
            f"Unknown model type: {model_type}. Use 'gemini', 'openai', or 'local'"
        )


# ─── Convenience function ─────────────────────────────────────────────────────
def summarize_text(transcript: str, model: str = "gemini", max_length: int = 300) -> str:
    summarizer = get_summarizer(model)
    return summarizer.summarize(transcript, max_length=max_length)


# ─── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Summarizer Module...")

    test_text = (
        "This is a test transcript about Python programming. Python is a high-level programming language "
        "that is widely used for web development, data science, and artificial intelligence. "
        "It was created by Guido van Rossum and first released in 1991. Python emphasizes code readability "
        "with its use of significant indentation. The language has a large standard library and "
        "supports multiple programming paradigms including object-oriented and functional programming."
    )

    for model_name in ["local", "gemini", "openai"]:
        print(f"\n--- Testing {model_name.upper()} ---")
        try:
            summarizer = get_summarizer(model_name)
            summary = summarizer.summarize(test_text, max_length=50)
            print(f"Summary: {summary[:200]}...")
        except Exception as e:
            print(f"Error with {model_name}: {e}")
