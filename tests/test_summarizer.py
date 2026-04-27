"""Tests for modules/summarizer.py — AI summarizers, factory, prompts, error handling."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.append(str(Path(__file__).parent.parent))

from modules.summarizer import (
    get_summarizer,
    summarize_text,
    SimpleLocalSummarizer,
    GeminiNewSDKSummarizer,
    GeminiLegacySummarizer,
    OpenAISummarizer,
    SummarizerError,
    APIKeyError,
    ModelNotAvailableError,
    SUMMARIZATION_PROMPT,
    BULLET_POINT_PROMPT,
)


class TestSimpleLocalSummarizer:
    """Test the fallback local summarizer (no API calls)."""

    def test_basic_summarization(self):
        summarizer = SimpleLocalSummarizer()
        text = "Python is great. It is used for data science. It supports AI. It has many libraries. It is easy to learn."
        result = summarizer.summarize(text, max_length=300)
        assert "Python" in result
        assert "Note: This is a basic extractive summary" in result

    def test_empty_text(self):
        summarizer = SimpleLocalSummarizer()
        result = summarizer.summarize("")
        assert result == "No transcript available."

    def test_bullet_points(self):
        summarizer = SimpleLocalSummarizer()
        text = "Point one. Point two. Point three. Point four. Point five. Point six."
        bullets = summarizer.get_bullet_points(text)
        assert len(bullets) > 0
        assert all(b.strip().endswith(".") for b in bullets)

    def test_bullet_points_empty(self):
        summarizer = SimpleLocalSummarizer()
        bullets = summarizer.get_bullet_points("")
        assert bullets == ["No content available."]

    def test_respects_max_length(self):
        summarizer = SimpleLocalSummarizer()
        long_text = "Word. " * 1000
        result = summarizer.summarize(long_text, max_length=10)
        # Should truncate with ellipsis if too long
        assert len(result) <= 10 * 6 + 50  # rough upper bound


class TestGeminiNewSDKSummarizer:
    """Test Gemini summarizer with mocked google-genai SDK."""

    @patch("modules.summarizer.GOOGLE_GENAI_AVAILABLE", True)
    @patch("modules.summarizer.google_genai.Client")
    def test_successful_summarization(self, mock_client_class):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a generated summary."
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        summarizer = GeminiNewSDKSummarizer(api_key="test-key")
        text = "Python is a programming language. It is widely used."
        result = summarizer.summarize(text, max_length=100)
        assert result == "This is a generated summary."
        mock_client.models.generate_content.assert_called_once()

    @patch("modules.summarizer.GOOGLE_GENAI_AVAILABLE", True)
    @patch("modules.summarizer.google_genai.Client")
    def test_short_text_returns_message(self, mock_client_class):
        summarizer = GeminiNewSDKSummarizer(api_key="test-key")
        result = summarizer.summarize("Hi", max_length=100)
        assert "too short" in result.lower()

    @patch("modules.summarizer.GOOGLE_GENAI_AVAILABLE", True)
    @patch("modules.summarizer.google_genai.Client")
    def test_empty_response_fallback(self, mock_client_class):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = ""
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        summarizer = GeminiNewSDKSummarizer(api_key="test-key")
        text = "A" * 100
        result = summarizer.summarize(text, max_length=100)
        assert "unable to generate" in result.lower()

    @patch("modules.summarizer.GOOGLE_GENAI_AVAILABLE", True)
    @patch("modules.summarizer.google_genai.Client")
    def test_api_error_raises(self, mock_client_class):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API quota exceeded")
        mock_client_class.return_value = mock_client

        summarizer = GeminiNewSDKSummarizer(api_key="test-key")
        with pytest.raises(SummarizerError) as exc_info:
            summarizer.summarize("A" * 100, max_length=100)
        assert "failed" in str(exc_info.value).lower()

    @patch("modules.summarizer.GOOGLE_GENAI_AVAILABLE", True)
    @patch("modules.summarizer.google_genai.Client")
    def test_bullet_points(self, mock_client_class):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "- Point one\n- Point two\n- Point three"
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        summarizer = GeminiNewSDKSummarizer(api_key="test-key")
        bullets = summarizer.get_bullet_points("Some transcript text")
        assert len(bullets) == 3
        assert "Point one" in bullets

    @patch("modules.summarizer.GOOGLE_GENAI_AVAILABLE", True)
    @patch("modules.summarizer.google_genai.Client")
    def test_long_text_truncation(self, mock_client_class):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Summary"
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client

        summarizer = GeminiNewSDKSummarizer(api_key="test-key")
        very_long_text = "A" * 50000
        summarizer.summarize(very_long_text, max_length=100)

        # Check that the prompt was called with truncated text
        call_args = mock_client.models.generate_content.call_args
        prompt = call_args.kwargs.get("contents") or call_args[1].get("contents")
        assert "..." in prompt or len(prompt) < 50000

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(APIKeyError):
                GeminiNewSDKSummarizer(api_key=None)

    def test_sdk_not_available_raises(self):
        with patch("modules.summarizer.GOOGLE_GENAI_AVAILABLE", False):
            with pytest.raises(ModelNotAvailableError):
                GeminiNewSDKSummarizer(api_key="test-key")


class TestGeminiLegacySummarizer:
    """Test legacy Gemini summarizer with mocked google-generativeai SDK."""

    @patch("modules.summarizer.LEGACY_GENAI_AVAILABLE", True)
    @patch("modules.summarizer.legacy_genai")
    def test_successful_summarization(self, mock_legacy):
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Legacy summary."
        mock_model.generate_content.return_value = mock_response
        mock_legacy.GenerativeModel.return_value = mock_model

        summarizer = GeminiLegacySummarizer(api_key="test-key")
        text = "This is a test transcript about Python programming. Python is a high-level language widely used for web development and AI."
        result = summarizer.summarize(text, max_length=100)
        assert result == "Legacy summary."

    @patch("modules.summarizer.LEGACY_GENAI_AVAILABLE", True)
    @patch("modules.summarizer.legacy_genai")
    def test_api_error_raises(self, mock_legacy):
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("Network error")
        mock_legacy.GenerativeModel.return_value = mock_model

        summarizer = GeminiLegacySummarizer(api_key="test-key")
        text = "This is a test transcript about Python programming. Python is a high-level language widely used for web development and AI."
        with pytest.raises(SummarizerError) as exc_info:
            summarizer.summarize(text, max_length=100)
        assert "failed" in str(exc_info.value).lower()

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(APIKeyError):
                GeminiLegacySummarizer(api_key=None)

    def test_sdk_not_available_raises(self):
        with patch("modules.summarizer.LEGACY_GENAI_AVAILABLE", False):
            with pytest.raises(ModelNotAvailableError):
                GeminiLegacySummarizer(api_key="test-key")


class TestOpenAISummarizer:
    """Test OpenAI summarizer with mocked OpenAI client."""

    @patch("modules.summarizer.OPENAI_AVAILABLE", True)
    @patch("modules.summarizer.OpenAI")
    def test_successful_summarization(self, mock_openai_class):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "OpenAI generated summary."
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        summarizer = OpenAISummarizer(api_key="test-key")
        text = "This is a test transcript about Python programming. Python is a high-level language widely used for web development and AI."
        result = summarizer.summarize(text, max_length=100)
        assert result == "OpenAI generated summary."
        mock_client.chat.completions.create.assert_called_once()

    @patch("modules.summarizer.OPENAI_AVAILABLE", True)
    @patch("modules.summarizer.OpenAI")
    def test_api_error_raises(self, mock_openai_class):
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Rate limit")
        mock_openai_class.return_value = mock_client

        summarizer = OpenAISummarizer(api_key="test-key")
        text = "This is a test transcript about Python programming. Python is a high-level language widely used for web development and AI."
        with pytest.raises(SummarizerError) as exc_info:
            summarizer.summarize(text, max_length=100)
        assert "failed" in str(exc_info.value).lower()

    @patch("modules.summarizer.OPENAI_AVAILABLE", True)
    @patch("modules.summarizer.OpenAI")
    def test_short_text_returns_message(self, mock_openai_class):
        summarizer = OpenAISummarizer(api_key="test-key")
        result = summarizer.summarize("Hi", max_length=100)
        assert "too short" in result.lower()

    @patch("modules.summarizer.OPENAI_AVAILABLE", True)
    @patch("modules.summarizer.OpenAI")
    def test_bullet_points(self, mock_openai_class):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "- Key insight one\n- Key insight two"
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        summarizer = OpenAISummarizer(api_key="test-key")
        bullets = summarizer.get_bullet_points("Some text")
        assert len(bullets) == 2
        assert "Key insight one" in bullets

    def test_missing_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(APIKeyError):
                OpenAISummarizer(api_key=None)

    def test_sdk_not_available_raises(self):
        with patch("modules.summarizer.OPENAI_AVAILABLE", False):
            with pytest.raises(ModelNotAvailableError):
                OpenAISummarizer(api_key="test-key")


class TestGetSummarizerFactory:
    """Test the summarizer factory function."""

    @patch("modules.summarizer.GOOGLE_GENAI_AVAILABLE", True)
    @patch("modules.summarizer.GeminiNewSDKSummarizer")
    def test_factory_gemini_new_sdk(self, mock_gemini_class):
        mock_instance = MagicMock()
        mock_gemini_class.return_value = mock_instance
        result = get_summarizer("gemini")
        assert result == mock_instance

    @patch("modules.summarizer.GOOGLE_GENAI_AVAILABLE", False)
    @patch("modules.summarizer.LEGACY_GENAI_AVAILABLE", True)
    @patch("modules.summarizer.GeminiLegacySummarizer")
    def test_factory_gemini_legacy_fallback(self, mock_legacy_class):
        mock_instance = MagicMock()
        mock_legacy_class.return_value = mock_instance
        result = get_summarizer("gemini")
        assert result == mock_instance

    @patch("modules.summarizer.GOOGLE_GENAI_AVAILABLE", False)
    @patch("modules.summarizer.LEGACY_GENAI_AVAILABLE", False)
    def test_factory_gemini_local_fallback(self):
        result = get_summarizer("gemini")
        assert isinstance(result, SimpleLocalSummarizer)

    @patch("modules.summarizer.OPENAI_AVAILABLE", True)
    @patch("modules.summarizer.OpenAISummarizer")
    def test_factory_openai(self, mock_openai_class):
        mock_instance = MagicMock()
        mock_openai_class.return_value = mock_instance
        result = get_summarizer("openai")
        assert result == mock_instance

    @patch("modules.summarizer.OPENAI_AVAILABLE", False)
    def test_factory_openai_fallback(self):
        result = get_summarizer("openai")
        assert isinstance(result, SimpleLocalSummarizer)

    def test_factory_local(self):
        result = get_summarizer("local")
        assert isinstance(result, SimpleLocalSummarizer)

    def test_factory_unknown_model_raises(self):
        with pytest.raises(ModelNotAvailableError):
            get_summarizer("unknown_model")

    def test_factory_case_insensitive(self):
        with patch("modules.summarizer.GOOGLE_GENAI_AVAILABLE", True):
            with patch("modules.summarizer.GeminiNewSDKSummarizer") as mock_gemini:
                mock_instance = MagicMock()
                mock_gemini.return_value = mock_instance
                result = get_summarizer("GEMINI")
                assert result == mock_instance


class TestSummarizeText:
    """Test the convenience summarize_text function."""

    @patch("modules.summarizer.get_summarizer")
    def test_summarize_text(self, mock_get_summarizer):
        mock_summarizer = MagicMock()
        mock_summarizer.summarize.return_value = "Final summary"
        mock_get_summarizer.return_value = mock_summarizer

        result = summarize_text("Some transcript", model="gemini", max_length=200)
        assert result == "Final summary"
        mock_get_summarizer.assert_called_once_with("gemini")
        mock_summarizer.summarize.assert_called_once_with("Some transcript", max_length=200)


class TestPromptTemplates:
    """Test prompt template formatting."""

    def test_summarization_prompt_formatting(self):
        prompt = SUMMARIZATION_PROMPT.format(
            transcript="Test transcript",
            max_length=300,
            min_length=100
        )
        assert "Test transcript" in prompt
        assert "300" in prompt
        assert "100" in prompt
        assert "Summary:" in prompt

    def test_bullet_point_prompt_formatting(self):
        prompt = BULLET_POINT_PROMPT.format(transcript="Test content")
        assert "Test content" in prompt
        assert "Key Points:" in prompt


class TestSummarizerErrorHierarchy:
    """Test custom exception classes."""

    def test_summarizer_error_is_exception(self):
        assert issubclass(SummarizerError, Exception)

    def test_api_key_error_is_summarizer_error(self):
        assert issubclass(APIKeyError, SummarizerError)

    def test_model_not_available_error_is_summarizer_error(self):
        assert issubclass(ModelNotAvailableError, SummarizerError)

