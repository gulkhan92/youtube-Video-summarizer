"""Tests for modules/utils.py — caching, text cleaning, validation, helpers."""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

import sys
sys.path.append(str(Path(__file__).parent.parent))

from modules.utils import (
    clean_transcript,
    get_cache_key,
    load_cache,
    save_cache,
    check_cache,
    save_to_cache,
    clear_cache,
    estimate_reading_time,
    truncate_text,
    format_duration,
    validate_youtube_url,
)


class TestCleanTranscript:
    """Test transcript text cleaning and normalization."""

    def test_remove_extra_whitespace(self):
        raw = "  This   is    a   test  "
        cleaned = clean_transcript(raw)
        assert cleaned == "This is a test"

    def test_remove_bracketed_artifacts(self):
        raw = "Hello [Music] world [Applause]"
        cleaned = clean_transcript(raw)
        assert "[Music]" not in cleaned
        assert "[Applause]" not in cleaned

    def test_remove_parenthetical_artifacts(self):
        raw = "Hello (inaudible) world (laughter)"
        cleaned = clean_transcript(raw)
        assert "(inaudible)" not in cleaned
        assert "(laughter)" not in cleaned

    def test_fix_punctuation_spacing(self):
        raw = "Hello world , this is a test ."
        cleaned = clean_transcript(raw)
        assert "world ," not in cleaned
        assert "test ." not in cleaned

    def test_empty_string(self):
        assert clean_transcript("") == ""

    def test_capitalize_sentences(self):
        raw = "hello world. this is a test."
        cleaned = clean_transcript(raw)
        assert cleaned.startswith("Hello world")


class TestGetCacheKey:
    """Test cache key generation."""

    def test_consistent_hash(self):
        key1 = get_cache_key("https://youtube.com/watch?v=TEST", "gemini")
        key2 = get_cache_key("https://youtube.com/watch?v=TEST", "gemini")
        assert key1 == key2
        assert len(key1) == 32  # MD5 hex length

    def test_different_inputs_different_keys(self):
        key1 = get_cache_key("url1", "gemini")
        key2 = get_cache_key("url2", "gemini")
        assert key1 != key2

    def test_different_models_different_keys(self):
        key1 = get_cache_key("https://youtube.com/watch?v=TEST", "gemini")
        key2 = get_cache_key("https://youtube.com/watch?v=TEST", "openai")
        assert key1 != key2


class TestCacheOperations:
    """Test cache load, save, check, clear with temporary files."""

    @pytest.fixture(autouse=True)
    def temp_cache(self, tmp_path, monkeypatch):
        """Override CACHE_FILE to use a temporary path for all tests."""
        temp_file = tmp_path / "summaries.json"
        monkeypatch.setattr("modules.utils.CACHE_FILE", temp_file)
        yield temp_file

    def test_save_and_load_cache(self, temp_cache):
        data = {"key1": {"summary": "Test summary"}}
        save_cache(data)
        loaded = load_cache()
        assert loaded == data

    def test_check_cache_hit(self, temp_cache):
        key = "test_key"
        summary = "Cached summary"
        save_to_cache(key, summary)
        result = check_cache(key)
        assert result == summary

    def test_check_cache_miss(self, temp_cache):
        result = check_cache("nonexistent_key")
        assert result is None

    def test_cache_expiry(self, temp_cache):
        key = "expired_key"
        summary = "Old summary"
        old_time = (datetime.now() - timedelta(days=10)).isoformat()
        save_cache({key: {"summary": summary, "timestamp": old_time}})
        result = check_cache(key)
        assert result is None

    def test_cache_not_expired(self, temp_cache):
        key = "fresh_key"
        summary = "Fresh summary"
        save_to_cache(key, summary)
        result = check_cache(key)
        assert result == summary

    def test_clear_cache(self, temp_cache):
        save_to_cache("k1", "s1")
        save_to_cache("k2", "s2")
        count = clear_cache()
        assert count == 2
        assert load_cache() == {}

    def test_clear_empty_cache(self, temp_cache):
        count = clear_cache()
        assert count == 0


class TestEstimateReadingTime:
    """Test reading time estimation."""

    def test_short_text(self):
        assert estimate_reading_time("Hello world") == "1 min"

    def test_exact_one_minute(self):
        words = " ".join(["word"] * 200)
        assert estimate_reading_time(words) == "1 min"

    def test_two_minutes(self):
        words = " ".join(["word"] * 400)
        assert estimate_reading_time(words) == "2 mins"

    def test_empty_text(self):
        assert estimate_reading_time("") == "0 min"

    def test_custom_wpm(self):
        words = " ".join(["word"] * 100)
        assert estimate_reading_time(words, words_per_minute=100) == "1 min"


class TestTruncateText:
    """Test text truncation helper."""

    def test_no_truncation_needed(self):
        text = "Short text"
        assert truncate_text(text, max_length=100) == text

    def test_truncation(self):
        text = "A" * 100
        result = truncate_text(text, max_length=20)
        assert result.endswith("...")
        assert len(result) == 20

    def test_exact_length(self):
        text = "A" * 10
        assert truncate_text(text, max_length=10) == text


class TestFormatDuration:
    """Test duration formatting."""

    def test_seconds_only(self):
        assert format_duration(45) == "0:45"

    def test_minutes_and_seconds(self):
        assert format_duration(125) == "2:05"

    def test_hours_minutes_seconds(self):
        assert format_duration(3665) == "1:01:05"

    def test_zero_seconds(self):
        assert format_duration(0) == "Unknown"


class TestValidateYouTubeUrl:
    """Test YouTube URL validation."""

    def test_valid_standard_url(self):
        assert validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_valid_short_url(self):
        assert validate_youtube_url("https://youtu.be/dQw4w9WgXcQ") is True

    def test_valid_embed_url(self):
        assert validate_youtube_url("https://www.youtube.com/embed/dQw4w9WgXcQ") is True

    def test_valid_shorts_url(self):
        assert validate_youtube_url("https://www.youtube.com/shorts/AbCdEfGhIjK") is True

    def test_invalid_url(self):
        assert validate_youtube_url("https://example.com/video") is False

    def test_empty_string(self):
        assert validate_youtube_url("") is False

    def test_non_url_string(self):
        assert validate_youtube_url("not a url") is False

