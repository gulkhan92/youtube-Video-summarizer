"""Tests for modules/youtube.py — URL extraction, validation, transcript fetching, error handling."""

import pytest
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from modules.youtube import (
    extract_video_id,
    fetch_youtube_transcript,
    get_video_info,
    InvalidURLError,
    TranscriptUnavailableError,
    YouTubeError,
)


class TestExtractVideoId:
    """Test video ID extraction from various YouTube URL formats."""

    def test_standard_watch_url(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_short_url(self):
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_embed_url(self):
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        url = "https://www.youtube.com/shorts/AbCdEfGhIjK"
        assert extract_video_id(url) == "AbCdEfGhIjK"

    def test_watch_with_extra_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=share&t=120"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_invalid_url_raises(self):
        with pytest.raises(InvalidURLError):
            extract_video_id("https://example.com/video")

    def test_empty_string_raises(self):
        with pytest.raises(InvalidURLError):
            extract_video_id("")

    def test_video_id_with_hyphens(self):
        url = "https://www.youtube.com/watch?v=abc-123_xyz"
        assert extract_video_id(url) == "abc-123_xyz"


class TestFetchYouTubeTranscript:
    """Test transcript fetching with mocked YouTubeTranscriptApi."""

    @patch("modules.youtube.YouTubeTranscriptApi.get_transcript")
    def test_fetch_success(self, mock_get_transcript):
        mock_get_transcript.return_value = [
            {"text": "Hello world", "start": 0.0, "duration": 1.0},
            {"text": "This is a test", "start": 1.0, "duration": 2.0},
        ]
        result = fetch_youtube_transcript("https://www.youtube.com/watch?v=TEST123")
        assert result == "Hello world This is a test"
        mock_get_transcript.assert_called_once_with("TEST123", languages=["en"])

    @patch("modules.youtube.YouTubeTranscriptApi.get_transcript")
    def test_fetch_with_custom_languages(self, mock_get_transcript):
        mock_get_transcript.return_value = [{"text": "Hola", "start": 0.0, "duration": 1.0}]
        result = fetch_youtube_transcript("https://youtu.be/TEST123", languages=["es", "en"])
        assert result == "Hola"
        mock_get_transcript.assert_called_once_with("TEST123", languages=["es", "en"])

    @patch("modules.youtube.YouTubeTranscriptApi.get_transcript")
    def test_fetch_empty_transcript(self, mock_get_transcript):
        mock_get_transcript.return_value = []
        result = fetch_youtube_transcript("https://www.youtube.com/watch?v=TEST123")
        assert result == ""

    @patch("modules.youtube.YouTubeTranscriptApi.get_transcript")
    def test_fetch_transcripts_disabled(self, mock_get_transcript):
        from youtube_transcript_api._errors import TranscriptsDisabled
        mock_get_transcript.side_effect = TranscriptsDisabled("TEST123")
        with pytest.raises(TranscriptUnavailableError) as exc_info:
            fetch_youtube_transcript("https://www.youtube.com/watch?v=TEST123")
        assert "disabled" in str(exc_info.value).lower()

    @patch("modules.youtube.YouTubeTranscriptApi.get_transcript")
    def test_fetch_no_transcript_found(self, mock_get_transcript):
        from youtube_transcript_api._errors import NoTranscriptFound
        mock_get_transcript.side_effect = NoTranscriptFound("TEST123", [], ["en"])
        with pytest.raises(TranscriptUnavailableError) as exc_info:
            fetch_youtube_transcript("https://www.youtube.com/watch?v=TEST123")
        assert "no transcript" in str(exc_info.value).lower() or "languages" in str(exc_info.value).lower()

    @patch("modules.youtube.YouTubeTranscriptApi.get_transcript")
    def test_fetch_video_unavailable(self, mock_get_transcript):
        from youtube_transcript_api._errors import VideoUnavailable
        mock_get_transcript.side_effect = VideoUnavailable("TEST123")
        with pytest.raises(YouTubeError) as exc_info:
            fetch_youtube_transcript("https://www.youtube.com/watch?v=TEST123")
        assert "unavailable" in str(exc_info.value).lower()

    @patch("modules.youtube.YouTubeTranscriptApi.get_transcript")
    def test_fetch_generic_exception(self, mock_get_transcript):
        mock_get_transcript.side_effect = ConnectionError("Network failure")
        with pytest.raises(YouTubeError) as exc_info:
            fetch_youtube_transcript("https://www.youtube.com/watch?v=TEST123")
        assert "failed" in str(exc_info.value).lower()

    def test_fetch_invalid_url(self):
        with pytest.raises(InvalidURLError):
            fetch_youtube_transcript("not-a-url")


class TestGetVideoInfo:
    """Test get_video_info helper."""

    @patch("modules.youtube.YouTubeTranscriptApi.list_transcripts")
    def test_get_info_success(self, mock_list):
        mock_transcript = MagicMock()
        mock_transcript.language_code = "en"
        mock_list.return_value = [mock_transcript]

        info = get_video_info("https://www.youtube.com/watch?v=TEST123")
        assert info["video_id"] == "TEST123"
        assert info["has_transcript"] is True
        assert "en" in info["available_languages"]

    @patch("modules.youtube.YouTubeTranscriptApi.list_transcripts")
    def test_get_info_no_transcripts(self, mock_list):
        mock_list.return_value = []
        info = get_video_info("https://www.youtube.com/watch?v=TEST123")
        assert info["video_id"] == "TEST123"
        assert info["has_transcript"] is False

    def test_get_info_invalid_url(self):
        with pytest.raises(InvalidURLError):
            get_video_info("bad-url")

