"""Tests for modules/file_processor.py — file validation, audio extraction, transcription pipeline."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.append(str(Path(__file__).parent.parent))

from modules.file_processor import (
    validate_video_file,
    get_file_hash,
    UnsupportedFormatError,
    FileProcessorError,
    AudioExtractionError,
    TranscriptionError,
    SUPPORTED_FORMATS,
    extract_audio_with_ffmpeg,
    process_uploaded_file,
    transcribe_audio,
)


class TestValidateVideoFile:
    """Test video file format and size validation."""

    def test_valid_mp4(self):
        validate_video_file("video.mp4")

    def test_valid_mov(self):
        validate_video_file("recording.mov")

    def test_valid_mkv(self):
        validate_video_file("movie.mkv")

    def test_valid_uppercase_extension(self):
        validate_video_file("video.MP4")

    def test_unsupported_format_raises(self):
        with pytest.raises(UnsupportedFormatError) as exc_info:
            validate_video_file("video.flv")
        assert "flv" in str(exc_info.value)

    def test_unsupported_no_extension_raises(self):
        with pytest.raises(UnsupportedFormatError):
            validate_video_file("video")

    def test_empty_filename_raises(self):
        with pytest.raises(UnsupportedFormatError):
            validate_video_file("")

    def test_file_too_large_in_pipeline(self):
        with patch("modules.file_processor.os.path.getsize", return_value=15 * 1024 * 1024 * 1024):
            with pytest.raises(UnsupportedFormatError) as exc_info:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                    video_path = f.name
                try:
                    process_uploaded_file(video_path, max_file_size_mb=10240)
                finally:
                    if os.path.exists(video_path):
                        os.unlink(video_path)
            assert "too large" in str(exc_info.value).lower()


class TestGetFileHash:
    """Test MD5 hash generation for file content."""

    def test_hash_consistency(self):
        data = b"test content"
        hash1 = get_file_hash(data)
        hash2 = get_file_hash(data)
        assert hash1 == hash2
        assert len(hash1) == 32

    def test_different_content_different_hash(self):
        hash1 = get_file_hash(b"content1")
        hash2 = get_file_hash(b"content2")
        assert hash1 != hash2

    def test_empty_data(self):
        hash_value = get_file_hash(b"")
        assert len(hash_value) == 32


class TestExtractAudioWithFFmpeg:
    """Test FFmpeg audio extraction with mocked subprocess."""

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_successful_extraction(self, mock_temp, mock_run):
        mock_temp.return_value.name = "/tmp/test_audio.mp3"
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        result = extract_audio_with_ffmpeg("/tmp/video.mp4")
        assert result == "/tmp/test_audio.mp3"
        mock_run.assert_called_once()

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_ffmpeg_failure_raises(self, mock_temp, mock_run):
        mock_temp.return_value.name = "/tmp/test_audio.mp3"
        mock_run.return_value = MagicMock(returncode=1, stderr="Error decoding")

        with pytest.raises(AudioExtractionError) as exc_info:
            extract_audio_with_ffmpeg("/tmp/video.mp4")
        assert "FFmpeg failed" in str(exc_info.value)

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_ffmpeg_timeout_raises(self, mock_temp, mock_run):
        mock_temp.return_value.name = "/tmp/test_audio.mp3"
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300)

        with pytest.raises(AudioExtractionError) as exc_info:
            extract_audio_with_ffmpeg("/tmp/video.mp4")
        assert "timeout" in str(exc_info.value).lower()

    @patch("subprocess.run")
    @patch("tempfile.NamedTemporaryFile")
    def test_ffmpeg_not_found_raises(self, mock_temp, mock_run):
        mock_temp.return_value.name = "/tmp/test_audio.mp3"
        mock_run.side_effect = FileNotFoundError("ffmpeg not found")

        with pytest.raises(AudioExtractionError) as exc_info:
            extract_audio_with_ffmpeg("/tmp/video.mp4")
        assert "not found" in str(exc_info.value).lower()


class TestTranscribeAudio:
    """Test Whisper audio transcription with mocked model."""

    @patch("modules.file_processor.WHISPER_AVAILABLE", True)
    @patch("modules.file_processor.whisper.load_model")
    def test_successful_transcription(self, mock_load_model):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Hello world this is a test"}
        mock_load_model.return_value = mock_model

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio data")
            audio_path = f.name

        try:
            result = transcribe_audio(audio_path, model_size="base")
            assert result == "Hello world this is a test"
            mock_load_model.assert_called_once_with("base")
        finally:
            os.unlink(audio_path)

    @patch("modules.file_processor.WHISPER_AVAILABLE", False)
    def test_whisper_not_installed_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            audio_path = f.name

        try:
            with pytest.raises(TranscriptionError) as exc_info:
                transcribe_audio(audio_path)
            assert "not installed" in str(exc_info.value).lower()
        finally:
            os.unlink(audio_path)

    def test_audio_file_not_found_raises(self):
        with pytest.raises(TranscriptionError) as exc_info:
            transcribe_audio("/nonexistent/audio.mp3")
        assert "not found" in str(exc_info.value).lower()

    @patch("modules.file_processor.WHISPER_AVAILABLE", True)
    @patch("modules.file_processor.whisper.load_model")
    def test_transcription_with_progress_callback(self, mock_load_model):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Test transcript"}
        mock_load_model.return_value = mock_model

        progress_messages = []
        def callback(msg, pct=None):
            progress_messages.append(msg)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            audio_path = f.name

        try:
            transcribe_audio(audio_path, progress_callback=callback)
            assert len(progress_messages) > 0
        finally:
            os.unlink(audio_path)

    @patch("modules.file_processor.WHISPER_AVAILABLE", True)
    @patch("modules.file_processor.whisper.load_model")
    def test_transcription_error_raises(self, mock_load_model):
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("GPU out of memory")
        mock_load_model.return_value = mock_model

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            audio_path = f.name

        try:
            with pytest.raises(TranscriptionError) as exc_info:
                transcribe_audio(audio_path)
            assert "failed" in str(exc_info.value).lower()
        finally:
            os.unlink(audio_path)


class TestProcessUploadedFile:
    """Test the complete file processing pipeline."""

    @patch("modules.file_processor.os.path.getsize", return_value=100 * 1024 * 1024)
    @patch("shutil.disk_usage")
    @patch("modules.file_processor.extract_audio_with_ffmpeg")
    @patch("modules.file_processor.transcribe_audio")
    @patch("modules.file_processor.validate_video_file")
    def test_successful_pipeline(
        self, mock_validate, mock_transcribe, mock_extract, mock_disk, mock_filesize
    ):
        mock_disk.return_value = MagicMock(free=10 * 1024 * 1024 * 1024)
        mock_extract.return_value = "/tmp/audio.mp3"
        mock_transcribe.return_value = "This is the transcript"

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            result = process_uploaded_file(video_path, whisper_model_size="base")
            assert result == "This is the transcript"
            mock_validate.assert_called_once()
            mock_extract.assert_called_once_with(video_path)
            mock_transcribe.assert_called_once()
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

    @patch("modules.file_processor.os.path.getsize", return_value=100 * 1024 * 1024)
    @patch("shutil.disk_usage")
    def test_insufficient_disk_space(self, mock_disk, mock_filesize):
        mock_disk.return_value = MagicMock(free=50 * 1024 * 1024)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            with pytest.raises(FileProcessorError) as exc_info:
                process_uploaded_file(video_path)
            assert "disk space" in str(exc_info.value).lower()
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

    @patch("modules.file_processor.os.path.getsize", return_value=100 * 1024 * 1024)
    @patch("shutil.disk_usage")
    @patch("modules.file_processor.validate_video_file")
    def test_validation_failure(self, mock_validate, mock_disk, mock_filesize):
        mock_disk.return_value = MagicMock(free=10 * 1024 * 1024 * 1024)
        mock_validate.side_effect = UnsupportedFormatError("Bad format")

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            video_path = f.name

        try:
            with pytest.raises(UnsupportedFormatError):
                process_uploaded_file(video_path)
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

    @patch("modules.file_processor.os.path.getsize", return_value=100 * 1024 * 1024)
    @patch("shutil.disk_usage")
    @patch("modules.file_processor.extract_audio_with_ffmpeg")
    @patch("modules.file_processor.transcribe_audio")
    @patch("modules.file_processor.validate_video_file")
    def test_temp_file_cleanup(
        self, mock_validate, mock_transcribe, mock_extract, mock_disk, mock_filesize
    ):
        mock_disk.return_value = MagicMock(free=10 * 1024 * 1024 * 1024)
        audio_path = "/tmp/test_cleanup.mp3"
        mock_extract.return_value = audio_path
        mock_transcribe.return_value = "Transcript"

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        try:
            with patch("os.path.exists", return_value=True):
                with patch("os.unlink") as mock_unlink:
                    process_uploaded_file(video_path)
                    mock_unlink.assert_called_with(audio_path)
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)

