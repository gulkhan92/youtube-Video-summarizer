"""
File processing module for uploaded videos
Handles audio extraction and transcription using Whisper
"""

import tempfile
import os
import hashlib
from typing import Optional, Callable
from pathlib import Path

# Import with error handling for optional dependencies
try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: moviepy not installed. File upload feature will not work.")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: whisper not installed. Transcription feature will not work.")


# Custom exceptions
class FileProcessorError(Exception):
    """Base exception for file processing errors"""
    pass

class AudioExtractionError(FileProcessorError):
    """Failed to extract audio from video"""
    pass

class TranscriptionError(FileProcessorError):
    """Failed to transcribe audio"""
    pass

class UnsupportedFormatError(FileProcessorError):
    """Video format not supported"""
    pass


# Supported video formats
SUPPORTED_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}  # .mov already supported for screen recording


def validate_video_file(file_name: str, max_size_mb: int = 10240) -> None:
    """
    Validate uploaded video file format and size.
    
    Args:
        file_name: Name of the uploaded file
        max_size_mb: Maximum allowed file size in MB
        
    Raises:
        UnsupportedFormatError: If file format is not supported
    """
    # Check file extension
    file_ext = Path(file_name).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Unsupported format: {file_ext}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )


def get_file_hash(file_data: bytes) -> str:
    """
    Generate MD5 hash of file content for caching.
    
    Args:
        file_data: Binary file content
        
    Returns:
        MD5 hash string
    """
    return hashlib.md5(file_data).hexdigest()





def transcribe_audio(
    audio_path: str, 
    model_size: str = 'base',
    language: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Transcribe audio file using Whisper.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        language: Optional language code (auto-detected if None)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Transcribed text
        
    Raises:
        TranscriptionError: If transcription fails
    """
    
    if not WHISPER_AVAILABLE:
        raise TranscriptionError(
            "whisper is not installed. Run: pip install openai-whisper"
        )
    
    if not os.path.exists(audio_path):
        raise TranscriptionError(f"Audio file not found: {audio_path}")
    
    try:
        # Load model (can be cached at app level)
        if progress_callback:
            progress_callback(f"Loading Whisper model ({model_size})...")
        
        model = whisper.load_model(model_size)
        
        if progress_callback:
            progress_callback("Transcribing audio (this may take a minute)...")
        
        # Transcribe
        result = model.transcribe(
            audio_path,
            language=language,
            verbose=False,  # Don't print progress to console
            task='transcribe',
            fp16=False  # Use FP32 for better CPU compatibility
        )
        
        if progress_callback:
            progress_callback("Transcription complete!")
        
        return result["text"]
        
    except Exception as e:
        raise TranscriptionError(f"Failed to transcribe audio: {str(e)}")


def process_uploaded_file(
    video_path: str,
    whisper_model_size: str = 'base',
    max_file_size_mb: int = 10240,
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Complete pipeline for processing video file from disk path.
    Optimized for large files using FFmpeg.
    
    Args:
        video_path: Path to video file on disk
        whisper_model_size: Whisper model size to use
        max_file_size_mb: Maximum allowed file size
        progress_callback: Optional callback for progress updates
        
    Returns:
        Transcribed text from the video
    """
    audio_path = None
    
    try:
        # Disk space check (~2x file size needed)
        file_size_bytes = os.path.getsize(video_path)
        import shutil
        available_bytes = shutil.disk_usage('/').free
        if available_bytes < file_size_bytes * 2:
            raise FileProcessorError(f"Insufficient disk space. Need ~{file_size_bytes*2/(1024**3):.1f}GB available.")

        
        # Validate file first
        validate_video_file(video_path, max_file_size_mb)
        
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        if progress_callback:
            progress_callback(f"✅ Disk OK, size: {file_size_mb:.1f}MB")
        
        if file_size_mb > max_file_size_mb:
            raise UnsupportedFormatError(
                f"File too large: {file_size_mb:.1f}MB. "
                f"Max: {max_file_size_mb}MB"
            )

        
        if progress_callback:
            progress_callback("🔄 Extracting audio with FFmpeg...")
        
        # Extract audio using FFmpeg (memory efficient, fast)
        audio_path = extract_audio_with_ffmpeg(video_path)
        
        if progress_callback:
            progress_callback("✅ Audio extracted, starting transcription...")
        
        # Transcribe with Whisper
        transcript = transcribe_audio(
            audio_path, 
            model_size=whisper_model_size,
            progress_callback=progress_callback
        )
        
        return transcript
        
    finally:
        # Cleanup audio temp file
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass


def extract_audio_with_ffmpeg(video_path: str, audio_format: str = 'mp3') -> str:
    """
    Extract audio from video using FFmpeg subprocess (memory-efficient).
    
    Args:
        video_path: Input video file path
        audio_format: Output format ('mp3', 'wav', etc.)
        
    Returns:
        Path to extracted audio file
    """
    import subprocess
    import tempfile
    
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_format}').name
    
    cmd = [
        'ffmpeg', '-i', video_path,
        '-vn',  # No video
        '-acodec', 'mp3',
        '-ac', '1',  # Mono channel
        '-ar', '16000',  # Whisper optimal sample rate
        '-y',  # Overwrite
        audio_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise AudioExtractionError(
                f"FFmpeg failed: {result.stderr}. Install with: brew install ffmpeg"
            )
        
        return audio_path
        
    except subprocess.TimeoutExpired:
        raise AudioExtractionError("FFmpeg timeout - video too long or corrupted")
    except FileNotFoundError:
        raise AudioExtractionError("FFmpeg not found. Install with: brew install ffmpeg")


# Simple test function
if __name__ == "__main__":
    print("File Processor Module")
    print(f"MoviePy available: {MOVIEPY_AVAILABLE}")
    print(f"Whisper available: {WHISPER_AVAILABLE}")
    print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")