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
SUPPORTED_FORMATS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}


def validate_video_file(file_name: str, max_size_mb: int = 200) -> None:
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


def extract_audio_from_video(
    video_path: str, 
    audio_format: str = 'mp3',
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Extract audio track from video file.
    
    Args:
        video_path: Path to the video file
        audio_format: Output audio format ('mp3', 'wav')
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Path to extracted audio file
        
    Raises:
        AudioExtractionError: If audio extraction fails
    """
    
    if not MOVIEPY_AVAILABLE:
        raise AudioExtractionError(
            "moviepy is not installed. Run: pip install moviepy"
        )
    
    audio_path = None
    
    try:
        if progress_callback:
            progress_callback("Loading video file...")
        
        # Load video
        video = VideoFileClip(video_path)
        
        # Get audio duration for progress tracking
        duration = video.duration
        
        # Create temporary audio file
        audio_path = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f'.{audio_format}'
        ).name
        
        if progress_callback:
            progress_callback(f"Extracting audio ({duration:.1f} seconds)...")
        
        # Extract and write audio
        # Use ffmpeg_params for better compatibility
        video.audio.write_audiofile(
            audio_path,
            logger=None,  # Disable moviepy's logger
            verbose=False,
            ffmpeg_params=['-ac', '1']  # Convert to mono for better Whisper performance
        )
        
        # Clean up
        video.close()
        
        if progress_callback:
            progress_callback("Audio extraction complete!")
        
        return audio_path
        
    except Exception as e:
        # Clean up partial files
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass
        
        raise AudioExtractionError(f"Failed to extract audio: {str(e)}")


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
    uploaded_file,
    whisper_model_size: str = 'base',
    max_file_size_mb: int = 200,
    progress_callback: Optional[Callable] = None
) -> str:
    """
    Complete pipeline for processing uploaded video file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        whisper_model_size: Whisper model size to use
        max_file_size_mb: Maximum allowed file size
        progress_callback: Optional callback for progress updates
        
    Returns:
        Transcribed text from the video
    """
    
    video_path = None
    audio_path = None
    
    try:
        # Validate file
        validate_video_file(uploaded_file.name, max_file_size_mb)
        
        # Check file size
        uploaded_file.seek(0, 2)  # Seek to end
        file_size_mb = uploaded_file.tell() / (1024 * 1024)
        uploaded_file.seek(0)  # Reset to beginning
        
        if file_size_mb > max_file_size_mb:
            raise UnsupportedFormatError(
                f"File too large: {file_size_mb:.1f}MB. "
                f"Maximum size: {max_file_size_mb}MB"
            )
        
        if progress_callback:
            progress_callback(f"File size: {file_size_mb:.1f}MB")
            progress_callback("Saving uploaded file...")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(uploaded_file.read())
            video_path = tmp_video.name
        
        if progress_callback:
            progress_callback("Extracting audio from video...")
        
        # Extract audio
        audio_path = extract_audio_from_video(video_path, progress_callback=progress_callback)
        
        if progress_callback:
            progress_callback("Transcribing audio...")
        
        # Transcribe
        transcript = transcribe_audio(
            audio_path, 
            model_size=whisper_model_size,
            progress_callback=progress_callback
        )
        
        return transcript
        
    finally:
        # Cleanup temporary files
        for path in [video_path, audio_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass


# Simple test function
if __name__ == "__main__":
    print("File Processor Module")
    print(f"MoviePy available: {MOVIEPY_AVAILABLE}")
    print(f"Whisper available: {WHISPER_AVAILABLE}")
    print(f"Supported formats: {', '.join(SUPPORTED_FORMATS)}")