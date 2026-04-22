"""
Video Summarizer Modules
A collection of utilities for YouTube and file-based video summarization
"""

from .youtube import fetch_youtube_transcript, extract_video_id
from .file_processor import process_uploaded_file, transcribe_audio
from .summarizer import summarize_text
from .utils import clean_transcript, get_cache_key, check_cache, save_to_cache

__all__ = [
    # YouTube
    'fetch_youtube_transcript',
    'extract_video_id',
    
    # File processing
    'process_uploaded_file',
    'transcribe_audio',
    
    # Summarization
    'summarize_text',
    
    # Utils
    'clean_transcript',
    'get_cache_key',
    'check_cache',
    'save_to_cache',
]

__version__ = '1.0.0'