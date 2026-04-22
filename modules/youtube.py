"""
YouTube transcript fetching module
Handles extraction of video ID and fetching transcripts from YouTube videos
"""

import re
from typing import Optional
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)

# Custom exceptions for better error handling
class YouTubeError(Exception):
    """Base exception for YouTube-related errors"""
    pass

class InvalidURLError(YouTubeError):
    """Raised when YouTube URL is invalid"""
    pass

class TranscriptUnavailableError(YouTubeError):
    """Raised when no transcript is available for the video"""
    pass


def extract_video_id(url: str) -> str:
    """
    Extract YouTube video ID from various URL formats.
    
    Supports:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    
    Args:
        url: YouTube video URL
        
    Returns:
        Video ID string
        
    Raises:
        InvalidURLError: If URL doesn't contain a valid video ID
    """
    
    # Patterns for different YouTube URL formats
    patterns = [
        # Standard watch URL: youtube.com/watch?v=VIDEO_ID
        r'(?:youtube\.com\/watch\?v=)([\w-]+)',
        
        # Short youtu.be URL: youtu.be/VIDEO_ID
        r'(?:youtu\.be\/)([\w-]+)',
        
        # Embed URL: youtube.com/embed/VIDEO_ID
        r'(?:youtube\.com\/embed\/)([\w-]+)',
        
        # Shorts URL: youtube.com/shorts/VIDEO_ID
        r'(?:youtube\.com\/shorts\/)([\w-]+)',
        
        # Watch with extra params: youtube.com/watch?v=VIDEO_ID&feature=share
        r'(?:youtube\.com\/watch\?.*v=)([\w-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    raise InvalidURLError(f"Could not extract video ID from URL: {url}")


def fetch_youtube_transcript(url: str, languages: list = None) -> str:
    """
    Fetch transcript from a YouTube video.
    
    Args:
        url: YouTube video URL
        languages: List of language codes to try (default: ['en'])
        
    Returns:
        Combined transcript text as a single string
        
    Raises:
        InvalidURLError: If URL is invalid
        TranscriptUnavailableError: If no transcript is available
        YouTubeError: For other YouTube API errors
    """
    
    if languages is None:
        languages = ['en']  # Default to English
    
    try:
        # Extract video ID from URL
        video_id = extract_video_id(url)
        
        # Fetch transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, 
            languages=languages
        )
        
        # Combine all transcript segments into one string
        # Each segment has: {'text': '...', 'start': timestamp, 'duration': seconds}
        full_transcript = " ".join([entry['text'] for entry in transcript_list])
        
        return full_transcript
        
    except InvalidURLError:
        # Re-raise as is
        raise
        
    except TranscriptsDisabled:
        raise TranscriptUnavailableError(
            "Subtitles/closed captions are disabled for this video. "
            "Try a different video or use the file upload option."
        )
        
    except NoTranscriptFound:
        # Try to get available language info
        try:
            video_id = extract_video_id(url)
            transcript_info = YouTubeTranscriptApi.list_transcripts(video_id)
            available_langs = [t.language_code for t in transcript_info]
            
            raise TranscriptUnavailableError(
                f"No transcript found in languages {languages}. "
                f"Available languages: {available_langs[:5]}..."
                if available_langs else "No transcripts available at all."
            )
        except:
            raise TranscriptUnavailableError(
                f"No transcript found for this video in {languages}. "
                "The video might not have captions available."
            )
            
    except VideoUnavailable:
        raise YouTubeError("Video is unavailable. It might be private or deleted.")
        
    except Exception as e:
        raise YouTubeError(f"Failed to fetch transcript: {str(e)}")


def get_video_info(url: str) -> dict:
    """
    Get basic video information (title, duration, etc.)
    Note: This is a simplified version - for full info, you'd need youtube-dl or yt-dlp
    
    Args:
        url: YouTube video URL
        
    Returns:
        Dictionary with video information
    """
    
    try:
        video_id = extract_video_id(url)
        
        # Basic info from transcript API
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Get available languages
        available_languages = [t.language_code for t in transcript_list]
        
        return {
            'video_id': video_id,
            'url': url,
            'available_languages': available_languages,
            'has_transcript': len(available_languages) > 0
        }
        
    except Exception as e:
        return {
            'video_id': extract_video_id(url) if url else None,
            'url': url,
            'error': str(e),
            'has_transcript': False
        }


# Simple test function to verify module works
if __name__ == "__main__":
    # Test with a sample video (Python tutorial with captions)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with a real video
    
    print("Testing YouTube module...")
    
    # Test video ID extraction
    try:
        video_id = extract_video_id(test_url)
        print(f"✅ Extracted video ID: {video_id}")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
    
    # Test transcript fetching
    try:
        transcript = fetch_youtube_transcript(test_url)
        print(f"✅ Fetched transcript: {len(transcript)} characters")
        print(f"Preview: {transcript[:200]}...")
    except Exception as e:
        print(f"❌ Transcript fetch failed: {e}")