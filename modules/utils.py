"""
Utility functions for caching, text cleaning, and helper operations
"""

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta


# Cache configuration
CACHE_DIR = Path("./cache")
CACHE_DIR.mkdir(exist_ok=True)
CACHE_FILE = CACHE_DIR / "summaries.json"
CACHE_EXPIRY_DAYS = 7  # Cache expires after 7 days


def clean_transcript(text: str) -> str:
    """
    Clean transcript text by removing noise and normalizing spacing.
    
    Args:
        text: Raw transcript text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common transcript artifacts
    text = re.sub(r'\[.*?\]', '', text)  # Remove [music], [applause], etc.
    text = re.sub(r'\(.*?\)', '', text)  # Remove (inaudible), (laughter), etc.
    
    # Fix common punctuation issues
    text = text.replace(' ,', ',')
    text = text.replace(' .', '.')
    text = text.replace('  ', ' ')
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Capitalize first letter of sentences (simple version)
    sentences = re.split(r'([.!?] +)', text)
    cleaned_sentences = []
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            sentence = sentences[i].strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                cleaned_sentences.append(sentence)
            if i + 1 < len(sentences):
                cleaned_sentences.append(sentences[i + 1])
    
    return ''.join(cleaned_sentences)


def get_cache_key(input_source: str, model: str = "gemini") -> str:
    """
    Generate a unique cache key based on input source.
    
    Args:
        input_source: YouTube URL or file hash
        model: Summarization model used
        
    Returns:
        MD5 hash key
    """
    content = f"{input_source}_{model}"
    return hashlib.md5(content.encode()).hexdigest()


def load_cache() -> Dict[str, Any]:
    """
    Load cached summaries from JSON file.
    
    Returns:
        Dictionary of cached summaries
    """
    if not CACHE_FILE.exists():
        return {}
    
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        
        # Remove expired entries
        now = datetime.now()
        expired_keys = []
        
        for key, data in cache.items():
            if 'timestamp' in data:
                cache_time = datetime.fromisoformat(data['timestamp'])
                if now - cache_time > timedelta(days=CACHE_EXPIRY_DAYS):
                    expired_keys.append(key)
        
        for key in expired_keys:
            del cache[key]
        
        # Save cleaned cache if we removed anything
        if expired_keys:
            save_cache(cache)
        
        return cache
        
    except (json.JSONDecodeError, KeyError, ValueError):
        return {}


def save_cache(cache: Dict[str, Any]) -> None:
    """
    Save cache to JSON file.
    
    Args:
        cache: Cache dictionary to save
    """
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save cache: {e}")


def check_cache(key: str) -> Optional[str]:
    """
    Check if a summary exists in cache.
    
    Args:
        key: Cache key
        
    Returns:
        Summary text if found and not expired, None otherwise
    """
    cache = load_cache()
    
    if key in cache:
        data = cache[key]
        
        # Check expiry
        if 'timestamp' in data:
            cache_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cache_time <= timedelta(days=CACHE_EXPIRY_DAYS):
                return data.get('summary')
    
    return None


def save_to_cache(key: str, summary: str, metadata: Optional[Dict] = None) -> None:
    """
    Save summary to cache.
    
    Args:
        key: Cache key
        summary: Summary text to cache
        metadata: Optional metadata to store
    """
    cache = load_cache()
    
    cache[key] = {
        'summary': summary,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    
    save_cache(cache)


def clear_cache() -> int:
    """
    Clear all cached summaries.
    
    Returns:
        Number of entries cleared
    """
    if not CACHE_FILE.exists():
        return 0
    
    try:
        cache = load_cache()
        count = len(cache)
        save_cache({})
        return count
    except Exception as e:
        print(f"Error clearing cache: {e}")
        return 0


def estimate_reading_time(text: str, words_per_minute: int = 200) -> str:
    """
    Estimate reading time for text.
    
    Args:
        text: Text to estimate reading time for
        words_per_minute: Average reading speed
        
    Returns:
        Formatted reading time string
    """
    if not text:
        return "0 min"
    
    word_count = len(text.split())
    minutes = max(1, round(word_count / words_per_minute))
    
    if minutes == 1:
        return "1 min"
    else:
        return f"{minutes} mins"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def format_duration(seconds: int) -> str:
    """
    Format duration in seconds to readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2:30" or "1:15:45")
    """
    if not seconds:
        return "Unknown"
    
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def validate_youtube_url(url: str) -> bool:
    """
    Validate if a string is a valid YouTube URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid YouTube URL, False otherwise
    """
    youtube_patterns = [
        r'^https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'^https?://(?:www\.)?youtu\.be/[\w-]+',
        r'^https?://(?:www\.)?youtube\.com/embed/[\w-]+',
        r'^https?://(?:www\.)?youtube\.com/shorts/[\w-]+',
    ]
    
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    return False


# Simple test
if __name__ == "__main__":
    test_text = "  This is a   test   transcript.   [Music]   It has some noise.   "
    cleaned = clean_transcript(test_text)
    print(f"Original: '{test_text}'")
    print(f"Cleaned: '{cleaned}'")
    
    key = get_cache_key("test_video_123", "gemini")
    print(f"Cache key: {key}")
    
    reading_time = estimate_reading_time("This is a test sentence with several words.", 200)
    print(f"Reading time: {reading_time}")