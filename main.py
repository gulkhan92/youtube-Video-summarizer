#!/usr/bin/env python3
"""
Video File Summarizer CLI - Transcript.txt + Summary.txt
Usage: python3 main.py [video.mp4] [-o output]
"""

import sys
import argparse
from pathlib import Path
import os
from google import genai

# Add modules path
sys.path.append(str(Path(__file__).parent / 'modules'))

from modules.file_processor import process_uploaded_file
from modules.utils import clean_transcript

def main():
    parser = argparse.ArgumentParser(description='Video → transcript.txt + summary.txt')
    parser.add_argument('video_file', nargs='?', default=None, help='Video path')
    parser.add_argument('-o', '--output', default=None, help='Output base name')
    parser.add_argument('-l', '--length', type=int, default=300)
    parser.add_argument('--whisper', default='medium', choices=['tiny','base','small','medium','large'])
    
    args = parser.parse_args()
    
    # Video file
    if not args.video_file:
        args.video_file = input("Video file path: ").strip()
    video_path = Path(args.video_file)
    if not video_path.exists():
        print(f"❌ {video_path} not found")
        return 1
    
    # Output base name
    if args.output is None:
        args.output = video_path.stem  # Use filename without extension
    
    print(f"🎬 {video_path.name} → {args.output}_*.txt")
    
    try:
        print("🔄 FFmpeg + Whisper...")
        raw_transcript = process_uploaded_file(str(video_path), args.whisper)
        
        if len(raw_transcript.strip()) < 50:
            print("❌ No audio")
            return 1
        
        cleaned = clean_transcript(raw_transcript)
        
        # Transcript file
        transcript_file = f"{args.output}_transcript.txt"
        Path(transcript_file).write_text(f"""# Video Transcript
{ video_path.name } ({video_path.stat().st_size/(1024**3):.1f}GB)
Whisper: {args.whisper}
Generated: {os.popen('date').read().strip()}

RAW:
{raw_transcript}

CLEANED:
{cleaned}
""")
        
        print("🤖 Gemini summary...")
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=f"Summarize video transcript ({args.length} words max, bullet points):\\n{cleaned[:10000]}"
        )
        summary = response.text.strip()
        
        # Summary file
        summary_file = f"{args.output}_summary.txt"
        Path(summary_file).write_text(f"""# Video Summary
{ video_path.name } ({video_path.stat().st_size/(1024**3):.1f}GB)
Summary: {args.length} words max
Generated: {os.popen('date').read().strip()}

{summary}

Transcript: {transcript_file}
---
Gemini CLI Summarizer
""")
        
        print(f"✅ {summary_file}")
        print(f"✅ {transcript_file}")
        print(summary[:700] + "...")
        
    except Exception as e:
        print(f"❌ {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

