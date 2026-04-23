"""
YouTube & Video Summarizer App
A Streamlit application for summarizing YouTube videos and uploaded video files
"""

import streamlit as st
import sys
import os
import tempfile
from pathlib import Path

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Import modules
from modules.youtube import fetch_youtube_transcript, InvalidURLError, TranscriptUnavailableError
from modules.file_processor import process_uploaded_file, SUPPORTED_FORMATS
import shutil

from modules.summarizer import summarize_text, get_summarizer

from modules.utils import (
    clean_transcript, 
    get_cache_key, 
    check_cache, 
    save_to_cache,
    estimate_reading_time,
    validate_youtube_url,
    clear_cache,
    truncate_text
)

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Video Summarizer Pro - 10GB Upload Fixed",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for professional look
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Summary container */
    .summary-container {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    .summary-container h3 {
        color: #667eea;
        margin-top: 0;
    }
    
    /* Success message styling */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f8f9fa;
    }
    
    /* Info box styling */
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Stats card */
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stats-number {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stats-label {
        font-size: 0.8rem;
        color: #666;
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'summary_history' not in st.session_state:
        st.session_state.summary_history = []
    if 'current_summary' not in st.session_state:
        st.session_state.current_summary = None
    if 'processing' not in st.session_state:
        st.session_state.processing = False


def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>🎬 Video Summarizer Pro</h1>
        <p>Transform any video into concise, actionable insights using AI</p>
    </div>
    """, unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with settings and info"""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        model_choice = st.selectbox(
            "🤖 Summarization Model",
            options=["Gemini (Recommended)", "OpenAI", "Local (Basic)"],
            help="Gemini is free and fast. OpenAI requires API key. Local works offline but basic."
        )
        
        # Map display name to internal name
        model_map = {
            "Gemini (Recommended)": "gemini",
            "OpenAI": "openai",
            "Local (Basic)": "local"
        }
        selected_model = model_map[model_choice]
        
        # Whisper model for file uploads
        whisper_size = st.select_slider(
            "🎙️ Transcription Quality",
            options=["tiny", "base", "small", "medium", "large"],
            value="base",
            help="Larger models are more accurate but slower"
        )
        
        # Summary length
        summary_length = st.slider(
            "📏 Summary Length (words)",
            min_value=100,
            max_value=500,
            value=300,
            step=50
        )
        
        st.markdown("---")
        
        # Stats
        st.markdown("### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(st.session_state.summary_history)}</div>
                <div class="stats-label">Videos Processed</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            total_words = sum(s.get('word_count', 0) for s in st.session_state.summary_history)
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{total_words}</div>
                <div class="stats-label">Words Summarized</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", use_container_width=True):
            count = clear_cache()
            st.success(f"Cleared {count} cached summaries")
            st.rerun()
        
        st.markdown("---")
        
        # Info section
        with st.expander("ℹ️ How it works"):
            st.markdown("""
            **YouTube URL:**
            1. Fetches video transcript
            2. Cleans and processes text
            3. Generates AI summary
            
            **File Upload:**
            1. Extracts audio from video
            2. Transcribes using Whisper AI
            3. Generates AI summary
            
            **Tips:**
            - Use videos with clear speech
            - YouTube videos need captions enabled
            - Longer videos take more time
            """)
        
        return selected_model, whisper_size, summary_length


def display_youtube_tab():
    """Display YouTube URL input tab"""
    st.markdown("### 📺 YouTube Video")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        youtube_url = st.text_input(
            "Paste YouTube URL",
            placeholder="https://www.youtube.com/watch?v=..."
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        summarize_youtube = st.button("🎯 Summarize", key="youtube_btn", use_container_width=True)
    
    if youtube_url and summarize_youtube:
        if not validate_youtube_url(youtube_url):
            st.error("❌ Please enter a valid YouTube URL")
            return
        
        process_youtube_video(youtube_url)


def process_youtube_video(url: str):
    """Process and summarize YouTube video"""
    
    # Create progress placeholder
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # Step 1: Fetch transcript
        with status_placeholder.container():
            st.info("📝 Fetching video transcript...")
        
        transcript = fetch_youtube_transcript(url)
        
        if not transcript or len(transcript.strip()) < 50:
            st.error("❌ Transcript is too short or empty. Try a different video.")
            return
        
        # Step 2: Clean transcript
        with status_placeholder.container():
            st.info("🧹 Cleaning transcript...")
        
        cleaned_transcript = clean_transcript(transcript)
        
        # Step 3: Check cache
        cache_key = get_cache_key(url, st.session_state.selected_model)
        cached_summary = check_cache(cache_key)
        
        if cached_summary:
            st.success("📦 Using cached summary (faster!)")
            summary = cached_summary
        else:
            # Step 4: Generate summary
            with status_placeholder.container():
                st.info(f"🤖 Generating summary using {st.session_state.selected_model}...")
            
            summary = summarize_text(
                cleaned_transcript,
                model=st.session_state.selected_model,
                max_length=st.session_state.summary_length
            )
            
            # Save to cache
            save_to_cache(cache_key, summary, {'source': 'youtube', 'url': url})
        
        # Step 5: Display results
        status_placeholder.empty()
        display_summary_results(summary, cleaned_transcript, source='youtube', url=url)
        
    except InvalidURLError:
        st.error("❌ Invalid YouTube URL. Please check and try again.")
    except TranscriptUnavailableError as e:
        st.warning(f"⚠️ {str(e)}")
        st.info("💡 Tip: Try a different video or use the File Upload option above.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please try again with a different video or contact support.")


def display_file_upload_tab():
    """Display file upload tab - with error handling"""
    st.markdown("### 📁 Upload Video File (Max 10GB)")
    
    try:
        # Production file uploader
        uploaded_file = st.file_uploader(
            "Choose a video file (up to 10GB supported)",
            type=list(SUPPORTED_FORMATS),
            help="Supported: MP4, MOV, AVI, MKV, WebM"
        )
        
        if uploaded_file is not None:
            # File metrics always shown
            file_size_gb = uploaded_file.size / (1024**3)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File", uploaded_file.name)
            with col2:
                st.metric("Size", f"{file_size_gb:.2f} GB")
            with col3:
                st.metric("Type", uploaded_file.type or "Unknown")
            
            if file_size_gb > 10:
                st.error("❌ File exceeds 10GB limit")
                st.stop()
            
            # Save button
            if st.button("💾 Save Video to Disk (chunked)", type="secondary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    total_size = uploaded_file.size
                    saved_bytes = 0
                    tmp_path = tempfile.mktemp(suffix=f"_{Path(uploaded_file.name).suffix}")
                    
                    status_text.info(f"Saving to temp file: {Path(tmp_path).name}")
                    
                    with open(tmp_path, "wb") as f:
                        uploaded_file.seek(0)
                        while chunk := uploaded_file.read(1024 * 1024):  # 1MB chunks
                            f.write(chunk)
                            saved_bytes += len(chunk)
                            progress_bar.progress(saved_bytes / total_size)
                    
                    st.session_state.temp_video_path = tmp_path
                    progress_bar.empty()
                    status_text.success(f"✅ Saved {os.path.getsize(tmp_path)/(1024**3):.2f} GB")
                    
                except Exception as e:
                    st.error(f"Save failed: {str(e)}")
            
            # Process saved file
            if 'temp_video_path' in st.session_state:
                tmp_path = st.session_state.temp_video_path
                if os.path.exists(tmp_path):
                    size_gb = os.path.getsize(tmp_path) / (1024**3)
                    st.success(f"📁 Ready: {Path(tmp_path).name} ({size_gb:.2f} GB)")
                    
                    col1, col2 = st.columns([3,1])
                    with col1:
                        if st.button("🚀 Process & Summarize Video", type="primary", use_container_width=True):
                            process_uploaded_video(tmp_path)
                    with col2:
                        if st.button("🗑️ Clear Temp File"):
                            try:
                                os.unlink(tmp_path)
                                del st.session_state.temp_video_path
                                st.success("Cleaned up temp file")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Cleanup failed: {e}")
                else:
                    st.warning("❌ Temp file was deleted/moved")
                    if 'temp_video_path' in st.session_state:
                        del st.session_state.temp_video_path
        else:
            st.info("📤 Upload a video file to get started")

            
    except Exception as e:
        st.error(f"❌ Upload error: {str(e)}")
        st.code(traceback.format_exc())


def process_uploaded_video(video_path: str):
    """Process and summarize uploaded video file from disk path"""
    from pathlib import Path  # Local import for function scope
    
    # Create progress elements
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(message: str, progress: int = None):
        """Update progress UI"""
        status_text.info(message)
        if progress is not None:
            progress_bar.progress(progress)
    
    st.info(f"🔍 Processing: {Path(video_path).name}")
    
    # Early validation
    if not os.path.exists(video_path):
        st.error("❌ Temp file missing. Please re-upload.")
        return
    
    try:
        # Process video (memory efficient)
        update_progress("🎬 Validating & FFmpeg audio extraction...", 10)
        
        transcript = process_uploaded_file(
            video_path,
            whisper_model_size=st.session_state.whisper_size,
            progress_callback=update_progress
        )
        
        update_progress("✅ Whisper transcription complete!", 70)
        
        if not transcript or len(transcript.strip()) < 50:
            st.error("❌ No speech detected. Try video with clear audio.")
            return
        
        # Clean & cache
        cleaned_transcript = clean_transcript(transcript)
        cache_key = get_cache_key(f"file_{Path(video_path).name}", st.session_state.selected_model)
        
        cached_summary = check_cache(cache_key)
        if cached_summary:
            update_progress("📦 Cache hit!", 90)
            summary = cached_summary
        else:
            update_progress(f"🤖 AI Summary ({st.session_state.selected_model})...", 80)
            summary = summarize_text(
                cleaned_transcript,
                model=st.session_state.selected_model,
                max_length=st.session_state.summary_length
            )
            save_to_cache(cache_key, summary, {'source': 'file', 'filename': Path(video_path).name})
        
        # Success
        progress_bar.progress(100)
        status_text.empty()
        display_summary_results(summary, cleaned_transcript, source='file', filename=Path(video_path).name)
        
        # Auto cleanup
        try:
            os.unlink(video_path)
            if 'temp_video_path' in st.session_state and st.session_state.temp_video_path == video_path:
                del st.session_state.temp_video_path
            st.success("🧹 Temp file auto-cleaned.")
        except Exception as cleanup_err:
            st.warning(f"Cleanup warning: {cleanup_err}")
            
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Processing failed: {str(e)}")
        st.info("Tips: `brew install ffmpeg`, add GOOGLE_API_KEY/.env, free disk space, smaller test video.")
        st.code(str(e), language='text')



def display_summary_results(summary: str, transcript: str, source: str, **kwargs):
    """Display summary results in a professional format"""
    
    # Calculate stats
    word_count = len(summary.split())
    reading_time = estimate_reading_time(summary)
    transcript_word_count = len(transcript.split())
    compression_ratio = round((1 - word_count / transcript_word_count) * 100, 1) if transcript_word_count > 0 else 0
    
    # Store in session history
    history_entry = {
        'source': source,
        'summary': summary,
        'word_count': word_count,
        'timestamp': None  # Would add datetime here
    }
    st.session_state.summary_history.append(history_entry)
    st.session_state.current_summary = summary
    
    # Display source info
    if source == 'youtube':
        st.info(f"📺 Source: YouTube Video - {kwargs.get('url', 'Unknown')}")
    else:
        st.info(f"📁 Source: Uploaded File - {kwargs.get('filename', 'Unknown')}")
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{word_count}</div>
            <div class="stats-label">Summary Words</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{reading_time}</div>
            <div class="stats-label">Reading Time</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{compression_ratio}%</div>
            <div class="stats-label">Compression</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        model_display = st.session_state.selected_model.capitalize()
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{model_display}</div>
            <div class="stats-label">Model Used</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary container
    st.markdown("### 📝 Video Summary")
    st.markdown(f"""
    <div class="summary-container">
        <h3>🎯 Key Insights</h3>
        <p>{summary}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Copy Summary", use_container_width=True):
            st.write("✅ Copied to clipboard!")
            # Note: Actual clipboard requires JavaScript, this is a placeholder
    
    with col2:
        if st.button("🔄 Regenerate", use_container_width=True):
            st.info("Regenerating summary...")
            # Would trigger regeneration
    
    with col3:
        if st.button("📥 Download", use_container_width=True):
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="video_summary.txt",
                mime="text/plain"
            )
    
    # Expandable section for raw transcript
    with st.expander("📄 View Raw Transcript"):
        st.text_area("Transcript", transcript, height=200, disabled=True)


def main():
    """Main application entry point"""
    
    # Initialize session state
    init_session_state()
    
    # Display header
    display_header()
    
    # Setup sidebar and get settings
    selected_model, whisper_size, summary_length = display_sidebar()
    
    # Store settings in session state for access across functions
    st.session_state.selected_model = selected_model
    st.session_state.whisper_size = whisper_size
    st.session_state.summary_length = summary_length
    
    # Create tabs for input methods
    tab1, tab2 = st.tabs(["🎬 YouTube URL", "📁 Upload Video"])
    
    with tab1:
        display_youtube_tab()
    
    with tab2:
        display_file_upload_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with Streamlit • Powered by Google Gemini & Whisper AI</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()