"""
Configuration settings for the Podcast Insight Engine.
Modify these settings to customize the system behavior.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Model configurations
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Options: all-MiniLM-L6-v2, all-mpnet-base-v2
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"  # Options: facebook/bart-large-cnn, t5-small
SPACY_MODEL = "en_core_web_sm"  # Options: en_core_web_sm, en_core_web_md, en_core_web_lg

# Processing settings
BATCH_SIZE = 32  # For embedding generation
MIN_SEGMENT_LENGTH = 50  # Minimum characters for text segments
MAX_INPUT_LENGTH = 1024  # Maximum tokens for summarization input

# Search settings
DEFAULT_SEARCH_RESULTS = 5
MIN_SIMILARITY_THRESHOLD = 0.0
MAX_SEARCH_RESULTS = 50

# Summarization settings
SUMMARY_MAX_LENGTH = 150  # tokens
SUMMARY_MIN_LENGTH = 50   # tokens
NUM_HIGHLIGHTS = 3

# Topic extraction settings
NUM_TOPICS = 10
MAX_TFIDF_FEATURES = 100
MIN_DOCUMENT_FREQUENCY = 2
NGRAM_RANGE = (1, 2)  # Include unigrams and bigrams

# Entity extraction settings
MAX_DOCS_FOR_NER = 100  # Process max N documents for NER (for speed)
ENTITY_MIN_LENGTH = 2   # Minimum characters for entity

# API settings (if using external APIs)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Caching settings
ENABLE_EMBEDDING_CACHE = True
EMBEDDING_CACHE_FILE = CACHE_DIR / "embeddings.npy"
METADATA_CACHE_FILE = CACHE_DIR / "metadata.pkl"

# UI settings
STREAMLIT_THEME = {
    "primaryColor": "#FF4B4B",
    "backgroundColor": "#FFFFFF",
    "secondaryBackgroundColor": "#F0F2F6",
    "textColor": "#262730",
    "font": "sans serif"
}

# Logging settings
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FILE = BASE_DIR / "podcast_engine.log"

# Performance settings
USE_GPU = False  # Set to True if GPU available
DEVICE = 0 if USE_GPU else -1  # 0 for GPU, -1 for CPU

# Evaluation settings
TEST_QUERIES = [
    "stories about forgiveness",
    "episodes featuring immigrants",
    "conversations about mental health",
    "tales from the workplace",
    "stories about childhood"
]

# Column name mappings (for auto-detection)
COLUMN_MAPPINGS = {
    "text": ["text", "transcript", "content", "body", "speech"],
    "episode_title": ["episode_title", "title", "episode_name", "name"],
    "episode_id": ["episode_id", "ep_id", "id", "episode_number"],
    "speaker": ["speaker", "person", "name", "author"],
    "start_time": ["start_time", "start", "begin_time"],
    "end_time": ["end_time", "end", "finish_time"],
    "date": ["date", "air_date", "published", "publish_date"]
}

# Error messages
ERROR_MESSAGES = {
    "no_text_column": "Could not identify text column in dataset. Please ensure your CSV has a column containing transcript text.",
    "no_embeddings": "Embeddings not generated yet. Please click 'Generate Embeddings' in the sidebar.",
    "empty_query": "Please enter a search query.",
    "no_results": "No results found for your query. Try different search terms.",
    "episode_not_found": "Episode not found in dataset.",
    "model_load_error": "Error loading model. Please check your internet connection and try again."
}

# Helper functions
def get_cache_path(filename: str) -> Path:
    """Get path for cache file."""
    return CACHE_DIR / filename

def get_output_path(filename: str) -> Path:
    """Get path for output file."""
    return OUTPUT_DIR / filename

def get_data_path(filename: str) -> Path:
    """Get path for data file."""
    return DATA_DIR / filename

# Validate configuration
def validate_config():
    """Validate configuration settings."""
    errors = []
    
    if BATCH_SIZE < 1:
        errors.append("BATCH_SIZE must be at least 1")
    
    if MIN_SEGMENT_LENGTH < 10:
        errors.append("MIN_SEGMENT_LENGTH should be at least 10")
    
    if SUMMARY_MAX_LENGTH < SUMMARY_MIN_LENGTH:
        errors.append("SUMMARY_MAX_LENGTH must be greater than SUMMARY_MIN_LENGTH")
    
    if NUM_TOPICS < 1:
        errors.append("NUM_TOPICS must be at least 1")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True

# Run validation on import
validate_config()