#!/usr/bin/env python3
"""
Setup script to download and preprocess the This American Life dataset from Kaggle.
This script handles the full dataset preparation including downloading, processing, and caching.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")
    
    required_packages = {
        'kagglehub': 'kagglehub',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sentence_transformers': 'sentence-transformers',
        'transformers': 'transformers',
        'spacy': 'spacy',
        'streamlit': 'streamlit',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(pip_name)
    
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("✓ All dependencies installed\n")
    return True

def download_kaggle_dataset(force_redownload=False):
    """
    Download the This American Life dataset from Kaggle.
    
    Args:
        force_redownload: Force redownload even if file exists
        
    Returns:
        DataFrame with the dataset
    """
    print("="*60)
    print("DOWNLOADING KAGGLE DATASET")
    print("="*60)
    
    cache_file = Path("cache/kaggle_dataset.pkl")
    
    # Check if already downloaded
    if cache_file.exists() and not force_redownload:
        print(f"Found cached dataset at {cache_file}")
        print("Loading from cache...")
        df = pd.read_pickle(cache_file)
        print(f"✓ Loaded {len(df)} records from cache\n")
        return df
    
    print("\nDownloading from Kaggle (this may take a few minutes)...")
    print("Dataset: This American Life Podcast Transcripts and Alignments")
    
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        
        # Download the dataset
        print("\nAttempting to load dataset...")
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "shuyangli94/this-american-life-podcast-transcriptsalignments",
            "",  # Empty string loads the default file
        )
        
        print(f"\n✓ Downloaded successfully!")
        print(f"  Records: {len(df)}")
        print(f"  Columns: {', '.join(df.columns.tolist())}")
        print(f"  Shape: {df.shape}")
        
        # Cache the dataset
        cache_file.parent.mkdir(exist_ok=True)
        df.to_pickle(cache_file)
        print(f"  Cached to: {cache_file}")
        
        return df
    
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have Kaggle API credentials set up")
        print("2. Create ~/.kaggle/kaggle.json with your API token")
        print("3. Or manually download from:")
        print("   https://www.kaggle.com/datasets/shuyangli94/this-american-life-podcast-transcriptsalignments")
        print("4. Place the CSV file in the data/ directory")
        sys.exit(1)

def preprocess_dataset(df):
    """
    Preprocess the dataset for use with the Podcast Insight Engine.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Processed dataframe
    """
    print("\n" + "="*60)
    print("PREPROCESSING DATASET")
    print("="*60)
    
    from data_processor import DataProcessor
    
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    
    processor = DataProcessor()
    
    print("\nProcessing...")
    processed_df = processor.process_dataframe(df)
    
    print(f"\n✓ Processing complete!")
    print(f"  Processed shape: {processed_df.shape}")
    print(f"  Episodes: {processed_df['episode_title'].nunique() if 'episode_title' in processed_df.columns else 'N/A'}")
    print(f"  Average text length: {processed_df['text'].str.len().mean():.0f} characters")
    
    # Save processed data
    output_file = Path("data/processed_transcripts.csv")
    output_file.parent.mkdir(exist_ok=True)
    processed_df.to_csv(output_file, index=False)
    print(f"  Saved to: {output_file}")
    
    return processed_df

def generate_embeddings(processed_df, batch_size=32):
    """
    Generate embeddings for the dataset.
    
    Args:
        processed_df: Processed dataframe
        batch_size: Batch size for encoding
        
    Returns:
        Tuple of (embeddings, texts)
    """
    print("\n" + "="*60)
    print("GENERATING EMBEDDINGS")
    print("="*60)
    print("\nThis may take 30-60 minutes depending on your hardware...")
    
    from embedding_generator import EmbeddingGenerator
    
    # Check if embeddings already exist
    cache_file = Path("cache/embeddings.npy")
    texts_file = Path("cache/embedding_texts.pkl")
    
    if cache_file.exists() and texts_file.exists():
        print("\nFound cached embeddings!")
        response = input("Use cached embeddings? (y/n): ").lower()
        if response == 'y':
            print("Loading cached embeddings...")
            embeddings = np.load(cache_file)
            texts = pd.read_pickle(texts_file)
            print(f"✓ Loaded embeddings: {embeddings.shape}")
            return embeddings, texts
    
    print("\nInitializing embedding model...")
    embed_gen = EmbeddingGenerator()
    
    print(f"\nGenerating embeddings for {len(processed_df)} segments...")
    print(f"Batch size: {batch_size}")
    
    embeddings, texts = embed_gen.generate_embeddings(processed_df, batch_size=batch_size)
    
    print(f"\n✓ Embeddings generated!")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
    
    # Cache embeddings
    cache_file.parent.mkdir(exist_ok=True)
    np.save(cache_file, embeddings)
    pd.to_pickle(texts, texts_file)
    print(f"  Cached to: {cache_file}")
    
    return embeddings, texts

def create_summary_stats(processed_df, embeddings):
    """
    Create and save summary statistics.
    
    Args:
        processed_df: Processed dataframe
        embeddings: Generated embeddings
    """
    print("\n" + "="*60)
    print("GENERATING SUMMARY STATISTICS")
    print("="*60)
    
    stats = {
        'total_records': len(processed_df),
        'total_episodes': processed_df['episode_title'].nunique() if 'episode_title' in processed_df.columns else 'N/A',
        'avg_text_length': processed_df['text'].str.len().mean(),
        'total_words': processed_df['text'].str.split().str.len().sum(),
        'embedding_shape': embeddings.shape,
        'embedding_dimension': embeddings.shape[1],
        'speakers': processed_df['speaker'].nunique() if 'speaker' in processed_df.columns else 'N/A'
    }
    
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save stats
    import json
    stats_file = Path("cache/dataset_stats.json")
    
    # Convert numpy types to Python types for JSON serialization
    json_stats = {}
    for key, value in stats.items():
        if isinstance(value, (np.integer, np.floating)):
            json_stats[key] = float(value)
        elif isinstance(value, tuple):
            json_stats[key] = list(value)
        else:
            json_stats[key] = value
    
    with open(stats_file, 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    print(f"\n✓ Stats saved to: {stats_file}")

def run_test_search(processed_df, embeddings, texts):
    """
    Run a test search to verify everything works.
    
    Args:
        processed_df: Processed dataframe
        embeddings: Generated embeddings
        texts: List of texts
    """
    print("\n" + "="*60)
    print("RUNNING TEST SEARCH")
    print("="*60)
    
    from semantic_search import SemanticSearch
    
    searcher = SemanticSearch(embeddings, texts, processed_df)
    
    test_queries = [
        "stories about forgiveness",
        "episodes featuring immigrants",
        "childhood memories"
    ]
    
    print("\nTesting with sample queries:")
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        results = searcher.search(query, top_k=3)
        if results:
            print(f"  ✓ Found {len(results)} results")
            print(f"    Top result: {results[0]['episode_title'][:50]}...")
        else:
            print(f"  ✗ No results found")

def main():
    parser = argparse.ArgumentParser(description='Setup This American Life dataset')
    parser.add_argument('--skip-download', action='store_true', 
                       help='Skip downloading, use existing data')
    parser.add_argument('--skip-embeddings', action='store_true',
                       help='Skip embedding generation')
    parser.add_argument('--force-redownload', action='store_true',
                       help='Force redownload even if cached')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for embedding generation')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("THIS AMERICAN LIFE DATASET SETUP")
    print("="*60)
    print("\nThis script will:")
    print("1. Check dependencies")
    print("2. Download the Kaggle dataset (~700 episodes)")
    print("3. Preprocess and clean the data")
    print("4. Generate semantic embeddings")
    print("5. Create summary statistics")
    print("\nEstimated time: 30-60 minutes")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n✗ Please install missing dependencies first")
        sys.exit(1)
    
    # Download dataset
    if args.skip_download:
        print("\nSkipping download, loading existing data...")
        cache_file = Path("cache/kaggle_dataset.pkl")
        if not cache_file.exists():
            print(f"✗ No cached dataset found at {cache_file}")
            sys.exit(1)
        df = pd.read_pickle(cache_file)
    else:
        df = download_kaggle_dataset(force_redownload=args.force_redownload)
    
    # Preprocess
    processed_df = preprocess_dataset(df)
    
    # Generate embeddings
    if args.skip_embeddings:
        print("\nSkipping embedding generation")
        embeddings = None
        texts = None
    else:
        embeddings, texts = generate_embeddings(processed_df, batch_size=args.batch_size)
        
        # Create stats
        create_summary_stats(processed_df, embeddings)
        
        # Test search
        run_test_search(processed_df, embeddings, texts)
    
    # Final summary
    print("\n" + "="*60)
    print("SETUP COMPLETE! ✓")
    print("="*60)
    print("\nYour dataset is ready to use!")
    print("\nNext steps:")
    print("1. Run: streamlit run main.py")
    print("2. Or test with: python test_system.py")
    print("\nFiles created:")
    print(f"  - data/processed_transcripts.csv (processed dataset)")
    if not args.skip_embeddings:
        print(f"  - cache/embeddings.npy (semantic embeddings)")
        print(f"  - cache/embedding_texts.pkl (embedding texts)")
        print(f"  - cache/dataset_stats.json (statistics)")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()