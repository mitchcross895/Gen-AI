#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json


def check_dependencies():
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
        print(f"\nMissing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    print("✓ All dependencies installed\n")
    return True


def load_json_file(json_path):
    """
    Intelligently load a JSON file by inspecting its structure.
    Handles various JSON formats including nested structures.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Case 1: Already a list of records
    if isinstance(data, list):
        return pd.DataFrame(data)
    
    # Case 2: Dictionary with a single key containing the records
    if isinstance(data, dict):
        # If it's a simple key-value dict (like speaker map), convert to records
        if all(not isinstance(v, (list, dict)) for v in data.values()):
            return pd.DataFrame([{"key": k, "value": v} for k, v in data.items()])
        
        # If one of the dict values is a list, try that
        for key, value in data.items():
            if isinstance(value, list):
                df = pd.DataFrame(value)
                df['__data_key'] = key
                return df
        
        # Try to convert dict directly to DataFrame
        try:
            return pd.DataFrame(data)
        except:
            # Last resort: wrap in a list
            return pd.DataFrame([data])
    
    raise ValueError(f"Unsupported JSON structure: {type(data)}")


def download_kaggle_dataset(force_redownload=False):
    """
    Download dataset using kagglehub.dataset_download and allow
    the user to specify a custom output directory. Supports JSON and CSV files.
    """
    import kagglehub

    print("=" * 60)
    print("DOWNLOADING KAGGLE DATASET")
    print("=" * 60)

    print("\nEnter custom download directory (optional).")
    print("Press ENTER to use the default kagglehub cache.")
    user_path = input("Download path: ").strip()

    try:
        print("\nDownloading from Kaggle...")
        kaggle_path = kagglehub.dataset_download(
            "shuyangli94/this-american-life-podcast-transcriptsalignments"
        )
        print(f"\n✓ Dataset downloaded to: {kaggle_path}")

        final_path = Path(user_path) if user_path else Path(kaggle_path)

        if user_path:
            import shutil
            final_path.mkdir(parents=True, exist_ok=True)
            shutil.copytree(kaggle_path, final_path, dirs_exist_ok=True)
            print(f"✓ Copied dataset to: {final_path}")

        # Load both JSON and CSV files
        json_files = list(final_path.glob("*.json"))
        csv_files = list(final_path.glob("*.csv"))
        
        # Skip certain metadata files
        skip_files = {'kaggle.json', 'full-speaker-map.json'}
        json_files = [f for f in json_files if f.name not in skip_files]
        
        all_files = json_files + csv_files
        
        if not all_files:
            raise FileNotFoundError(f"No JSON or CSV files found in {final_path}")

        print("\nFound data files:")
        for f in all_files:
            print(f"  - {f.name}")

        frames = []
        
        # Process JSON files
        for jf in json_files:
            try:
                print(f"\nProcessing {jf.name}...")
                df = load_json_file(jf)
                df["__source_file"] = jf.name
                frames.append(df)
                print(f"✓ Loaded {jf.name}: {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"  Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
            except Exception as e:
                print(f"✗ Error loading {jf.name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Process CSV files
        for cf in csv_files:
            try:
                print(f"\nProcessing {cf.name}...")
                df = pd.read_csv(cf)
                df["__source_file"] = cf.name
                frames.append(df)
                print(f"✓ Loaded {cf.name}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"✗ Error loading {cf.name}: {e}")

        if not frames:
            raise ValueError("No valid JSON or CSV files could be loaded.")

        # Combine all dataframes
        df = pd.concat(frames, ignore_index=True, sort=False)
        print(f"\n✓ Combined dataset shape: {df.shape}")
        print(f"✓ Total columns: {len(df.columns)}")
        print(f"✓ Columns: {list(df.columns)}")

        # Cache dataframe for later processing
        cache_file = Path("cache/kaggle_dataset.pkl")
        cache_file.parent.mkdir(exist_ok=True)
        df.to_pickle(cache_file)
        print(f"✓ Cached dataframe: {cache_file}")

        return df

    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def preprocess_dataset(df):
    """
    Preprocess the dataset - with fallback if DataProcessor doesn't exist.
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING DATASET")
    print("=" * 60)

    try:
        from data_processor import DataProcessor
        processor = DataProcessor()
        processed_df = processor.process_dataframe(df)
    except (ImportError, AttributeError) as e:
        print(f"Note: DataProcessor not available ({e})")
        print("Using basic preprocessing instead...")
        
        # Basic preprocessing fallback
        processed_df = df.copy()
        
        # Common text field names to look for
        text_fields = ['text', 'transcript', 'content', 'utterance', 'dialogue']
        text_col = None
        for field in text_fields:
            if field in processed_df.columns:
                text_col = field
                break
        
        if text_col:
            # Basic text cleaning
            processed_df['text'] = processed_df[text_col].astype(str).str.strip()
            processed_df = processed_df[processed_df['text'].str.len() > 0]
            print(f"✓ Using '{text_col}' as text field")
        else:
            print(f"Available columns: {list(processed_df.columns)}")
            print("Warning: No standard text field found")
        
        # Try to identify other useful fields
        if 'speaker' not in processed_df.columns:
            for col in ['speaker_id', 'speaker_name', 'author']:
                if col in processed_df.columns:
                    processed_df['speaker'] = processed_df[col]
                    break
        
        if 'episode_title' not in processed_df.columns:
            for col in ['title', 'episode', 'episode_name']:
                if col in processed_df.columns:
                    processed_df['episode_title'] = processed_df[col]
                    break
        
        print(f"✓ Processed {len(processed_df)} records")

    output_file = Path("data/processed_transcripts.csv")
    output_file.parent.mkdir(exist_ok=True)
    processed_df.to_csv(output_file, index=False)

    print(f"✓ Saved processed dataset: {output_file}")
    return processed_df


def generate_embeddings(processed_df, batch_size=32):
    """
    Generate embeddings - with fallback if EmbeddingGenerator doesn't exist.
    """
    print("\n" + "=" * 60)
    print("GENERATING EMBEDDINGS")
    print("=" * 60)

    cache_file = Path("cache/embeddings.npy")
    texts_file = Path("cache/embedding_texts.pkl")

    if cache_file.exists() and texts_file.exists():
        response = input("Use cached embeddings? (y/n): ").lower()
        if response == "y":
            embeddings = np.load(cache_file)
            texts = pd.read_pickle(texts_file)
            print(f"✓ Loaded cached embeddings: {embeddings.shape}")
            return embeddings, texts

    try:
        from embedding_generator import EmbeddingGenerator
        embed_gen = EmbeddingGenerator()
        embeddings, texts = embed_gen.generate_embeddings(processed_df, batch_size=batch_size)
    except (ImportError, AttributeError) as e:
        print(f"Note: EmbeddingGenerator not available ({e})")
        print("Using basic sentence-transformers instead...")
        
        from sentence_transformers import SentenceTransformer
        
        # Find text column
        text_col = 'text' if 'text' in processed_df.columns else processed_df.columns[0]
        texts = processed_df[text_col].astype(str).tolist()
        
        print(f"Loading model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        embeddings = np.array(embeddings)
        
        print(f"✓ Generated embeddings: {embeddings.shape}")

    cache_file.parent.mkdir(exist_ok=True)
    np.save(cache_file, embeddings)
    pd.to_pickle(texts, texts_file)

    print("✓ Embeddings generated and cached.")
    return embeddings, texts


def create_summary_stats(processed_df, embeddings):
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY STATISTICS")
    print("=" * 60)

    stats = {
        "total_records": len(processed_df),
        "total_episodes": processed_df["episode_title"].nunique()
        if "episode_title" in processed_df.columns else "N/A",
        "avg_text_length": processed_df["text"].str.len().mean()
        if "text" in processed_df.columns else "N/A",
        "total_words": processed_df["text"].str.split().str.len().sum()
        if "text" in processed_df.columns else "N/A",
        "embedding_shape": embeddings.shape,
        "embedding_dimension": embeddings.shape[1],
        "speakers": processed_df["speaker"].nunique()
        if "speaker" in processed_df.columns else "N/A",
    }

    stats_file = Path("cache/dataset_stats.json")

    json_stats = {}
    for k, v in stats.items():
        if isinstance(v, (np.integer, np.floating)):
            json_stats[k] = float(v)
        elif isinstance(v, tuple):
            json_stats[k] = list(v)
        else:
            json_stats[k] = v

    with open(stats_file, "w") as f:
        json.dump(json_stats, f, indent=2)

    print(f"✓ Stats saved: {stats_file}")
    
    # Display summary
    print("\nDataset Summary:")
    for key, value in json_stats.items():
        print(f"  {key}: {value}")


def run_test_search(processed_df, embeddings, texts):
    print("\n" + "=" * 60)
    print("RUNNING TEST SEARCH")
    print("=" * 60)

    try:
        from semantic_search import SemanticSearch
        searcher = SemanticSearch(embeddings, texts, processed_df)

        test_queries = [
            "stories about forgiveness",
            "episodes featuring immigrants",
            "childhood memories",
        ]

        for q in test_queries:
            results = searcher.search(q, top_k=3)
            print(f"✓ Query '{q}' → {len(results)} results")
    except (ImportError, AttributeError) as e:
        print(f"Note: SemanticSearch not available ({e})")
        print("Skipping test search - you can implement this later")


def main():
    parser = argparse.ArgumentParser(description="Setup This American Life dataset")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not check_dependencies():
        sys.exit(1)

    if args.skip_download:
        cache_file = Path("cache/kaggle_dataset.pkl")
        if not cache_file.exists():
            print("✗ No cached dataset found.")
            sys.exit(1)
        df = pd.read_pickle(cache_file)
        print(f"✓ Loaded cached dataset: {df.shape}")
    else:
        df = download_kaggle_dataset(force_redownload=args.force_redownload)

    processed_df = preprocess_dataset(df)

    if not args.skip_embeddings:
        embeddings, texts = generate_embeddings(processed_df, batch_size=args.batch_size)
        create_summary_stats(processed_df, embeddings)
        run_test_search(processed_df, embeddings, texts)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE! ✓")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - cache/kaggle_dataset.pkl (raw data)")
    print("  - data/processed_transcripts.csv (processed data)")
    if not args.skip_embeddings:
        print("  - cache/embeddings.npy (embeddings)")
        print("  - cache/embedding_texts.pkl (texts)")
        print("  - cache/dataset_stats.json (statistics)")


if __name__ == "__main__":
    main()