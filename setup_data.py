#!/usr/bin/env python3
"""
Updated setup script using kagglehub.dataset_download(), allowing
user-defined download path, and correctly loading JSON files.
"""

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


def download_kaggle_dataset(force_redownload=False):
    """
    Download dataset using kagglehub.dataset_download and allow
    the user to specify a custom output directory. Supports JSON files.
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

        # Load JSON files
        json_files = list(final_path.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {final_path}")

        print("\nFound JSON files:")
        for jf in json_files:
            print(f"  - {jf.name}")

        frames = []
        for jf in json_files:
            try:
                data = pd.read_json(jf, orient="records")
                data["__source_file"] = jf.name
                frames.append(data)
                print(f"✓ Loaded {jf.name}: {data.shape[0]} rows")
            except Exception as e:
                print(f"Skipping {jf.name}: {e}")

        if not frames:
            raise ValueError("No valid JSON files could be loaded.")

        df = pd.concat(frames, ignore_index=True)
        print(f"\n✓ Combined dataset shape: {df.shape}")

        # Cache dataframe for later processing
        cache_file = Path("cache/kaggle_dataset.pkl")
        cache_file.parent.mkdir(exist_ok=True)
        df.to_pickle(cache_file)
        print(f"✓ Cached dataframe: {cache_file}")

        return df

    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        sys.exit(1)


def preprocess_dataset(df):
    print("\n" + "=" * 60)
    print("PREPROCESSING DATASET")
    print("=" * 60)

    from data_processor import DataProcessor
    processor = DataProcessor()

    processed_df = processor.process_dataframe(df)

    output_file = Path("data/processed_transcripts.csv")
    output_file.parent.mkdir(exist_ok=True)
    processed_df.to_csv(output_file, index=False)

    print(f"✓ Saved processed dataset: {output_file}")
    return processed_df


def generate_embeddings(processed_df, batch_size=32):
    print("\n" + "=" * 60)
    print("GENERATING EMBEDDINGS")
    print("=" * 60)

    from embedding_generator import EmbeddingGenerator

    cache_file = Path("cache/embeddings.npy")
    texts_file = Path("cache/embedding_texts.pkl")

    if cache_file.exists() and texts_file.exists():
        response = input("Use cached embeddings? (y/n): ").lower()
        if response == "y":
            embeddings = np.load(cache_file)
            texts = pd.read_pickle(texts_file)
            return embeddings, texts

    embed_gen = EmbeddingGenerator()
    embeddings, texts = embed_gen.generate_embeddings(processed_df, batch_size=batch_size)

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

    import json

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


def run_test_search(processed_df, embeddings, texts):
    print("\n" + "=" * 60)
    print("RUNNING TEST SEARCH")
    print("=" * 60)

    from semantic_search import SemanticSearch
    searcher = SemanticSearch(embeddings, texts, processed_df)

    test_queries = [
        "stories about forgiveness",
        "episodes featuring immigrants",
        "childhood memories",
    ]

    for q in test_queries:
        results = searcher.search(q, top_k=3)
        print(f"- Query '{q}' → {len(results)} results")


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
    else:
        df = download_kaggle_dataset(force_redownload=args.force_redownload)

    processed_df = preprocess_dataset(df)

    if not args.skip_embeddings:
        embeddings, texts = generate_embeddings(processed_df, batch_size=args.batch_size)
        create_summary_stats(processed_df, embeddings)
        run_test_search(processed_df, embeddings, texts)

    print("\nSetup Complete! ✓")


if __name__ == "__main__":
    main()
