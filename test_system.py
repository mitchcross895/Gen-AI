"""
System Test Script
Verifies that all components are working correctly
"""

import os
import sys
from pathlib import Path

def check_files():
    """Check if required files exist"""
    print("Checking required files...")
    
    required_files = [
        "requirements.txt",
        "data_preprocessing.py",
        "embedding_generation.py",
        "summarization.py",
        "app.py",
        "setup_data.py",
        "create_sample_data.py"
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            missing.append(file)
    
    return len(missing) == 0

def check_data_files():
    """Check if data files exist"""
    print("\nChecking data files...")
    
    data_files = {
        "this_american_life_transcripts.csv": "Source dataset",
        "preprocessed_transcripts.csv": "Preprocessed data",
        "faiss_index.idx": "Search index",
        "metadata.pkl": "Search metadata",
        "episode_summaries.csv": "Episode summaries"
    }
    
    for file, desc in data_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            size_mb = size / (1024 * 1024)
            print(f"  ✓ {file} ({size_mb:.2f} MB) - {desc}")
        else:
            print(f"  ✗ {file} - NOT FOUND ({desc})")

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking Python packages...")
    
    required_packages = [
        "streamlit",
        "pandas",
        "numpy",
        "sentence_transformers",
        "transformers",
        "torch",
        "faiss",
        "nltk"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    return len(missing) == 0

def test_preprocessing():
    """Test data preprocessing"""
    print("\nTesting data preprocessing module...")
    try:
        from data_processor import TranscriptPreprocessor
        preprocessor = TranscriptPreprocessor()
        
        # Test text cleaning
        test_text = "  Hello,   this is a test!  "
        cleaned = preprocessor.clean_text(test_text)
        assert len(cleaned) > 0
        print("  ✓ Text cleaning works")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_embeddings():
    """Test embedding generation"""
    print("\nTesting embedding generation...")
    try:
        from embedding_generator import EmbeddingGenerator
        
        print("  Loading model (this may take a moment)...")
        embed_gen = EmbeddingGenerator()
        
        # Test embedding generation
        test_texts = ["This is a test sentence.", "Another test sentence."]
        embeddings = embed_gen.generate_embeddings(test_texts)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == embed_gen.dimension
        print(f"  ✓ Generated embeddings with shape {embeddings.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_summarization():
    """Test summarization"""
    print("\nTesting summarization module...")
    try:
        from summarizer import EpisodeSummarizer
        
        print("  Loading model (this may take a moment)...")
        summarizer = EpisodeSummarizer(use_api=False)
        
        test_text = """
        This is a test transcript. It talks about a person who went on a journey.
        They traveled to many places and met interesting people along the way.
        In the end, they learned important lessons about life and friendship.
        The story is both touching and inspiring.
        """
        
        summary = summarizer.summarize_episode(test_text, "brief")
        assert len(summary) > 0
        print(f"  ✓ Generated summary: {summary[:100]}...")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def run_full_test():
    """Run complete system test"""
    print("=" * 60)
    print("PODCAST INSIGHT ENGINE - SYSTEM TEST")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Files
    if not check_files():
        print("\n❌ Some required files are missing!")
        all_passed = False
    
    # Test 2: Dependencies
    if not check_dependencies():
        print("\n❌ Some required packages are not installed!")
        print("Run: pip install -r requirements.txt")
        all_passed = False
    
    # Test 3: Data files
    check_data_files()
    
    # Test 4: Components (if dependencies are available)
    if all_passed:
        print("\n" + "=" * 60)
        print("TESTING COMPONENTS")
        print("=" * 60)
        
        if not test_preprocessing():
            all_passed = False
        
        if not test_embeddings():
            all_passed = False
        
        if not test_summarization():
            all_passed = False
    
    # Final report
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour system is ready to use.")
        print("\nNext steps:")
        if not os.path.exists("preprocessed_transcripts.csv"):
            print("1. Run: python create_sample_data.py")
            print("2. Run: python setup_data.py --sample")
        print("3. Run: streamlit run app.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before proceeding.")
    
    return all_passed

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)