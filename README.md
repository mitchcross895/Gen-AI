# Podcast Insight Engine

A powerful semantic search, summarization, and analysis tool for This American Life podcast transcripts. Built with Python, leveraging state-of-the-art NLP models for deep content exploration.

## Features

- **Semantic Search**: Find episodes and moments by meaning, not just keywords
- **AI Summarization**: Generate brief or detailed episode summaries
- **Topic Extraction**: Automatically identify themes and named entities
- **Similar Episodes**: Discover related content based on semantic similarity
- **Analytics**: View statistics and insights across the entire dataset
- **Multiple Interfaces**: CLI, interactive mode, and integration options

## Quick Start

### Prerequisites

- **Python 3.8+** (3.9 or 3.10 recommended)
- **8GB+ RAM** (16GB recommended for full dataset)
- **5GB+ disk space** for models and data

### Installation

#### Option 1: Automated Installation (Recommended)

**Linux/Mac:**
```bash
chmod +x install.sh
./install.sh
```

**Windows:**
```cmd
install.bat
```

#### Option 2: Manual Installation

1. **Clone or download the project**
```bash
cd podcast-insight-engine
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download spaCy model**
```bash
python -m spacy download en_core_web_sm
```

### Initial Setup

#### Step 1: Get Kaggle Credentials
```bash
python setup_kaggle.py
```
Follow the prompts to configure your Kaggle API access.

#### Step 2: Download and Process Data
```bash
python setup_data.py
```
This will:
- Download the This American Life dataset from Kaggle
- Process transcripts into searchable format
- Generate semantic embeddings
- Create search indices

**Options:**
- `--skip-download`: Use cached dataset
- `--skip-embeddings`: Skip embedding generation
- `--batch-size 16`: Reduce batch size for lower memory usage

## Usage Guide

### 1. View Episode Summaries

After running the summarizer, view your summaries:

```bash
# Interactive mode
python view_summaries.py

# List all episodes
python view_summaries.py --list

# Show specific episode
python view_summaries.py --show 5

# Search summaries
python view_summaries.py --search "immigration"

# Export to HTML
python view_summaries.py --export html
```

### 2. Generate Summaries

Create AI-powered episode summaries:

```bash
# Run the summarizer (processes all episodes)
python summarizer.py

# Summaries are saved to: output/episode_summaries.csv
```

**Note**: Summarization can take 1-2 hours for the full dataset. Start with a few episodes to test.

### 3. Semantic Search

#### Integrated Search (Recommended)

Search both episodes and specific moments:

```bash
# Interactive mode
python integrated_search.py

# Then use commands like:
> search episodes forgiveness stories
> search segments childhood memory  
> search all family dynamics
> similar Episode 742
> details The Dropout
```

**Command-line usage:**
```bash
# Find episodes by topic
python integrated_search.py --episodes "immigration stories"

# Find specific moments
python integrated_search.py --segments "I realized that"

# Search both
python integrated_search.py --all "love and loss"

# Find similar episodes
python integrated_search.py --similar "Episode 123"
```

#### Basic Transcript Search

For lower-level segment search:

```bash
python run_cli.py --mode search --data data/processed_transcripts.csv --query "your search term"
```

### 4. Topic Extraction

Extract themes and entities from episodes:

```bash
# CLI interface
python cli.py topics --num 20

# For specific episode
python cli.py topics --episode "Episode Title"

# Using run_cli.py
python run_cli.py --mode topics --data data/processed_transcripts.csv
```

### 5. Statistics and Analysis

View dataset statistics:

```bash
python cli.py stats --detailed

# Episode list
python cli.py episodes --limit 50

# System status
python cli.py status
```

## Project Structure

```
podcast-insight-engine/
├── setup_kaggle.py          # Kaggle API setup
├── setup_data.py            # Data download & processing
├── summarizer.py            # Episode summarization
├── view_summaries.py        # Summary browser
├── integrated_search.py     # Combined search system
├── semantic_search.py       # Core search engine
├── embedding_generator.py   # Embedding creation
├── data_processor.py        # Data preprocessing
├── topic_extractor.py       # Topic/entity extraction
├── cli.py                   # Main CLI interface
├── test_system.py          # System tests
├── config.py               # Configuration
├── requirements.txt        # Dependencies
├── install.sh              # Linux/Mac installer
├── install.bat             # Windows installer
└── README.md               # This file

Generated directories:
├── cache/                  # Embeddings & cached data
├── data/                   # Processed transcripts
└── output/                 # Summaries & exports
```

## Common Workflows

### Workflow 1: Explore Episode Topics

```bash
# 1. Generate summaries
python summarizer.py

# 2. Search for topic
python integrated_search.py
> search episodes mental health

# 3. Get details on interesting episode
> details Episode Title

# 4. Find similar episodes
> similar Episode Title
```

### Workflow 2: Find Specific Moments

```bash
# Search for specific phrases or themes
python integrated_search.py
> search segments "turning point in my life"
> search segments confrontation
```

### Workflow 3: Export for External Use

```bash
# Export summaries to HTML
python view_summaries.py --export html

# Export to JSON
python view_summaries.py --export json

# Export to Markdown
python view_summaries.py --export md
```

### Workflow 4: Analyze Dataset

```bash
# Interactive CLI
python cli.py interactive

# Then explore:
> stats --detailed
> topics --num 30
> episodes 100
```

## Configuration

### Memory Management

If you encounter memory issues:

1. **Reduce batch size** in `setup_data.py`:
```bash
python setup_data.py --batch-size 8
```

2. **Process fewer episodes** in `summarizer.py`:
```python
summaries_df = summarizer.batch_summarize_episodes(df, sample_size=10)
```

3. **Use sample mode**:
```bash
python setup_data.py --sample
```

### Model Selection

Edit the model in respective files:

**Embeddings** (`embedding_generator.py`):
```python
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, efficient
# or
model = SentenceTransformer('all-mpnet-base-v2')  # Better quality, slower
```

**Summarization** (`summarizer.py`):
```python
model_name = "facebook/bart-large-cnn"  # Default
# or  
model_name = "facebook/bart-large-xsum"  # More extractive
```

## Troubleshooting

### Issue: "No cached dataset found"

**Solution**: Run the data setup first:
```bash
python setup_data.py
```

### Issue: "Summaries file not found"

**Solution**: Generate summaries:
```bash
python summarizer.py
```

### Issue: "Out of memory"

**Solutions**:
1. Reduce batch size: `--batch-size 8`
2. Process fewer episodes: edit `sample_size` parameter
3. Use smaller model: switch to `all-MiniLM-L6-v2`
4. Close other applications

### Issue: "CUDA out of memory" (GPU)

**Solution**: Force CPU usage:
```python
# In summarizer.py and embedding_generator.py
device=-1  # CPU only
```

### Issue: "Kaggle API error"

**Solution**: Verify credentials:
```bash
python setup_kaggle.py
```

### Issue: "Module not found"

**Solution**: Reinstall dependencies:
```bash
pip install -r requirements.txt --force-reinstall
```

## Performance Tips

1. **First Run**: Use `--sample` mode to test everything works
2. **Embeddings**: Generate once, reuse (cached in `cache/`)
3. **Summaries**: Start with 5-10 episodes to gauge time
4. **Search**: Integrated search is faster than separate tools
5. **Batch Size**: Lower = slower but uses less memory

## Key Files Explained

| File | Purpose | When to Use |
|------|---------|-------------|
| `setup_kaggle.py` | Configure Kaggle API | First time setup |
| `setup_data.py` | Download & process data | After Kaggle setup |
| `summarizer.py` | Generate episode summaries | For overview analysis |
| `view_summaries.py` | Browse/export summaries | After summarization |
| `integrated_search.py` | Search episodes & segments | Primary search tool |
| `cli.py` | Full-featured CLI | Advanced operations |

## Output Files

### Generated Data

- `cache/kaggle_dataset.pkl` - Raw downloaded data
- `data/processed_transcripts.csv` - Cleaned transcripts
- `cache/embeddings.npy` - Semantic embeddings
- `cache/embedding_texts.pkl` - Text corresponding to embeddings
- `output/episode_summaries.csv` - AI-generated summaries

### Export Formats

From `view_summaries.py`:
- `output/summaries_export.txt` - Plain text
- `output/summaries_export.json` - JSON
- `output/summaries_export.md` - Markdown
- `output/summaries_export.html` - HTML (with search!)

## Contributing

This is a portfolio/educational project. Feel free to:
- Report issues
- Suggest improvements
- Fork and customize
- Use as reference for your own projects

## License

This project is for educational purposes. The This American Life transcripts are subject to their own copyright and terms of use.

## Acknowledgments

- **This American Life** for the amazing podcast content
- **Sentence Transformers** for semantic embeddings
- **Hugging Face** for transformers and BART models
- **spaCy** for NLP capabilities
- **Kaggle** for dataset hosting

## Support

Having issues? Check:
1. This README troubleshooting section
2. Run `python test_system.py` to diagnose issues
3. Check that all dependencies are installed
4. Verify Python version (3.8+)

##  Next Steps After Installation

1. ✅ Run `python test_system.py` to verify setup
2. ✅ Generate summaries: `python summarizer.py`
3. ✅ Try searching: `python integrated_search.py`
4. ✅ Explore: `python view_summaries.py`
5. ✅ Analyze: `python cli.py interactive`

---

**Happy exploring!**

*Built with ❤️ using Python, Transformers, and Sentence-BERT*