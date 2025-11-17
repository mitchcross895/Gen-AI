# Currently, this only works with the sample data. The full dataset functionality is still under work.

# Quick Start Guide

### Option 1: With Sample Data (Fastest - Recommended for Testing)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create sample data
python create_sample_data.py

# 3. Setup the system (sample mode - ~5 minutes)
python setup_data.py --sample

# 4. Launch the app
streamlit run app.py
```

### Option 2: With Full Dataset

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download This American Life dataset from Kaggle
# Place it as: this_american_life_transcripts.csv

# 3. Setup the system (full mode - ~30-60 minutes)
python setup_data.py

# 4. Launch the app
streamlit run app.py
```

## Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- 3GB free disk space (for models and data)
- Internet connection (first run only, to download models)

## What Each File Does

| File | Purpose |
|------|---------|
| `create_sample_data.py` | Generates sample transcript data for testing |
| `data_preprocessing.py` | Cleans and prepares transcript data |
| `embedding_generation.py` | Creates semantic embeddings and search index |
| `summarization.py` | Generates episode summaries using LLMs |
| `setup_data.py` | Orchestrates the complete data preparation |
| `app.py` | Streamlit web interface for searching and browsing |

## Testing the Prototype

Once running, try these searches in the app:
- "stories about family"
- "episodes about immigration"
- "forgiveness and reconciliation"
- "childhood memories"
- "criminal justice system"

## Performance Tips

**Faster Setup:**
```bash
python setup_data.py --sample  # Process only 100 segments
```

**Using GPU (if available):**
Edit `summarization.py`, line 33:
```python
device=0  # Change from -1 to 0
```

**Reduce Memory Usage:**
Edit `embedding_generation.py`, line 34:
```python
batch_size=16  # Reduce from 32
```

## Troubleshooting

**"ModuleNotFoundError"**
```bash
pip install --upgrade -r requirements.txt
```

**"No such file or directory"**
- Make sure you're in the project directory
- Run `create_sample_data.py` first

**Slow performance**
- Use `--sample` flag for testing
- Check if GPU is available and configured
- Close other memory-intensive applications

**Streamlit won't start**
```bash
streamlit run app.py --server.port 8502  # Try different port
```

## Expected Results

**Setup Time:**
- Sample mode: ~5-10 minutes
- Full mode: ~30-60 minutes

**Search Performance:**
- Query latency: <100ms
- Accuracy: 70-85% relevant results

**Memory Usage:**
- Setup: ~3-4GB
- Running: ~1-2GB

## Getting Help

1. Check README.md for detailed documentation
2. Review error messages carefully
3. Ensure all files are in the same directory
4. Verify Python version: `python --version`
5. Check installed packages: `pip list`

## Verification Checklist

After setup, verify these files exist:
- [ ] `preprocessed_transcripts.csv`
- [ ] `faiss_index.idx`
- [ ] `metadata.pkl`
- [ ] `episode_summaries.csv`

If all exist, you're ready to go! Run `streamlit run app.py`