"""
Summarization Module
Generates episode summaries using local Hugging Face models only
Improved to handle long texts without token limits
"""

import pandas as pd
from typing import List, Dict
import os
from transformers import pipeline, AutoTokenizer
import re

class EpisodeSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Initialize summarizer with local Hugging Face model
        """
        print("Loading local summarization model...")
        self.model_name = model_name
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            device=-1  # CPU, use 0 for GPU
        )
        # Load tokenizer to properly count tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # BART max input is 1024 tokens, but we'll use 1000 to be safe
        self.max_input_tokens = 1000
        print("Local model loaded successfully")
    
    def count_tokens(self, text: str) -> int:
        """Count actual tokens in text"""
        return len(self.tokenizer.encode(text, truncation=False))
    
    def split_into_token_chunks(self, text: str, max_tokens: int = 1000) -> List[str]:
        """
        Split text into chunks that fit within token limit
        Uses sentence boundaries to avoid cutting mid-sentence
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds limit, force split it
            if sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence by words
                words = sentence.split()
                temp_chunk = []
                for word in words:
                    temp_chunk.append(word)
                    if self.count_tokens(' '.join(temp_chunk)) > max_tokens:
                        chunks.append(' '.join(temp_chunk[:-1]))
                        temp_chunk = [word]
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                continue
            
            # Check if adding this sentence exceeds limit
            if current_tokens + sentence_tokens > max_tokens:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def summarize_with_local_model(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """
        Summarize using local Hugging Face model
        Handles texts of any length by chunking if necessary
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary in tokens
            min_length: Minimum length of summary in tokens
            
        Returns:
            Summary string
        """
        text = text.strip()
        
        # Check if text is too short
        if len(text.split()) < 20:
            return text  # Return as-is if very short
        
        token_count = self.count_tokens(text)
        
        # If text fits within limit, summarize directly
        if token_count <= self.max_input_tokens:
            try:
                summary = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min(min_length, len(text.split()) // 2),
                    do_sample=False,
                    truncation=True
                )
                return summary[0]['summary_text']
            except Exception as e:
                print(f"Summarization error: {e}")
                return text[:500] + "..."
        
        # Text is too long - use chunking strategy
        print(f"  Text has {token_count} tokens, splitting into chunks...")
        chunks = self.split_into_token_chunks(text, self.max_input_tokens)
        print(f"  Created {len(chunks)} chunks")
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                # Use shorter summaries for chunks to keep final summary manageable
                chunk_max_length = min(max_length // 2, 130)
                summary = self.summarizer(
                    chunk,
                    max_length=chunk_max_length,
                    min_length=30,
                    do_sample=False,
                    truncation=True
                )
                chunk_summaries.append(summary[0]['summary_text'])
                print(f"    Chunk {i+1}/{len(chunks)} summarized")
            except Exception as e:
                print(f"    Error on chunk {i+1}: {e}")
                # Use first part of chunk as fallback
                chunk_summaries.append(chunk[:200] + "...")
        
        # Combine chunk summaries
        combined = ' '.join(chunk_summaries)
        
        # If combined summary is still too long, summarize again
        combined_tokens = self.count_tokens(combined)
        if combined_tokens > self.max_input_tokens:
            print(f"  Combined summary has {combined_tokens} tokens, creating final summary...")
            return self.summarize_with_local_model(combined, max_length, min_length)
        elif combined_tokens > 100:  # Only re-summarize if there's enough content
            print(f"  Creating final summary from {len(chunk_summaries)} chunk summaries...")
            try:
                final = self.summarizer(
                    combined,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )
                return final[0]['summary_text']
            except Exception as e:
                print(f"  Final summary error: {e}")
                return combined
        
        return combined
    
    def summarize_episode(self, episode_text: str, summary_type: str = "brief") -> str:
        """
        Main summarization method
        
        Args:
            episode_text: Full text to summarize
            summary_type: 'brief' (150 tokens), 'detailed' (300 tokens), or 'comprehensive' (500 tokens)
            
        Returns:
            Summary string
        """
        length_map = {
            "brief": (150, 50),
            "detailed": (300, 100),
            "comprehensive": (500, 150)
        }
        
        max_len, min_len = length_map.get(summary_type, (150, 50))
        return self.summarize_with_local_model(episode_text, max_len, min_len)
    
    def batch_summarize_episodes(self, df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
        """
        Generate summaries for all episodes in dataset
        
        Args:
            df: DataFrame with episode data
            sample_size: Optional limit on number of episodes to process
            
        Returns:
            DataFrame with episode summaries
        """
        print(f"Generating summaries for episodes...")
        
        episode_summaries = []
        
        # Determine grouping column
        if 'episode_id' in df.columns:
            group_col = 'episode_id'
        elif 'episode_title' in df.columns:
            group_col = 'episode_title'
        else:
            print("Warning: No episode identifier found")
            return pd.DataFrame()
        
        # Get unique episodes
        unique_episodes = df[group_col].unique()
        
        if sample_size:
            unique_episodes = unique_episodes[:sample_size]
            print(f"Processing {sample_size} episodes (sampled)")
        else:
            print(f"Processing {len(unique_episodes)} episodes")
        
        for idx, episode_id in enumerate(unique_episodes, 1):
            episode_data = df[df[group_col] == episode_id]
            
            if len(episode_data) == 0:
                continue
            
            # Combine all text for episode
            full_text = ' '.join(episode_data['text'].astype(str).tolist())
            
            print(f"\n[{idx}/{len(unique_episodes)}] Processing: {episode_id}")
            print(f"  Full text: {len(full_text)} characters, {len(full_text.split())} words")
            
            # Generate brief summary
            brief_summary = self.summarize_episode(full_text, "brief")
            
            # Get episode metadata
            first_row = episode_data.iloc[0]
            episode_title = first_row.get('episode_title', f'Episode {episode_id}')
            
            episode_summaries.append({
                'episode_id': episode_id,
                'episode_title': episode_title,
                'brief_summary': brief_summary,
                'word_count': len(full_text.split()),
                'character_count': len(full_text),
                'date': first_row.get('date', None)
            })
            
            print(f"  ✓ Summary generated ({len(brief_summary.split())} words)")
        
        return pd.DataFrame(episode_summaries)
    
    def summarize_with_extractive_fallback(self, text: str, num_sentences: int = 5) -> str:
        """
        Extractive summarization fallback for very long texts
        Selects the most representative sentences
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) <= num_sentences:
            return text
        
        # Simple scoring: sentences in the beginning and end are often important
        # Also prefer longer sentences (more content)
        scored_sentences = []
        for i, sent in enumerate(sentences):
            # Position score (higher for beginning and end)
            position_score = 1.0
            if i < len(sentences) * 0.2:  # First 20%
                position_score = 2.0
            elif i > len(sentences) * 0.8:  # Last 20%
                position_score = 1.5
            
            # Length score (prefer medium length)
            word_count = len(sent.split())
            length_score = min(word_count / 20.0, 1.0)
            
            total_score = position_score * length_score
            scored_sentences.append((total_score, i, sent))
        
        # Sort by score and take top sentences
        scored_sentences.sort(reverse=True)
        top_sentences = scored_sentences[:num_sentences]
        
        # Re-sort by original position to maintain flow
        top_sentences.sort(key=lambda x: x[1])
        
        return ' '.join([sent for _, _, sent in top_sentences])


def generate_highlights(text: str, num_highlights: int = 3) -> List[str]:
    """
    Extract key highlights from transcript
    Improved to find more meaningful moments
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Expanded keywords for important moments
    keywords = [
        'important', 'realized', 'discovered', 'incredible', 'amazing', 
        'shocking', 'surprised', 'never', 'always', 'remember', 'learned',
        'understand', 'believe', 'moment', 'suddenly', 'finally', 'decided',
        'changed', 'transformed', 'revelation', 'breakthrough', 'turning point',
        'significant', 'crucial', 'essential', 'profound', 'powerful'
    ]
    
    scored_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) < 8:  # Skip very short sentences
            continue
        
        # Count keyword matches
        keyword_score = sum(1 for kw in keywords if kw in sent.lower())
        
        # Bonus for quotes (often contain important statements)
        quote_score = 2 if '"' in sent else 0
        
        # Bonus for emotional punctuation
        emotion_score = sent.count('!') + sent.count('?')
        
        total_score = keyword_score + quote_score + emotion_score
        
        if total_score > 0:  # Only include sentences with some relevance
            scored_sentences.append((total_score, sent))
    
    # Sort by score and return top highlights
    scored_sentences.sort(reverse=True)
    highlights = [sent for score, sent in scored_sentences[:num_highlights] if sent]
    
    return highlights if highlights else ["No significant highlights found."]


def extract_key_quotes(text: str, num_quotes: int = 5) -> List[str]:
    """
    Extract potential key quotes from text
    Improved extraction logic
    """
    # Find sentences in quotes
    quoted = re.findall(r'"([^"]{20,})"', text)  # At least 20 chars
    
    if quoted and len(quoted) >= num_quotes:
        return quoted[:num_quotes]
    
    # Fallback: find sentences with first-person narrative
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    first_person_patterns = [
        r'\bI\b', r"\bI'm\b", r"\bI've\b", r"\bI'd\b", r"\bI'll\b",
        r'\bwe\b', r"\bwe're\b", r"\bwe've\b"
    ]
    
    first_person = []
    for sent in sentences:
        sent = sent.strip()
        # Check for first-person pronouns
        if any(re.search(pattern, sent, re.IGNORECASE) for pattern in first_person_patterns):
            if 10 <= len(sent.split()) <= 30:  # Good quote length
                first_person.append(sent)
    
    # Combine quoted and first-person
    all_quotes = quoted + first_person
    
    return all_quotes[:num_quotes] if all_quotes else ["No quotes found."]


if __name__ == "__main__":
    print("Testing EpisodeSummarizer...")
    
    if os.path.exists("data/processed_transcripts.csv"):
        df = pd.read_csv("data/processed_transcripts.csv")
        
        summarizer = EpisodeSummarizer()
        
        # Test with first episode
        if 'episode_title' in df.columns or 'episode_id' in df.columns:
            group_col = 'episode_title' if 'episode_title' in df.columns else 'episode_id'
            first_episode = df[group_col].iloc[0]
            episode_data = df[df[group_col] == first_episode]
            episode_text = ' '.join(episode_data['text'].astype(str).tolist())
            
            print(f"\nSummarizing: {first_episode}")
            print(f"Text length: {len(episode_text)} characters, {len(episode_text.split())} words")
            
            # Brief summary
            print("\nGenerating brief summary...")
            brief = summarizer.summarize_episode(episode_text, "brief")
            print(f"\nBrief Summary ({len(brief.split())} words):\n{brief}")
            
            # Detailed summary
            print("\nGenerating detailed summary...")
            detailed = summarizer.summarize_episode(episode_text, "detailed")
            print(f"\nDetailed Summary ({len(detailed.split())} words):\n{detailed}")
            
            # Extract highlights
            highlights = generate_highlights(episode_text, 3)
            print(f"\nKey Highlights:")
            for i, highlight in enumerate(highlights, 1):
                print(f"{i}. {highlight}")
        
        # Batch summarize (process 3 episodes as example)
        print("\n" + "="*60)
        print("Batch summarizing first 3 episodes...")
        summaries_df = summarizer.batch_summarize_episodes(df, sample_size=3)
        
        # Save summaries
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        summaries_df.to_csv(f"{output_dir}/episode_summaries.csv", index=False)
        print(f"\n✓ Summaries saved to {output_dir}/episode_summaries.csv")
        
        # Display results
        print("\n" + "="*60)
        print("SUMMARY RESULTS")
        print("="*60)
        for i, row in summaries_df.iterrows():
            print(f"\n{row['episode_title']}:")
            print(f"  Words: {row['word_count']:,}")
            print(f"  Summary: {row['brief_summary']}")
    else:
        print("\nNo processed data found. Please run setup_data.py first.")
        print("\nTesting with sample text:")
        
        sample_text = """
        In this episode, we explore the story of a family who moved across the country 
        to start a new life. They faced many challenges along the way, including financial 
        difficulties and cultural adjustments. However, through perseverance and support 
        from their community, they were able to build a successful business. The father 
        says, "I never imagined we would make it this far, but we believed in ourselves 
        and kept pushing forward." Their journey is a testament to the power of determination 
        and the importance of family bonds. Today, they help other immigrant families 
        navigate similar challenges. The mother adds, "Our struggles made us stronger, 
        and now we can give back to others who are just starting their journey."
        """ * 50  # Repeat to make it long
        
        summarizer = EpisodeSummarizer()
        
        print(f"Sample text length: {len(sample_text.split())} words")
        
        summary = summarizer.summarize_episode(sample_text, "brief")
        print(f"\nBrief Summary: {summary}")
        
        detailed = summarizer.summarize_episode(sample_text, "detailed")
        print(f"\nDetailed Summary: {detailed}")
        
        highlights = generate_highlights(sample_text, 3)
        print(f"\nHighlights:")
        for h in highlights:
            print(f"  - {h}")