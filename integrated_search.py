#!/usr/bin/env python3
"""
Integrated Search System
Combines semantic search with episode summaries for powerful exploration
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict
from semantic_search import SemanticSearch
from sentence_transformers import SentenceTransformer


class IntegratedSearch:
    """Enhanced search combining transcripts and summaries"""
    
    def __init__(
        self,
        embeddings_path="cache/embeddings.npy",
        texts_path="cache/embedding_texts.pkl",
        transcripts_path="data/processed_transcripts.csv",
        summaries_path="output/episode_summaries.csv"
    ):
        """
        Initialize integrated search system
        
        Args:
            embeddings_path: Path to transcript embeddings
            texts_path: Path to transcript texts
            transcripts_path: Path to processed transcripts
            summaries_path: Path to episode summaries
        """
        print("Loading search system...")
        
        # Load transcript search
        if Path(embeddings_path).exists() and Path(texts_path).exists():
            self.embeddings = np.load(embeddings_path)
            self.texts = pd.read_pickle(texts_path)
            self.transcripts_df = pd.read_csv(transcripts_path)
            self.transcript_search = SemanticSearch(
                self.embeddings, 
                self.texts, 
                self.transcripts_df
            )
            print(f"✓ Loaded {len(self.texts)} transcript segments")
        else:
            print("⚠ Transcript embeddings not found")
            self.transcript_search = None
        
        # Load summaries
        if Path(summaries_path).exists():
            self.summaries_df = pd.read_csv(summaries_path)
            print(f"✓ Loaded {len(self.summaries_df)} episode summaries")
            
            # Create embeddings for summaries
            print("Creating summary embeddings...")
            self.summary_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.summary_embeddings = self.summary_model.encode(
                self.summaries_df['brief_summary'].tolist(),
                show_progress_bar=True
            )
            print("✓ Summary embeddings ready")
        else:
            print("⚠ Episode summaries not found")
            self.summaries_df = None
    
    def search_episodes(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for episodes by semantic similarity to summaries
        This finds EPISODES (not segments) that match your query
        
        Args:
            query: Search query
            top_k: Number of episodes to return
            
        Returns:
            List of matching episodes with summaries
        """
        if self.summaries_df is None:
            print("❌ No summaries available")
            return []
        
        # Encode query
        query_embedding = self.summary_model.encode([query])
        
        # Compute similarities with summaries
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, self.summary_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Compile results
        results = []
        for idx in top_indices:
            row = self.summaries_df.iloc[idx]
            results.append({
                'episode_title': row['episode_title'],
                'episode_id': row.get('episode_id', 'N/A'),
                'summary': row['brief_summary'],
                'word_count': row.get('word_count', 'N/A'),
                'date': row.get('date', 'N/A'),
                'similarity': float(similarities[idx]),
                'rank': len(results) + 1
            })
        
        return results
    
    def search_segments(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Search for specific transcript segments
        This finds SPECIFIC MOMENTS in episodes
        
        Args:
            query: Search query
            top_k: Number of segments to return
            
        Returns:
            List of matching transcript segments
        """
        if self.transcript_search is None:
            print("❌ No transcript search available")
            return []
        
        return self.transcript_search.search(query, top_k=top_k)
    
    def search_combined(self, query: str, episodes: int = 3, segments: int = 5) -> Dict:
        """
        Combined search: find both relevant episodes and specific moments
        
        Args:
            query: Search query
            episodes: Number of episodes to return
            segments: Number of segments to return
            
        Returns:
            Dictionary with both episode and segment results
        """
        return {
            'episodes': self.search_episodes(query, top_k=episodes),
            'segments': self.search_segments(query, top_k=segments)
        }
    
    def find_similar_episodes(self, episode_title: str, top_k: int = 5) -> List[Dict]:
        """
        Find episodes similar to a given episode
        
        Args:
            episode_title: Title of reference episode
            top_k: Number of similar episodes to return
            
        Returns:
            List of similar episodes
        """
        if self.summaries_df is None:
            print("❌ No summaries available")
            return []
        
        # Find the episode
        matches = self.summaries_df[
            self.summaries_df['episode_title'].str.contains(episode_title, case=False, na=False)
        ]
        
        if len(matches) == 0:
            print(f"❌ Episode not found: {episode_title}")
            return []
        
        # Get its embedding
        episode_idx = matches.index[0]
        episode_embedding = self.summary_embeddings[episode_idx:episode_idx+1]
        
        # Find similar
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(episode_embedding, self.summary_embeddings)[0]
        
        # Exclude the episode itself
        similarities[episode_idx] = -1
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                row = self.summaries_df.iloc[idx]
                results.append({
                    'episode_title': row['episode_title'],
                    'summary': row['brief_summary'],
                    'similarity': float(similarities[idx]),
                    'rank': len(results) + 1
                })
        
        return results
    
    def get_episode_details(self, episode_title: str) -> Dict:
        """
        Get full details about an episode including summary and stats
        
        Args:
            episode_title: Episode title (partial match OK)
            
        Returns:
            Dictionary with episode details
        """
        result = {}
        
        # Get summary
        if self.summaries_df is not None:
            matches = self.summaries_df[
                self.summaries_df['episode_title'].str.contains(episode_title, case=False, na=False)
            ]
            if len(matches) > 0:
                row = matches.iloc[0]
                result['summary'] = {
                    'title': row['episode_title'],
                    'summary': row['brief_summary'],
                    'word_count': row.get('word_count', 'N/A'),
                    'date': row.get('date', 'N/A')
                }
        
        # Get transcript stats
        if self.transcript_search is not None:
            stats = self.transcript_search.get_episode_summary_stats(episode_title)
            if stats:
                result['transcript_stats'] = stats
        
        return result


def display_episode_results(results: List[Dict]):
    """Display episode search results"""
    print("\n" + "="*70)
    print("EPISODE RESULTS")
    print("="*70)
    
    for result in results:
        print(f"\n[{result['rank']}] {result['episode_title']}")
        print(f"    Similarity: {result['similarity']:.3f}")
        if result.get('date') and result['date'] != 'N/A':
            print(f"    Date: {result['date']}")
        if result.get('word_count') and result['word_count'] != 'N/A':
            print(f"    Length: {int(result['word_count']):,} words")
        print(f"\n    Summary: {result['summary']}")
        print("-" * 70)


def display_segment_results(results: List[Dict]):
    """Display transcript segment results"""
    print("\n" + "="*70)
    print("TRANSCRIPT SEGMENTS")
    print("="*70)
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Similarity: {result['similarity']:.3f}")
        if 'episode_title' in result:
            print(f"    Episode: {result['episode_title']}")
        if 'speaker' in result:
            print(f"    Speaker: {result['speaker']}")
        if 'timestamp' in result:
            print(f"    Time: {result['timestamp']}")
        print(f"\n    {result['text']}")
        print("-" * 70)


def interactive_mode(searcher: IntegratedSearch):
    """Interactive search interface"""
    print("\n" + "="*70)
    print("INTEGRATED SEARCH SYSTEM")
    print("="*70)
    print("\nCommands:")
    print("  search episodes <query>   - Find episodes by topic")
    print("  search segments <query>   - Find specific moments")
    print("  search all <query>        - Search both")
    print("  similar <episode>         - Find similar episodes")
    print("  details <episode>         - Get episode details")
    print("  quit                      - Exit")
    print("\n" + "="*70)
    
    while True:
        try:
            cmd = input("\n> ").strip()
            
            if not cmd:
                continue
            
            cmd_lower = cmd.lower()
            
            if cmd_lower in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif cmd_lower.startswith('search episodes '):
                query = cmd[16:].strip()
                results = searcher.search_episodes(query, top_k=5)
                display_episode_results(results)
            
            elif cmd_lower.startswith('search segments '):
                query = cmd[16:].strip()
                results = searcher.search_segments(query, top_k=10)
                display_segment_results(results)
            
            elif cmd_lower.startswith('search all '):
                query = cmd[11:].strip()
                results = searcher.search_combined(query, episodes=3, segments=5)
                display_episode_results(results['episodes'])
                display_segment_results(results['segments'])
            
            elif cmd_lower.startswith('similar '):
                episode = cmd[8:].strip()
                results = searcher.find_similar_episodes(episode, top_k=5)
                if results:
                    print(f"\nEpisodes similar to '{episode}':")
                    display_episode_results(results)
            
            elif cmd_lower.startswith('details '):
                episode = cmd[8:].strip()
                details = searcher.get_episode_details(episode)
                
                if 'summary' in details:
                    s = details['summary']
                    print(f"\n{'='*70}")
                    print(f"{s['title']}")
                    print(f"{'='*70}")
                    if s.get('date') and s['date'] != 'N/A':
                        print(f"Date: {s['date']}")
                    if s.get('word_count') and s['word_count'] != 'N/A':
                        print(f"Length: {int(s['word_count']):,} words")
                    print(f"\nSummary:\n{s['summary']}")
                
                if 'transcript_stats' in details:
                    print(f"\nTranscript Statistics:")
                    for key, value in details['transcript_stats'].items():
                        print(f"  {key}: {value}")
                
                if not details:
                    print(f"❌ Episode not found: {episode}")
            
            elif cmd_lower == 'help':
                print("\nCommands:")
                print("  search episodes <query>   - Find episodes by topic")
                print("  search segments <query>   - Find specific moments")
                print("  search all <query>        - Search both")
                print("  similar <episode>         - Find similar episodes")
                print("  details <episode>         - Get episode details")
                print("  quit                      - Exit")
            
            else:
                print(f"Unknown command. Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Integrated search for episodes and transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python integrated_search.py                                    # Interactive mode
  python integrated_search.py --episodes "immigration stories"   # Find episodes
  python integrated_search.py --segments "childhood memory"      # Find moments
  python integrated_search.py --all "family dynamics"            # Search both
  python integrated_search.py --similar "Episode Title"          # Find similar
        """
    )
    
    parser.add_argument('--episodes', '-e', metavar='QUERY',
                        help='Search for episodes by topic')
    parser.add_argument('--segments', '-s', metavar='QUERY',
                        help='Search transcript segments')
    parser.add_argument('--all', '-a', metavar='QUERY',
                        help='Search both episodes and segments')
    parser.add_argument('--similar', metavar='EPISODE',
                        help='Find episodes similar to given episode')
    parser.add_argument('--details', '-d', metavar='EPISODE',
                        help='Get details about an episode')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                        help='Number of results to return')
    
    args = parser.parse_args()
    
    # Initialize search system
    try:
        searcher = IntegratedSearch()
    except Exception as e:
        print(f"❌ Error initializing search: {e}")
        return 1
    
    # Execute command
    if args.episodes:
        results = searcher.search_episodes(args.episodes, top_k=args.top_k)
        display_episode_results(results)
    
    elif args.segments:
        results = searcher.search_segments(args.segments, top_k=args.top_k)
        display_segment_results(results)
    
    elif args.all:
        results = searcher.search_combined(args.all, episodes=args.top_k, segments=args.top_k)
        display_episode_results(results['episodes'])
        display_segment_results(results['segments'])
    
    elif args.similar:
        results = searcher.find_similar_episodes(args.similar, top_k=args.top_k)
        display_episode_results(results)
    
    elif args.details:
        details = searcher.get_episode_details(args.details)
        
        if 'summary' in details:
            s = details['summary']
            print(f"\n{'='*70}")
            print(f"{s['title']}")
            print(f"{'='*70}")
            if s.get('date') and s['date'] != 'N/A':
                print(f"Date: {s['date']}")
            if s.get('word_count') and s['word_count'] != 'N/A':
                print(f"Length: {int(s['word_count']):,} words")
            print(f"\nSummary:\n{s['summary']}")
        
        if 'transcript_stats' in details:
            print(f"\nTranscript Statistics:")
            for key, value in details['transcript_stats'].items():
                print(f"  {key}: {value}")
    
    else:
        # No arguments - enter interactive mode
        interactive_mode(searcher)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())