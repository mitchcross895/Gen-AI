#!/usr/bin/env python3
"""
Simple command-line runner for the Podcast Insight Engine.
This script allows you to run the system without the Streamlit UI.
"""

import argparse
import pandas as pd
from data_processor import DataProcessor
from embedding_generator import EmbeddingGenerator
from semantic_search import SemanticSearch
from summarizer import EpisodeSummarizer
from topic_extractor import TopicExtractor
import json

def main():
    parser = argparse.ArgumentParser(description='Podcast Insight Engine CLI')
    parser.add_argument('--data', required=True, help='Path to CSV dataset')
    parser.add_argument('--mode', choices=['process', 'search', 'summarize', 'topics'], 
                       required=True, help='Operation mode')
    parser.add_argument('--query', help='Search query (for search mode)')
    parser.add_argument('--episode', help='Episode title (for summarize mode)')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    # Process data
    print("Processing data...")
    processor = DataProcessor()
    processed_df = processor.process_dataframe(df)
    
    if args.mode == 'process':
        print("\n=== Data Processing Complete ===")
        print(f"Processed {len(processed_df)} transcript segments")
        print(f"Columns: {', '.join(processed_df.columns)}")
        
        if args.output:
            processed_df.to_csv(args.output, index=False)
            print(f"Saved to {args.output}")
    
    elif args.mode == 'search':
        if not args.query:
            print("Error: --query required for search mode")
            return
        
        print("Generating embeddings...")
        embed_gen = EmbeddingGenerator()
        embeddings, texts = embed_gen.generate_embeddings(processed_df)
        
        print(f"\nSearching for: {args.query}")
        searcher = SemanticSearch(embeddings, texts, processed_df)
        results = searcher.search(args.query, top_k=5)
        
        print("\n=== Search Results ===")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Similarity: {result['similarity']:.3f}")
            if 'episode_title' in result:
                print(f"   Episode: {result['episode_title']}")
            print(f"   Text: {result['text'][:200]}...")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    elif args.mode == 'summarize':
        if not args.episode:
            print("Error: --episode required for summarize mode")
            return
        
        print(f"Summarizing episode: {args.episode}")
        
        # Get episode text
        episode_df = processed_df[processed_df['episode_title'] == args.episode]
        if len(episode_df) == 0:
            print(f"Error: Episode '{args.episode}' not found")
            return
        
        episode_text = ' '.join(episode_df['text'].tolist())
        
        summarizer = EpisodeSummarizer()
        summary = summarizer.summarize_episode(episode_text, args.episode)
        
        print("\n=== Summary ===")
        print(summary)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(summary)
            print(f"\nSummary saved to {args.output}")
    
    elif args.mode == 'topics':
        print("Extracting topics and entities...")
        extractor = TopicExtractor()
        topics, entities = extractor.extract_topics_and_entities(processed_df)
        
        print("\n=== Top Topics ===")
        for topic, score in topics[:10]:
            print(f"- {topic}: {score:.3f}")
        
        print("\n=== Named Entities ===")
        for entity_type, entity_set in entities.items():
            print(f"\n{entity_type}:")
            for entity in list(entity_set)[:10]:
                print(f"  - {entity}")
        
        if args.output:
            output_data = {
                'topics': topics,
                'entities': {k: list(v) for k, v in entities.items()}
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()