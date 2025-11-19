#!/usr/bin/env python3
"""
Summary Viewer
Browse, search, and export episode summaries
"""

import pandas as pd
import os
from pathlib import Path
import argparse
import json


def load_summaries(filepath="output/episode_summaries.csv"):
    """Load the summaries CSV file"""
    if not os.path.exists(filepath):
        print(f"❌ Summaries file not found at: {filepath}")
        print("\nTo generate summaries, run:")
        print("  python summarizer.py")
        return None
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} episode summaries")
    return df


def display_summary(row, index=None, detailed=False):
    """Display a single summary in a nice format"""
    print("\n" + "="*70)
    if index is not None:
        print(f"[{index + 1}] {row['episode_title']}")
    else:
        print(f"{row['episode_title']}")
    print("="*70)
    
    if 'episode_id' in row and pd.notna(row['episode_id']):
        print(f"Episode ID: {row['episode_id']}")
    
    if 'date' in row and pd.notna(row['date']):
        print(f"Date: {row['date']}")
    
    if 'word_count' in row and pd.notna(row['word_count']):
        print(f"Word Count: {int(row['word_count']):,} words")
    
    print("\nSummary:")
    print("-" * 70)
    print(row['brief_summary'])
    print("-" * 70)


def list_all_summaries(df, show_full=False):
    """List all summaries"""
    print("\n" + "="*70)
    print(f"ALL EPISODE SUMMARIES ({len(df)} episodes)")
    print("="*70)
    
    if show_full:
        for idx, row in df.iterrows():
            display_summary(row, idx)
    else:
        print("\n{:<5} {:<50} {:<10}".format("No.", "Episode Title", "Words"))
        print("-" * 70)
        for idx, row in df.iterrows():
            title = row['episode_title'][:47] + "..." if len(str(row['episode_title'])) > 50 else row['episode_title']
            word_count = f"{int(row['word_count']):,}" if 'word_count' in row and pd.notna(row['word_count']) else "N/A"
            print(f"{idx+1:<5} {title:<50} {word_count:<10}")
        
        print("\nTo see full summaries, use: --full")


def search_summaries(df, query):
    """Search summaries by keyword"""
    query_lower = query.lower()
    
    # Search in title and summary
    matches = df[
        df['episode_title'].str.lower().str.contains(query_lower, na=False) |
        df['brief_summary'].str.lower().str.contains(query_lower, na=False)
    ]
    
    print("\n" + "="*70)
    print(f"SEARCH RESULTS for '{query}' ({len(matches)} matches)")
    print("="*70)
    
    if len(matches) == 0:
        print("No matches found.")
        return
    
    for idx, row in matches.iterrows():
        display_summary(row, idx)


def show_statistics(df):
    """Show statistics about the summaries"""
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    stats = {
        "Total Episodes": len(df),
        "Average Word Count": f"{df['word_count'].mean():,.0f}" if 'word_count' in df else "N/A",
        "Total Words": f"{df['word_count'].sum():,.0f}" if 'word_count' in df else "N/A",
        "Shortest Episode": f"{df['word_count'].min():,.0f} words" if 'word_count' in df else "N/A",
        "Longest Episode": f"{df['word_count'].max():,.0f} words" if 'word_count' in df else "N/A",
        "Average Summary Length": f"{df['brief_summary'].str.split().str.len().mean():.0f} words",
    }
    
    for key, value in stats.items():
        print(f"  {key:<25}: {value}")
    
    # Show date range if available
    if 'date' in df and df['date'].notna().any():
        dates = pd.to_datetime(df['date'], errors='coerce')
        if dates.notna().any():
            print(f"\n  Date Range:")
            print(f"    First Episode: {dates.min()}")
            print(f"    Last Episode: {dates.max()}")


def export_summaries(df, format='txt', output_file=None):
    """Export summaries to different formats"""
    if output_file is None:
        output_file = f"output/summaries_export.{format}"
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'txt':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("THIS AMERICAN LIFE - EPISODE SUMMARIES\n")
            f.write("=" * 70 + "\n\n")
            
            for idx, row in df.iterrows():
                f.write(f"[{idx + 1}] {row['episode_title']}\n")
                f.write("-" * 70 + "\n")
                if 'date' in row and pd.notna(row['date']):
                    f.write(f"Date: {row['date']}\n")
                if 'word_count' in row and pd.notna(row['word_count']):
                    f.write(f"Word Count: {int(row['word_count']):,}\n")
                f.write(f"\nSummary:\n{row['brief_summary']}\n")
                f.write("\n" + "=" * 70 + "\n\n")
    
    elif format == 'json':
        df.to_json(output_path, orient='records', indent=2)
    
    elif format == 'md':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# This American Life - Episode Summaries\n\n")
            
            for idx, row in df.iterrows():
                f.write(f"## {idx + 1}. {row['episode_title']}\n\n")
                if 'date' in row and pd.notna(row['date']):
                    f.write(f"**Date:** {row['date']}  \n")
                if 'word_count' in row and pd.notna(row['word_count']):
                    f.write(f"**Word Count:** {int(row['word_count']):,}  \n")
                f.write(f"\n{row['brief_summary']}\n\n")
                f.write("---\n\n")
    
    elif format == 'html':
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>This American Life - Episode Summaries</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        .episode { margin: 30px 0; padding: 20px; background: white; border-left: 4px solid #3498db; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .episode-title { color: #2c3e50; margin-top: 0; }
        .meta { color: #7f8c8d; font-size: 0.9em; margin: 10px 0; }
        .summary { line-height: 1.6; color: #34495e; }
        .search-box { margin: 20px 0; }
        .search-box input { width: 100%; padding: 10px; font-size: 16px; border: 2px solid #ddd; border-radius: 4px; }
    </style>
    <script>
        function searchSummaries() {
            const query = document.getElementById('search').value.toLowerCase();
            const episodes = document.getElementsByClassName('episode');
            
            for (let ep of episodes) {
                const text = ep.textContent.toLowerCase();
                ep.style.display = text.includes(query) ? 'block' : 'none';
            }
        }
    </script>
</head>
<body>
    <h1>This American Life - Episode Summaries</h1>
    <div class="search-box">
        <input type="text" id="search" placeholder="Search summaries..." onkeyup="searchSummaries()">
    </div>
""")
            
            for idx, row in df.iterrows():
                f.write(f'    <div class="episode">\n')
                f.write(f'        <h2 class="episode-title">{idx + 1}. {row["episode_title"]}</h2>\n')
                f.write(f'        <div class="meta">\n')
                if 'date' in row and pd.notna(row['date']):
                    f.write(f'            <strong>Date:</strong> {row["date"]} | ')
                if 'word_count' in row and pd.notna(row['word_count']):
                    f.write(f'<strong>Word Count:</strong> {int(row["word_count"]):,}')
                f.write(f'        </div>\n')
                f.write(f'        <div class="summary">{row["brief_summary"]}</div>\n')
                f.write(f'    </div>\n')
            
            f.write("</body>\n</html>")
    
    print(f"\n✓ Summaries exported to: {output_path}")


def interactive_mode(df):
    """Interactive command-line browser"""
    print("\n" + "="*70)
    print("INTERACTIVE SUMMARY BROWSER")
    print("="*70)
    print("\nCommands:")
    print("  list              - List all episodes")
    print("  show <number>     - Show specific episode (e.g., 'show 5')")
    print("  search <keyword>  - Search summaries (e.g., 'search family')")
    print("  stats             - Show statistics")
    print("  export <format>   - Export (txt, json, md, html)")
    print("  quit              - Exit")
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
            
            elif cmd_lower == 'list':
                list_all_summaries(df, show_full=False)
            
            elif cmd_lower.startswith('show '):
                try:
                    num = int(cmd.split()[1]) - 1
                    if 0 <= num < len(df):
                        display_summary(df.iloc[num], num)
                    else:
                        print(f"Invalid episode number. Choose 1-{len(df)}")
                except (ValueError, IndexError):
                    print("Usage: show <number>")
            
            elif cmd_lower.startswith('search '):
                query = ' '.join(cmd.split()[1:])
                if query:
                    search_summaries(df, query)
                else:
                    print("Usage: search <keyword>")
            
            elif cmd_lower == 'stats':
                show_statistics(df)
            
            elif cmd_lower.startswith('export '):
                try:
                    format = cmd.split()[1].lower()
                    if format in ['txt', 'json', 'md', 'html']:
                        export_summaries(df, format)
                    else:
                        print("Supported formats: txt, json, md, html")
                except IndexError:
                    print("Usage: export <format>")
            
            elif cmd_lower == 'help':
                print("\nCommands:")
                print("  list              - List all episodes")
                print("  show <number>     - Show specific episode")
                print("  search <keyword>  - Search summaries")
                print("  stats             - Show statistics")
                print("  export <format>   - Export summaries")
                print("  quit              - Exit")
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="View and search episode summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_summaries.py                    # Interactive mode
  python view_summaries.py --list             # List all summaries
  python view_summaries.py --show 5           # Show episode #5
  python view_summaries.py --search family    # Search for keyword
  python view_summaries.py --stats            # Show statistics
  python view_summaries.py --export html      # Export to HTML
  python view_summaries.py --full             # Show all summaries
        """
    )
    
    parser.add_argument('--file', '-f', default='output/episode_summaries.csv',
                        help='Path to summaries CSV file')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List all episode titles')
    parser.add_argument('--full', action='store_true',
                        help='Show full summaries (with --list)')
    parser.add_argument('--show', '-s', type=int, metavar='N',
                        help='Show specific episode number')
    parser.add_argument('--search', metavar='QUERY',
                        help='Search summaries for keyword')
    parser.add_argument('--stats', action='store_true',
                        help='Show summary statistics')
    parser.add_argument('--export', '-e', choices=['txt', 'json', 'md', 'html'],
                        help='Export summaries to format')
    parser.add_argument('--output', '-o',
                        help='Output file for export')
    
    args = parser.parse_args()
    
    # Load summaries
    df = load_summaries(args.file)
    if df is None:
        return 1
    
    # Execute command
    if args.list:
        list_all_summaries(df, show_full=args.full)
    
    elif args.show:
        if 1 <= args.show <= len(df):
            display_summary(df.iloc[args.show - 1], args.show - 1)
        else:
            print(f"Invalid episode number. Choose 1-{len(df)}")
    
    elif args.search:
        search_summaries(df, args.search)
    
    elif args.stats:
        show_statistics(df)
    
    elif args.export:
        export_summaries(df, args.export, args.output)
    
    else:
        # No arguments - enter interactive mode
        interactive_mode(df)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())