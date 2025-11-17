import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from data_processor import DataProcessor
from embedding_generator import EmbeddingGenerator
from semantic_search import SemanticSearch
from summarizer import EpisodeSummarizer
from topic_extractor import TopicExtractor
import json

# Page configuration
st.set_page_config(
    page_title="Podcast Insight Engine",
    page_icon="🎙️",
    layout="wide"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'embeddings_generated' not in st.session_state:
    st.session_state.embeddings_generated = False

def main():
    st.title("🎙️ Podcast Insight Engine")
    st.markdown("Search, explore, and discover podcast content using AI")
    
    # Sidebar for data loading and configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Data upload
        uploaded_file = st.file_uploader(
            "Upload Podcast Transcript Dataset (CSV)",
            type=['csv']
        )
        
        if uploaded_file is not None:
            try:
                # Load and process data
                with st.spinner("Loading and processing data..."):
                    df = pd.read_csv(uploaded_file)
                    st.session_state.raw_data = df
                    
                    # Initialize processor
                    processor = DataProcessor()
                    st.session_state.processed_data = processor.process_dataframe(df)
                    st.session_state.data_loaded = True
                    
                st.success(f"✅ Loaded {len(df)} transcript entries")
                
                # Show data info
                st.subheader("Dataset Info")
                st.write(f"Columns: {', '.join(df.columns.tolist())}")
                st.write(f"Shape: {df.shape}")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return
        
        # Generate embeddings button
        if st.session_state.data_loaded and not st.session_state.embeddings_generated:
            if st.button("Generate Embeddings"):
                with st.spinner("Generating embeddings... This may take a few minutes."):
                    try:
                        embed_gen = EmbeddingGenerator()
                        embeddings, texts = embed_gen.generate_embeddings(
                            st.session_state.processed_data
                        )
                        st.session_state.embeddings = embeddings
                        st.session_state.embedding_texts = texts
                        st.session_state.embeddings_generated = True
                        st.success("✅ Embeddings generated!")
                    except Exception as e:
                        st.error(f"Error generating embeddings: {str(e)}")
    
    # Main content area
    if not st.session_state.data_loaded:
        st.info("👈 Please upload a podcast transcript dataset to begin")
        st.markdown("""
        ### Expected Dataset Format
        Your CSV should contain columns like:
        - `text` or `transcript`: The transcript content
        - `episode_title` or `title`: Episode name
        - `speaker` (optional): Speaker identification
        - `start_time`, `end_time` (optional): Timestamps
        - `episode_id` (optional): Unique episode identifier
        """)
        return
    
    # Create tabs for different features
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search", "📊 Topics", "📝 Summaries", "📈 Analytics"])
    
    with tab1:
        st.header("Semantic Search")
        
        if not st.session_state.embeddings_generated:
            st.warning("⚠️ Please generate embeddings first (see sidebar)")
        else:
            search_query = st.text_input(
                "Search for topics, concepts, or themes:",
                placeholder="e.g., stories about forgiveness, episodes featuring immigrants"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                num_results = st.slider("Number of results:", 1, 20, 5)
            with col2:
                search_button = st.button("Search", type="primary")
            
            if search_button and search_query:
                with st.spinner("Searching..."):
                    try:
                        searcher = SemanticSearch(
                            st.session_state.embeddings,
                            st.session_state.embedding_texts,
                            st.session_state.processed_data
                        )
                        results = searcher.search(search_query, top_k=num_results)
                        
                        st.subheader(f"Found {len(results)} relevant results")
                        
                        for idx, result in enumerate(results, 1):
                            with st.expander(f"{idx}. {result['episode_title']} (Similarity: {result['similarity']:.3f})"):
                                st.markdown(f"**Text:** {result['text'][:500]}...")
                                if 'speaker' in result and result['speaker']:
                                    st.markdown(f"**Speaker:** {result['speaker']}")
                                if 'timestamp' in result and result['timestamp']:
                                    st.markdown(f"**Timestamp:** {result['timestamp']}")
                                
                                # Feedback mechanism
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.button("👍 Relevant", key=f"relevant_{idx}")
                                with col2:
                                    st.button("👎 Not Relevant", key=f"not_relevant_{idx}")
                    
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
    
    with tab2:
        st.header("Topic Extraction & Entity Recognition")
        
        if st.button("Extract Topics and Entities"):
            with st.spinner("Analyzing topics and entities..."):
                try:
                    extractor = TopicExtractor()
                    topics, entities = extractor.extract_topics_and_entities(
                        st.session_state.processed_data
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🏷️ Main Topics")
                        for topic, score in topics[:10]:
                            st.markdown(f"- **{topic}** ({score:.2f})")
                    
                    with col2:
                        st.subheader("👤 Named Entities")
                        if 'PERSON' in entities:
                            st.markdown("**People:**")
                            for person in list(entities['PERSON'])[:10]:
                                st.markdown(f"- {person}")
                        if 'GPE' in entities:
                            st.markdown("**Places:**")
                            for place in list(entities['GPE'])[:10]:
                                st.markdown(f"- {place}")
                
                except Exception as e:
                    st.error(f"Topic extraction error: {str(e)}")
    
    with tab3:
        st.header("Episode Summaries")
        
        # Get unique episodes
        if 'episode_title' in st.session_state.processed_data.columns:
            episodes = st.session_state.processed_data['episode_title'].unique()
            selected_episode = st.selectbox("Select an episode:", episodes)
            
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    try:
                        summarizer = EpisodeSummarizer()
                        
                        # Get episode text
                        episode_data = st.session_state.processed_data[
                            st.session_state.processed_data['episode_title'] == selected_episode
                        ]
                        episode_text = " ".join(episode_data['text'].tolist())
                        
                        summary = summarizer.summarize_episode(episode_text, selected_episode)
                        
                        st.subheader("📝 Summary")
                        st.markdown(summary)
                        
                        # Feedback
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.button("👍 Accurate")
                        with col2:
                            st.button("👎 Inaccurate")
                        with col3:
                            st.button("⚠️ Report Issue")
                    
                    except Exception as e:
                        st.error(f"Summarization error: {str(e)}")
        else:
            st.warning("Episode title column not found in dataset")
    
    with tab4:
        st.header("Analytics Dashboard")
        
        if st.session_state.data_loaded:
            df = st.session_state.processed_data
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Episodes", df['episode_title'].nunique() if 'episode_title' in df.columns else "N/A")
            with col2:
                st.metric("Total Transcript Lines", len(df))
            with col3:
                avg_length = df['text'].str.len().mean() if 'text' in df.columns else 0
                st.metric("Avg Text Length", f"{avg_length:.0f} chars")
            
            # Word count distribution
            if 'text' in df.columns:
                st.subheader("Text Length Distribution")
                df['word_count'] = df['text'].str.split().str.len()
                st.bar_chart(df['word_count'].value_counts().head(20))

if __name__ == "__main__":
    main()