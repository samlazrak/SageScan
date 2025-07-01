import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import json
import os
import tempfile
from typing import Dict, List, Optional
import time
from datetime import datetime

# Import our Deep Researcher Agent
try:
    from src.deep_researcher_agent import DeepResearcherAgent, ResearchResult, ScrapedContent, ExtractedLink
except ImportError:
    st.error("Please install the required dependencies by running: pip install -r requirements.txt")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="🧠 Deep Researcher Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-text {
        color: #28a745;
        font-weight: bold;
    }
    .error-text {
        color: #dc3545;
        font-weight: bold;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'research_result' not in st.session_state:
        st.session_state.research_result = None
    if 'research_completed' not in st.session_state:
        st.session_state.research_completed = False
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0

def create_network_visualization(result: ResearchResult) -> go.Figure:
    """Create an interactive network visualization of the content relationships"""
    G = result.content_network
    
    if G.number_of_nodes() == 0:
        return go.Figure().add_annotation(text="No network data available", 
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
    
    # Create layout
    try:
        pos = nx.spring_layout(G, k=1, iterations=50)
    except:
        pos = nx.random_layout(G)
    
    # Extract node and edge data
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = [G.nodes[node].get('title', node)[:50] + '...' if len(G.nodes[node].get('title', node)) > 50 
                 else G.nodes[node].get('title', node) for node in G.nodes()]
    node_success = [G.nodes[node].get('success', False) for node in G.nodes()]
    node_depth = [G.nodes[node].get('depth', 0) for node in G.nodes()]
    
    # Create edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=1, color='#888'),
                            hoverinfo='none',
                            mode='lines',
                            name='Links'))
    
    # Add nodes
    colors = ['green' if success else 'red' for success in node_success]
    fig.add_trace(go.Scatter(x=node_x, y=node_y,
                            mode='markers+text',
                            hoverinfo='text',
                            text=node_text,
                            textposition="middle center",
                            hovertext=[f"Title: {title}<br>Depth: {depth}<br>Success: {success}" 
                                     for title, depth, success in zip(node_text, node_depth, node_success)],
                            marker=dict(size=20,
                                      color=colors,
                                      line=dict(width=2, color='white')),
                            name='Pages'))
    
    fig.update_layout(title="Content Network Graph",
                     showlegend=True,
                     hovermode='closest',
                     margin=dict(b=20,l=5,r=5,t=40),
                     annotations=[ dict(text="Green = Successfully scraped, Red = Failed",
                                       showarrow=False,
                                       xref="paper", yref="paper",
                                       x=0.005, y=-0.002 ) ],
                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    
    return fig

def create_domain_analysis_chart(domain_analysis: Dict[str, int]) -> go.Figure:
    """Create a bar chart of domain distribution"""
    if not domain_analysis:
        return go.Figure().add_annotation(text="No domain data available", 
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
    
    # Take top 15 domains
    top_domains = dict(list(domain_analysis.items())[:15])
    
    fig = px.bar(x=list(top_domains.values()), 
                 y=list(top_domains.keys()),
                 orientation='h',
                 labels={'x': 'Number of Pages', 'y': 'Domain'},
                 title="Domain Distribution (Top 15)")
    
    fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
    return fig

def create_scraping_success_chart(scraped_content: List[ScrapedContent]) -> go.Figure:
    """Create a pie chart showing scraping success rates"""
    successful = sum(1 for content in scraped_content if content.success)
    failed = len(scraped_content) - successful
    
    fig = px.pie(values=[successful, failed], 
                 names=['Successful', 'Failed'],
                 title="Scraping Success Rate",
                 color_discrete_map={'Successful': 'green', 'Failed': 'red'})
    return fig

def create_content_depth_chart(scraped_content: List[ScrapedContent]) -> go.Figure:
    """Create a bar chart showing content distribution by depth"""
    depth_counts = {}
    for content in scraped_content:
        depth = content.depth
        depth_counts[depth] = depth_counts.get(depth, 0) + 1
    
    if not depth_counts:
        return go.Figure().add_annotation(text="No depth data available", 
                                        xref="paper", yref="paper",
                                        x=0.5, y=0.5, showarrow=False)
    
    fig = px.bar(x=list(depth_counts.keys()), 
                 y=list(depth_counts.values()),
                 labels={'x': 'Depth Level', 'y': 'Number of Pages'},
                 title="Content Distribution by Scraping Depth")
    return fig

def display_scraped_content_table(scraped_content: List[ScrapedContent]):
    """Display detailed table of scraped content"""
    if not scraped_content:
        st.write("No content to display.")
        return
    
    # Prepare data for DataFrame
    data = []
    for content in scraped_content:
        data.append({
            'URL': content.url,
            'Title': content.title[:100] + '...' if len(content.title) > 100 else content.title,
            'Success': '✅' if content.success else '❌',
            'Depth': content.depth,
            'Content Length': len(content.content),
            'Images': len(content.images),
            'Error': content.error if content.error else 'None'
        })
    
    df = pd.DataFrame(data)
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    with col1:
        success_filter = st.selectbox("Filter by Success", ['All', 'Successful', 'Failed'])
    with col2:
        depth_filter = st.selectbox("Filter by Depth", ['All'] + sorted(list(set(df['Depth']))))
    with col3:
        min_content_length = st.number_input("Minimum Content Length", min_value=0, value=0)
    
    # Apply filters
    filtered_df = df.copy()
    if success_filter == 'Successful':
        filtered_df = filtered_df[filtered_df['Success'] == '✅']
    elif success_filter == 'Failed':
        filtered_df = filtered_df[filtered_df['Success'] == '❌']
    
    if depth_filter != 'All':
        filtered_df = filtered_df[filtered_df['Depth'] == int(depth_filter)]
    
    filtered_df = filtered_df[filtered_df['Content Length'] >= min_content_length]
    
    st.dataframe(filtered_df, use_container_width=True)

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Deep Researcher Agent</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <strong>Welcome to the Deep Researcher Agent!</strong><br>
    This advanced tool extracts links from PDF documents and performs deep, multi-level web content analysis. 
    Upload a PDF to discover and analyze all referenced sources with network visualization and comprehensive insights.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF document to extract and analyze links from"
    )
    
    # Basic configuration
    st.sidebar.subheader("Basic Settings")
    max_links = st.sidebar.slider("Maximum Links to Process", 5, 100, 20, 
                                 help="Maximum number of initial links to extract and process")
    
    use_multi_level = st.sidebar.checkbox("Enable Multi-Level Scraping", value=True,
                                        help="Recursively follow links found in scraped content")
    
    include_images = st.sidebar.checkbox("Include Images", value=True,
                                       help="Extract and analyze images from scraped content")
    
    use_selenium = st.sidebar.checkbox("Use Selenium (for JS-heavy sites)", value=False,
                                     help="Enable Selenium for JavaScript-heavy websites (slower but more comprehensive)")
    
    # Advanced configuration
    with st.sidebar.expander("🔬 Advanced Settings"):
        max_depth = st.slider("Maximum Scraping Depth", 1, 5, 2,
                             help="How many levels deep to follow links")
        
        max_links_per_level = st.slider("Max Links Per Level", 5, 50, 10,
                                       help="Maximum links to follow at each depth level")
        
        delay_between_requests = st.slider("Delay Between Requests (seconds)", 0.5, 5.0, 1.0,
                                         help="Delay to avoid overwhelming target servers")
        
        max_workers = st.slider("Concurrent Workers", 1, 10, 5,
                               help="Number of concurrent scraping workers")
    
    # Domain filtering
    with st.sidebar.expander("🌐 Domain Filtering"):
        allowed_domains_text = st.text_area(
            "Allowed Domains (one per line)",
            placeholder="arxiv.org\nnature.com\nscholar.google.com",
            help="Only scrape from these domains (leave empty to allow all)"
        )
        
        blocked_domains_text = st.text_area(
            "Blocked Domains (one per line)", 
            placeholder="facebook.com\ntwitter.com\nsocial-media.com",
            help="Never scrape from these domains"
        )
        
        allowed_domains = [d.strip() for d in allowed_domains_text.split('\n') if d.strip()] if allowed_domains_text else None
        blocked_domains = [d.strip() for d in blocked_domains_text.split('\n') if d.strip()] if blocked_domains_text else None
    
    # Main content area
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Display file info
        st.success(f"📄 PDF uploaded: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        
        # Research button
        if st.button("🚀 Start Deep Research", type="primary", use_container_width=True):
            try:
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize agent
                status_text.text("🔧 Initializing Deep Researcher Agent...")
                progress_bar.progress(10)
                
                agent = DeepResearcherAgent(
                    use_selenium=use_selenium,
                    delay_between_requests=delay_between_requests,
                    max_workers=max_workers
                )
                
                # Start research
                status_text.text("🔍 Starting deep research analysis...")
                progress_bar.progress(20)
                
                start_time = time.time()
                
                result = agent.deep_research(
                    pdf_path=tmp_file_path,
                    max_links=max_links,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                    include_images=include_images,
                    max_depth=max_depth,
                    max_links_per_level=max_links_per_level,
                    use_multi_level=use_multi_level
                )
                
                end_time = time.time()
                duration = end_time - start_time
                
                progress_bar.progress(100)
                status_text.text(f"✅ Research completed in {duration:.2f} seconds!")
                
                # Store result in session state
                st.session_state.research_result = result
                st.session_state.research_completed = True
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                st.success("🎉 Deep research analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error during research: {str(e)}")
                if 'tmp_file_path' in locals():
                    os.unlink(tmp_file_path)
    
    # Display results if available
    if st.session_state.research_completed and st.session_state.research_result:
        result = st.session_state.research_result
        
        st.markdown("---")
        st.header("📊 Research Results")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📎 Total Links Found", result.total_links_found)
        
        with col2:
            st.metric("✅ Successful Scrapes", result.successful_scrapes)
        
        with col3:
            st.metric("❌ Failed Scrapes", result.failed_scrapes)
        
        with col4:
            success_rate = (result.successful_scrapes / max(result.total_links_found, 1)) * 100
            st.metric("📈 Success Rate", f"{success_rate:.1f}%")
        
        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("🔍 Max Depth Reached", result.max_depth_reached)
        
        with col6:
            st.metric("🌐 Unique Domains", len(result.domain_analysis))
        
        with col7:
            network_nodes = result.content_network.number_of_nodes()
            st.metric("🕸️ Network Nodes", network_nodes)
        
        with col8:
            network_edges = result.content_network.number_of_edges()
            st.metric("🔗 Network Edges", network_edges)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Summary", "🕸️ Network Visualization", "📊 Analytics", "📑 Content Details", "💾 Export Data"
        ])
        
        with tab1:
            st.subheader("📝 Research Summary")
            st.markdown(result.summary)
            
            if result.key_insights:
                st.subheader("💡 Key Insights")
                for insight in result.key_insights:
                    st.write(f"• {insight}")
        
        with tab2:
            st.subheader("🕸️ Content Network Visualization")
            if result.content_network.number_of_nodes() > 0:
                network_fig = create_network_visualization(result)
                st.plotly_chart(network_fig, use_container_width=True)
            else:
                st.info("No network connections found to visualize.")
        
        with tab3:
            st.subheader("📊 Analytics Dashboard")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Domain analysis
                domain_fig = create_domain_analysis_chart(result.domain_analysis)
                st.plotly_chart(domain_fig, use_container_width=True)
                
                # Content depth distribution
                depth_fig = create_content_depth_chart(result.scraped_content)
                st.plotly_chart(depth_fig, use_container_width=True)
            
            with col2:
                # Success rate pie chart
                success_fig = create_scraping_success_chart(result.scraped_content)
                st.plotly_chart(success_fig, use_container_width=True)
                
                # Content length distribution
                successful_content = [c for c in result.scraped_content if c.success]
                if successful_content:
                    content_lengths = [len(c.content) for c in successful_content]
                    length_fig = px.histogram(x=content_lengths, 
                                            labels={'x': 'Content Length (characters)', 'y': 'Count'},
                                            title="Content Length Distribution")
                    st.plotly_chart(length_fig, use_container_width=True)
        
        with tab4:
            st.subheader("📑 Detailed Content Analysis")
            display_scraped_content_table(result.scraped_content)
        
        with tab5:
            st.subheader("💾 Export Research Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export summary as markdown
                if st.button("📄 Download Summary (MD)", use_container_width=True):
                    st.download_button(
                        label="📄 Download Summary",
                        data=result.summary,
                        file_name=f"research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
            
            with col2:
                # Export data as JSON
                if st.button("📊 Download Data (JSON)", use_container_width=True):
                    export_data = {
                        'pdf_path': result.pdf_path,
                        'total_links_found': result.total_links_found,
                        'successful_scrapes': result.successful_scrapes,
                        'failed_scrapes': result.failed_scrapes,
                        'max_depth_reached': result.max_depth_reached,
                        'domain_analysis': result.domain_analysis,
                        'key_insights': result.key_insights,
                        'scraped_content': [
                            {
                                'url': content.url,
                                'title': content.title,
                                'success': content.success,
                                'depth': content.depth,
                                'content_length': len(content.content),
                                'num_images': len(content.images),
                                'error': content.error
                            }
                            for content in result.scraped_content
                        ]
                    }
                    
                    st.download_button(
                        label="📊 Download Data",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col3:
                # Export content as CSV
                if st.button("📈 Download Table (CSV)", use_container_width=True):
                    df_data = []
                    for content in result.scraped_content:
                        df_data.append({
                            'URL': content.url,
                            'Title': content.title,
                            'Success': content.success,
                            'Depth': content.depth,
                            'Content_Length': len(content.content),
                            'Images_Count': len(content.images),
                            'Error': content.error or 'None'
                        })
                    
                    df = pd.DataFrame(df_data)
                    csv = df.to_csv(index=False)
                    
                    st.download_button(
                        label="📈 Download CSV",
                        data=csv,
                        file_name=f"research_content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
    🧠 Deep Researcher Agent - Advanced PDF Link Analysis & Web Content Scraping<br>
    Built with Streamlit, NetworkX, and advanced web scraping technologies
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()