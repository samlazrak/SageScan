"""
Visualization utilities for the Deep Researcher Agent
"""

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from typing import Dict, List, Optional, Tuple
import pandas as pd

def create_network_layout(G: nx.DiGraph, layout_type: str = "spring") -> Dict:
    """Create layout positions for network visualization"""
    if G.number_of_nodes() == 0:
        return {}
    
    layout_functions = {
        "spring": lambda: nx.spring_layout(G, k=1, iterations=50),
        "circular": lambda: nx.circular_layout(G),
        "random": lambda: nx.random_layout(G),
        "shell": lambda: nx.shell_layout(G),
        "spectral": lambda: nx.spectral_layout(G) if G.number_of_nodes() > 1 else nx.random_layout(G)
    }
    
    try:
        return layout_functions.get(layout_type, layout_functions["spring"])()
    except Exception:
        return nx.random_layout(G)

def calculate_node_metrics(G: nx.DiGraph) -> Dict[str, Dict]:
    """Calculate various metrics for network nodes"""
    metrics = {}
    
    if G.number_of_nodes() == 0:
        return metrics
    
    # Basic metrics
    degree_centrality = nx.degree_centrality(G)
    in_degree_centrality = nx.in_degree_centrality(G)
    out_degree_centrality = nx.out_degree_centrality(G)
    
    # Advanced metrics (if graph is not empty and connected)
    try:
        if nx.is_weakly_connected(G):
            closeness_centrality = nx.closeness_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
        else:
            closeness_centrality = {node: 0 for node in G.nodes()}
            betweenness_centrality = {node: 0 for node in G.nodes()}
    except Exception:
        closeness_centrality = {node: 0 for node in G.nodes()}
        betweenness_centrality = {node: 0 for node in G.nodes()}
    
    for node in G.nodes():
        metrics[node] = {
            'degree_centrality': degree_centrality.get(node, 0),
            'in_degree_centrality': in_degree_centrality.get(node, 0),
            'out_degree_centrality': out_degree_centrality.get(node, 0),
            'closeness_centrality': closeness_centrality.get(node, 0),
            'betweenness_centrality': betweenness_centrality.get(node, 0),
            'in_degree': G.in_degree(node),
            'out_degree': G.out_degree(node)
        }
    
    return metrics

def create_interactive_network_plot(G: nx.DiGraph, 
                                  node_attributes: Optional[Dict] = None,
                                  layout_type: str = "spring",
                                  node_size_metric: str = "degree_centrality",
                                  color_metric: str = "success") -> go.Figure:
    """Create an interactive network plot with advanced features"""
    
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.add_annotation(text="No network data available", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False,
                          font=dict(size=16))
        return fig
    
    # Calculate layout
    pos = create_network_layout(G, layout_type)
    node_metrics = calculate_node_metrics(G)
    
    # Extract node positions
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    # Extract node information
    node_text = []
    node_hover = []
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        # Get node attributes
        attrs = G.nodes[node]
        metrics = node_metrics.get(node, {})
        
        # Node text (shortened title)
        title = attrs.get('title', node)
        if len(title) > 30:
            display_text = title[:30] + '...'
        else:
            display_text = title
        node_text.append(display_text)
        
        # Hover information
        hover_info = f"<b>{title}</b><br>"
        hover_info += f"URL: {node}<br>"
        hover_info += f"Success: {'✅' if attrs.get('success', False) else '❌'}<br>"
        hover_info += f"Depth: {attrs.get('depth', 0)}<br>"
        hover_info += f"Content Length: {attrs.get('content_length', 0)}<br>"
        hover_info += f"Images: {attrs.get('num_images', 0)}<br>"
        hover_info += f"Degree Centrality: {metrics.get('degree_centrality', 0):.3f}<br>"
        hover_info += f"In-Degree: {metrics.get('in_degree', 0)}<br>"
        hover_info += f"Out-Degree: {metrics.get('out_degree', 0)}"
        node_hover.append(hover_info)
        
        # Node color based on success/depth/etc
        if color_metric == "success":
            node_colors.append('green' if attrs.get('success', False) else 'red')
        elif color_metric == "depth":
            depth = attrs.get('depth', 0)
            node_colors.append(depth)
        elif color_metric in metrics:
            node_colors.append(metrics[color_metric])
        else:
            node_colors.append('blue')
        
        # Node size based on centrality or other metrics
        if node_size_metric in metrics:
            size = max(10, metrics[node_size_metric] * 50)  # Scale and ensure minimum size
        else:
            size = 20
        node_sizes.append(size)
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Edge information
        edge_info.append(f"{edge[0]} → {edge[1]}")
    
    # Create the plot
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                            line=dict(width=1, color='rgba(50,50,50,0.5)'),
                            hoverinfo='none',
                            mode='lines',
                            name='Connections',
                            showlegend=False))
    
    # Add nodes
    fig.add_trace(go.Scatter(x=node_x, y=node_y,
                            mode='markers+text',
                            hoverinfo='text',
                            text=node_text,
                            textposition="middle center",
                            textfont=dict(size=8),
                            hovertext=node_hover,
                            marker=dict(size=node_sizes,
                                      color=node_colors,
                                      colorscale='Viridis' if color_metric not in ['success'] else None,
                                      line=dict(width=2, color='white'),
                                      showscale=color_metric not in ['success'],
                                      colorbar=dict(title=color_metric.replace('_', ' ').title()) if color_metric not in ['success'] else None),
                            name='Web Pages',
                            showlegend=False))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Content Network Analysis",
            x=0.5,
            font=dict(size=20)
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(text="Node size: Centrality | Color: Success/Depth",
                 showarrow=False,
                 xref="paper", yref="paper",
                 x=0.005, y=-0.002,
                 font=dict(size=12))
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )
    
    return fig

def create_content_metrics_dashboard(scraped_content: List) -> Dict[str, go.Figure]:
    """Create a comprehensive dashboard of content metrics"""
    figures = {}
    
    if not scraped_content:
        empty_fig = go.Figure()
        empty_fig.add_annotation(text="No data available", 
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False)
        return {"empty": empty_fig}
    
    # Prepare data
    df_data = []
    for content in scraped_content:
        df_data.append({
            'url': content.url,
            'title': content.title,
            'success': content.success,
            'depth': content.depth,
            'content_length': len(content.content) if hasattr(content, 'content') else 0,
            'num_images': len(content.images) if hasattr(content, 'images') else 0,
            'has_error': bool(content.error) if hasattr(content, 'error') else False
        })
    
    df = pd.DataFrame(df_data)
    
    # 1. Success rate by depth
    success_by_depth = df.groupby('depth').agg({
        'success': ['count', 'sum']
    }).round(2)
    success_by_depth.columns = ['total', 'successful']
    success_by_depth['success_rate'] = (success_by_depth['successful'] / success_by_depth['total'] * 100).round(1)
    
    figures['success_by_depth'] = px.bar(
        x=success_by_depth.index,
        y=success_by_depth['success_rate'],
        labels={'x': 'Depth Level', 'y': 'Success Rate (%)'},
        title="Success Rate by Scraping Depth"
    )
    
    # 2. Content length distribution
    successful_df = df[df['success'] == True]
    if not successful_df.empty:
        figures['content_length_dist'] = px.histogram(
            successful_df,
            x='content_length',
            title="Content Length Distribution",
            labels={'x': 'Content Length (characters)', 'y': 'Count'},
            nbins=20
        )
    
    # 3. Images distribution
    if not successful_df.empty:
        figures['images_dist'] = px.histogram(
            successful_df,
            x='num_images',
            title="Images Per Page Distribution",
            labels={'x': 'Number of Images', 'y': 'Count'},
            nbins=10
        )
    
    # 4. Content quality heatmap (if we have enough data)
    if len(df) > 5:
        # Create quality score based on content length and success
        df['quality_score'] = df.apply(lambda row: 
            (row['content_length'] / 1000) * (2 if row['success'] else 0.5), axis=1)
        
        quality_matrix = df.pivot_table(
            values='quality_score',
            index='depth',
            columns=pd.cut(df['content_length'], bins=5, labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long']),
            aggfunc='mean',
            fill_value=0
        )
        
        figures['quality_heatmap'] = px.imshow(
            quality_matrix.values,
            x=quality_matrix.columns,
            y=quality_matrix.index,
            title="Content Quality Score by Depth and Length",
            labels={'x': 'Content Length Category', 'y': 'Depth Level', 'color': 'Quality Score'},
            aspect='auto'
        )
    
    return figures

def create_domain_treemap(domain_analysis: Dict[str, int]) -> go.Figure:
    """Create a treemap visualization of domain distribution"""
    if not domain_analysis:
        fig = go.Figure()
        fig.add_annotation(text="No domain data available", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Prepare data for treemap
    domains = list(domain_analysis.keys())[:20]  # Top 20 domains
    counts = [domain_analysis[domain] for domain in domains]
    
    # Create treemap
    fig = go.Figure(go.Treemap(
        labels=domains,
        values=counts,
        parents=[""] * len(domains),
        textinfo="label+value+percent parent",
        hovertemplate="<b>%{label}</b><br>Pages: %{value}<br>Percentage: %{percentParent}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Domain Distribution (Treemap)",
        font_size=12
    )
    
    return fig

def create_temporal_analysis(scraped_content: List) -> go.Figure:
    """Create temporal analysis if timestamp data is available"""
    # This is a placeholder for temporal analysis
    # In a real implementation, you'd analyze scraping timestamps, 
    # content publication dates, etc.
    
    fig = go.Figure()
    fig.add_annotation(text="Temporal analysis requires timestamp data", 
                      xref="paper", yref="paper",
                      x=0.5, y=0.5, showarrow=False)
    return fig

def export_network_data(G: nx.DiGraph, format_type: str = "gexf") -> str:
    """Export network data in various formats"""
    if format_type == "gexf":
        # Export as GEXF (Gephi format)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.gexf', delete=False) as f:
            nx.write_gexf(G, f.name)
            return f.name
    
    elif format_type == "graphml":
        # Export as GraphML
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as f:
            nx.write_graphml(G, f.name)
            return f.name
    
    elif format_type == "json":
        # Export as JSON
        from networkx.readwrite import json_graph
        import json
        import tempfile
        
        data = json_graph.node_link_data(G)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2)
            return f.name
    
    else:
        raise ValueError(f"Unsupported format: {format_type}")