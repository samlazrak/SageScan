# 🧠 Deep Researcher Agent

A comprehensive research tool that extracts links from PDF documents and performs deep, multi-level web content analysis with interactive Streamlit interface.

![Deep Researcher Agent](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

### Core Capabilities
- **📄 PDF Link Extraction**: Automatically extracts hyperlinks from PDF files with context and page numbers
- **🕷️ Advanced Web Scraping**: Multi-method content scraping with fallback mechanisms
- **🔍 Deep Analysis**: Recursive multi-level scraping with configurable depth
- **🕸️ Network Analysis**: Builds interactive network graphs of content relationships
- **📊 Rich Visualization**: Interactive charts, network graphs, and analytics dashboards
- **🎯 Smart Filtering**: Domain-based filtering and relevance scoring
- **💾 Multiple Export Formats**: JSON, CSV, Markdown, and network formats

### Scraping Methods
1. **Trafilatura**: Best for main content extraction
2. **Newspaper3k**: Optimized for news articles and blogs
3. **Beautiful Soup**: Reliable fallback for general web content
4. **Selenium**: For JavaScript-heavy websites (optional)

### Network Analysis
- Interactive network visualization with Plotly
- Node centrality metrics (degree, betweenness, closeness)
- Community detection and connectivity analysis
- Export to Gephi, GraphML, and JSON formats

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd deep-researcher-agent

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

### 3. Upload and Analyze

1. Upload a PDF document through the web interface
2. Configure analysis parameters (depth, domains, etc.)
3. Click "Start Deep Research"
4. Explore results in interactive dashboards

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (recommended)
- Internet connection for web scraping

### Key Dependencies
```
streamlit>=1.28.0
PyMuPDF>=1.23.0
PyPDF2>=3.0.0
beautifulsoup4>=4.12.0
requests>=2.31.0
networkx>=3.0
plotly>=5.17.0
pandas>=2.0.0
trafilatura>=1.6.0
newspaper3k>=0.2.8
validators>=0.22.0
selenium>=4.15.0 (optional)
```

## 🎮 Usage

### Streamlit Web Interface

The easiest way to use the Deep Researcher Agent is through the interactive web interface:

```bash
streamlit run streamlit_app.py
```

**Interface Features:**
- 📂 Drag-and-drop PDF upload
- ⚙️ Configurable analysis parameters
- 📊 Real-time progress tracking
- 🎨 Interactive visualizations
- 💾 Multiple export options

### Programmatic Usage

```python
from src.deep_researcher_agent import DeepResearcherAgent

# Initialize agent
agent = DeepResearcherAgent(
    use_selenium=False,
    delay_between_requests=1.0,
    max_workers=5
)

# Perform deep research
result = agent.deep_research(
    pdf_path="research_paper.pdf",
    max_links=20,
    allowed_domains=["arxiv.org", "nature.com"],
    max_depth=2,
    use_multi_level=True
)

# Access results
print(f"Found {result.total_links_found} links")
print(f"Success rate: {result.successful_scrapes}/{result.total_links_found}")
print(result.summary)
```

### Command Line Example

```bash
python examples/deep_research_example.py
```

## 🔧 Configuration Options

### Basic Settings
- **max_links**: Maximum number of initial links to process (5-100)
- **use_multi_level**: Enable recursive link following (True/False)
- **include_images**: Extract images from scraped content (True/False)
- **use_selenium**: Enable Selenium for JS-heavy sites (True/False)

### Advanced Settings
- **max_depth**: Maximum recursion depth (1-5)
- **max_links_per_level**: Links to follow per depth level (5-50)
- **delay_between_requests**: Delay between HTTP requests (0.5-5.0 seconds)
- **max_workers**: Concurrent scraping workers (1-10)

### Domain Filtering
- **allowed_domains**: Whitelist of domains to scrape
- **blocked_domains**: Blacklist of domains to avoid
- **relevance_threshold**: Minimum relevance score for links (0.0-1.0)

## 📊 Output and Analysis

### Research Result Structure

```python
@dataclass
class ResearchResult:
    pdf_path: str
    total_links_found: int
    successful_scrapes: int
    failed_scrapes: int
    max_depth_reached: int
    content_network: nx.DiGraph
    scraped_content: List[ScrapedContent]
    summary: str
    key_insights: List[str]
    domain_analysis: Dict[str, int]
```

### Visualization Types

1. **Network Graph**: Interactive node-link diagram
2. **Domain Distribution**: Bar chart and treemap
3. **Success Rate Analysis**: Pie charts and metrics
4. **Content Quality Metrics**: Histograms and heatmaps
5. **Depth Analysis**: Multi-level scraping results

### Export Formats

- **Markdown Summary**: Comprehensive research report
- **JSON Data**: Complete structured data
- **CSV Table**: Tabular content analysis
- **Network Files**: GEXF, GraphML for network analysis tools

## 🛠️ Advanced Features

### Custom Scraping Agents

```python
from src.deep_researcher_agent import WebScraperAgent

# Custom scraper with specific settings
with WebScraperAgent(use_selenium=True, delay=2.0) as scraper:
    content = scraper.scrape_content(url, include_images=True)
```

### Network Analysis

```python
import networkx as nx
from src.visualization_utils import calculate_node_metrics

# Analyze network properties
metrics = calculate_node_metrics(result.content_network)
centrality = nx.degree_centrality(result.content_network)
```

### Custom Visualizations

```python
from src.visualization_utils import create_interactive_network_plot

# Create custom network plot
fig = create_interactive_network_plot(
    G=result.content_network,
    layout_type="spring",
    node_size_metric="degree_centrality",
    color_metric="depth"
)
```

## 🔍 Use Cases

### Academic Research
- Analyze citation networks in research papers
- Map literature review connections
- Discover related work and resources

### Market Research
- Analyze competitor website networks
- Track industry reference patterns
- Identify key information sources

### Journalism
- Investigate source networks in documents
- Verify reference chains and credibility
- Map information flow patterns

### Legal Analysis
- Analyze legal document references
- Track case law citations
- Map regulatory connections

## 🚨 Best Practices

### Ethical Scraping
- Respect robots.txt files
- Use appropriate delays between requests
- Don't overwhelm target servers
- Follow website terms of service

### Performance Optimization
- Start with smaller link limits for testing
- Use domain filtering to focus analysis
- Adjust worker count based on system resources
- Enable Selenium only when necessary

### Troubleshooting
- Check PDF contains actual hyperlinks
- Verify internet connectivity
- Review domain filtering settings
- Monitor memory usage for large analyses

## 📈 Performance Metrics

### Typical Performance
- **Small PDFs** (5-10 links): 30-60 seconds
- **Medium PDFs** (10-25 links): 1-3 minutes
- **Large PDFs** (25+ links): 3-10 minutes
- **Deep Analysis** (depth 3+): 5-15 minutes

### Memory Usage
- **Base application**: ~100MB
- **Per scraped page**: ~1-5MB
- **Network visualization**: ~10-50MB
- **Large analysis** (100+ pages): ~500MB-1GB

## 🐛 Troubleshooting

### Common Issues

**PDF Link Extraction Fails**
```
Solution: Ensure PDF contains actual hyperlinks, not just text URLs
Check: PDF version, security settings, and file integrity
```

**Selenium ChromeDriver Errors**
```
Solution: Install ChromeDriver or disable Selenium
Command: pip install webdriver-manager
Or set: use_selenium=False
```

**Memory Issues with Large Analyses**
```
Solution: Reduce max_links, max_depth, or max_workers
Monitor: System memory usage during analysis
```

**Network Connection Timeouts**
```
Solution: Increase delay_between_requests
Check: Internet connectivity and DNS resolution
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black src/ streamlit_app.py

# Lint code
flake8 src/ streamlit_app.py
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyMuPDF**: PDF processing and link extraction
- **Trafilatura**: High-quality web content extraction
- **Streamlit**: Interactive web interface framework
- **NetworkX**: Network analysis and graph algorithms
- **Plotly**: Interactive visualizations
- **Beautiful Soup**: HTML parsing and content extraction

## 📞 Support

- 📧 Email: [your-email@domain.com]
- 🐛 Issues: [GitHub Issues](link-to-issues)
- 📖 Documentation: [Full Documentation](link-to-docs)
- 💬 Discussions: [GitHub Discussions](link-to-discussions)

## 🔮 Roadmap

### Upcoming Features
- [ ] AI-powered content summarization
- [ ] Advanced sentiment analysis
- [ ] Real-time collaborative analysis
- [ ] Cloud deployment options
- [ ] API for programmatic access
- [ ] Database storage for results
- [ ] Scheduled analysis workflows

### Version History
- **v1.0.0**: Initial release with core functionality
- **v1.1.0**: Enhanced visualization and export options
- **v1.2.0**: Multi-level scraping and network analysis
- **v2.0.0**: Streamlit interface and advanced features

---

**🧠 Deep Researcher Agent** - Transforming PDF analysis into actionable research insights through advanced web scraping and network analysis.