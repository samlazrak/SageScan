# 🧠 Deep Researcher Agent - Setup Instructions

This guide will help you set up and run the Deep Researcher Agent in various environments.

## 📋 Prerequisites

- **Python 3.8+** (required)
- **4GB+ RAM** (recommended)
- **Internet connection** (for web scraping)
- **PDF files with links** (for testing)

## 🚀 Quick Setup (Recommended)

### Option 1: Using the Test Script

```bash
# 1. Run the test script first
python3 test_deep_researcher.py

# 2. If tests pass, install dependencies manually:
pip install --break-system-packages -r requirements.txt

# 3. Launch the application
python3 run_deep_researcher.py
```

### Option 2: Direct Launch

```bash
# If you have Streamlit installed:
streamlit run streamlit_app.py
```

## 🐳 Environment-Specific Setup

### Linux/Ubuntu

```bash
# Install Python venv (if needed)
sudo apt update
sudo apt install python3-venv python3-pip

# Create virtual environment
python3 -m venv deep_researcher_env
source deep_researcher_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run streamlit_app.py
```

### macOS

```bash
# Install using pip
pip3 install -r requirements.txt

# Or using conda
conda install --file requirements_conda.txt

# Launch application
streamlit run streamlit_app.py
```

### Windows

```cmd
# Create virtual environment
python -m venv deep_researcher_env
deep_researcher_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run streamlit_app.py
```

### Docker (Advanced)

```bash
# Build Docker image
docker build -t deep-researcher-agent .

# Run container
docker run -p 8501:8501 deep-researcher-agent
```

## 🛠️ Manual Dependency Installation

If automatic installation fails, install dependencies manually:

### Core Dependencies

```bash
# PDF processing
pip install PyMuPDF PyPDF2

# Web scraping
pip install beautifulsoup4 requests trafilatura newspaper3k

# Network analysis
pip install networkx

# Visualization
pip install streamlit plotly pandas

# Utilities
pip install validators tqdm

# Optional (for JavaScript-heavy sites)
pip install selenium webdriver-manager
```

### Minimal Installation (Core Features Only)

```bash
# For basic functionality without full visualization
pip install PyMuPDF requests beautifulsoup4 streamlit pandas
```

## ⚙️ Configuration

### Default Settings

The application comes with sensible defaults, but you can customize:

1. **Edit `config_sample.json`** (created by test script)
2. **Use Streamlit sidebar** (runtime configuration)
3. **Modify code directly** (advanced users)

### Key Settings

- **max_links**: Start with 5-10 for testing
- **max_depth**: Begin with 1-2 levels
- **delay_between_requests**: Use 1-2 seconds to be respectful
- **use_selenium**: Only enable if needed (slower)

## 📄 Testing with Sample PDFs

### Academic Papers (Recommended)

```bash
# Download sample PDFs with links:
wget https://arxiv.org/pdf/1706.03762.pdf  # Transformer paper
wget https://arxiv.org/pdf/2010.11929.pdf  # Vision Transformer
```

### Creating Test PDFs

1. **From web articles**: Save any article with references as PDF
2. **Research papers**: Use Google Scholar, IEEE, arXiv
3. **Reports**: Use any document with web references

## 🌐 Network Requirements

### Firewall Settings

Ensure outbound HTTPS/HTTP access for:
- Target domains you want to scrape
- Academic domains (arxiv.org, scholar.google.com, etc.)

### Proxy Configuration

If behind a corporate proxy:

```python
# Add to your environment or modify the agent code
import os
os.environ['HTTP_PROXY'] = 'http://proxy.company.com:8080'
os.environ['HTTPS_PROXY'] = 'http://proxy.company.com:8080'
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Check Python version
python3 --version

# Verify installations
python3 -c "import streamlit; print('Streamlit OK')"
python3 -c "import fitz; print('PyMuPDF OK')"
```

#### 2. Permission Errors

```bash
# On Linux/macOS
chmod +x run_deep_researcher.py
chmod +x test_deep_researcher.py

# Or run with python explicitly
python3 run_deep_researcher.py
```

#### 3. Memory Issues

```python
# Reduce these settings in the app:
max_links = 5           # Instead of 20
max_depth = 1           # Instead of 2
max_workers = 2         # Instead of 5
```

#### 4. Selenium/ChromeDriver Issues

```bash
# Option 1: Disable Selenium
# Set use_selenium=False in the app

# Option 2: Install ChromeDriver
pip install webdriver-manager

# Option 3: System installation
sudo apt install chromium-chromedriver  # Ubuntu
brew install chromedriver               # macOS
```

#### 5. PDF Processing Errors

- Ensure PDF contains actual hyperlinks (not just text)
- Try different PDF files
- Check file permissions and encoding

### Performance Optimization

#### For Large PDFs

```python
# Start with conservative settings:
max_links = 5
max_depth = 1
delay_between_requests = 2.0
```

#### For Slow Networks

```python
# Increase timeouts and delays:
delay_between_requests = 3.0
max_workers = 2
```

## 🚀 Deployment Options

### Local Development

```bash
# Standard local setup
streamlit run streamlit_app.py
```

### Cloud Deployment

#### Streamlit Cloud

1. Push to GitHub repository
2. Connect to Streamlit Cloud
3. Deploy directly from repository

#### Heroku

```bash
# Add Procfile:
echo "web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0" > Procfile

# Deploy to Heroku
heroku create your-app-name
git push heroku main
```

#### Google Cloud Run

```bash
# Build and deploy
gcloud run deploy deep-researcher --source .
```

### Server Deployment

```bash
# Run on specific port and host
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0

# Run in background
nohup streamlit run streamlit_app.py &
```

## 📊 Usage Examples

### Basic Usage

1. **Upload PDF**: Drag and drop in the web interface
2. **Configure**: Set max_links=10, max_depth=2
3. **Run**: Click "Start Deep Research"
4. **Analyze**: Explore results in different tabs

### Advanced Usage

```python
# Programmatic usage
from src.deep_researcher_agent import DeepResearcherAgent

agent = DeepResearcherAgent()
result = agent.deep_research(
    pdf_path="paper.pdf",
    max_links=15,
    allowed_domains=["arxiv.org", "nature.com"]
)
```

### Batch Processing

```python
# Process multiple PDFs
import os
for pdf_file in os.listdir("pdfs/"):
    if pdf_file.endswith(".pdf"):
        result = agent.deep_research(f"pdfs/{pdf_file}")
        # Save results...
```

## 💡 Best Practices

### Ethical Scraping

- Use reasonable delays (1-2 seconds minimum)
- Respect robots.txt files
- Don't overwhelm target servers
- Follow website terms of service

### Performance

- Start with small limits for testing
- Use domain filtering to focus analysis
- Monitor system resources during large analyses
- Cache results when possible

### Data Management

- Export results regularly
- Use version control for configurations
- Backup important analysis results
- Document your research methodology

## 📞 Support

### Getting Help

1. **Run the test script**: `python3 test_deep_researcher.py`
2. **Check logs**: Look for error messages in the terminal
3. **Review documentation**: Read `DEEP_RESEARCHER_README.md`
4. **Try minimal settings**: Reduce complexity for testing

### Reporting Issues

When reporting issues, include:
- Python version
- Operating system
- Error messages (full traceback)
- PDF file characteristics
- Configuration settings used

## 🔄 Updates and Maintenance

### Updating Dependencies

```bash
# Update all packages
pip install --upgrade -r requirements.txt

# Update specific packages
pip install --upgrade streamlit plotly
```

### Backing Up Data

```bash
# Backup configurations and results
tar -czf backup_$(date +%Y%m%d).tar.gz config_sample.json results/ *.md
```

---

**🧠 Deep Researcher Agent** - Comprehensive setup guide for advanced PDF link analysis and research automation.