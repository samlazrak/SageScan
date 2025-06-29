# ScanSage - AI-Powered Document Analysis

ScanSage is a comprehensive document processing system that combines OCR (Optical Character Recognition), text preprocessing, and LLM (Large Language Model) analysis to extract insights from scanned documents and digital notes.

## Features

- **OCR Processing**: Extract text from scanned images with high accuracy
- **Text Preprocessing**: Clean and normalize OCR-extracted text
- **LLM Analysis**: Sentiment analysis, summarization, and key topic extraction
- **Multi-format Support**: Images (JPG, PNG, TIFF) and digital documents (PDF, DOCX, TXT)
- **Local LLM Support**: Run with Ollama or llama.cpp for privacy
- **Batch Processing**: Process multiple files efficiently
- **Docker Support**: Easy deployment with Docker Compose

## Quick Start with Docker

### Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM (for larger models) or 4GB RAM (for smaller models)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ScanSage
```

### 2. Start Services

```bash
# Start Ollama and ScanSage containers
docker-compose up -d ollama

# Wait for Ollama to be ready (check with: docker-compose ps)
```

### 3. Download LLM Models

```bash
# Pull a small model that fits in memory (recommended for testing)
docker-compose exec ollama ollama pull tinyllama

# Or pull larger models if you have sufficient RAM
docker-compose exec ollama ollama pull llama2:7b
```

### 4. Run Sentiment Analysis

Use the provided script for easy one-off jobs:

```bash
# Make the script executable (first time only)
chmod +x run_scansage_job.sh

# Run sentiment analysis on an image
./run_scansage_job.sh \
  --input examples/image.png \
  --output results/sentiment_analysis.json \
  --llm-provider ollama \
  --llm-model tinyllama \
  --stats
```

### 5. View Results

The results will be saved to `results/sentiment_analysis.json` with:
- Extracted text from OCR
- Sentiment analysis (positive/negative/neutral)
- Key topics and action items
- Processing statistics

## Usage Examples

### Basic Image Processing

```bash
# Process a single image with default settings
./run_scansage_job.sh --input your_image.jpg --output results/output.json
```

### Digital Notes Processing

```bash
# Process text files (PDF, DOCX, TXT) instead of images
./run_scansage_job.sh \
  --input examples/sample_notes/sample_note.txt \
  --output results/digital_notes.json \
  --digital \
  --llm-provider ollama \
  --llm-model tinyllama
```

### Batch Processing

```bash
# Process all images in a folder
./run_scansage_job.sh \
  --input your_images_folder/ \
  --output results/batch_results.json \
  --stats
```

### Advanced Configuration

```bash
# Use different LLM models and providers
./run_scansage_job.sh \
  --input examples/image.png \
  --output results/advanced.json \
  --llm-provider ollama \
  --llm-model llama2:7b \
  --no-stopwords \
  --no-lemmatize \
  --stats
```

## Available Models

### Memory Requirements

| Model | Size | RAM Required | Use Case |
|-------|------|--------------|----------|
| `tinyllama` | 637 MB | ~2 GB | Testing, quick analysis |
| `llama2:7b` | 3.8 GB | ~8 GB | Better quality analysis |
| `llama2:latest` | 3.8 GB | ~8 GB | Latest features |

### Pulling Models

```bash
# Check available models
docker-compose exec ollama ollama list

# Pull a specific model
docker-compose exec ollama ollama pull <model_name>
```

## Configuration

### Environment Variables

The following environment variables can be set in `docker-compose.yml`:

```yaml
environment:
  - LOG_LEVEL=INFO
  - DEFAULT_LLM_PROVIDER=local
  - DEFAULT_LLM_MODEL=tinyllama
  - OLLAMA_BASE_URL=http://ollama:11434
  - TESSERACT_CMD=/usr/bin/tesseract
```

### Volume Mounts

The following directories are mounted for data persistence:

- `./data` → `/app/data` - Application data
- `./uploads` → `/app/uploads` - Upload directory
- `./results` → `/app/results` - Output results
- `./examples` → `/app/examples` - Example files

## Troubleshooting

### Memory Issues

If you encounter memory errors:

```bash
# Check available memory
docker-compose exec ollama free -h

# Use a smaller model
docker-compose exec ollama ollama pull tinyllama

# Or increase Docker memory limits in Docker Desktop
```

### Container Issues

```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs scansage
docker-compose logs ollama

# Restart services
docker-compose restart
```

### OCR Issues

```bash
# Check if Tesseract is working
docker-compose exec scansage tesseract --version

# Test OCR on a simple image
docker-compose exec scansage python -c "
from src.ocr_processor import OCRProcessor
processor = OCRProcessor()
result = processor.extract_text('examples/image.png')
print(f'OCR Confidence: {result[\"confidence\"]}%')
"
```

## Development

### Local Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLP data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm

# Run tests
python -m pytest tests/
```

### Running Without Docker

```bash
# Set up environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run directly
python main.py --input examples/image.png --output results/local_test.json
```

## API Reference

### Command Line Arguments

```bash
python main.py --help
```

Key arguments:
- `--input`: Input file or folder path
- `--output`: Output JSON file path
- `--digital`: Process digital files (PDF, DOCX, TXT) instead of images
- `--llm-provider`: LLM provider (openai, anthropic, local, ollama, llama-cpp)
- `--llm-model`: Model name to use
- `--stats`: Show processing statistics
- `--verbose`: Enable verbose logging

### Python API

```python
from src.note_processor import NoteProcessor

# Initialize processor
processor = NoteProcessor(
    llm_provider="ollama",
    llm_model="tinyllama"
)

# Process a single file
result = processor.process_single_note("path/to/image.jpg")

# Process multiple files
results = processor.process_batch("path/to/folder/")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the logs with `docker-compose logs`
- Open an issue on GitHub

---

**Note**: This system requires sufficient memory to run LLM models. For production use, consider using cloud-based LLM services or dedicated hardware with adequate RAM. 