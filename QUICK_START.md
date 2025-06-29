# Quick Start Guide

Get up and running with the Scanned Notes Processor in minutes!

## Prerequisites

1. **Python 3.8+** installed on your system (tested with Python 3.13)
2. **Tesseract OCR** installed:
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

## Installation

### Quick Setup (Recommended)

For automated setup, use one of our setup scripts:

**macOS/Linux:**
```bash
./setup.sh
```

**Windows:**
```cmd
setup.bat
```

These scripts will automatically handle the entire setup process for you!

### Manual Setup

If you prefer manual setup or the scripts don't work for your system:

### Option 1: Using Conda (Recommended)

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd scanned-notes-processor
   ```

2. **Create and activate conda environment**
   ```bash
   conda create -n ScanSage python=3.8
   conda activate ScanSage
   ```

3. **Install pip in conda environment (if needed)**
   ```bash
   conda install pip
   ```

4. **Install Python dependencies**
   ```bash
   conda activate ScanSage
   conda install -y -c conda-forge numpy pandas scikit-learn scipy pillow opencv pytorch transformers nltk spacy python-dotenv requests pydantic pytesseract openai fastapi uvicorn python-multipart
   pip install langchain langchain-openai
   ```

5. **Download required NLP data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   python -m spacy download en_core_web_sm
   ```

6. **Set up API keys** (optional, for LLM features)
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

### Option 2: Using pip (Alternative)

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd scanned-notes-processor
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required NLP data**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   python -m spacy download en_core_web_sm
   ```

4. **Set up API keys** (optional, for LLM features)
   ```bash
   cp env.example .env
   # Edit .env with your API keys
   ```

## Quick Examples

### 1. Process a Single Scanned Note

```bash
python main.py --input your_scan.jpg --output result.json --stats
```

### 2. Process All Scanned Notes in a Folder

```bash
python main.py --input scanned_notes/ --output results/ --stats
```

### 3. Process Digital Notes (PDF, DOCX, TXT)

```bash
python main.py --input digital_notes/ --output results/ --digital --stats
```

### 4. Use Different LLM Provider

```bash
python main.py --input notes/ --output results/ --llm-provider openai --llm-model gpt-4 --stats
```

## Python API Usage

```python
from src.note_processor import NoteProcessor

# Initialize processor
processor = NoteProcessor()

# Process a single note
result = processor.process_single_note("scan.jpg")
print(f"Summary: {result.summary}")
print(f"Sentiment: {result.sentiment}")

# Process multiple notes
results = processor.process_batch("notes_folder/")
for result in results:
    print(f"{result.filename}: {result.summary}")
```

## Expected Output

The processor will generate a JSON file with:

```json
[
  {
    "filename": "scan.jpg",
    "extracted_text": "Original OCR text...",
    "processed_text": "Cleaned and preprocessed text...",
    "summary": "Brief summary of the note content",
    "sentiment": "positive",
    "sentiment_score": 0.8,
    "key_topics": ["topic1", "topic2"],
    "action_items": ["action1", "action2"],
    "confidence": 0.9,
    "ocr_confidence": 85.5,
    "word_count": 150,
    "sentence_count": 8,
    "processing_time": 2.5,
    "model_used": "gpt-3.5-turbo"
  }
]
```

## Troubleshooting

### Common Issues

1. **"Tesseract not found"**
   - Install Tesseract OCR (see Prerequisites)
   - Ensure it's in your system PATH

2. **"pip is not found" in conda environment**
   - Run: `conda install pip` after activating your environment
   - Verify with: `which pip` and `pip --version`

3. **"No text extracted"**
   - Check image quality (should be clear, high contrast)
   - Try different image formats (JPG, PNG, TIFF)
   - Ensure text is readable by human eyes

4. **"LLM analysis failed"**
   - Check your API keys in `.env` file
   - Verify internet connection
   - Try using fallback mode (no API key required)

5. **"NLTK data not found"**
   - Run: `python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"`

6. **Package installation issues**
   - Try: `pip install --upgrade pip`
   - Use: `pip install -r requirements.txt --no-cache-dir`
   - Ensure you're in the correct conda environment: `conda activate ScanSage`

7. **Verify installation**
   - Run: `python test_installation.py` to test all dependencies and OCR functionality

### Performance Tips

- Use high-quality scans (300+ DPI)
- Ensure good lighting and contrast
- Process in batches for efficiency
- Use SSD storage for faster I/O

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [DIGITAL_NOTES_EXPANSION.md](DIGITAL_NOTES_EXPANSION.md) for advanced features
- Run `python examples/example_usage.py` to see more examples
- Explore the test suite in the `tests/` directory

## Support

- Check the [Issues](https://github.com/yourusername/scanned-notes-processor/issues) page
- Review the test files for usage examples
- Read the inline documentation in the source code

Happy processing! 🚀 