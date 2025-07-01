# OCR Image Analyzer with MLX-based Text Analysis

A powerful Python application that extracts text from images using OCR (Optical Character Recognition) and then uses small MLX language models to summarize the content and analyze sentiment.

## 🌟 Features

- **Advanced OCR**: Uses Tesseract OCR with image preprocessing for accurate text extraction
- **MLX Integration**: Leverages Apple's MLX framework for efficient on-device language model inference
- **Text Summarization**: Automatically generates concise summaries of extracted text
- **Sentiment Analysis**: Analyzes the emotional tone of the text content
- **Beautiful CLI**: Rich terminal interface with progress bars, tables, and colored output
- **Batch Processing**: Process multiple images simultaneously
- **Flexible Output**: Save results to JSON files for further analysis
- **Cross-platform**: Works on macOS (with MLX) and other platforms (with fallback analysis)

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- Tesseract OCR installed on your system

### Platform-Specific Requirements

#### macOS (Apple Silicon recommended)
- MLX framework for optimal performance
- Apple Silicon Macs get the best performance with MLX models

#### Linux/Windows
- Uses fallback text analysis methods when MLX is not available
- Still provides full OCR functionality

## 🚀 Installation

### 1. Install System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng
```

#### macOS:
```bash
brew install tesseract
```

#### Windows:
Download and install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Install Python Dependencies

#### Basic Installation:
```bash
pip install -r requirements.txt
```

#### For Apple Silicon Macs (recommended):
```bash
pip install -r requirements.txt
pip install mlx mlx-lm
```

#### Using setup.py:
```bash
# Basic installation
pip install -e .

# With MLX support (Apple Silicon)
pip install -e ".[mlx]"

# With transformers support
pip install -e ".[transformers]"
```

## 💻 Usage

### Basic Usage

Process a single image:
```bash
python ocr_analyzer.py image.jpg
```

Process multiple images:
```bash
python ocr_analyzer.py image1.jpg image2.png document.pdf
```

### Advanced Options

```bash
# Use a specific MLX model
python ocr_analyzer.py --model "mlx-community/Qwen2.5-1.5B-Instruct-4bit" image.jpg

# Save results to JSON file
python ocr_analyzer.py --output results.json image.jpg

# Enable verbose logging
python ocr_analyzer.py --verbose image.jpg

# Combine options
python ocr_analyzer.py -m "mlx-community/Qwen2.5-0.5B-Instruct-4bit" -o analysis.json -v *.jpg
```

### Example Output

```
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Image         ┃ Text Length ┃ Summary                                         ┃  Sentiment  ┃ Confidence ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ document.jpg  │ 245         │ This document discusses quarterly sales...      │  positive   │ 0.82       │
│ receipt.png   │ 87          │ Receipt shows purchase of office supplies...    │  neutral    │ 0.65       │
└───────────────┴─────────────┴─────────────────────────────────────────────────┴─────────────┴────────────┘
```

## 🧠 Supported Models

### MLX Models (Apple Silicon)
- `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (default - fast and efficient)
- `mlx-community/Qwen2.5-1.5B-Instruct-4bit` (larger, more accurate)
- `mlx-community/Phi-3-mini-4k-instruct-4bit`
- Any compatible MLX model from Hugging Face

### Fallback Analysis
When MLX is not available, the application uses:
- Rule-based text summarization
- Keyword-based sentiment analysis
- Still provides accurate OCR results

## 🔧 Configuration

### Tesseract Configuration
The application automatically configures Tesseract for optimal text extraction:
- Uses OEM 3 (LSTM engine) for best accuracy
- PSM 6 (uniform block of text) for documents
- Custom character whitelist for common text

### Image Preprocessing
Automatic image enhancement includes:
- Grayscale conversion
- Gaussian blur for noise reduction
- Adaptive thresholding
- Morphological operations for text clarity

## 📊 Output Format

### JSON Output Structure
```json
{
  "image_path": "path/to/image.jpg",
  "extracted_text": "Full extracted text content...",
  "summary": "Concise summary of the text...",
  "sentiment_analysis": {
    "sentiment": "positive",
    "confidence": 0.85,
    "explanation": "Text contains positive language indicators..."
  },
  "success": true
}
```

## 🎯 Use Cases

- **Document Digitization**: Convert physical documents to searchable text
- **Receipt Processing**: Extract and analyze purchase information
- **Social Media Analysis**: Analyze sentiment from image-based posts
- **Research**: Process academic papers and extract key insights
- **Business Intelligence**: Analyze customer feedback from images
- **Content Moderation**: Automatically review image-based content

## 🛠️ Development

### Project Structure
```
ocr-analyzer/
├── ocr_analyzer.py          # Main application
├── requirements.txt         # Python dependencies
├── setup.py                # Installation script
├── README.md               # This file
└── examples/               # Example images and outputs
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

#### "pytesseract.TesseractNotFoundError"
- **Solution**: Install Tesseract OCR on your system
- **macOS**: `brew install tesseract`
- **Ubuntu**: `sudo apt install tesseract-ocr`

#### "MLX not available"
- **Solution**: This is normal on non-Apple Silicon systems
- **Workaround**: The application will use fallback analysis methods

#### Poor OCR Results
- **Solution**: Ensure images have good contrast and resolution
- **Tip**: Use high-resolution scans (300+ DPI) for best results

#### Model Loading Errors
- **Solution**: Check internet connection for model downloads
- **Alternative**: Use a smaller model like the default 0.5B parameter model

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text extraction
- [MLX](https://github.com/ml-explore/mlx) for efficient model inference
- [Rich](https://github.com/Textualize/rich) for beautiful terminal output
- [OpenCV](https://opencv.org/) for image processing

## 📞 Support

For issues, questions, or contributions, please open an issue on the GitHub repository.