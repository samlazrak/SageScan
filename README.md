# Text Summarizer with OCR and MLX

A Python application that combines Optical Character Recognition (OCR) with small language models in MLX format to extract and summarize text from images and documents.

## Features

- 🔍 **OCR Text Extraction**: Extract text from images using advanced preprocessing
- 🤖 **MLX Language Models**: Use efficient small language models for summarization
- 📁 **Multiple Input Types**: Support for images, text files, and direct text input
- ⚡ **Optimized Performance**: MLX provides fast inference on Apple Silicon and other platforms
- 🎨 **Rich CLI Interface**: Beautiful command-line interface with progress indicators
- 💾 **Flexible Output**: Save results in various formats

## Installation

### Prerequisites

1. **Python 3.8+** is required
2. **Tesseract OCR** must be installed on your system:

```bash
# On macOS
brew install tesseract

# On Ubuntu/Debian
sudo apt-get install tesseract-ocr

# On CentOS/RHEL
sudo yum install tesseract
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### MLX Installation

MLX is optimized for Apple Silicon but also works on other platforms:

```bash
# For Apple Silicon Macs
pip install mlx mlx-lm

# For other platforms, MLX will still work but may not be as optimized
```

## Usage

### Command Line Interface

```bash
# Summarize text from an image
python text_summarizer.py -i path/to/image.png

# Summarize direct text input
python text_summarizer.py -t "Your text content here..."

# Summarize a text file
python text_summarizer.py -i document.txt

# Use a specific model and save output
python text_summarizer.py -i image.jpg -m mlx-community/phi-2-4bit -o summary.txt

# Adjust summary length
python text_summarizer.py -i document.pdf --max-tokens 200
```

### Demo Mode

Run without arguments to see a demonstration with example text:

```bash
python text_summarizer.py
```

### Python API

```python
from text_summarizer import TextSummarizer

# Initialize the summarizer
summarizer = TextSummarizer()

# Process an image
result = summarizer.process_file("image.png")
print(result['summary'])

# Summarize direct text
summary = summarizer.summarize_text("Your text here...")
print(summary)
```

## Supported File Formats

### Images
- `.jpg`, `.jpeg` - JPEG images
- `.png` - PNG images  
- `.bmp` - Bitmap images
- `.tiff`, `.tif` - TIFF images
- `.webp` - WebP images

### Text Files
- `.txt` - Plain text files
- `.md` - Markdown files
- `.rst` - reStructuredText files
- `.csv` - CSV files (text content will be extracted)

## Configuration

Edit `config.py` to customize:

- **Model Selection**: Choose from different MLX models
- **OCR Settings**: Adjust preprocessing parameters
- **Summarization**: Control output length and style
- **File Processing**: Add support for new formats

### Available Models

The application supports various MLX-compatible models:

- `tiny`: TinyLlama-1.1B (fastest, lower quality)
- `phi`: Phi-2 (balanced performance)
- `mistral`: Mistral-7B (higher quality, slower)
- Custom models from Hugging Face Hub

## Examples

### Example 1: Processing Meeting Notes Image

```bash
python text_summarizer.py -i meeting_notes.jpg -o summary.md
```

Output:
```
📊 Processing Results
├─ Source: image
├─ Original length: 1,250 characters  
└─ Summary length: 145 characters

📋 Summary
Meeting covered Q4 budget planning, team restructuring, and new product launch timeline. Key decisions: 15% budget increase approved, hiring freeze lifted, product launch moved to March 2024.
```

### Example 2: Summarizing the Think Tool Documentation

Using the provided text about the think tool:

```bash
python text_summarizer.py -t "## Using the think tool..."
```

Expected output:
```
The think tool serves as a planning scratchpad before taking actions. Users should list applicable rules, verify required information is collected, ensure policy compliance, and check tool results for accuracy. Examples include flight cancellations (requiring user verification and rule checking) and booking tickets (involving membership tiers, baggage calculations, and payment method validation).
```

## Advanced Features

### Custom Models

```bash
# Use a custom model from Hugging Face
python text_summarizer.py -i doc.txt -m "your-username/custom-model-mlx"
```

### Batch Processing

```python
import os
from text_summarizer import TextSummarizer

summarizer = TextSummarizer()

# Process all images in a directory
image_dir = "documents/"
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        result = summarizer.process_file(os.path.join(image_dir, filename))
        print(f"{filename}: {result['summary']}")
```

### Image Preprocessing

The application automatically applies several preprocessing steps for better OCR:

1. **Denoising**: Removes image noise
2. **Grayscale Conversion**: Improves text detection
3. **Adaptive Thresholding**: Enhances text contrast
4. **Resolution Enhancement**: Upscales images when needed

## Troubleshooting

### Common Issues

1. **"No text found in image"**
   - Ensure image has sufficient contrast
   - Try preprocessing the image manually
   - Check if text is clearly visible

2. **MLX model loading fails**
   - Verify internet connection for model download
   - Check available disk space
   - Try a smaller model (e.g., "tiny")

3. **Tesseract not found**
   - Install Tesseract OCR system-wide
   - Add Tesseract to your PATH environment variable

### Performance Tips

- Use smaller models (`tiny` or `phi`) for faster processing
- Process images at higher resolution for better OCR
- Use GPU acceleration when available (Apple Silicon Macs)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) - Efficient ML framework
- [Tesseract](https://github.com/tesseract-ocr/tesseract) - OCR engine
- [OpenCV](https://opencv.org/) - Image processing
- [Rich](https://github.com/Textualize/rich) - Terminal formatting
