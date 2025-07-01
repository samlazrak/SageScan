# OCR + MLX Text Analyzer

A Python application that combines Optical Character Recognition (OCR) with MLX language models to extract text from images and provide intelligent analysis including summarization and sentiment analysis.

## Features

- **OCR Text Extraction**: Uses Tesseract OCR with image preprocessing for optimal text extraction
- **MLX Model Integration**: Leverages small, efficient MLX models for text analysis
- **Text Summarization**: Generates concise summaries of extracted text
- **Sentiment Analysis**: Analyzes emotional tone with confidence scores
- **Batch Processing**: Process multiple images efficiently
- **Command Line Interface**: Easy-to-use CLI with flexible options

## Requirements

- Python 3.8+
- macOS with Apple Silicon (for MLX support) or compatible system
- Tesseract OCR engine

## Installation

### 1. Install System Dependencies

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**CentOS/RHEL:**
```bash
sudo yum install tesseract
```

### 2. Install Python Package

```bash
# Install from source
git clone <repository-url>
cd ocr-mlx-analyzer
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### 3. Verify Installation

```bash
python ocr_mlx_analyzer.py --help
```

## Usage

### Basic Usage

```bash
# Analyze a single image
python ocr_mlx_analyzer.py path/to/your/image.jpg

# Use a specific model
python ocr_mlx_analyzer.py path/to/image.png --model "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

# Save results to file
python ocr_mlx_analyzer.py image.jpg --output results.json

# Enable verbose logging
python ocr_mlx_analyzer.py image.jpg --verbose
```

### Advanced Usage

```bash
# Process multiple images with batch script
python batch_processor.py images_folder/ --output-dir results/

# Use custom model for specific domain
python ocr_mlx_analyzer.py document.jpg --model "custom-model-name"
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)
- BMP (.bmp)
- GIF (.gif)
- WebP (.webp)

## MLX Models

The application supports various MLX models optimized for different use cases:

- **Default**: `mlx-community/Llama-3.2-1B-Instruct-4bit` (balanced performance)
- **Lightweight**: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (faster, lower memory)
- **Custom**: Any compatible MLX instruction-tuned model

## Output Format

The application provides structured output including:

```json
{
  "image_path": "path/to/image.jpg",
  "extracted_text": "Full text extracted from image...",
  "summary": "Concise summary of the content...",
  "sentiment": {
    "sentiment": "positive|negative|neutral",
    "confidence": 0.85,
    "explanation": "Detailed analysis explanation..."
  }
}
```

## Performance Tips

1. **Image Quality**: Higher resolution images with clear text provide better OCR results
2. **Preprocessing**: The application automatically optimizes images for OCR
3. **Model Selection**: Choose smaller models for faster processing, larger for better quality
4. **Batch Processing**: Use the batch processor for multiple images to amortize model loading time

## Troubleshooting

### Common Issues

**Tesseract not found:**
```bash
# Check if tesseract is in PATH
which tesseract

# If not found, install or add to PATH
export PATH="/usr/local/bin:$PATH"
```

**MLX import errors:**
- Ensure you're on a compatible system (Apple Silicon recommended)
- Install MLX: `pip install mlx mlx-lm`

**Poor OCR results:**
- Ensure image has sufficient resolution (min 300 DPI recommended)
- Check image contrast and clarity
- Try different image preprocessing techniques

**Memory issues:**
- Use smaller MLX models
- Process images one at a time
- Reduce image resolution if necessary

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text extraction
- [MLX](https://github.com/ml-explore/mlx) for efficient language model inference
- [OpenCV](https://opencv.org/) for image processing