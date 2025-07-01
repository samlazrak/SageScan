# OCR Text Summarizer with MLX

A Python application that combines Optical Character Recognition (OCR) with small language models in MLX format to extract and summarize text from images.

## Features

- **OCR Text Extraction**: Extract text from images using Tesseract OCR with preprocessing
- **MLX Language Models**: Use efficient MLX-format language models for text summarization  
- **Image Preprocessing**: Automatic image enhancement for better OCR accuracy
- **Flexible Input**: Process images or raw text input
- **Multiple Output Formats**: Console output and file export options
- **Model Selection**: Choose from different MLX models based on your needs

## Architecture

### Components

1. **OCRProcessor**: Handles image preprocessing and text extraction
   - Image noise reduction and contrast enhancement
   - Tesseract OCR integration with optimized settings
   - Text cleaning and normalization

2. **MLXSummarizer**: Manages language model inference for summarization
   - MLX model loading with fallback support
   - Prompt engineering for effective summarization
   - Response post-processing and formatting

3. **OCRSummarizerApp**: Main orchestration class
   - Workflow management
   - Result formatting and export
   - Error handling and logging

## Installation

### Quick Setup

Run the automated setup script:

```bash
chmod +x setup.sh
./setup.sh
```

### Manual Installation

1. **Install system dependencies**:
```bash
# Update packages
sudo apt-get update

# Install Tesseract OCR
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng libtesseract-dev

# Install system libraries for OpenCV
sudo apt-get install -y python3-dev python3-pip libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
```

2. **Install Python packages**:
```bash
pip3 install -r requirements.txt
```

## Usage

### Basic Examples

**Summarize text from an image:**
```bash
python3 ocr_summarizer.py --image document.jpg
```

**Summarize raw text:**
```bash
python3 ocr_summarizer.py --text "Your long text content here..."
```

**Save results to file:**
```bash
python3 ocr_summarizer.py --image scan.png --output results.txt
```

### Advanced Options

**Use a different MLX model:**
```bash
python3 ocr_summarizer.py --image photo.jpg --model "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
```

**Enable verbose logging:**
```bash
python3 ocr_summarizer.py --image document.jpg --verbose
```

### Command Line Arguments

```
usage: ocr_summarizer.py [-h] (--image IMAGE | --text TEXT) [--output OUTPUT] 
                        [--model MODEL] [--verbose]

options:
  -h, --help            show this help message and exit
  --image IMAGE, -i IMAGE
                        Path to image file for OCR text extraction
  --text TEXT, -t TEXT  Raw text to summarize (instead of using OCR)
  --output OUTPUT, -o OUTPUT
                        Output file to save results (optional)
  --model MODEL, -m MODEL
                        MLX model name for summarization
  --verbose, -v         Enable verbose logging
```

## Supported Formats

### Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff, .tif)
- BMP (.bmp)
- GIF (.gif)

### MLX Models
The application supports various MLX-format language models:

- **Default**: `mlx-community/Llama-3.2-1B-Instruct-4bit`
- **Lightweight**: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`
- **Custom**: Any compatible MLX model from Hugging Face

## Implementation Details

### OCR Pipeline

1. **Image Preprocessing**:
   - Convert to grayscale
   - Apply noise reduction (median blur)
   - Optimize contrast with Otsu thresholding

2. **Text Extraction**:
   - Use Tesseract with optimized PSM (Page Segmentation Mode)
   - Clean extracted text (remove artifacts, normalize whitespace)

3. **Quality Control**:
   - Validate extracted text
   - Handle empty or corrupted results

### Summarization Pipeline

1. **Prompt Engineering**:
   - Structured prompts for consistent results
   - Context-aware instructions
   - Length control parameters

2. **Model Inference**:
   - MLX-optimized inference
   - Temperature and token control
   - Fallback model support

3. **Post-processing**:
   - Response cleaning and formatting
   - Extract summary from model output
   - Handle edge cases and errors

## Performance Considerations

### Memory Usage
- MLX models are optimized for Apple Silicon but work on other platforms
- Smaller models (0.5B-1B parameters) recommended for resource-constrained environments
- Image preprocessing uses minimal memory with streaming processing

### Processing Speed
- OCR: ~1-3 seconds per image (depends on size and complexity)
- Summarization: ~2-10 seconds (depends on text length and model size)
- Model loading: ~5-30 seconds (one-time cost per session)

## Error Handling

The application includes comprehensive error handling:

- **Image Issues**: Invalid formats, corrupted files, unreadable images
- **OCR Failures**: No text detected, processing errors
- **Model Issues**: Loading failures, inference errors, fallback mechanisms
- **System Issues**: Missing dependencies, permission errors

## Troubleshooting

### Common Issues

1. **Tesseract not found**:
   ```bash
   sudo apt-get install tesseract-ocr
   ```

2. **OpenCV import errors**:
   ```bash
   pip3 install opencv-python-headless
   ```

3. **MLX model download issues**:
   - Check internet connection
   - Verify model name is correct
   - Try fallback model

4. **Poor OCR results**:
   - Ensure image has good contrast
   - Check image resolution (minimum 300 DPI recommended)
   - Try different image preprocessing

### Debug Mode

Enable verbose logging to diagnose issues:
```bash
python3 ocr_summarizer.py --image test.jpg --verbose
```

## Contributing

Feel free to contribute improvements:

1. **OCR Enhancement**: Better preprocessing algorithms
2. **Model Support**: Additional MLX model compatibility
3. **Performance**: Optimization for specific use cases
4. **Features**: Batch processing, GUI interface, etc.

## License

This project is open source. See the code for implementation details and feel free to modify for your needs.

## Dependencies

### System Dependencies
- Tesseract OCR engine
- Python 3.8+
- Standard Linux image processing libraries

### Python Dependencies
- `mlx` and `mlx-lm`: MLX framework and language models
- `pytesseract`: Python wrapper for Tesseract OCR
- `opencv-python`: Image processing and computer vision
- `pillow`: Python Imaging Library
- `numpy`: Numerical computing

## Examples

### Example 1: Process a Document
```bash
python3 ocr_summarizer.py --image invoice.pdf --output invoice_summary.txt
```

### Example 2: Quick Text Summary
```bash
python3 ocr_summarizer.py --text "$(cat long_article.txt)" --output summary.txt
```

### Example 3: Batch Processing (Shell Script)
```bash
#!/bin/bash
for img in *.jpg; do
    python3 ocr_summarizer.py --image "$img" --output "${img%.jpg}_summary.txt"
done
```

This application transforms the traditional "think tool" concept into a practical OCR and summarization workflow, leveraging modern MLX language models for efficient text processing.