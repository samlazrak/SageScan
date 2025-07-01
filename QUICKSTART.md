# 🚀 Quick Start Guide

Get up and running with OCR Image Analyzer in 5 minutes!

## 1. System Requirements

- **Python 3.8+** (check with `python3 --version`)
- **Operating System**: Linux, macOS, or Windows

## 2. One-Command Installation

### Linux/macOS (Recommended):
```bash
curl -sSL https://raw.githubusercontent.com/your-repo/ocr-analyzer/main/install.sh | bash
```

Or clone and run:
```bash
git clone <your-repo-url>
cd ocr-analyzer
chmod +x install.sh
./install.sh
```

### Manual Installation:
```bash
# 1. Install Tesseract OCR
# Ubuntu/Debian:
sudo apt install tesseract-ocr tesseract-ocr-eng

# macOS:
brew install tesseract

# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki

# 2. Install Python dependencies
pip3 install -r requirements.txt

# 3. For Apple Silicon Macs (optional but recommended):
pip3 install mlx mlx-lm
```

## 3. Test Your Installation

```bash
python3 test_installation.py
```

You should see:
```
🎉 Installation successful!
```

## 4. Your First OCR Analysis

### Basic Usage:
```bash
# Analyze a single image
python3 ocr_analyzer.py my_document.jpg

# Analyze multiple images
python3 ocr_analyzer.py photo1.jpg photo2.png document.pdf
```

### With Options:
```bash
# Save results to JSON file
python3 ocr_analyzer.py --output results.json image.jpg

# Use a different MLX model (Apple Silicon)
python3 ocr_analyzer.py --model "mlx-community/Qwen2.5-1.5B-Instruct-4bit" document.jpg

# Enable verbose output
python3 ocr_analyzer.py --verbose image.jpg
```

## 5. Sample Output

When you run the analyzer, you'll see:

```
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Image         ┃ Text Length ┃ Summary                                         ┃  Sentiment  ┃ Confidence ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ document.jpg  │ 245         │ This document discusses quarterly sales...      │  positive   │ 0.82       │
└───────────────┴─────────────┴─────────────────────────────────────────────────┴─────────────┴────────────┘

📖 Extracted Text - document.jpg
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ The quarterly sales report shows significant growth across all product categories. Customer satisfaction ratings have improved ┃
┃ substantially, with positive feedback highlighting our improved customer service and product quality...                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

📝 Summary
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ The quarterly sales report demonstrates strong performance with growth across product categories and improved customer        ┃
┃ satisfaction due to better service and quality.                                                                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

💭 Sentiment Analysis
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Sentiment: POSITIVE                                                                                                           ┃
┃ Confidence: 0.85                                                                                                              ┃
┃ Explanation: The text contains positive language indicators such as "growth", "improved", and "positive feedback"           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## 6. Tips for Best Results

### Image Quality
- Use high-resolution images (300+ DPI)
- Ensure good contrast between text and background
- Avoid skewed or rotated text when possible

### Supported Formats
- PNG, JPG, JPEG, TIFF, BMP, GIF
- PDF files (experimental)

### Model Selection (Apple Silicon)
- **Small/Fast**: `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (default)
- **Larger/Better**: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- **Alternative**: `mlx-community/Phi-3-mini-4k-instruct-4bit`

## 7. Common Issues

### "Command not found"
Make sure Python 3.8+ is installed:
```bash
python3 --version
```

### "Tesseract not found"
Install Tesseract OCR for your system (see step 2 above)

### "MLX not available"
This is normal on non-Apple Silicon systems. The app will use fallback analysis.

## 8. Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check out example use cases and advanced configuration
- Contribute to the project or report issues

## Need Help?

- Check the [troubleshooting section](README.md#troubleshooting) in README.md
- Run `python3 ocr_analyzer.py --help` for command options
- Open an issue on GitHub for bugs or feature requests

Happy analyzing! 🎉