# Quick Start Guide - Text Summarizer with OCR and MLX

Get started with the Text Summarizer application in just a few minutes!

## 🚀 One-Line Installation

```bash
bash install.sh
```

This will automatically:
- Check your Python version
- Install Tesseract OCR 
- Install all Python dependencies
- Set up the application

## ⚡ Immediate Usage

After installation, try these commands:

### 1. Summarize the Think Tool Example (from your request)
```bash
python3 text_summarizer.py -t "## Using the think tool

Before taking any action or responding to the user after receiving tool results, use the think tool as a scratchpad to:
- List the specific rules that apply to the current request
- Check if all required information is collected
- Verify that the planned action complies with all policies
- Iterate over tool results for correctness"
```

### 2. Run the Demo
```bash
python3 demo.py
```

### 3. Get Help
```bash
python3 text_summarizer.py --help
```

## 📁 Example Files You Can Test

Create a test image with text and try:
```bash
python3 text_summarizer.py -i your_image.png
```

Create a text file and try:
```bash
echo "Machine learning is revolutionizing technology across industries..." > test.txt
python3 text_summarizer.py -i test.txt
```

## 🔧 Manual Installation (if needed)

If the automatic installer doesn't work:

1. **Install Tesseract OCR:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # macOS
   brew install tesseract
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Key Features Demo

```bash
# Summarize with custom length
python3 text_summarizer.py -t "Your long text here..." --max-tokens 50

# Use different model
python3 text_summarizer.py -t "Your text..." -m mlx-community/phi-2-4bit

# Save output to file
python3 text_summarizer.py -t "Your text..." -o summary.md
```

## 📱 What the Application Does

1. **OCR Processing**: Extracts text from images using advanced preprocessing
2. **Text Summarization**: Uses MLX-optimized language models for fast inference
3. **Multiple Inputs**: Handles images, text files, and direct text input
4. **Smart Output**: Provides formatted summaries with metadata

## 🔍 Expected Output Example

```
🔤 Text Summarizer with OCR and MLX
════════════════════════════════════════════════════════════

📊 Processing Results
┌─ Source: direct_text
├─ Original length: 485 characters
└─ Summary length: 142 characters

📋 Summary
┌─ The think tool serves as a planning scratchpad for users to ─┐
│ list applicable rules, verify information collection,         │
│ ensure policy compliance, and check tool results before      │
│ taking actions. It helps structure decision-making           │
│ processes systematically.                                    │
└─────────────────────────────────────────────────────────────┘
```

## ⚠️ Troubleshooting

- **"Tesseract not found"**: Install Tesseract OCR system-wide
- **"MLX import error"**: MLX works best on Apple Silicon, but functions on other platforms
- **"No text found in image"**: Ensure image has clear, readable text

## 📖 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [config.py](config.py) to customize settings
- Explore [demo.py](demo.py) for programming examples

Happy summarizing! 🎉