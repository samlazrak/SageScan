# SageScan - Experimental AI Note Scanner

## About
**First attempt at AI Agentic Note scanner and sentiment analysis**

This is an **experimental project** exploring OCR, text summarization, and sentiment analysis using local AI models.

## What This Project Actually Is
- **Experimental/Proof of Concept**: This is a learning project, not production software
- **Local AI Integration**: Uses MLX framework for local model inference
- **OCR + Summarization**: Combines Tesseract OCR with AI-powered text summarization
- **Learning Exercise**: Built to understand AI/ML integration patterns

## Current Capabilities
- ✅ Image-to-text conversion using Tesseract OCR
- ✅ Text summarization using local MLX models
- ✅ Basic sentiment analysis
- ✅ Web interface for file upload
- ✅ Support for multiple file formats

## Technologies Used
- Python
- OpenCV (image processing)
- Tesseract OCR
- MLX (local AI models)
- spaCy (NLP)
- Flask (web interface)

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python text_summarizer.py -i your_image.jpg -o summary.md
```

## Available Models
- `tiny`: TinyLlama-1.1B (fastest, lower quality)
- `phi`: Phi-2 (balanced performance)
- `mistral`: Mistral-7B (higher quality, slower)

## Project Status
- **Development Phase**: Early experimental stage
- **Purpose**: Learning AI/ML integration
- **Production Ready**: No
- **Accuracy Claims**: Not validated

## Key Learnings
- Local AI model integration
- OCR preprocessing techniques
- Text summarization implementation
- Web application development with AI features

## Future Improvements
- Better error handling
- More robust OCR preprocessing
- Comprehensive testing
- Performance optimization

## Note for Interviewers
This project demonstrates my interest in AI/ML and willingness to experiment with new technologies. It's a learning exercise that shows my ability to integrate multiple AI components, but should not be considered production-ready software.
