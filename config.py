"""
Configuration file for Text Summarizer application
"""

# MLX Model Configuration
DEFAULT_MODELS = {
    "tiny": "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit",
    "small": "mlx-community/Llama-2-7b-chat-hf-4bit",
    "phi": "mlx-community/phi-2-4bit",
    "mistral": "mlx-community/Mistral-7B-Instruct-v0.1-4bit"
}

# OCR Configuration
OCR_CONFIG = {
    "tesseract_config": "--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?:;()[]{}\"' -",
    "preprocessing": {
        "denoise": True,
        "threshold": True,
        "resize_factor": 2.0
    }
}

# Summarization Settings
SUMMARIZATION_CONFIG = {
    "max_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "system_prompt": """You are a helpful assistant that creates concise, accurate summaries. 
Focus on the main points and key information while maintaining clarity."""
}

# File Processing
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
SUPPORTED_TEXT_FORMATS = ['.txt', '.md', '.rst', '.csv']

# Output Configuration
OUTPUT_CONFIG = {
    "include_metadata": True,
    "include_original_text": False,  # Set to True to include original text in output
    "markdown_format": True
}