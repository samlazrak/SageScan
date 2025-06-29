"""
Utility functions for the scanned notes processor.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_image_file(file_path: str) -> bool:
    """Validate if file is a supported image format."""
    supported_formats = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif'}
    file_ext = Path(file_path).suffix.lower()
    return file_ext in supported_formats

def validate_digital_file(file_path: str) -> bool:
    """Validate if file is a supported digital format."""
    supported_formats = {'.pdf', '.docx', '.txt', '.rtf', '.md'}
    file_ext = Path(file_path).suffix.lower()
    return file_ext in supported_formats

def get_files_from_folder(folder_path: str, file_validator) -> List[str]:
    """Get all valid files from a folder."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    files = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and file_validator(file_path):
            files.append(file_path)
    
    return sorted(files)

def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """Save processing results to JSON file."""
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Results saved to: {output_path}")

def load_results(input_path: str) -> List[Dict[str, Any]]:
    """Load processing results from JSON file."""
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_output_filename(input_path: str, suffix: str = "_processed") -> str:
    """Create output filename based on input path."""
    input_path = Path(input_path)
    return str(input_path.parent / f"{input_path.stem}{suffix}.json")

def format_timestamp() -> str:
    """Get formatted timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations."""
    # Remove or replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip() 