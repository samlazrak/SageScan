# Local LLM Setup Guide

This guide explains how to set up and use ScanSage with local LLMs using Ollama and llama.cpp, eliminating the need for API keys.

## Overview

ScanSage now supports local LLM inference through:
- **Ollama**: Easy-to-use local LLM server with many pre-trained models
- **llama.cpp**: High-performance inference server for GGUF models
- **Docker Compose**: Automated setup and orchestration

## Quick Start

### 1. Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM (16GB recommended)
- 10GB+ free disk space for models

### 2. Automated Setup

```bash
# Clone the repository
git clone <repository-url>
cd ScanSage

# Run the automated setup script
./start_local_llm.sh
```

This script will:
- Create necessary directories
- Start all services with Docker Compose
- Download popular models
- Wait for services to be ready

### 3. Manual Setup

If you prefer manual setup:

```bash
# Create directories
mkdir -p data uploads results models

# Copy environment file
cp env.example .env

# Start services
docker-compose up -d

# Wait for services to be ready
docker-compose logs -f
```

## Services

### Ollama (Port 11434)
- **Purpose**: Primary local LLM server
- **Models**: llama2, mistral, codellama, and many more
- **API**: RESTful API compatible with OpenAI format
- **Web UI**: Available at http://localhost:3000 (optional)

### llama.cpp (Port 8080)
- **Purpose**: High-performance inference for GGUF models
- **Models**: Any GGUF format model
- **Performance**: Optimized for CPU/GPU inference
- **Memory**: Efficient memory usage

### ScanSage (Port 8000)
- **Purpose**: Main application
- **Features**: OCR, text processing, LLM analysis
- **Integration**: Connects to both Ollama and llama.cpp

## Available Models

### Ollama Models
```bash
# List available models
docker exec scansage-ollama ollama list

# Pull additional models
docker exec scansage-ollama ollama pull llama2:7b
docker exec scansage-ollama ollama pull mistral:7b
docker exec scansage-ollama ollama pull codellama:7b
docker exec scansage-ollama ollama pull phi:2.7b
```

### llama.cpp Models
Download GGUF models from Hugging Face:
- [Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF)
- [Mistral-7B-Instruct-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
- [CodeLlama-7B-Instruct-GGUF](https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF)

Place downloaded `.gguf` files in the `models/` directory.

## Configuration

### Environment Variables

Edit `.env` file to customize settings:

```env
# LLM Provider Configuration
DEFAULT_LLM_PROVIDER=local
DEFAULT_LLM_MODEL=llama2

# Local LLM Server URLs
OLLAMA_BASE_URL=http://localhost:11434
LLAMA_CPP_BASE_URL=http://localhost:8080

# Model Configuration
TEMPERATURE=0.3
MAX_TOKENS=500
MAX_TEXT_LENGTH=4000

# Application Configuration
LOG_LEVEL=INFO
UPLOAD_DIR=./uploads
RESULTS_DIR=./results
DATA_DIR=./data
```

### Provider Options

- `local` or `ollama`: Use Ollama server
- `llama-cpp`: Use llama.cpp server
- `openai`: Use OpenAI API (requires API key)
- `anthropic`: Use Anthropic API (requires API key)

## Usage

### Command Line

```bash
# Process with Ollama (default)
python main.py --input scan.jpg --output result.json --llm-provider local

# Process with specific model
python main.py --input scan.jpg --output result.json --llm-provider ollama --llm-model mistral

# Process with llama.cpp
python main.py --input scan.jpg --output result.json --llm-provider llama-cpp --llm-model llama-2-7b-chat

# Process with custom server URL
python main.py --input scan.jpg --output result.json --llm-provider ollama --server-url http://192.168.1.100:11434
```

### Python API

```python
from src.note_processor import NoteProcessor

# Initialize with local LLM
processor = NoteProcessor(
    llm_provider="local",
    llm_model="llama2",
    server_url="http://localhost:11434"
)

# Process a note
result = processor.process_single_note("scan.jpg")
print(f"Summary: {result.summary}")
print(f"Sentiment: {result.sentiment}")
```

## Performance Optimization

### Hardware Requirements

| Model Size | RAM | GPU | Performance |
|------------|-----|-----|-------------|
| 7B         | 8GB | CPU | Good        |
| 13B        | 16GB| CPU | Fair        |
| 7B         | 8GB | GPU | Excellent   |
| 13B        | 16GB| GPU | Excellent   |

### Docker Compose Optimization

Edit `docker-compose.yml` for your hardware:

```yaml
# For GPU support
services:
  ollama:
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    runtime: nvidia

  llama-cpp:
    environment:
      - N_GPU=1
      - N_THREADS=8
```

### Model Selection

- **llama2**: Good general-purpose model
- **mistral**: Excellent reasoning and analysis
- **codellama**: Best for technical documents
- **phi**: Fast and efficient for simple tasks

## Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   # Check logs
   docker-compose logs -f
   
   # Restart services
   docker-compose down
   docker-compose up -d
   ```

2. **Out of memory**
   ```bash
   # Use smaller model
   docker exec scansage-ollama ollama pull llama2:7b
   
   # Or reduce Docker memory limit
   # Edit docker-compose.yml
   ```

3. **Slow inference**
   ```bash
   # Check if GPU is being used
   docker exec scansage-ollama nvidia-smi
   
   # Optimize llama.cpp settings
   # Edit docker-compose.yml N_THREADS and N_GPU
   ```

4. **Model not found**
   ```bash
   # List available models
   docker exec scansage-ollama ollama list
   
   # Pull missing model
   docker exec scansage-ollama ollama pull <model-name>
   ```

### Health Checks

```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Check llama.cpp
curl http://localhost:8080/health

# Check ScanSage
curl http://localhost:8000/health
```

## Advanced Configuration

### Custom Models

1. **Ollama Custom Model**
   ```bash
   # Create Modelfile
   FROM llama2:7b
   TEMPLATE """{{ .Prompt }}"""
   
   # Build and run
   docker exec scansage-ollama ollama create custom -f Modelfile
   ```

2. **llama.cpp Custom Model**
   ```bash
   # Download GGUF model
   wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf -O models/custom.gguf
   
   # Update docker-compose.yml
   command: --model /models/custom.gguf
   ```

### Load Balancing

For high-throughput applications:

```yaml
services:
  ollama-1:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
  
  ollama-2:
    image: ollama/ollama:latest
    ports:
      - "11435:11434"
  
  scansage:
    environment:
      - OLLAMA_BASE_URL=http://ollama-1:11434
      - OLLAMA_BASE_URL_2=http://ollama-2:11434
```

## Monitoring

### Logs
```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f ollama
docker-compose logs -f llama-cpp
docker-compose logs -f scansage
```

### Metrics
```bash
# Ollama metrics
curl http://localhost:11434/api/tags

# llama.cpp metrics
curl http://localhost:8080/metrics
```

## Security

### Network Security
- Services are exposed on localhost only
- Use reverse proxy for external access
- Implement authentication if needed

### Model Security
- Download models from trusted sources
- Verify model checksums
- Use isolated containers

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker logs
3. Check model compatibility
4. Verify hardware requirements

## Migration from API-based Setup

If you're migrating from API keys:

1. **Update environment**
   ```bash
   # Change provider
   sed -i 's/DEFAULT_LLM_PROVIDER=openai/DEFAULT_LLM_PROVIDER=local/' .env
   ```

2. **Update code**
   ```python
   # Old
   processor = NoteProcessor(llm_provider="openai", api_key="sk-...")
   
   # New
   processor = NoteProcessor(llm_provider="local")
   ```

3. **Test setup**
   ```bash
   python test_installation.py
   ``` 