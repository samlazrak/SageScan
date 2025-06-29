#!/bin/bash

# ScanSage Local LLM Startup Script
# This script helps set up and start local LLM services

set -e

echo "🚀 ScanSage Local LLM Setup"
echo "============================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are available
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_success "Docker and Docker Compose found"

# Create necessary directories
print_status "Creating directories..."
mkdir -p data uploads results models

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating .env file from template..."
    cp env.example .env
    print_success ".env file created"
else
    print_success ".env file already exists"
fi

# Start the services
print_status "Starting ScanSage services with Docker Compose..."
docker-compose up -d

print_success "Services started! Waiting for them to be ready..."

# Wait for services to be healthy
print_status "Waiting for Ollama to be ready..."
timeout=120
counter=0
while [ $counter -lt $timeout ]; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama is ready!"
        break
    fi
    sleep 2
    counter=$((counter + 2))
    echo -n "."
done

if [ $counter -ge $timeout ]; then
    print_warning "Ollama took too long to start. You may need to wait a bit longer."
fi

print_status "Waiting for llama.cpp to be ready..."
counter=0
while [ $counter -lt $timeout ]; do
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        print_success "llama.cpp is ready!"
        break
    fi
    sleep 2
    counter=$((counter + 2))
    echo -n "."
done

if [ $counter -ge $timeout ]; then
    print_warning "llama.cpp took too long to start. You may need to wait a bit longer."
fi

# Download models
print_status "Setting up models..."

# Download Ollama models
print_status "Downloading Ollama models..."
docker exec scansage-ollama ollama pull llama2
docker exec scansage-ollama ollama pull mistral
docker exec scansage-ollama ollama pull codellama

print_success "Ollama models downloaded"

# Download llama.cpp models (optional)
print_status "Downloading llama.cpp models..."
if [ ! -f "models/llama-2-7b-chat.gguf" ]; then
    print_warning "llama.cpp models not found. You can download them manually:"
    echo "  - Visit: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF"
    echo "  - Download llama-2-7b-chat.gguf"
    echo "  - Place it in the models/ directory"
else
    print_success "llama.cpp models found"
fi

print_success "Local LLM setup completed!"
echo ""
echo "Services running:"
echo "  - Ollama: http://localhost:11434"
echo "  - llama.cpp: http://localhost:8080"
echo "  - ScanSage: http://localhost:8000"
echo "  - Ollama Web UI: http://localhost:3000 (optional)"
echo ""
echo "Available models:"
echo "  - llama2 (Ollama)"
echo "  - mistral (Ollama)"
echo "  - codellama (Ollama)"
echo "  - llama-2-7b-chat (llama.cpp)"
echo ""
echo "To use ScanSage with local LLMs:"
echo "  python main.py --input your_file.jpg --output result.json --llm-provider local"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f" 