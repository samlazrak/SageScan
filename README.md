# ScanSage

A tool that reads text from scanned documents and analyzes it using language models. Built this to help process handwritten notes and scanned papers - turns out it's pretty useful for extracting insights from any kind of document.

## What it does

- Reads text from images using OCR (works with handwriting too, though results vary)
- Cleans up the extracted text to make it more readable
- Runs sentiment analysis, summarization, and topic extraction using local LLMs
- Handles images (JPG, PNG, TIFF) and digital files (PDF, DOCX, TXT)
- Can process single files or entire folders

The main advantage is running everything locally with Ollama - no need to send your documents to external APIs.

## Getting started

You'll need Docker installed. That's it.

### Basic setup

```bash
git clone <repository-url>
cd ScanSage

# Start the services
docker-compose up -d ollama

# Download a model (this one's small and fast)
docker-compose exec ollama ollama pull tinyllama
```

### Try it out

```bash
# Make the script executable
chmod +x run_scansage_job.sh

# Process an image
./run_scansage_job.sh \
  --input examples/image.png \
  --output results/analysis.json \
  --llm-provider ollama \
  --llm-model tinyllama \
  --stats
```

Check `results/analysis.json` for the extracted text and analysis.

## More examples

```bash
# Process text files instead of images
./run_scansage_job.sh \
  --input my_notes.txt \
  --output results/text_analysis.json \
  --digital

# Process a whole folder of images
./run_scansage_job.sh \
  --input photos/ \
  --output results/batch.json

# Use a bigger model (needs more RAM)
./run_scansage_job.sh \
  --input document.png \
  --output results/detailed.json \
  --llm-model llama2:7b
```

## Models and memory

| Model | Download Size | RAM Needed | Notes |
|-------|---------------|------------|-------|
| `tinyllama` | 637 MB | ~2 GB | Good for testing |
| `llama2:7b` | 3.8 GB | ~8 GB | Better quality |

```bash
# See what models you have
docker-compose exec ollama ollama list

# Get a new one
docker-compose exec ollama ollama pull llama2:7b
```

## When things break

**Out of memory?**
```bash
# Use the tiny model
docker-compose exec ollama ollama pull tinyllama
```

**Containers not starting?**
```bash
# Check what's happening
docker-compose ps
docker-compose logs scansage
```

**OCR not working well?**
The OCR works best with clear, high-contrast text. Handwriting recognition is hit-or-miss depending on legibility.

## Running without Docker

If you prefer to run everything locally:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download language data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
python -m spacy download en_core_web_sm

# Run it
python main.py --input your_file.jpg --output results.json
```

## Configuration

The `docker-compose.yml` file has some settings you can tweak:

```yaml
environment:
  - LOG_LEVEL=INFO  # Change to DEBUG if something's not working
  - DEFAULT_LLM_MODEL=tinyllama
  - OLLAMA_BASE_URL=http://ollama:11434
```

## Python API

```python
from src.note_processor import NoteProcessor

processor = NoteProcessor(
    llm_provider="ollama",
    llm_model="tinyllama"
)

result = processor.process_single_note("my_image.jpg")
print(result['sentiment'])
```

## Contributing

Found a bug or want to add something? Pull requests are welcome. The code could probably use some cleanup in places.

## License

MIT License - do whatever you want with it.

---

**Heads up**: The language models need a decent amount of RAM. If you're on a laptop, stick with `tinyllama`. For production stuff, you might want to use cloud APIs instead. 