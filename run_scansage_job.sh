#!/bin/bash
# run_scansage_job.sh - Run a one-off ScanSage job for testing or batch processing
#
# Usage:
#   ./run_scansage_job.sh --input <input_path> --output <output_path> [extra args]
#
# If no arguments are provided, defaults to sample note and results file.
#
# Example:
#   ./run_scansage_job.sh --input examples/sample_notes/sample_note.txt --output results/sample_note_sentiment.json --stats

set -e

# Default values
INPUT="examples/sample_notes/sample_note.txt"
OUTPUT="results/sample_note_sentiment.json"
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --input)
      INPUT="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --help|-h)
      echo "Usage: $0 --input <input_path> --output <output_path> [extra args]"
      echo "Defaults:"
      echo "  --input $INPUT"
      echo "  --output $OUTPUT"
      echo "Any extra arguments will be passed to main.py."
      exit 0
      ;;
    *)
      EXTRA_ARGS+=" $1"
      shift
      ;;
  esac
done

# Stop the scansage container if running
if docker-compose ps | grep -q scansage-app; then
  echo "Stopping running scansage container..."
  docker-compose stop scansage
fi

# Run the one-off job
CMD="docker-compose run --rm scansage python main.py --input $INPUT --output $OUTPUT $EXTRA_ARGS"
echo "Running: $CMD"
if $CMD; then
  echo "\nScanSage job completed successfully."
  echo "Output saved to: $OUTPUT"
else
  echo "\nScanSage job failed. Check the logs above."
  exit 1
fi 