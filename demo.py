#!/usr/bin/env python3
"""
Demonstration script for Text Summarizer with OCR and MLX

This script shows various ways to use the text summarizer:
1. Direct text summarization
2. Text file processing
3. Batch processing example
4. Different model configurations
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path to import our module
sys.path.insert(0, str(Path(__file__).parent))

from text_summarizer import TextSummarizer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def demo_direct_text():
    """Demonstrate direct text summarization"""
    console.print(Panel.fit("📝 Demo 1: Direct Text Summarization", style="bold blue"))
    
    # Example text about the think tool (from user's request)
    think_tool_text = """## Using the think tool

Before taking any action or responding to the user after receiving tool results, use the think tool as a scratchpad to:
- List the specific rules that apply to the current request
- Check if all required information is collected
- Verify that the planned action complies with all policies
- Iterate over tool results for correctness 

Here are some examples of what to iterate over inside the think tool:

User wants to cancel flight ABC123
- Need to verify: user ID, reservation ID, reason
- Check cancellation rules:
  * Is it within 24h of booking?
  * If not, check ticket class and insurance
- Verify no segments flown or are in the past
- Plan: collect missing info, verify rules, get confirmation

User wants to book 3 tickets to NYC with 2 checked bags each
- Need user ID to check:
  * Membership tier for baggage allowance
  * Which payments methods exist in profile
- Baggage calculation:
  * Economy class × 3 passengers
  * If regular member: 1 free bag each → 3 extra bags = $150
  * If silver member: 2 free bags each → 0 extra bags = $0
  * If gold member: 3 free bags each → 0 extra bags = $0
- Payment rules to verify:
  * Max 1 travel certificate, 1 credit card, 3 gift cards
  * All payment methods must be in profile
  * Travel certificate remainder goes to waste
- Plan:
1. Get user ID
2. Verify membership level for bag fees
3. Check which payment methods in profile and if their combination is allowed
4. Calculate total: ticket price + any bag fees
5. Get explicit confirmation for booking"""

    try:
        summarizer = TextSummarizer()
        summary = summarizer.summarize_text(think_tool_text, max_tokens=100)
        
        console.print(Panel(
            f"**Original length:** {len(think_tool_text)} characters\n"
            f"**Summary length:** {len(summary)} characters\n\n"
            f"**Summary:**\n{summary}",
            title="📋 Think Tool Summary",
            style="green"
        ))
        
    except Exception as e:
        console.print(f"❌ Error in direct text demo: {e}")

def demo_text_file():
    """Demonstrate text file processing"""
    console.print(Panel.fit("📄 Demo 2: Text File Processing", style="bold blue"))
    
    # Create a sample text file
    sample_file = "sample_document.txt"
    sample_content = """Machine Learning Operations (MLOps) Best Practices

MLOps is a set of practices that combines Machine Learning, DevOps, and Data Engineering to reliably and efficiently deploy and maintain ML models in production.

Key Components:
1. Version Control: Track changes in code, data, and models
2. Continuous Integration: Automated testing for ML pipelines
3. Continuous Deployment: Automated model deployment
4. Monitoring: Track model performance and data drift
5. Reproducibility: Ensure experiments can be replicated

Benefits:
- Faster time to market for ML models
- Improved collaboration between teams
- Better model governance and compliance
- Reduced risk of model failures in production
- Scalable ML workflows

Implementation Steps:
1. Set up version control for all ML artifacts
2. Create automated testing pipelines
3. Implement model registry and deployment automation
4. Set up monitoring and alerting systems
5. Establish model retraining procedures

Common Tools:
- MLflow for experiment tracking
- Kubeflow for ML pipelines
- Docker for containerization
- Kubernetes for orchestration
- Prometheus for monitoring"""

    try:
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        
        summarizer = TextSummarizer()
        result = summarizer.process_file(sample_file, max_tokens=120)
        
        console.print(Panel(
            f"**File:** {result['file_path']}\n"
            f"**Source type:** {result['source_type']}\n"
            f"**Original length:** {len(result['original_text'])} characters\n"
            f"**Summary length:** {len(result['summary'])} characters\n\n"
            f"**Summary:**\n{result['summary']}",
            title="📊 MLOps Document Summary",
            style="cyan"
        ))
        
        # Cleanup
        os.remove(sample_file)
        
    except Exception as e:
        console.print(f"❌ Error in text file demo: {e}")

def demo_model_comparison():
    """Demonstrate different model configurations"""
    console.print(Panel.fit("🤖 Demo 3: Model Comparison", style="bold blue"))
    
    sample_text = """Artificial Intelligence (AI) is transforming industries worldwide. From healthcare to finance, AI applications are becoming increasingly sophisticated. Machine learning algorithms can now process vast amounts of data, identify patterns, and make predictions with remarkable accuracy. However, challenges remain in areas like explainability, bias mitigation, and ethical AI deployment."""
    
    models = [
        "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit",
        "mlx-community/phi-2-4bit"
    ]
    
    table = Table(title="Model Comparison Results")
    table.add_column("Model", style="cyan")
    table.add_column("Summary", style="green")
    table.add_column("Length", style="yellow")
    
    for model_path in models:
        try:
            summarizer = TextSummarizer(model_path=model_path)
            summary = summarizer.summarize_text(sample_text, max_tokens=50)
            
            model_name = model_path.split('/')[-1]
            table.add_row(model_name, summary, str(len(summary)))
            
        except Exception as e:
            console.print(f"⚠️ Could not test model {model_path}: {e}")
    
    console.print(table)

def demo_batch_processing():
    """Demonstrate batch processing of multiple text files"""
    console.print(Panel.fit("📚 Demo 4: Batch Processing", style="bold blue"))
    
    # Create multiple sample files
    files_data = {
        "ai_overview.txt": "Artificial Intelligence encompasses machine learning, natural language processing, computer vision, and robotics. These technologies are revolutionizing how we interact with computers and process information.",
        "python_guide.txt": "Python is a versatile programming language known for its simplicity and readability. It's widely used in web development, data science, artificial intelligence, and automation tasks.",
        "cloud_computing.txt": "Cloud computing provides on-demand access to computing resources over the internet. Major providers include AWS, Azure, and Google Cloud Platform, offering services like storage, compute, and databases."
    }
    
    created_files = []
    try:
        # Create sample files
        for filename, content in files_data.items():
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            created_files.append(filename)
        
        # Process all files
        summarizer = TextSummarizer()
        results = []
        
        for filename in created_files:
            try:
                result = summarizer.process_file(filename, max_tokens=30)
                results.append({
                    "file": filename,
                    "summary": result['summary'],
                    "original_length": len(result['original_text'])
                })
            except Exception as e:
                console.print(f"⚠️ Error processing {filename}: {e}")
        
        # Display results in a table
        table = Table(title="Batch Processing Results")
        table.add_column("File", style="cyan")
        table.add_column("Original Length", style="yellow")
        table.add_column("Summary", style="green")
        
        for result in results:
            table.add_row(
                result["file"],
                str(result["original_length"]),
                result["summary"]
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"❌ Error in batch processing demo: {e}")
    
    finally:
        # Cleanup created files
        for filename in created_files:
            try:
                os.remove(filename)
            except FileNotFoundError:
                pass

def main():
    """Run all demonstrations"""
    console.print(Panel.fit(
        "🚀 Text Summarizer with OCR and MLX - Demonstration",
        style="bold magenta"
    ))
    
    console.print("\n[bold yellow]Note:[/bold yellow] Some demos may take time to load models on first run.")
    console.print("Press Ctrl+C to stop at any time.\n")
    
    try:
        # Run demos
        demo_direct_text()
        console.print("\n" + "="*80 + "\n")
        
        demo_text_file()
        console.print("\n" + "="*80 + "\n")
        
        demo_batch_processing()
        console.print("\n" + "="*80 + "\n")
        
        # Model comparison is optional (may fail if models aren't available)
        try:
            demo_model_comparison()
        except Exception as e:
            console.print(f"⚠️ Model comparison demo skipped: {e}")
        
        console.print(Panel.fit(
            "✅ All demonstrations completed successfully!",
            style="bold green"
        ))
        
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Demo interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n\n❌ Demo failed: {e}")
        console.print("\nTry installing dependencies with: pip install -r requirements.txt")

if __name__ == "__main__":
    main()