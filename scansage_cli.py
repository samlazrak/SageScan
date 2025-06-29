#!/usr/bin/env python3
"""
Interactive CLI for ScanSage - AI-Powered Document Analysis
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from tabulate import tabulate

class ScanSageCLI:
    """Interactive CLI for ScanSage document analysis."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.results_dir = self.project_root / "results"
        self.examples_dir = self.project_root / "examples"
        self.uploads_dir = self.project_root / "uploads"
        self.dropbox_dir = self.project_root / "dropbox"
        
        # Ensure directories exist
        self.results_dir.mkdir(exist_ok=True)
        self.uploads_dir.mkdir(exist_ok=True)
        self.dropbox_dir.mkdir(exist_ok=True)
        
        # Set up dropbox with test files if empty
        self.setup_dropbox()
        
        # Default configuration
        self.config = {
            "llm_provider": "ollama",
            "llm_model": "tinyllama",
            "show_stats": True,
            "digital_mode": False
        }
    
    def setup_dropbox(self):
        """Set up dropbox with test files if it's empty."""
        dropbox_files = list(self.dropbox_dir.glob("*"))
        if not dropbox_files:
            print("📦 Setting up dropbox with test files...")
            
            # Copy example files to dropbox
            try:
                # Copy image file
                if (self.examples_dir / "image.png").exists():
                    import shutil
                    shutil.copy2(self.examples_dir / "image.png", self.dropbox_dir / "test_image.png")
                    print("   ✅ Added test_image.png")
                
                # Copy text file
                if (self.examples_dir / "sample_notes" / "sample_note.txt").exists():
                    import shutil
                    shutil.copy2(self.examples_dir / "sample_notes" / "sample_note.txt", self.dropbox_dir / "test_note.txt")
                    print("   ✅ Added test_note.txt")
                
                print("   📁 Dropbox ready for testing!")
                
            except Exception as e:
                print(f"   ⚠️  Could not set up test files: {e}")
                print("   💡 You can manually add files to the dropbox directory.")
    
    def run(self):
        """Main CLI loop."""
        self.print_banner()
        
        while True:
            try:
                self.show_main_menu()
                choice = input("\nEnter your choice (1-9): ").strip()
                
                if choice == "1":
                    self.process_single_file()
                elif choice == "2":
                    self.process_folder()
                elif choice == "3":
                    self.quick_test()
                elif choice == "4":
                    self.configure_settings()
                elif choice == "5":
                    self.manage_models()
                elif choice == "6":
                    self.view_results()
                elif choice == "7":
                    self.check_status()
                elif choice == "8":
                    self.show_help()
                elif choice == "9":
                    print("\n👋 Goodbye! Thanks for using ScanSage.")
                    break
                else:
                    print("\n❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using ScanSage.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                input("Press Enter to continue...")
    
    def print_banner(self):
        """Print the ScanSage banner."""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    ScanSage CLI                              ║
║              AI-Powered Document Analysis                    ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def show_main_menu(self):
        """Display the main menu."""
        menu = """
📋 MAIN MENU
══════════════════════════════════════════════════════════════

1. 📄 Process Single File
2. 📁 Process Folder (Batch)
3. ⚡ Quick Test (Auto-select files)
4. ⚙️  Configure Settings
5. 🧠 Manage LLM Models
6. 📊 View Results
7. 🔍 Check System Status
8. ❓ Help
9. 🚪 Exit

Current Settings:
   • LLM Provider: {provider}
   • Model: {model}
   • Mode: {mode}
   • Stats: {stats}
        """.format(
            provider=self.config["llm_provider"],
            model=self.config["llm_model"],
            mode="Digital" if self.config["digital_mode"] else "Image",
            stats="Enabled" if self.config["show_stats"] else "Disabled"
        )
        print(menu)
    
    def process_single_file(self):
        """Process a single file interactively."""
        print("\n📄 PROCESS SINGLE FILE")
        print("══════════════════════════════════════════════════")
        
        # File selection
        file_path = self.select_file()
        if not file_path:
            return
        
        # Determine if it's a digital file
        digital_extensions = {'.txt', '.pdf', '.docx', '.rtf', '.md'}
        is_digital = Path(file_path).suffix.lower() in digital_extensions
        
        if is_digital and not self.config["digital_mode"]:
            print(f"\n📝 Detected digital file: {Path(file_path).suffix}")
            use_digital = input("Switch to digital mode? (y/n): ").lower().startswith('y')
            if use_digital:
                self.config["digital_mode"] = True
        
        # Generate output filename
        output_file = self.generate_output_filename(file_path)
        
        # Confirm and run
        print(f"\n📋 Job Summary:")
        print(f"   Input: {file_path}")
        print(f"   Output: {output_file}")
        print(f"   Mode: {'Digital' if self.config['digital_mode'] else 'Image'}")
        print(f"   Model: {self.config['llm_model']}")
        
        confirm = input("\n🚀 Start processing? (y/n): ").lower().startswith('y')
        if not confirm:
            return
        
        self.run_job(file_path, str(output_file))
    
    def process_folder(self):
        """Process a folder of files."""
        print("\n📁 PROCESS FOLDER (BATCH)")
        print("══════════════════════════════════════════════════")
        
        # Folder selection
        folder_path = self.select_folder()
        if not folder_path:
            return
        
        # Generate output filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"results/batch_results_{timestamp}.json"
        
        # Confirm and run
        print(f"\n📋 Batch Job Summary:")
        print(f"   Input Folder: {folder_path}")
        print(f"   Output: {output_file}")
        print(f"   Mode: {'Digital' if self.config['digital_mode'] else 'Image'}")
        print(f"   Model: {self.config['llm_model']}")
        
        confirm = input("\n🚀 Start batch processing? (y/n): ").lower().startswith('y')
        if not confirm:
            return
        
        self.run_job(folder_path, output_file)
    
    def quick_test(self):
        """Quick test with auto-selected files."""
        print("\n⚡ QUICK TEST")
        print("══════════════════════════════════════════════════")
        
        # Auto-detect available test files
        test_files = []
        
        # Check for image files in dropbox
        image_files = list(self.dropbox_dir.glob("*.png")) + list(self.dropbox_dir.glob("*.jpg")) + list(self.dropbox_dir.glob("*.jpeg"))
        if image_files:
            test_files.append(("Image", str(image_files[0].relative_to(self.project_root))))
        
        # Check for text files in dropbox
        text_files = list(self.dropbox_dir.glob("*.txt")) + list(self.dropbox_dir.glob("*.pdf"))
        if text_files:
            test_files.append(("Digital", str(text_files[0].relative_to(self.project_root))))
        
        if not test_files:
            print("❌ No test files found in dropbox directory!")
            print("💡 Add some files to the dropbox directory for testing.")
            return
        
        print("🔍 Auto-detected test files:")
        for i, (file_type, file_path) in enumerate(test_files, 1):
            print(f"   {i}. {file_type}: {file_path}")
        
        # Run tests
        for file_type, file_path in test_files:
            print(f"\n🚀 Testing {file_type} file: {file_path}")
            
            # Set mode based on file type
            original_mode = self.config["digital_mode"]
            self.config["digital_mode"] = (file_type == "Digital")
            
            # Generate output filename
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = f"results/quick_test_{file_type.lower()}_{timestamp}.json"
            
            # Run the test
            self.run_job(file_path, output_file)
            
            # Restore original mode
            self.config["digital_mode"] = original_mode
        
        print("\n✅ Quick test completed!")
        print("📊 Check the results directory for output files.")
    
    def select_file(self) -> Optional[str]:
        """Interactive file selection."""
        print("\n📂 File Selection:")
        print("1. Browse examples directory")
        print("2. Browse uploads directory")
        print("3. Browse dropbox directory")
        print("4. Enter custom path")
        print("5. Cancel")
        
        choice = input("Choose option (1-5): ").strip()
        
        if choice == "1":
            return self.browse_directory(self.examples_dir, "examples")
        elif choice == "2":
            return self.browse_directory(self.uploads_dir, "uploads")
        elif choice == "3":
            return self.browse_directory(self.dropbox_dir, "dropbox")
        elif choice == "4":
            path = input("Enter file path: ").strip()
            if os.path.exists(path):
                return path
            else:
                print("❌ File not found!")
                return None
        else:
            return None
    
    def select_folder(self) -> Optional[str]:
        """Interactive folder selection."""
        print("\n📂 Folder Selection:")
        print("1. Browse examples directory")
        print("2. Browse uploads directory")
        print("3. Browse dropbox directory")
        print("4. Enter custom path")
        print("5. Cancel")
        
        choice = input("Choose option (1-5): ").strip()
        
        if choice == "1":
            return self.browse_directory(self.examples_dir, "examples", is_folder=True)
        elif choice == "2":
            return self.browse_directory(self.uploads_dir, "uploads", is_folder=True)
        elif choice == "3":
            return self.browse_directory(self.dropbox_dir, "dropbox", is_folder=True)
        elif choice == "4":
            path = input("Enter folder path: ").strip()
            if os.path.isdir(path):
                return path
            else:
                print("❌ Folder not found!")
                return None
        else:
            return None
    
    def browse_directory(self, directory: Path, name: str, is_folder: bool = False) -> Optional[str]:
        """Browse a directory for files or folders."""
        if not directory.exists():
            print(f"❌ {name} directory not found!")
            return None
        
        items = []
        if is_folder:
            items = [d for d in directory.iterdir() if d.is_dir()]
        else:
            # Get files based on mode
            if self.config["digital_mode"]:
                extensions = {'.txt', '.pdf', '.docx', '.rtf', '.md'}
            else:
                extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.bmp', '.gif'}
            
            items = [f for f in directory.iterdir() 
                    if f.is_file() and f.suffix.lower() in extensions]
        
        if not items:
            print(f"❌ No {'folders' if is_folder else 'files'} found in {name} directory!")
            return None
        
        print(f"\n📂 Available {'folders' if is_folder else 'files'} in {name}:")
        for i, item in enumerate(items, 1):
            print(f"   {i}. {item.name}")
        
        try:
            choice = int(input(f"\nSelect {'folder' if is_folder else 'file'} (1-{len(items)}): "))
            if 1 <= choice <= len(items):
                selected_item = items[choice - 1]
                # Return the relative path that will work with the Docker container
                return str(selected_item.relative_to(self.project_root))
        except ValueError:
            pass
        
        return None
    
    def configure_settings(self):
        """Configure ScanSage settings."""
        print("\n⚙️  CONFIGURE SETTINGS")
        print("══════════════════════════════════════════════════")
        
        while True:
            print(f"\nCurrent Settings:")
            print(f"1. LLM Provider: {self.config['llm_provider']}")
            print(f"2. LLM Model: {self.config['llm_model']}")
            print(f"3. Processing Mode: {'Digital' if self.config['digital_mode'] else 'Image'}")
            print(f"4. Show Statistics: {'Yes' if self.config['show_stats'] else 'No'}")
            print("5. Back to main menu")
            
            choice = input("\nSelect setting to change (1-5): ").strip()
            
            if choice == "1":
                self.config["llm_provider"] = self.select_llm_provider()
            elif choice == "2":
                self.config["llm_model"] = self.select_llm_model()
            elif choice == "3":
                self.config["digital_mode"] = not self.config["digital_mode"]
                print(f"✅ Mode changed to: {'Digital' if self.config['digital_mode'] else 'Image'}")
            elif choice == "4":
                self.config["show_stats"] = not self.config["show_stats"]
                print(f"✅ Statistics: {'Enabled' if self.config['show_stats'] else 'Disabled'}")
            elif choice == "5":
                break
    
    def select_llm_provider(self) -> str:
        """Select LLM provider."""
        print("\n🤖 LLM Provider Selection:")
        print("1. ollama (local)")
        print("2. llama-cpp (local)")
        print("3. openai (cloud)")
        print("4. anthropic (cloud)")
        
        choice = input("Select provider (1-4): ").strip()
        
        providers = {
            "1": "ollama",
            "2": "llama-cpp", 
            "3": "openai",
            "4": "anthropic"
        }
        
        return providers.get(choice, "ollama")
    
    def select_llm_model(self) -> str:
        """Select LLM model."""
        print("\n🧠 LLM Model Selection:")
        print("1. tinyllama (637 MB - Fast)")
        print("2. llama2:7b (3.8 GB - Better quality)")
        print("3. llama2:latest (3.8 GB - Latest)")
        print("4. Custom model name")
        
        choice = input("Select model (1-4): ").strip()
        
        models = {
            "1": "tinyllama",
            "2": "llama2:7b",
            "3": "llama2:latest"
        }
        
        if choice in models:
            return models[choice]
        elif choice == "4":
            return input("Enter custom model name: ").strip()
        else:
            return "tinyllama"
    
    def manage_models(self):
        """Manage LLM models in Ollama."""
        print("\n🧠 MANAGE LLM MODELS")
        print("══════════════════════════════════════════════════")
        
        # Check if Ollama is running
        if not self.check_ollama_status():
            print("❌ Ollama is not running. Start it first with: docker-compose up -d ollama")
            return
        
        while True:
            print("\n1. List available models")
            print("2. Pull new model")
            print("3. Remove model")
            print("4. Back to main menu")
            
            choice = input("\nSelect option (1-4): ").strip()
            
            if choice == "1":
                self.list_models()
            elif choice == "2":
                self.pull_model()
            elif choice == "3":
                self.remove_model()
            elif choice == "4":
                break
    
    def list_models(self):
        """List available models in Ollama."""
        print("\n📋 Available Models:")
        try:
            result = subprocess.run(
                ["docker-compose", "exec", "-T", "ollama", "ollama", "list"],
                capture_output=True, text=True, check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error listing models: {e}")
    
    def pull_model(self):
        """Pull a new model."""
        model_name = input("Enter model name to pull (e.g., tinyllama): ").strip()
        if not model_name:
            return
        
        print(f"\n📥 Pulling {model_name}...")
        try:
            result = subprocess.run(
                ["docker-compose", "exec", "-T", "ollama", "ollama", "pull", model_name],
                capture_output=True, text=True, check=True
            )
            print("✅ Model pulled successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error pulling model: {e}")
    
    def remove_model(self):
        """Remove a model."""
        model_name = input("Enter model name to remove: ").strip()
        if not model_name:
            return
        
        confirm = input(f"⚠️  Are you sure you want to remove {model_name}? (y/n): ").lower().startswith('y')
        if not confirm:
            return
        
        try:
            result = subprocess.run(
                ["docker-compose", "exec", "-T", "ollama", "ollama", "rm", model_name],
                capture_output=True, text=True, check=True
            )
            print("✅ Model removed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error removing model: {e}")
    
    def view_results(self):
        """View processing results."""
        print("\n📊 VIEW RESULTS")
        print("══════════════════════════════════════════════════")
        
        if not self.results_dir.exists():
            print("❌ No results directory found!")
            return
        
        result_files = list(self.results_dir.glob("*.json"))
        if not result_files:
            print("❌ No result files found!")
            return
        
        print(f"\n📂 Found {len(result_files)} result files:")
        for i, file in enumerate(result_files, 1):
            print(f"   {i}. {file.name}")
        
        try:
            choice = int(input(f"\nSelect file to view (1-{len(result_files)}): "))
            if 1 <= choice <= len(result_files):
                self.display_result(result_files[choice - 1])
        except ValueError:
            pass
    
    def display_result(self, result_file: Path):
        """Display a result file with pretty print and save pretty output."""
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            print(f"\n📄 Results from {result_file.name}:")
            print("══════════════════════════════════════════════════")
            
            # Ask user for pretty or raw view
            view_mode = input("View as (1) Pretty Table, (2) Raw JSON [1/2]? ").strip() or "1"
            if view_mode == "2":
                print(json.dumps(data, indent=2))
            else:
                self.pretty_print_results(data)
            
            # Save pretty and JSON outputs
            self.save_pretty_and_json_outputs(result_file, data)
            print(f"\n✅ Pretty and JSON outputs saved to results/prettyprint_*.txt and results/json_*.json")
        except Exception as e:
            print(f"❌ Error reading result file: {e}")
            input("\n⏸️  Press Enter to continue...")
    
    def pretty_print_results(self, data):
        """Pretty print results as a table and details."""
        if isinstance(data, list):
            # Table summary
            table = []
            for item in data:
                table.append([
                    item.get('filename', ''),
                    item.get('sentiment', ''),
                    item.get('sentiment_score', ''),
                    item.get('confidence', ''),
                    item.get('ocr_confidence', ''),
                    item.get('word_count', ''),
                    item.get('summary', '')[:40] + ('...' if len(item.get('summary', '')) > 40 else '')
                ])
            headers = ["File", "Sentiment", "Score", "LLM Conf.", "OCR Conf.", "Words", "Summary"]
            print(tabulate(table, headers, tablefmt="fancy_grid"))
            # Details
            for i, item in enumerate(data, 1):
                print(f"\n--- Result {i} ---")
                self.pretty_print_single(item)
        else:
            self.pretty_print_single(data)
    
    def pretty_print_single(self, result: Dict[str, Any]):
        """Pretty print a single result."""
        print(f"📁 File: {result.get('filename', 'Unknown')}")
        print(f"📝 Summary: {result.get('summary', 'N/A')}")
        print(f"😊 Sentiment: {result.get('sentiment', 'N/A')} (score: {result.get('sentiment_score', 'N/A')})")
        print(f"🎯 Key Topics: {', '.join(result.get('key_topics', []))}")
        print(f"✅ Action Items: {', '.join(result.get('action_items', []))}")
        print(f"📊 LLM Confidence: {result.get('confidence', 'N/A')}")
        print(f"🔍 OCR Confidence: {result.get('ocr_confidence', 'N/A')}")
        print(f"📈 Word Count: {result.get('word_count', 'N/A')}")
        print(f"⏱️  Processing Time: {result.get('processing_time', 'N/A')}s")
        print(f"🤖 Model Used: {result.get('model_used', 'N/A')}")
        if result.get('error'):
            print(f"❌ Error: {result.get('error')}")
    
    def save_pretty_and_json_outputs(self, result_file: Path, data):
        """Save pretty and JSON outputs to results/prettyprint_*.txt and results/json_*.json."""
        base = result_file.stem.replace(' ', '_')
        pretty_path = Path("results") / f"prettyprint_{base}.txt"
        json_path = Path("results") / f"json_{base}.json"
        # Save pretty
        with open(pretty_path, 'w') as f:
            from io import StringIO
            buf = StringIO()
            if isinstance(data, list):
                table = []
                for item in data:
                    table.append([
                        item.get('filename', ''),
                        item.get('sentiment', ''),
                        item.get('sentiment_score', ''),
                        item.get('confidence', ''),
                        item.get('ocr_confidence', ''),
                        item.get('word_count', ''),
                        item.get('summary', '')[:40] + ('...' if len(item.get('summary', '')) > 40 else '')
                    ])
                headers = ["File", "Sentiment", "Score", "LLM Conf.", "OCR Conf.", "Words", "Summary"]
                buf.write(tabulate(table, headers, tablefmt="fancy_grid"))
                for i, item in enumerate(data, 1):
                    buf.write(f"\n--- Result {i} ---\n")
                    for k, v in item.items():
                        buf.write(f"{k}: {v}\n")
            else:
                for k, v in data.items():
                    buf.write(f"{k}: {v}\n")
            f.write(buf.getvalue())
        # Save JSON
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def check_status(self):
        """Check system status."""
        print("\n🔍 SYSTEM STATUS")
        print("══════════════════════════════════════════════════")
        
        # Check Docker containers
        print("🐳 Docker Containers:")
        try:
            result = subprocess.run(
                ["docker-compose", "ps"],
                capture_output=True, text=True, check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error checking containers: {e}")
        
        # Check Ollama status
        print("\n🤖 Ollama Status:")
        if self.check_ollama_status():
            print("✅ Ollama is running")
            self.list_models()
        else:
            print("❌ Ollama is not running")
    
    def check_ollama_status(self) -> bool:
        """Check if Ollama is running."""
        try:
            result = subprocess.run(
                ["docker-compose", "exec", "-T", "ollama", "ollama", "list"],
                capture_output=True, text=True, check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def show_help(self):
        """Show help information."""
        help_text = """
❓ HELP
══════════════════════════════════════════════════

ScanSage is an AI-powered document analysis system that combines:
• OCR (Optical Character Recognition) for text extraction
• Text preprocessing for cleaning and normalization  
• LLM analysis for sentiment, summarization, and insights

Key Features:
• Process images (JPG, PNG, TIFF) and digital files (PDF, DOCX, TXT)
• Local LLM support via Ollama and llama.cpp
• Batch processing capabilities
• Interactive CLI interface

Getting Started:
1. Ensure Docker and Docker Compose are installed
2. Start services: docker-compose up -d ollama
3. Pull a model: docker-compose exec ollama ollama pull tinyllama
4. Run the CLI: python scansage_cli.py

For more information, see the README.md file.
        """
        print(help_text)
    
    def generate_output_filename(self, input_path: str) -> str:
        """Generate output filename based on input."""
        input_name = Path(input_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # Return container path
        return f"results/{input_name}_{timestamp}.json"
    
    def run_job(self, input_path: str, output_path: str):
        """Run a ScanSage job."""
        print(f"\n🚀 Starting ScanSage job...")
        print(f"⏳ This may take a few minutes...")
        
        # Convert host paths to container paths
        container_input = input_path
        container_output = output_path
        
        # If the path starts with a directory we mount, convert to container path
        if input_path.startswith('dropbox/'):
            container_input = f"dropbox/{input_path[8:]}"
        elif input_path.startswith('examples/'):
            container_input = f"examples/{input_path[9:]}"
        elif input_path.startswith('uploads/'):
            container_input = f"uploads/{input_path[8:]}"
        
        # Output should always be in results directory
        if not output_path.startswith('results/'):
            container_output = f"results/{Path(output_path).name}"
        
        # Build command
        cmd = [
            "./run_scansage_job.sh",
            "--input", container_input,
            "--output", container_output,
            "--llm-provider", self.config["llm_provider"],
            "--llm-model", self.config["llm_model"]
        ]
        
        if self.config["digital_mode"]:
            cmd.append("--digital")
        
        if self.config["show_stats"]:
            cmd.append("--stats")
        
        try:
            # Run the job
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            print("\n✅ Job completed successfully!")
            print(f"📄 Results saved to: {output_path}")
            
            # Show summary if stats were enabled
            if self.config["show_stats"]:
                print("\n📊 Processing Summary:")
                for line in result.stdout.split('\n'):
                    if any(keyword in line for keyword in ['Total files:', 'Success rate:', 'Avg processing time:']):
                        print(f"   {line}")
            
            # Save pretty and JSON outputs
            # Try to find the output file
            result_file = Path(output_path)
            if not result_file.exists() and Path("results").exists():
                # Try relative to results/
                result_file = Path("results") / Path(output_path).name
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                self.save_pretty_and_json_outputs(result_file, data)
                print(f"\n✅ Pretty and JSON outputs saved to results/prettyprint_*.txt and results/json_*.json")
            else:
                print(f"⚠️  Could not find output file to save pretty/JSON outputs.")
            
            # Ask if user wants to view results
            view_results = input("\n👀 View results now? (y/n): ").lower().startswith('y')
            if view_results:
                self.display_result(result_file)
                
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Job failed with error:")
            print(e.stderr)
            input("\n⏸️  Press Enter to continue...")
        except FileNotFoundError:
            print("\n❌ run_scansage_job.sh not found! Make sure it's executable.")
            print("Run: chmod +x run_scansage_job.sh")
            input("\n⏸️  Press Enter to continue...")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ScanSage Interactive CLI")
    parser.add_argument("--config", help="Path to configuration file")
    args = parser.parse_args()
    
    cli = ScanSageCLI()
    cli.run()

if __name__ == "__main__":
    main() 