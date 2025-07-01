#!/usr/bin/env python3
"""
Deep Researcher Agent - Startup Script

This script checks dependencies and launches the Streamlit application
with helpful error messages and setup guidance.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        print("Please upgrade Python: https://python.org/downloads/")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'plotly',
        'networkx',
        'requests',
        'beautifulsoup4',
        'fitz',  # PyMuPDF
        'trafilatura',
        'validators'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'fitz':
                import fitz
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("\n🔧 Installing dependencies...")
    
    try:
        # Install from requirements.txt if it exists
        if Path("requirements.txt").exists():
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully!")
                return True
            else:
                print(f"❌ Failed to install dependencies: {result.stderr}")
                return False
        else:
            print("❌ requirements.txt not found")
            print("Please ensure you're in the correct directory")
            return False
            
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        "streamlit_app.py",
        "src/deep_researcher_agent.py",
        "requirements.txt"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    return missing_files

def launch_streamlit():
    """Launch the Streamlit application"""
    print("\n🚀 Launching Deep Researcher Agent...")
    print("📱 The application will open in your default web browser")
    print("🌐 Default URL: http://localhost:8501")
    print("\n" + "="*50)
    print("Press Ctrl+C to stop the application")
    print("="*50 + "\n")
    
    try:
        # Launch Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("\n\n⏹️  Application stopped by user")
    except Exception as e:
        print(f"\n❌ Failed to launch Streamlit: {e}")
        print("\nTry running manually:")
        print("streamlit run streamlit_app.py")

def print_welcome():
    """Print welcome message and instructions"""
    print("🧠 Deep Researcher Agent")
    print("=" * 50)
    print("Advanced PDF link analysis and web content research")
    print("=" * 50)
    print()

def print_usage_info():
    """Print usage information"""
    print("\n📖 Quick Usage Guide:")
    print("1. Upload a PDF document with links/references")
    print("2. Configure analysis settings in the sidebar")
    print("3. Click 'Start Deep Research' to begin analysis")
    print("4. Explore results in interactive dashboards")
    print("\n💡 Tips:")
    print("• Start with small link limits for testing")
    print("• Use academic PDFs for best results")
    print("• Check domain filtering for focused analysis")
    print("• Enable Selenium only if needed (slower)")

def main():
    """Main startup function"""
    print_welcome()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print("\n📋 Checking required files...")
    missing_files = check_files()
    
    if missing_files:
        print(f"\n❌ Missing required files: {missing_files}")
        print("Please ensure you're in the correct directory")
        print("and all files are properly downloaded/created.")
        sys.exit(1)
    
    print("\n📦 Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        
        # Ask user if they want to install
        response = input("\nWould you like to install missing dependencies? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            if install_dependencies():
                print("\n✅ All dependencies installed!")
                
                # Re-check dependencies
                print("\n🔄 Re-checking dependencies...")
                missing_packages = check_dependencies()
                
                if missing_packages:
                    print(f"\n❌ Still missing: {missing_packages}")
                    print("Please install manually:")
                    print(f"pip install {' '.join(missing_packages)}")
                    sys.exit(1)
            else:
                print("\n❌ Installation failed. Please install manually:")
                print("pip install -r requirements.txt")
                sys.exit(1)
        else:
            print("\nPlease install dependencies manually:")
            print("pip install -r requirements.txt")
            sys.exit(1)
    
    print("\n✅ All checks passed!")
    print_usage_info()
    
    # Ask user if they want to proceed
    print("\n" + "="*50)
    response = input("Ready to launch Deep Researcher Agent? (y/n): ")
    
    if response.lower() in ['y', 'yes']:
        launch_streamlit()
    else:
        print("\n👋 Thanks! Run this script again when you're ready.")
        print("Or launch manually with: streamlit run streamlit_app.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Startup interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nFor help, please check:")
        print("• DEEP_RESEARCHER_README.md")
        print("• requirements.txt")
        print("• Python and package versions")