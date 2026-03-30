#!/usr/bin/env python3
"""
Quick setup script for the modernized music generation project.
Run this after installing Python 3.11+
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(e.stderr)
        return False

def main():
    """Main setup function."""
    print("🎵 Music Generation Project - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create s2s directory if it doesn't exist
    s2s_dir = Path("s2s")
    if not s2s_dir.exists():
        s2s_dir.mkdir()
        print("✅ Created s2s/ directory for model weights")
    
    # Install requirements
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies"
    ):
        print("\n⚠️  Installation had some issues, but continuing...")
    
    # Verify imports
    print("\n🔍 Verifying imports...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} ready")
        print(f"   GPU available: {torch.cuda.is_available()}")
        
        import music21
        print(f"✅ music21 ready")
        
        import numpy
        print(f"✅ NumPy ready")
        
        import sklearn
        print(f"✅ scikit-learn ready")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✨ Setup complete! You can now run:")
    print("   python auto_music_gen.py")
    print("=" * 50)

if __name__ == "__main__":
    main()
