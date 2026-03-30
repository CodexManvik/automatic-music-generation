#!/usr/bin/env python3
"""
Quick inference helper - Generate music instantly without training.
Useful if you already have a trained model.
"""

import subprocess
import sys
from pathlib import Path


def run_inference():
    """Run the inference script to generate music."""
    print("\n" + "=" * 60)
    print("🎵 Music Generation - Quick Inference")
    print("=" * 60)

    # Check if model exists
    model_path = Path("s2s/model.pth")
    if not model_path.exists():
        print("\n❌ Error: Trained model not found at s2s/model.pth")
        print("\nYou need to train the model first:")
        print("   python auto_music_gen.py")
        print("\nThis will take a few minutes but only needs to run once.")
        return False

    # Check if mappings exist
    mappings_path = Path("s2s/mappings.pkl")
    if not mappings_path.exists():
        print("\n❌ Error: Mappings file not found at s2s/mappings.pkl")
        print("\nTrain the model first: python auto_music_gen.py")
        return False

    # Run inference
    print("\n▶️  Starting inference...")
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=False,
            text=True,
        )
        if result.returncode == 0:
            print("\n✅ Success! Check pred_music.mid for your generated music.")
            return True
        else:
            print("\n❌ Inference failed!")
            return False
    except Exception as e:
        print(f"\n❌ Error running inference: {e}")
        return False


if __name__ == "__main__":
    success = run_inference()
    sys.exit(0 if success else 1)
