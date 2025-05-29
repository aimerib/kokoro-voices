#!/usr/bin/env python3
"""
Auto-generated optimized training script.
Generated from hyperparameter optimization using known Kokoro voices.
"""

import subprocess
import sys

def run_optimized_training():
    """Run training with auto-tuned optimal parameters."""
    
    cmd = [
        sys.executable, "training_styletts2.py",
        "--data", "./output/kokoro-dataset",
        "--name", "my_voice_optimized",
        "--epochs-projection", "150",  # More epochs since we have optimal params
        "--lr-projection", "0.00031304671498063696",
        "--max-style-samples", "100",
        "--wandb",
        "--log-audio-every", "5",
    ]
    
    print("🚀 Running OPTIMIZED StyleTTS2 → Kokoro Training")
    print("="*60)
    print("Using auto-tuned hyperparameters from synthetic voice optimization")
    print()
    print("Optimized parameters:")
    print(f"  Learning rate: 0.00031304671498063696")
    print(f"  Hidden dim: 512")
    print(f"  Num layers: 2")
    print(f"  Dropout: 0.1617133731590296")
    print(f"  Activation: relu")
    print(f"  Normalization: layer")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Optimized training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    run_optimized_training()
