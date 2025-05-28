#!/usr/bin/env python3
"""
Improved StyleTTS2 → Kokoro training with optimized parameters.

Based on diagnostic results showing StyleTTS2 is working well,
this script uses better training parameters for the projection layer.
"""

import subprocess
import sys
from pathlib import Path

def run_improved_training():
    """Run training with improved parameters."""
    
    print("🚀 Running Improved StyleTTS2 → Kokoro Training")
    print("="*60)
    
    # Improved parameters based on diagnostic results
    cmd = [
        sys.executable, "training_styletts2.py",
        "--data", "./output/kokoro-dataset",
        "--name", "my_voice_improved",
        "--epochs-projection", "100",  # More epochs since we have good embeddings
        "--lr-projection", "1e-3",     # Higher learning rate (original was too low)
        "--max-style-samples", "100",  # Fewer samples to avoid over-averaging
        "--wandb",                     # Enable logging
        "--log-audio-every", "5",      # More frequent audio logging
    ]
    
    print("Training command:")
    print(" ".join(cmd))
    print()
    
    # Run the training
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return False

def run_audio_feature_fallback():
    """Run training with audio feature fallback for comparison."""
    
    print("\n🔄 Running Audio Feature Fallback Training (for comparison)")
    print("="*60)
    
    # Temporarily disable StyleTTS2 to force fallback
    cmd = [
        sys.executable, "-c", 
        """
import subprocess
import sys
import os

# Temporarily rename styletts2 import to force fallback
original_path = sys.path[:]
sys.modules['styletts2'] = None

# Run training
cmd = [
    sys.executable, "training_styletts2.py",
    "--data", "./output/kokoro-dataset", 
    "--name", "my_voice_audio_features",
    "--epochs-projection", "50",
    "--lr-projection", "1e-3",
    "--max-style-samples", "100",
    "--wandb",
    "--log-audio-every", "10",
]

subprocess.run(cmd, check=True)
        """
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ Audio feature training completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Audio feature training failed: {e.returncode}")
        return False

def compare_results():
    """Compare the results of different training approaches."""
    
    print("\n📊 Comparing Training Results")
    print("="*60)
    
    voices_to_test = [
        ("my_voice_improved.pt", "StyleTTS2 with improved parameters"),
        ("my_voice_audio_features.pt", "Audio features fallback"),
    ]
    
    for voice_file, description in voices_to_test:
        voice_path = Path("output") / "my_voice_improved" / voice_file
        if voice_path.exists():
            print(f"\n🎵 Testing: {description}")
            print(f"   File: {voice_path}")
            
            # Generate test audio
            test_cmd = [
                sys.executable, "generate_with_styletts2_voice.py",
                str(voice_path),
                "Hello, this is a test of the improved voice training.",
                f"test_{voice_file.replace('.pt', '.wav')}"
            ]
            
            try:
                subprocess.run(test_cmd, check=True, capture_output=True)
                print(f"   ✅ Generated test audio: test_{voice_file.replace('.pt', '.wav')}")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Audio generation failed: {e}")
        else:
            print(f"\n❌ {description}: Voice file not found at {voice_path}")

def main():
    """Main function to run improved training pipeline."""
    
    print("StyleTTS2 → Kokoro Improved Training Pipeline")
    print("="*60)
    print("Based on diagnostic results showing StyleTTS2 is working well!")
    print()
    
    # Check if dataset exists
    dataset_path = Path("./output/kokoro-dataset")
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please run the dataset preparation first.")
        return
    
    print("📋 Training Plan:")
    print("1. Run improved StyleTTS2 training with better parameters")
    print("2. Run audio feature fallback for comparison")
    print("3. Generate test audio from both approaches")
    print("4. Compare results")
    print()
    
    input("Press Enter to start training...")
    
    # Run improved training
    success1 = run_improved_training()
    
    if success1:
        print("\n🎯 First training successful! Now trying audio features for comparison...")
        success2 = run_audio_feature_fallback()
        
        if success2:
            print("\n📊 Both trainings completed! Comparing results...")
            compare_results()
        else:
            print("\n⚠️ Audio feature training failed, but main training succeeded.")
    else:
        print("\n❌ Main training failed. Check the error messages above.")
    
    print("\n🏁 Training pipeline completed!")
    print("\nKey improvements made:")
    print("- Increased learning rate from 5e-5 to 1e-3")
    print("- Reduced max_style_samples from 1000 to 100")
    print("- Increased epochs to 100 (since embeddings are good)")
    print("- More frequent audio logging for monitoring")

if __name__ == "__main__":
    main() 