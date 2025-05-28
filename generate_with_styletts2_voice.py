#!/usr/bin/env python3
"""
Generate speech using a voice trained with the StyleTTS2 → Kokoro pipeline.

Usage:
    python generate_with_styletts2_voice.py \
        --voice output/my_voice/my_voice.pt \
        --text "Hello, this is my custom voice!" \
        --output my_speech.wav
"""

import argparse
import torch
import soundfile as sf
from pathlib import Path

def generate_speech(voice_path: str, text: str, output_path: str = "output.wav"):
    """
    Generate speech using a trained StyleTTS2 → Kokoro voice.
    
    Args:
        voice_path: Path to the trained voice embedding (.pt file)
        text: Text to synthesize
        output_path: Where to save the generated audio
    """
    
    try:
        from kokoro import KPipeline
    except ImportError:
        print("Error: Kokoro not installed. Please install it first.")
        return False
    
    print(f"Loading voice from: {voice_path}")
    
    # Load the trained voice embedding
    try:
        voice = torch.load(voice_path, map_location="cpu")
        print(f"✓ Loaded voice tensor with shape: {voice.shape}")
        
        # Verify the voice tensor has the correct format
        if voice.shape != (510, 1, 256):
            print(f"Warning: Unexpected voice shape {voice.shape}, expected (510, 1, 256)")
            
    except Exception as e:
        print(f"Error loading voice: {e}")
        return False
    
    print(f"Generating speech for: '{text}'")
    
    # Initialize Kokoro pipeline
    try:
        pipeline = KPipeline(lang_code="a")
    except Exception as e:
        print(f"Error initializing Kokoro pipeline: {e}")
        return False
    
    # Generate speech
    try:
        outputs = []
        for _, _, audio in pipeline(text, voice=voice):
            outputs.append(audio)
        
        if outputs:
            # Combine all audio segments
            full_audio = torch.cat(outputs)
            
            # Save to file
            sf.write(output_path, full_audio.numpy(), 24000)
            print(f"✓ Generated speech saved to: {output_path}")
            
            # Print some stats
            duration = len(full_audio) / 24000
            print(f"  Duration: {duration:.2f} seconds")
            print(f"  Sample rate: 24000 Hz")
            print(f"  Samples: {len(full_audio):,}")
            
            return True
        else:
            print("Error: No audio was generated")
            return False
            
    except Exception as e:
        print(f"Error during speech generation: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate speech with StyleTTS2 → Kokoro trained voice")
    
    parser.add_argument("--voice", required=True, help="Path to trained voice embedding (.pt file)")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", default="output.wav", help="Output audio file path")
    
    args = parser.parse_args()
    
    # Verify voice file exists
    if not Path(args.voice).exists():
        print(f"Error: Voice file not found: {args.voice}")
        return
    
    # Generate speech
    success = generate_speech(args.voice, args.text, args.output)
    
    if success:
        print("\n✓ Speech generation completed successfully!")
    else:
        print("\n✗ Speech generation failed!")

if __name__ == "__main__":
    main() 