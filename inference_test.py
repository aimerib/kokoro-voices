#!/usr/bin/env python3
"""
Quick inference test for trained voice embeddings.
Generates sample audio to verify voice quality during/after training.
"""

import argparse
import torch
from pathlib import Path
from kokoro import KModel, KPipeline
import torchaudio

def test_voice(voice_path: str, text: str = None, output: str = "test_output.wav"):
    """Test a trained voice embedding with sample text."""
    
    # Default test sentences for audiobook-style narration
    if text is None:
        text = [
            "Once upon a time, in a land far away, there lived a young prince.",
            "The morning sun cast long shadows across the ancient castle walls.",
            "Chapter one. The journey begins.",
            "And so, our hero set forth on an adventure that would change everything.",
        ]
    elif isinstance(text, str):
        text = [text]
    
    # Load model and voice
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = KModel().to(device)
    pipeline = KPipeline(model=model)
    
    # Load voice embedding
    voice_data = torch.load(voice_path, map_location=device)
    if isinstance(voice_data, dict):
        # Handle new format with metadata
        voice = voice_data["voice_embed"]
    else:
        # Handle raw tensor format
        voice = voice_data
    
    print(f"Loaded voice from: {voice_path}")
    print(f"Voice shape: {voice.shape}")
    print(f"Generating {len(text)} samples...\n")
    
    # Generate audio for each text
    for i, sentence in enumerate(text):
        print(f"[{i+1}/{len(text)}] {sentence}")
        
        try:
            audio, sr = pipeline(
                sentence,
                voice=voice,
                speed=1.0,
                lang="a",
            )
            
            # Save audio
            output_path = output.replace(".wav", f"_{i+1}.wav") if len(text) > 1 else output
            torchaudio.save(output_path, audio, sr)
            print(f"   ✓ Saved to: {output_path}")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    print("\nDone! Listen to the generated audio to evaluate voice quality.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained voice embeddings")
    parser.add_argument("voice", help="Path to voice embedding (.pt file)")
    parser.add_argument("--text", help="Custom text to synthesize", default=None)
    parser.add_argument("--output", help="Output WAV file", default="test_output.wav")
    
    args = parser.parse_args()
    test_voice(args.voice, args.text, args.output)
