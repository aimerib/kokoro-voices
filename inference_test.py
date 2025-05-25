#!/usr/bin/env python3
"""
Quick inference test for trained voice embeddings.
Generates sample audio to verify voice quality during/after training.
"""

import argparse
from os.path import isdir
import torch
from pathlib import Path
from kokoro import KModel, KPipeline
import torchaudio
import traceback
from utils import generate_with_custom_voice, generate_with_standard_voice

def test_voice(voice_path: str, text: str = None, output: str = "test_output.wav"):
    """Test a trained voice embedding with sample text."""
    try:
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
        
        # # Load model and voice
        # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # model = KModel().to(device)
        # pipeline = KPipeline(model=model, lang_code="a")
        
        # Load voice embedding
        if isdir(voice_path):
            voice_path = Path(voice_path)
            files = list(voice_path.glob("*.pt"))
            if len(files) == 0:
                raise ValueError(f"No .pt files found in directory: {voice_path}")
            for path in files:

                # voice_data = torch.load(path, map_location=device, weights_only=True)
                # if isinstance(voice_data, dict):
                #     # Handle new format with metadata
                #     print("Loading voice from new format")
                #     voice = voice_data["voice_embed"]
                # else:
                #     # Handle raw tensor format
                #     print("Loading voice from raw tensor format")
                #     voice = voice_data
        
                # print(f"Loaded voice from: {path}")
                generate_with_custom_voice(text, path, output.replace(".wav", f"_{path.name}.wav"))
        else:
            # voice_data = torch.load(voice_path, map_location=device, weights_only=True)
            # if isinstance(voice_data, dict):
            #     # Handle new format with metadata
            #     print("Loading voice from new format")
            #     voice = voice_data["voice_embed"]
            # else:
            #     # Handle raw tensor format
            #     print("Loading voice from raw tensor format")
            #     voice = voice_data
            # torch.save(voice, voice_path)
        
            # print(f"Loaded voice from: {voice_path}")
            generate_with_custom_voice(text, voice_path, output)
        # print(f"Voice shape: {voice.shape}")
        # print(f"Generating {len(text)} samples...\n")
        # pipeline.load_single_voice
        # # Generate audio for each text
        # for i, sentence in enumerate(text):
        #     print(f"[{i+1}/{len(text)}] {sentence}")
            
        #     try:
        #         # Pipeline returns a generator - iterate through it
        #         generator = pipeline(
        #             sentence,
        #             voice=voice,
        #             speed=1.0,
        #             split_pattern=r'\n+',
        #         )
                
        #         # Collect all audio chunks from the generator
        #         audio_chunks = []
        #         sample_rate = None
                
        #         for chunk_idx, (gs, ps, audio_chunk) in enumerate(generator):
        #             print(f"   Processing chunk {chunk_idx + 1}: gs={gs}, ps={ps}, audio_shape={audio_chunk.shape}")
        #             audio_chunks.append(audio_chunk)
        #             if sample_rate is None:
        #                 sample_rate = 24000  # Kokoro's default sample rate
                
        #         if not audio_chunks:
        #             print(f"   ✗ No audio generated for: {sentence}")
        #             continue
                
        #         # Concatenate all audio chunks
        #         if len(audio_chunks) == 1:
        #             final_audio = audio_chunks[0]
        #         else:
        #             final_audio = torch.cat(audio_chunks, dim=-1)
                
        #         # Ensure audio is 2D for torchaudio.save [channels, samples]
        #         if final_audio.dim() == 1:
        #             final_audio = final_audio.unsqueeze(0)
                
        #         # Save audio
        #         output_path = output.replace(".wav", f"_{i+1}.wav") if len(text) > 1 else output
        #         torchaudio.save(output_path, final_audio, sample_rate)
        #         print(f"   ✓ Saved to: {output_path} (duration: {final_audio.shape[-1]/sample_rate:.2f}s)")
                
        #     except Exception as e:
        #         traceback.print_exc()
        #         print(f"   ✗ Error: {e}")
        
        print("\nDone! Listen to the generated audio to evaluate voice quality.")
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test trained voice embeddings")
    parser.add_argument("voice", help="Path to voice embedding (.pt file)")
    parser.add_argument("--text", help="Custom text to synthesize", default=None)
    parser.add_argument("--output", help="Output WAV file", default="test_output.wav")
    
    args = parser.parse_args()
    test_voice(args.voice, args.text, args.output)
