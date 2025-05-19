#!/usr/bin/env python
"""
Prepare dataset from a long audio recording

This script automates the process of converting a single long audio recording into
a properly formatted Kokoro training dataset by:
1. Transcribing the audio using Whisper
2. Chunking the audio into sentence-level segments
3. Creating a properly formatted dataset with prompts.txt and WAV files

Usage:
    python prepare_dataset.py --input my_long_recording.wav --output ./my_dataset [--model tiny]
"""

import argparse
import os
import torch
import torchaudio
import numpy as np
import re
import random
from pathlib import Path
from tqdm import tqdm

def transcribe(audio, model_name="base"):
    # Load Whisper model
    print(f"Loading Whisper {model_name} model...")
    if torch.backends.mps.is_available():
        from whisper_mps import whisper
        return whisper.transcribe(
            audio,
            model=model_name,
            without_timestamps=False,
            verbose=False,
            language="en",
        )
    else:
        import whisper
        model = whisper.load_model(model_name)
        return model.transcribe(
            audio,
            word_timestamps=True,
            verbose=False,
            language="en",
        )


def ensure_dir(directory):
    """Ensure a directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def load_audio(file_path, sr=16000):
    """Load audio file and convert to the target sample rate"""
    # Handle different audio formats with torchaudio
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != sr:
            resampler = torchaudio.transforms.Resample(sample_rate, sr)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0).numpy()
    except Exception as e:
        print(f"Error loading audio file: {e}")
        raise


def segment_audio(audio, sample_rate, segments, min_duration=1.0, max_duration=10.0):
    """
    Split audio into segments based on Whisper transcription
    Returns a list of (audio_segment, text, start_time, end_time) tuples
    """
    results = []
    
    for segment in segments:
        start_sample = int(segment["start"] * sample_rate)
        end_sample = int(segment["end"] * sample_rate)
        duration = segment["end"] - segment["start"]
        
        # Skip segments that are too short or too long
        if duration < min_duration or duration > max_duration:
            continue
        
        # Get audio segment
        audio_segment = audio[start_sample:end_sample]
        
        # Get text
        text = segment["text"].strip()
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Remove multiple spaces
        text = re.sub(r'[^\w\s\.,\?!]', '', text)  # Keep only alphanumeric, spaces, and basic punctuation
        
        # Skip empty segments or those with just punctuation
        if not re.search(r'\w', text):
            continue
        
        results.append((audio_segment, text, segment["start"], segment["end"]))
    
    return results


def save_dataset(segments, output_dir, target_sr=24000):
    """Save audio segments and create prompts.txt"""
    ensure_dir(output_dir)
    
    # Initialize prompts.txt
    prompts_file = open(os.path.join(output_dir, "prompts.txt"), "w")
    
    # Track duplicate text to avoid exact duplicates
    seen_texts = set()
    
    # Process each segment
    for i, (audio_segment, text, start_time, end_time) in enumerate(tqdm(segments, desc="Saving segments")):
        # Skip if we've seen this exact text before (avoid duplicates)
        if text in seen_texts:
            continue
            
        seen_texts.add(text)
        
        # Convert audio to tensor and resample to target sample rate if needed
        audio_tensor = torch.from_numpy(audio_segment).float()
        
        if target_sr != 16000:  # Whisper uses 16kHz, Kokoro expects 24kHz
            resampler = torchaudio.transforms.Resample(16000, target_sr)
            audio_tensor = resampler(audio_tensor)
        
        # Determine filename
        filename = f"segment_{i:04d}.wav"
        filepath = os.path.join(output_dir, filename)
        
        # Save audio
        torchaudio.save(filepath, audio_tensor.unsqueeze(0), target_sr)
        
        # Write to prompts.txt (tab-separated: filename \t text)
        prompts_file.write(f"{filename}\t{text}\n")
    
    prompts_file.close()
    print(f"Created dataset with {len(seen_texts)} segments in {output_dir}")


def prepare_dataset(input_file, output_dir, model_name="base", seed=1985):
    """
    End-to-end preparation of dataset from a long audio file
    
    Args:
        input_file: Path to input audio file
        output_dir: Directory to save dataset
        model_name: Whisper model size (tiny, base, small, medium, large)
        seed: Random seed for shuffling
    """
    print(f"Processing {input_file}...")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Load audio file (Whisper expects 16kHz)
    print(f"Loading audio file...")
    audio = load_audio(input_file, sr=16000)
    
    # Transcribe with Whisper
    print(f"Transcribing audio (this may take a while)...")
    result = transcribe(audio, model_name=model_name)
    
    # Get segments
    print(f"Segmenting audio into chunks...")
    segments = segment_audio(audio, 16000, result["segments"])
    
    # Shuffle segments to improve training diversity
    random.shuffle(segments)
    
    # Save dataset
    print(f"Saving {len(segments)} segments to {output_dir}...")
    save_dataset(segments, output_dir)
    
    print(f"Done! Dataset ready for training.")
    print(f"To train: python training.py --data {output_dir}")
    
    return len(segments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a Kokoro voice dataset from a long audio recording")
    parser.add_argument("--input", required=True, help="Input audio file path")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large", "turbo"],
                       help="Whisper model size (default: base)")
    parser.add_argument("--seed", type=int, default=1985, help="Random seed for shuffling (default: 42)")
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Prepare dataset
    prepare_dataset(args.input, args.output, args.model, args.seed)
