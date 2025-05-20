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
import shutil

import inflect, unicodedata

import torchaudio.functional as F

from dotenv import load_dotenv

eng = inflect.engine()
load_dotenv()

# Check if espeak envs are passed
if not os.getenv("PHONEMIZER_ESPEAK_LIBRARY") or not os.getenv("PHONEMIZER_ESPEAK_PATH"):
    PHONEMIZER_AVAILABLE = False
else:
    from collections import Counter
    from phonemizer import phonemize 
    PHONEMIZER_AVAILABLE = True
    

def loudness_normalize(wav: torch.Tensor, peak_db: float = -0.1):
    """Normalise to fixed peak dBFS (simple, fast)."""
    peak = wav.abs().max()
    target_amp = 10 ** (peak_db / 20)
    return wav * (target_amp / peak)

def trim_silence(wav: torch.Tensor, sr: int, thr_db: float = -35.0):
    """Strip leading / trailing silence with energy below threshold."""
    db_per_sample = 20 * torch.log10(wav.abs() + 1e-9)
    mask = db_per_sample > thr_db
    if not mask.any():
        return wav      # leave as-is if all silent
    first, last = mask.nonzero()[0][0], mask.nonzero()[-1][0]
    return wav[first:last+1]

def phoneme_histogram(texts):
    phones = phonemize(texts, language='en-us', backend='espeak', strip=True)
    counts = Counter(' '.join(phones).split())
    return counts

# Used when exporting to HuggingFace
from huggingface_hub import HfApi, upload_folder
def upload_to_hf(dataset_dir: str, repo_id: str, private: bool = True, exists_ok:bool = False, duration: float = None, num_clips: int = None):
    """
    Push an entire dataset directory to the Hugging Face Hub.

    Args:
        dataset_dir: local directory (prompts.txt + wavs)
        repo_id: e.g. "aimeri/my-voice-demo"
        private: create a private repo (recommended for gated datasets)
        gated:   put the repo behind the Hub's "Access request" gate
    """
    api = HfApi()

    # Create the repo if it doesn't exist yet
    if not api.repo_exists(repo_id, repo_type="dataset"):
        api.create_repo(
            repo_id,
            private=private,
            exist_ok=exists_ok,
            repo_type="dataset",
        )
    write_dataset_card(dataset_dir, duration, num_clips)
    print(f"Uploading {dataset_dir} → hf://datasets/{repo_id} …")
    upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=dataset_dir,
        commit_message="Add prepared dataset",
    )
    print("✓ upload complete")

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

def normalise_text(t: str):
    t = unicodedata.normalize("NFKC", t)
    # Expand numbers → words (“2024”→“two thousand twenty-four”)
    t = re.sub(r'\d+', lambda m: eng.number_to_words(m.group(0)), t)
    t = re.sub(r'[^a-zA-Z0-9\.,\?!\' ]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t.capitalize()

def ensure_dir(directory):
    """Ensure a directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def write_dataset_card(out_dir: str, duration_min: float, num_clips: int):
    text = f"""---
license: cc-by-nc-nd-4.0
dataset_type: audio
pretty_name: Kokoro voice cloning dataset
size_categories:
- 100M<n<1B
---

### Summary
* **Total duration** ≈ {duration_min:.1f} min  
* **Clips** {num_clips}  
* Prepared with `prepare_dataset.py`.
* Contains train/validation/test splits (80/10/10) for proper evaluation.

### Intended use
Single-speaker voice embedding experiments with Kokoro-TTS. Dataset provided for
training script evaluation only.

### Dataset structure
The dataset is organized in a directory structure with three splits:

```
dataset/
├── train/               # Training data (~80%)
│   ├── metadata.jsonl    # Training metadata
│   └── segment_*.wav     # Audio files
├── validation/           # Validation data (~10%) 
│   ├── metadata.jsonl    # Validation metadata
│   └── segment_*.wav     # Audio files
└── test/                 # Test data (~10%)
    ├── metadata.jsonl    # Test metadata
    └── segment_*.wav     # Audio files
```

Each split directory contains its own metadata.jsonl file with the format:
```json
{{"file_name": "segment_000.wav", "text": "Transcription text"}}
```
"""
    (Path(out_dir)/"README.md").write_text(text)

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
    Split audio into segments based on Whisper transcription with natural word boundaries
    Returns a list of (audio_segment, text, start_time, end_time) tuples
    """

    results = []
    
    # Add padding to create natural boundaries (100ms before, 200ms after)
    padding_before = int(0.05 * sample_rate)  # 50ms before speech starts
    padding_after = int(0.1 * sample_rate)   # 100ms after speech ends
    
    for segment in segments:
        start_sample = int(segment["start"] * sample_rate)
        end_sample = int(segment["end"] * sample_rate)
        duration = segment["end"] - segment["start"]

        if segment.get("no_speech_prob", 0) > 0.4:  # uncertain speech
            continue
        if segment.get("avg_logprob", -10) < -1.0:  # low confidence
            continue
        
        # Skip segments that are too short or too long
        if duration < min_duration or duration > max_duration:
            continue
        
        # Adjust boundaries with padding for more natural sound
        padded_start = max(0, start_sample - padding_before)
        padded_end = min(len(audio), end_sample + padding_after)
        
        # Get audio segment with padding
        audio_segment = audio[padded_start:padded_end]
        
        rms = np.sqrt(np.mean(audio_segment.astype(np.float32) ** 2))
        
        if rms < 1e-4:        # extremely quiet
            continue
        if np.abs(audio_segment).max() > 0.99:  # clipped
            continue

        # Get text
        text = normalise_text(segment["text"])
        
        # Skip empty segments or those with just punctuation
        if not re.search(r'\w', text):
            continue
        
        # Store with original timestamps (for tracking purposes)
        results.append((audio_segment, text, segment["start"], segment["end"]))
    
    return results


def save_dataset(segments, output_dir, target_sr=24000, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, seed=42):
    """Save audio segments and create metadata.jsonl with train/validation/test splits"""
    ensure_dir(output_dir)
    
    # Ensure splits add up to 1.0
    assert abs(train_ratio + validation_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create directories for splits
    train_dir = os.path.join(output_dir, "train")
    validation_dir = os.path.join(output_dir, "validation")
    test_dir = os.path.join(output_dir, "test")
    
    ensure_dir(train_dir)
    ensure_dir(validation_dir)
    ensure_dir(test_dir)
    
    # Track duplicate text to avoid exact duplicates
    seen_texts = set()
    unique_segments = []
    
    # First pass: filter out duplicates
    for audio_segment, text, _, _ in tqdm(segments, desc="Filtering duplicates"):
        if text in seen_texts:
            continue
        seen_texts.add(text)
        unique_segments.append((audio_segment, text))
    
    # Shuffle segments
    random.shuffle(unique_segments)
    
    # Calculate split indices
    total_segments = len(unique_segments)
    train_end = int(total_segments * train_ratio)
    val_end = train_end + int(total_segments * validation_ratio)
    
    # Split segments
    train_segments = unique_segments[:train_end]
    validation_segments = unique_segments[train_end:val_end]
    test_segments = unique_segments[val_end:]
    
    print(f"Split distribution: {len(train_segments)} train, {len(validation_segments)} validation, {len(test_segments)} test")
    
    # Save segments to their respective splits
    save_split(train_segments, train_dir, "train", 0, target_sr)
    save_split(validation_segments, validation_dir, "validation", len(train_segments), target_sr)
    save_split(test_segments, test_dir, "test", len(train_segments) + len(validation_segments), target_sr)
    
    print(f"Created dataset with {total_segments} segments in {output_dir} (metadata.jsonl format with splits)")
    return total_segments


def save_split(segments, output_dir, split_name, start_index, target_sr):
    """Save a specific split (train/validation/test) with its own metadata.jsonl"""
    metadata_file = open(os.path.join(output_dir, "metadata.jsonl"), "w")
    
    for i, (audio_segment, text) in enumerate(tqdm(segments, desc=f"Saving {split_name} segments")):
        # Convert audio to tensor and resample to target sample rate if needed
        audio_tensor = torch.from_numpy(audio_segment).float()
        
        if target_sr != 16000:  # Whisper uses 16kHz, Kokoro expects 24kHz
            resampler = torchaudio.transforms.Resample(16000, target_sr)
            audio_tensor = resampler(audio_tensor)
        
        audio_tensor = trim_silence(audio_tensor, target_sr)
        audio_tensor = loudness_normalize(audio_tensor)

        # Determine filename
        filename = f"segment_{start_index + i:04d}.wav"
        filepath = os.path.join(output_dir, filename)
        
        # Save audio
        torchaudio.save(filepath, audio_tensor.unsqueeze(0), target_sr)
        
        # Create metadata entry with just the filename (relative to the split directory)
        import json
        entry = {"file_name": filename, "text": text, "split": split_name}
        metadata_file.write(json.dumps(entry) + "\n")
    
    metadata_file.close()


def prepare_dataset(input_path, output_dir, model_name="base", seed=1985):
    """
    End-to-end preparation of dataset from a long audio file
    
    Args:
        input_path: Path to input audio file or folder
        output_dir: Directory to save dataset
        model_name: Whisper model size (tiny, base, small, medium, large)
        seed: Random seed for shuffling
    """
    print(f"Preparing dataset from {input_path} → {output_dir} using Whisper {model_name}")
    ensure_dir(output_dir)

    audio_to_process = []
    segments = []
    total_runtime = 0.0

    if os.path.isdir(input_path):
        for file in os.listdir(input_path):
            if file.endswith(".wav") or file.endswith(".mp3") or file.endswith(".m4a"):
                audio_to_process.append(os.path.join(input_path, file))
    else:
        audio_to_process.append(input_path)
    
    if not audio_to_process:
        raise ValueError(f"No audio files found in {input_path}")

    # Process each audio file
    for i, audio_file in enumerate(audio_to_process):
        print(f"Loading audio file {i+1}/{len(audio_to_process)}: {audio_file}")
        audio = load_audio(audio_file, sr=16000)
        
        # Transcribe and segment
        print(f"Transcribing audio {i+1}/{len(audio_to_process)} (this may take a while)...")
        result = transcribe(audio, model_name=model_name)
        segments.extend(segment_audio(audio, 16000, result["segments"]))
        total_runtime += audio.shape[0] / 16000
    
    # Save dataset with train/validation/test splits
    print(f"Found {len(segments)} segments, saving to {output_dir} with train/validation/test splits...")
    num_clips = save_dataset(segments, output_dir, seed=seed)

    if PHONEMIZER_AVAILABLE:
        hist = phoneme_histogram([t for _,t,_,_ in segments])
        rare = [p for p,c in hist.items() if c < 5]
        print(f"Phonemes with <5 examples: {', '.join(rare)}")
    
    duration_min = total_runtime / 60
    print(f"Done! Dataset ready for training.")
    print(f"Total audio duration: {duration_min:.1f} minutes")
    print(f"To train: python training.py --data {output_dir}")

    return num_clips, duration_min


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a Kokoro voice dataset from a long audio recording")
    parser.add_argument("--input", required=True, help="Input audio file path or path to folder with audio files")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large", "turbo"], help="Whisper model size (default: base)")
    parser.add_argument("--seed", type=int, default=1985, help="Random seed for shuffling (default: 42)")
    parser.add_argument("--upload-hf", action="store_true", help="After preparing, upload the dataset to Hugging Face")
    parser.add_argument("--hf-repo", type=str, help='Target HF dataset repo (e.g. "aimeri/my-voice-demo")')
    parser.add_argument("--hf-public", action="store_true", help="Create public repo (default: private)")
    parser.add_argument("--hf-exists-ok", action="store_true", help="Allow overwriting existing repo")
    parser.add_argument("--cleanup-after-upload", action="store_true", help="Remove the dataset directory after uploading")
    args = parser.parse_args()
    
    # Validate input file or folder exists
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file or folder not found: {args.input}")
    

    # Prepare dataset
    num_clips, duration = prepare_dataset(args.input, args.output, args.model, args.seed)
    if args.upload_hf:
        if not args.hf_repo:
            raise ValueError("--upload-hf requires --hf-repo")
        upload_to_hf(
            dataset_dir=args.output,
            repo_id=args.hf_repo,
            private=not args.hf_public,
            exists_ok=args.hf_exists_ok,
            duration=duration,
            num_clips=num_clips,
        )
        if args.cleanup_after_upload:
            shutil.rmtree(args.output)