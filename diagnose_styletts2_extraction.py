#!/usr/bin/env python3
"""
Diagnostic script for StyleTTS2 voice extraction issues.

This script helps identify why StyleTTS2 might not be extracting
distinctive voice characteristics from your dataset.
"""

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import tempfile
import soundfile as sf
from typing import List, Tuple
import librosa

# ---------------------------------------------------------------------------
# DANGER-ZONE: force full pickle loading
# ---------------------------------------------------------------------------
# StyleTTS2 checkpoints embed full Python objects.  We acknowledge the risk
# and explicitly patch torch.load so that, when StyleTTS2 calls it without
# specifying `weights_only`, we switch it back to the old (pre-2.6) behaviour
# of allowing full unpickling.

import warnings as _warnings

_orig_torch_load = torch.load

def _unsafe_full_load(*args, **kwargs):  # noqa: D401
    """Wrapper around torch.load that disables the weights-only default.

    This is *unsafe* because it allows executing pickled code.  We enable it
    knowingly because StyleTTS2's official checkpoints rely on this.
    """
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)

# Emit a single warning so users are aware.
_warnings.warn(
    "⚠  Overriding torch.load to allow full pickle unpickling.  This is "
    "dangerous—only use with trusted checkpoints (StyleTTS2).",
    RuntimeWarning,
    stacklevel=2,
)

torch.load = _unsafe_full_load


# Try to import StyleTTS2
try:
    from styletts2 import tts
    STYLETTS2_AVAILABLE = True
except ImportError:
    STYLETTS2_AVAILABLE = False
    print("StyleTTS2 not available - install with: pip install styletts2")

def load_dataset_samples(data_root: str, max_samples: int = 10) -> List[Tuple[str, torch.Tensor, Path]]:
    """Load a few samples from the dataset for analysis."""
    data_path = Path(data_root)
    train_dir = data_path / "train"
    metadata_file = train_dir / "metadata.jsonl"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    samples = []
    with open(metadata_file) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            if not line.strip():
                continue
            
            try:
                entry = json.loads(line)
                filename = entry["file_name"]
                text = entry["text"]
                
                wav_path = train_dir / filename
                if wav_path.exists():
                    # Load audio
                    audio, sr = torchaudio.load(wav_path)
                    audio = audio.mean(0)  # Convert to mono
                    if sr != 24000:
                        audio = torchaudio.functional.resample(audio, sr, 24000)
                    
                    samples.append((text, audio, wav_path))
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping invalid entry: {e}")
                continue
    
    return samples

def analyze_audio_quality(samples: List[Tuple[str, torch.Tensor, Path]]):
    """Analyze the quality and characteristics of audio samples."""
    print("\n" + "="*60)
    print("AUDIO QUALITY ANALYSIS")
    print("="*60)
    
    durations = []
    rms_levels = []
    snr_estimates = []
    
    for i, (text, audio, path) in enumerate(samples):
        duration = len(audio) / 24000
        rms = torch.sqrt(torch.mean(audio ** 2)).item()
        
        # Estimate SNR (very rough)
        # Assume silence is noise floor
        sorted_audio = torch.sort(torch.abs(audio))[0]
        noise_floor = sorted_audio[:len(sorted_audio)//10].mean()  # Bottom 10%
        signal_level = sorted_audio[-len(sorted_audio)//10:].mean()  # Top 10%
        snr_db = 20 * torch.log10(signal_level / (noise_floor + 1e-8)).item()
        
        durations.append(duration)
        rms_levels.append(rms)
        snr_estimates.append(snr_db)
        
        print(f"Sample {i+1}: {path.name}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  RMS Level: {rms:.4f}")
        print(f"  Est. SNR: {snr_db:.1f} dB")
        print(f"  Text: {text[:50]}...")
        print()
    
    print("SUMMARY:")
    print(f"  Avg Duration: {np.mean(durations):.2f}s (range: {np.min(durations):.2f}-{np.max(durations):.2f}s)")
    print(f"  Avg RMS: {np.mean(rms_levels):.4f} (range: {np.min(rms_levels):.4f}-{np.max(rms_levels):.4f})")
    print(f"  Avg SNR: {np.mean(snr_estimates):.1f} dB (range: {np.min(snr_estimates):.1f}-{np.max(snr_estimates):.1f} dB)")
    
    # Quality recommendations
    print("\nQUALITY RECOMMENDATIONS:")
    if np.mean(durations) < 2.0:
        print("⚠ Short audio clips - StyleTTS2 works better with 3-10 second clips")
    if np.mean(rms_levels) < 0.01:
        print("⚠ Low audio levels - consider normalizing audio")
    if np.mean(snr_estimates) < 20:
        print("⚠ Low SNR - audio may be noisy, consider denoising")
    if np.std(rms_levels) > 0.02:
        print("⚠ Inconsistent levels - consider RMS normalization")

def test_styletts2_extraction(samples: List[Tuple[str, torch.Tensor, Path]]):
    """Test StyleTTS2 style extraction on samples."""
    if not STYLETTS2_AVAILABLE:
        print("StyleTTS2 not available - skipping extraction test")
        return None
    
    print("\n" + "="*60)
    print("STYLETTS2 EXTRACTION TEST")
    print("="*60)
    
    try:
        # Load StyleTTS2 model
        print("Loading StyleTTS2 model...")
        model = tts.StyleTTS2()
        print("✓ StyleTTS2 model loaded")
        
        embeddings = []
        
        for i, (text, audio, path) in enumerate(samples):
            print(f"\nExtracting from sample {i+1}: {path.name}")
            
            try:
                # Convert to numpy
                audio_np = audio.detach().cpu().numpy()
                
                # Save to temporary file (StyleTTS2 needs file path)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, audio_np, 24000)
                    tmp_path = tmp.name
                
                # Extract style
                style_vec = model.compute_style(tmp_path)
                
                # Clean up temp file
                import os
                os.remove(tmp_path)
                
                # Convert to tensor
                if isinstance(style_vec, np.ndarray):
                    style_tensor = torch.from_numpy(style_vec)
                elif isinstance(style_vec, torch.Tensor):
                    style_tensor = style_vec.detach().cpu()
                else:
                    style_tensor = torch.tensor(style_vec, dtype=torch.float32)
                
                style_tensor = style_tensor.float().squeeze()
                embeddings.append(style_tensor)
                
                print(f"  ✓ Extracted embedding shape: {style_tensor.shape}")
                print(f"  ✓ Mean: {style_tensor.mean():.6f}, Std: {style_tensor.std():.6f}")
                print(f"  ✓ Range: [{style_tensor.min():.6f}, {style_tensor.max():.6f}]")
                
            except Exception as e:
                print(f"  ✗ Extraction failed: {e}")
                continue
        
        if embeddings:
            # Analyze embedding diversity
            print(f"\n" + "="*40)
            print("EMBEDDING DIVERSITY ANALYSIS")
            print("="*40)
            
            embeddings_stacked = torch.stack(embeddings)
            
            # Overall statistics
            overall_mean = embeddings_stacked.mean()
            overall_std = embeddings_stacked.std()
            
            print(f"Overall embedding statistics:")
            print(f"  Mean: {overall_mean:.6f}")
            print(f"  Std: {overall_std:.6f}")
            print(f"  Shape: {embeddings_stacked.shape}")
            
            # Pairwise distances
            distances = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    dist = torch.norm(embeddings[i] - embeddings[j]).item()
                    distances.append(dist)
            
            if distances:
                avg_distance = np.mean(distances)
                print(f"  Average pairwise distance: {avg_distance:.6f}")
                
                if avg_distance < 0.01:
                    print("  ⚠ Very low diversity - embeddings are too similar!")
                elif avg_distance < 0.1:
                    print("  ⚠ Low diversity - limited voice variation captured")
                else:
                    print("  ✓ Good diversity - embeddings show variation")
            
            # Dimension analysis
            dim_stds = embeddings_stacked.std(dim=0)
            active_dims = (dim_stds > 0.001).sum().item()
            print(f"  Active dimensions (std > 0.001): {active_dims}/{len(dim_stds)}")
            
            if active_dims < 50:
                print("  ⚠ Few active dimensions - StyleTTS2 may not be capturing enough variation")
            
            return embeddings_stacked
        else:
            print("No embeddings extracted successfully")
            return None
            
    except Exception as e:
        print(f"StyleTTS2 extraction test failed: {e}")
        return None

def compare_with_reference_voices():
    """Compare with known good reference voices."""
    print("\n" + "="*60)
    print("REFERENCE VOICE COMPARISON")
    print("="*60)
    
    try:
        from huggingface_hub import hf_hub_download
        
        # Download a few reference voices
        reference_voices = ['af_heart.pt', 'af_sarah.pt', 'am_adam.pt']
        
        for voice_name in reference_voices:
            try:
                voice_path = hf_hub_download(
                    repo_id='hexgrad/Kokoro-82M', 
                    filename=f'voices/{voice_name}'
                )
                voice_tensor = torch.load(voice_path, weights_only=True)
                
                # Analyze reference voice
                if voice_tensor.dim() == 3:  # [510, 1, 256]
                    avg_embedding = voice_tensor.mean(dim=0).squeeze()  # [256]
                else:
                    avg_embedding = voice_tensor.squeeze()
                
                print(f"{voice_name}:")
                print(f"  Shape: {voice_tensor.shape}")
                print(f"  Mean: {avg_embedding.mean():.6f}")
                print(f"  Std: {avg_embedding.std():.6f}")
                print(f"  Range: [{avg_embedding.min():.6f}, {avg_embedding.max():.6f}]")
                print(f"  L2 Norm: {torch.norm(avg_embedding):.6f}")
                print()
                
            except Exception as e:
                print(f"Failed to load {voice_name}: {e}")
                
    except Exception as e:
        print(f"Reference voice comparison failed: {e}")

def recommend_improvements(embeddings_stacked=None):
    """Provide recommendations for improving voice extraction."""
    print("\n" + "="*60)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("="*60)
    
    print("1. AUDIO PREPROCESSING:")
    print("   - Normalize RMS levels to consistent range (e.g., 0.1-0.3)")
    print("   - Apply noise reduction if SNR < 20 dB")
    print("   - Ensure clips are 3-10 seconds long")
    print("   - Remove silence from beginning/end")
    
    print("\n2. DATASET IMPROVEMENTS:")
    print("   - Include more diverse emotional expressions")
    print("   - Vary speaking pace and pitch")
    print("   - Include different sentence types (questions, statements)")
    print("   - Ensure consistent recording conditions")
    
    print("\n3. STYLETTS2 ALTERNATIVES:")
    print("   - Try different StyleTTS2 model checkpoints")
    print("   - Use audio feature extraction fallback")
    print("   - Consider speaker verification models (e.g., resemblyzer)")
    
    if embeddings_stacked is not None:
        avg_distance = torch.pdist(embeddings_stacked).mean().item()
        if avg_distance < 0.01:
            print("\n4. CRITICAL ISSUE:")
            print("   ⚠ StyleTTS2 embeddings are nearly identical!")
            print("   - This suggests the model isn't capturing voice characteristics")
            print("   - Try the audio feature fallback instead")
            print("   - Check if your voice samples are too similar")

def main():
    """Main diagnostic function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose StyleTTS2 extraction issues")
    parser.add_argument("--data", required=True, help="Path to dataset directory")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to analyze")
    
    args = parser.parse_args()
    
    print("StyleTTS2 Voice Extraction Diagnostics")
    print("="*60)
    
    try:
        # Load samples
        print(f"Loading {args.samples} samples from {args.data}...")
        samples = load_dataset_samples(args.data, args.samples)
        print(f"✓ Loaded {len(samples)} samples")
        
        # Analyze audio quality
        analyze_audio_quality(samples)
        
        # Test StyleTTS2 extraction
        embeddings = test_styletts2_extraction(samples)
        
        # Compare with reference voices
        compare_with_reference_voices()
        
        # Provide recommendations
        recommend_improvements(embeddings)
        
    except Exception as e:
        print(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 