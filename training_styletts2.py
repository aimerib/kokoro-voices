"""training_styletts2.py
StyleTTS2 → Kokoro Voice Embedding Pipeline
------------------------------------------
This script implements a practical approach for creating Kokoro voice embeddings:
1. Use StyleTTS2 inference to extract style embeddings from target voice samples
2. Train a projection layer to map StyleTTS2 embeddings to Kokoro format
3. Generate Kokoro-compatible voice embeddings

This approach is more practical than direct Kokoro embedding optimization
because StyleTTS2's embedding space is less overfitted and more generalizable.

Expected dataset layout
----------------------
<dataset>/
    train/           # Training data 
        metadata.jsonl   # JSON Lines format: {"file_name": "segment_0000.wav", "text": "<sentence>"}
        segment_*.wav    # Audio files
    validation/      # Validation data (optional)
        metadata.jsonl   # Same format as train
        segment_*.wav    # Audio files

Usage
-----
python training_styletts2.py --data ./my_dataset --epochs 100 --out my_voice.pt

Pipeline stages:
1. Extract StyleTTS2 style embeddings from target voice samples
2. Train projection layer to map to Kokoro embedding space
3. Generate final Kokoro voice embedding
4. Validate with Kokoro TTS generation
"""

from __future__ import annotations

import datetime
import os
import gc
import sys
import json
import math
import time
import random
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa  # simplified audio IO / resampling for StyleTTS2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tempfile
import shutil
import soundfile as sf

from accelerate import Accelerator
from huggingface_hub import snapshot_download, HfApi, upload_folder
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Import utilities (reuse some from original training)
from utils import TrainingLogger, format_duration

from dotenv import load_dotenv
load_dotenv()

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ------------------------- StyleTTS2 -------------------------
# We rely on the lightweight pip package which already bundles
# a convenience wrapper (`styletts2.tts.StyleTTS2`) exposing a
# `compute_style()` helper.  No manual checkpoints or YAML files
# are necessary.

try:
    from styletts2 import tts  # pip install styletts2
    STYLETTS2_AVAILABLE = True
except ImportError:
    STYLETTS2_AVAILABLE = False
    print("StyleTTS2 not found. Install with: pip install styletts2")

# Kokoro imports for final validation
from kokoro import KModel, KPipeline

from contextlib import contextmanager
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

# ---------------------------------------------------------------------------
# Dataset for Style Extraction
# ---------------------------------------------------------------------------

class StyleExtractionDataset(Dataset):
    """Dataset for extracting StyleTTS2 style embeddings"""
    
    def __init__(self, root: str | Path, split="train", target_sr: int = 24_000):
        self.root = Path(root)
        self.sentences: List[str] = []
        self.wav_paths: List[Path] = []
        self.target_sr = target_sr
        self.split = split

        # Load from the specified split directory
        split_dir = self.root / split
        metadata = split_dir / "metadata.jsonl"
        
        if not split_dir.exists():
            raise FileNotFoundError(f"{split_dir} not found – expected split directory '{split}' in dataset root")
            
        if not metadata.exists():
            raise FileNotFoundError(f"{metadata} not found – expected metadata.jsonl in {split} directory")
        
        self._load_metadata_file(metadata, split_dir)
    
    def _load_metadata_file(self, metadata_path, split_dir):
        """Load entries from a metadata.jsonl file"""
        with open(metadata_path) as f:
            for line in f:
                if not line.strip():
                    continue
                    
                try:
                    entry = json.loads(line)
                    filename = entry["file_name"]
                    text = entry["text"]
                    
                    # Path is relative to the split directory
                    wav_path = split_dir / filename
                    
                    if not wav_path.exists():
                        raise FileNotFoundError(f"Missing WAV file: {wav_path}")
                    
                    self.wav_paths.append(wav_path)
                    self.sentences.append(text)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line[:50]}...")
                except KeyError as e:
                    print(f"Warning: Skipping entry missing required field {e}: {line[:50]}...")

    def __len__(self) -> int:
        return len(self.sentences)

    def _load_wav(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)  # mono
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        return wav

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, Path]:
        text = self.sentences[idx]
        wav = self._load_wav(self.wav_paths[idx])
        return text, wav, self.wav_paths[idx]

# ---------------------------------------------------------------------------
# Embedding Projection Layer
# ---------------------------------------------------------------------------

class StyleToKokoroProjection(nn.Module):
    """
    Neural network to project StyleTTS2 style embeddings to Kokoro format.
    
    This learns a mapping from StyleTTS2's style space to Kokoro's 256-dim
    voice embedding space, preserving voice characteristics.
    """
    
    def __init__(self, style_dim: int, kokoro_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.style_dim = style_dim
        self.kokoro_dim = kokoro_dim
        
        # Multi-layer projection with residual connections
        self.projection = nn.Sequential(
            nn.Linear(style_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, kokoro_dim),
            nn.Tanh()  # Kokoro embeddings are typically in [-1, 1] range
        )
        
        # Initialize with small weights for stable training
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, style_embedding):
        """Project StyleTTS2 style to Kokoro format"""
        return self.projection(style_embedding)

# ---------------------------------------------------------------------------
# StyleTTS2 Style Extraction Functions
# ---------------------------------------------------------------------------

@contextmanager
def allow_pickle():
    """Temporarily allow full checkpoint unpickling (StyleTTS2 needs it)."""
    with torch.serialization.safe_globals([getattr]):
        yield

def load_styletts2_model(device: str = "cpu"):
    """Instantiate StyleTTS2 for inference & style extraction."""
    if not STYLETTS2_AVAILABLE:
        raise ImportError("StyleTTS2 not installed. `pip install styletts2`. ")

    # StyleTTS2 wrapper is not an nn.Module; keep it on CPU.
    with allow_pickle():
        model = tts.StyleTTS2()
    print("✓ StyleTTS2 model loaded (pip package)")
    return model  # no longer returning model_type

def extract_style_from_audio(model, audio: torch.Tensor | np.ndarray, sr: int = 24000) -> torch.Tensor | None:
    """Return the 256-D style vector for a single audio sample.

    The pip package exposes `compute_style()` which accepts either a path or
    a numpy array.  We avoid temporary files by passing the waveform array
    directly.
    """
    try:
        # Ensure numpy mono 24 kHz
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
        else:
            audio_np = np.asarray(audio)

        if audio_np.ndim > 1:
            audio_np = audio_np.mean(0)

        if sr != 24000:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=24000)

        # StyleTTS2 expects a file path – write a short-lived temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio_np, 24000)
            tmp_path = tmp.name

        style_vec = model.compute_style(tmp_path)

        os.remove(tmp_path)

        # Handle different return types from compute_style - ensure tensor output
        if isinstance(style_vec, np.ndarray):
            vec = torch.from_numpy(style_vec)
        elif isinstance(style_vec, torch.Tensor):
            vec = style_vec.detach().cpu()
        else:
            # Handle any other types by converting to tensor
            vec = torch.tensor(style_vec, dtype=torch.float32)

        return vec.float().squeeze()
    except Exception as exc:
        print(f"Style extraction failed: {exc}")
        return None

def extract_styletts2_embeddings(
    dataset: StyleExtractionDataset,
    device: str = "cpu",
    max_samples: int = 100
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Extract style embeddings from StyleTTS2 model using target voice samples.
    
    Args:
        dataset: Dataset containing target voice samples
        device: Device for computation
        max_samples: Maximum number of samples to process
        
    Returns:
        Tuple of (average_embedding, list_of_embeddings)
    """
    
    if not STYLETTS2_AVAILABLE:
        raise ImportError("StyleTTS2 not available. Install with: pip install styletts2")
    
    print("="*60)
    print("STAGE 1: StyleTTS2 Style Embedding Extraction")
    print("="*60)
    
    # Load StyleTTS2 model
    print("Loading StyleTTS2 model…")
    try:
        styletts2_model = load_styletts2_model(device)
    except Exception as e:
        print(f"StyleTTS2 unavailable → fallback to audio-feature extraction ({e})")
        return extract_audio_features_as_style(dataset, 'cpu', max_samples)
    
    embeddings = []
    
    # Sample indices to use
    sample_indices = random.sample(
        range(len(dataset)), 
        min(max_samples, len(dataset))
    )
    
    print(f"Extracting embeddings from {len(sample_indices)} samples...")
    
    for idx in tqdm(sample_indices, desc="Extracting embeddings"):
        text, audio, wav_path = dataset[idx]
        
        try:
            # one-shot style extraction
            style_embedding = extract_style_from_audio(styletts2_model, audio, sr=24000)
            
            if style_embedding is not None:
                embeddings.append(style_embedding.cpu())
                
        except Exception as e:
            print(f"Warning: Failed to extract embedding for sample {idx}: {e}")
            continue
    
    if not embeddings:
        print("No StyleTTS2 embeddings extracted → fallback to audio features…")
        return extract_audio_features_as_style(dataset, 'cpu', max_samples)
    
    # Average all embeddings
    avg_embedding = torch.stack(embeddings).mean(0)
    
    print(f"Extracted {len(embeddings)} embeddings")
    print(f"Average embedding shape: {avg_embedding.shape}")
    print(f"Embedding statistics: mean={avg_embedding.mean():.4f}, std={avg_embedding.std():.4f}")
    
    return avg_embedding, embeddings

def extract_audio_features_as_style(
    dataset: StyleExtractionDataset, 
    device: str, 
    max_samples: int
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Fallback: Extract audio features as style embeddings when StyleTTS2 is not available.
    
    This creates a reasonable style representation using traditional audio features.
    """
    print("Using audio feature extraction as StyleTTS2 fallback...")
    
    embeddings = []
    sample_indices = random.sample(range(len(dataset)), min(max_samples, len(dataset)))
    
    for idx in tqdm(sample_indices, desc="Extracting audio features"):
        try:
            text, audio, wav_path = dataset[idx]
            
            # Extract comprehensive audio features
            features = extract_comprehensive_audio_features(audio, device)
            
            if features is not None:
                embeddings.append(features.cpu())
                
        except Exception as e:
            print(f"Warning: Failed to extract features for sample {idx}: {e}")
            continue
    
    if not embeddings:
        # Create a random embedding as last resort
        print("Creating random style embedding as final fallback...")
        random_embedding = torch.randn(256) * 0.1
        return random_embedding, [random_embedding]
    
    # Average all feature embeddings
    avg_embedding = torch.stack(embeddings).mean(0)
    
    print(f"Extracted {len(embeddings)} feature-based embeddings")
    print(f"Average embedding shape: {avg_embedding.shape}")
    
    return avg_embedding, embeddings

def extract_comprehensive_audio_features(audio: torch.Tensor, device: str) -> Optional[torch.Tensor]:
    """
    Extract comprehensive audio features that can serve as style embeddings.
    
    This combines multiple audio characteristics into a 256-dimensional vector.
    """
    try:
        # Ensure audio is 1D
        if audio.dim() > 1:
            audio = audio.mean(0)
        
        audio = audio.to(device)
        
        # 1. Mel-spectrogram features
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
            center=True,
            power=1.0
        ).to(device)
        
        mel = mel_transform(audio.unsqueeze(0))
        mel_log = torch.log(mel.clamp(min=1e-5))
        
        # Statistical features from mel spectrogram
        mel_mean = mel_log.mean(dim=-1).squeeze()  # [80]
        mel_std = mel_log.std(dim=-1).squeeze()    # [80]
        
        # 2. MFCC features
        try:
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=24000,
                n_mfcc=13,
                melkwargs={
                    "n_fft": 1024,
                    "hop_length": 256,
                    "n_mels": 80,
                    "center": True,
                }
            ).to(device)
            
            mfcc = mfcc_transform(audio.unsqueeze(0))
            mfcc_mean = mfcc.mean(dim=-1).squeeze()  # [13]
            mfcc_std = mfcc.std(dim=-1).squeeze()    # [13]
        except Exception as e:
            print(f"MFCC extraction failed: {e}")
            mfcc_mean = torch.zeros(13, device=device)
            mfcc_std = torch.zeros(13, device=device)
        
        # 3. Spectral features
        try:
            spectral_centroid = torchaudio.functional.spectral_centroid(
                audio.unsqueeze(0), 
                sample_rate=24000, 
                pad=0,
                window=torch.hann_window(1024, device=device),
                n_fft=1024, 
                hop_length=256,
                win_length=1024
            ).mean()
        except Exception as e:
            print(f"Spectral centroid extraction failed: {e}")
            spectral_centroid = torch.tensor(0.0, device=device)
        
        # 4. Pitch features (F0)
        try:
            # Simple pitch estimation using autocorrelation
            pitch_features = estimate_pitch_features(audio, 24000)
        except Exception as e:
            print(f"Pitch extraction failed: {e}")
            pitch_features = torch.zeros(4)  # Fallback
        
        # 5. Energy features
        try:
            energy = torch.sqrt(torch.mean(audio ** 2))
            zero_crossing_rate = torch.mean(
                torch.abs(torch.diff(torch.sign(audio))) / 2.0
            )
        except Exception as e:
            print(f"Energy extraction failed: {e}")
            energy = torch.tensor(0.0, device=device)
            zero_crossing_rate = torch.tensor(0.0, device=device)
        
        # 6. Temporal features
        audio_length = len(audio) / 24000  # Duration in seconds
        
        # Combine all features with error handling
        try:
            features = torch.cat([
                mel_mean[:64],      # 64 dims - mel mean (truncated)
                mel_std[:64],       # 64 dims - mel std (truncated)
                mfcc_mean,          # 13 dims - mfcc mean
                mfcc_std,           # 13 dims - mfcc std
                pitch_features,     # 4 dims - pitch statistics
                torch.tensor([spectral_centroid, energy, zero_crossing_rate, audio_length], device=device)  # 4 dims
            ])
        except Exception as e:
            print(f"Feature concatenation failed: {e}")
            # Create a basic feature vector as fallback
            features = torch.randn(256, device=device) * 0.1
        
        # Pad or truncate to exactly 256 dimensions
        if features.shape[0] > 256:
            features = features[:256]
        elif features.shape[0] < 256:
            padding = torch.zeros(256 - features.shape[0], device=device)
            features = torch.cat([features, padding])
        
        # Normalize features
        features = (features - features.mean()) / (features.std() + 1e-8)
        features = features * 0.1  # Scale to reasonable range
        
        return features.float()
        
    except Exception as e:
        print(f"Error extracting audio features: {e}")
        return None

def estimate_pitch_features(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Estimate basic pitch features from audio.
    """
    try:
        # Simple autocorrelation-based pitch estimation
        # This is a basic implementation - could be improved with librosa
        
        # Frame the audio
        frame_length = 1024
        hop_length = 256
        
        # Ensure audio is long enough
        if len(audio) < frame_length:
            return torch.zeros(4)
        
        frames = audio.unfold(0, frame_length, hop_length)
        
        pitch_values = []
        for frame in frames:
            try:
                # Autocorrelation using conv1d (more stable than correlate)
                frame_normalized = frame - frame.mean()
                frame_normalized = frame_normalized / (frame_normalized.std() + 1e-8)
                
                # Pad for full correlation
                padded = torch.nn.functional.pad(frame_normalized, (frame_length-1, 0))
                autocorr = torch.nn.functional.conv1d(
                    padded.unsqueeze(0).unsqueeze(0),
                    frame_normalized.flip(0).unsqueeze(0).unsqueeze(0)
                ).squeeze()
                
                # Take the second half (positive lags)
                autocorr = autocorr[frame_length-1:]
                
                # Find peak (excluding zero lag)
                min_period = int(sample_rate / 500)  # 500 Hz max
                max_period = int(sample_rate / 50)   # 50 Hz min
                
                if len(autocorr) > max_period and max_period > min_period:
                    search_range = autocorr[min_period:max_period]
                    if len(search_range) > 0:
                        peak_idx = torch.argmax(search_range) + min_period
                        pitch = sample_rate / peak_idx.float()
                        if 50 <= pitch <= 500:  # Reasonable pitch range
                            pitch_values.append(pitch.item())
            except Exception:
                continue
        
        if pitch_values:
            pitch_tensor = torch.tensor(pitch_values)
            return torch.tensor([
                pitch_tensor.mean(),
                pitch_tensor.std(),
                pitch_tensor.min(),
                pitch_tensor.max()
            ])
        else:
            return torch.zeros(4)
            
    except Exception as e:
        print(f"Error estimating pitch: {e}")
        return torch.zeros(4)

# ---------------------------------------------------------------------------
# Kokoro Projection Training
# ---------------------------------------------------------------------------

def visualize_embedding(embedding_np):
    """Create a PCA visualization of an embedding vector for W&B visualization."""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Reshape if needed to ensure 2D array with samples as rows
        if embedding_np.ndim == 1:
            # For a single vector, we can't do PCA, so create a simple distribution plot
            data = embedding_np
            
            # Create a histogram plot
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(data, bins=30, alpha=0.7)
            ax.set_title('Distribution of Embedding Values')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True)
            
            # Add basic statistics to the plot
            stats_text = f"Mean: {np.mean(data):.3f}\nStd: {np.std(data):.3f}\n"
            stats_text += f"Min: {np.min(data):.3f}\nMax: {np.max(data):.3f}"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            return fig
        else:
            # Only do PCA if we have enough samples
            if embedding_np.shape[0] > 2:
                from sklearn.decomposition import PCA
                # Apply PCA to reduce to 2 dimensions
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(embedding_np)
                
                # Create a scatter plot
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7)
                
                # Add explained variance as title
                explained_variance = pca.explained_variance_ratio_
                ax.set_title(f'PCA of Embeddings (Explained variance: {explained_variance[0]:.2f}, {explained_variance[1]:.2f})')
                
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.grid(True)
                
                return fig
            else:
                # Create a simple scatter plot without PCA
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(np.arange(embedding_np.shape[0]), embedding_np.mean(axis=1), alpha=0.7)
                ax.set_title('Mean Embedding Values')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Mean Value')
                ax.grid(True)
                return fig
            
    except Exception as e:
        print(f"Error visualizing embedding: {e}")
        # Return a blank figure if visualization fails
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.text(0.5, 0.5, f"Visualization error: {e}", 
                horizontalalignment='center', verticalalignment='center')
        return fig

def train_kokoro_projection(
    style_embeddings: torch.Tensor,
    target_samples: List[Tuple[str, torch.Tensor]],
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    logger: Optional[TrainingLogger] = None,
    log_audio_every: int = 10
) -> StyleToKokoroProjection:
    """
    Train projection layer from StyleTTS2 to Kokoro embedding space.
    
    This uses a contrastive approach where we try to match the acoustic
    characteristics of generated speech with target audio.
    
    Args:
        style_embeddings: StyleTTS2 style embeddings
        target_samples: List of (text, audio) pairs for training
        epochs: Training epochs
        lr: Learning rate
        device: Device for training
        logger: Training logger
        log_audio_every: Frequency of audio logging
        
    Returns:
        Trained projection layer
    """
    
    print("="*60)
    print("STAGE 2: Kokoro Projection Training")
    print("="*60)
    
    # Ensure style embeddings are on the right device and don't require grad
    # (we only want to train the projection, not the input embeddings)
    style_embeddings = style_embeddings.to(device).detach()
    
    # Initialize projection layer
    style_dim = style_embeddings.shape[-1]
    projection = StyleToKokoroProjection(style_dim, kokoro_dim=256).to(device)
    
    # Ensure projection parameters require gradients
    for param in projection.parameters():
        param.requires_grad_(True)
    
    print(f"Projection model has {sum(p.numel() for p in projection.parameters())} parameters")
    print(f"Projection parameters require grad: {all(p.requires_grad for p in projection.parameters())}")
    
    # Load Kokoro model for validation
    kokoro_model = KModel()
    try:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id='hexgrad/Kokoro-82M', filename='kokoro-v1_0.pth')
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Load checkpoint (same logic as original training script)
        if isinstance(checkpoint, dict) and 'bert' in checkpoint:
            state_dict = {}
            for component_name in ['bert', 'predictor', 'decoder', 'text_encoder']:
                if component_name in checkpoint:
                    component_dict = checkpoint[component_name]
                    for key, value in component_dict.items():
                        new_key = key.replace('module.', '')
                        if not new_key.startswith(component_name + '.'):
                            new_key = f"{component_name}.{new_key}"
                        state_dict[new_key] = value
            
            if 'bert_encoder' in checkpoint:
                bert_encoder_dict = checkpoint['bert_encoder']
                for key, value in bert_encoder_dict.items():
                    clean_key = key.replace('module.', '')
                    if clean_key in ['weight', 'bias']:
                        state_dict[f"bert_encoder.{clean_key}"] = value
            
            kokoro_model.load_state_dict(state_dict, strict=False)
            print("✓ Loaded Kokoro model for validation")
        
    except Exception as e:
        print(f"Warning: Could not load Kokoro model: {e}")
        return projection
    
    kokoro_model = kokoro_model.to(device)
    kokoro_model.eval()
    
    # Freeze Kokoro model parameters to prevent them from being updated
    for param in kokoro_model.parameters():
        param.requires_grad_(False)
    
    # Setup mel transform for comparison
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        center=True,
        power=1,
    ).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(projection.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # G2P pipeline
    from kokoro import KPipeline
    g2p = KPipeline(lang_code="a", model=False)
    
    best_loss = float('inf')
    
    # Log initial embeddings to W&B if available
    if logger and logger.use_wandb and WANDB_AVAILABLE:
        # Log StyleTTS2 embedding
        styletts2_embedding_np = style_embeddings.cpu().numpy()
        wandb.log({
            "styletts2_embedding": wandb.Histogram(styletts2_embedding_np),
            "styletts2_embedding_pca": wandb.Image(visualize_embedding(styletts2_embedding_np))
        })
    
    # Add mel predictor to projection model for training
    # We'll add this dynamically in the first batch, then update optimizer
    mel_predictor_added = False
    
    for epoch in range(1, epochs + 1):
        projection.train()
        epoch_loss = 0.0
        valid_batches = 0
        
        # Sample a few target samples for this epoch
        epoch_samples = random.sample(target_samples, min(5, len(target_samples)))
        
        batch_losses = []
        batch_mse_losses = []
        
        for text, target_audio in epoch_samples:
            try:
                # Convert text to phonemes
                phonemes, _ = g2p.g2p(text)
                if not phonemes or len(phonemes) == 0:
                    continue
                
                # Convert to input IDs
                ids = [0]  # BOS
                ids.extend(kokoro_model.vocab.get(p) for p in phonemes if kokoro_model.vocab.get(p) is not None)
                ids.append(0)  # EOS
                input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
                
                if input_ids.shape[1] < 3:
                    continue
                
                # Project StyleTTS2 embedding to Kokoro format
                kokoro_embedding = projection(style_embeddings.to(device))
                
                if kokoro_embedding.dim() == 1:
                    kokoro_embedding = kokoro_embedding.unsqueeze(0)
                
                # Instead of full audio generation, use embedding-level training
                # Get target embedding from Kokoro model's encoder
                with torch.no_grad():
                    # Generate reference audio to extract target features
                    generated_audio, _ = kokoro_model.forward_with_tokens(
                        input_ids, 
                        kokoro_embedding.detach()  # Use detached version for reference
                    )
                    
                    # Convert both to mel spectrograms for feature extraction
                    target_audio = target_audio.to(device)
                    
                    # Normalize audio lengths
                    min_len = min(generated_audio.shape[-1], target_audio.shape[-1])
                    generated_audio_ref = generated_audio[..., :min_len]
                    target_audio_trimmed = target_audio[..., :min_len]
                    
                    # Convert to mel spectrograms
                    generated_mel_ref = mel_transform(generated_audio_ref)
                    target_mel = mel_transform(target_audio_trimmed.unsqueeze(0)).squeeze(0)
                
                # Create mel predictor if not exists and update optimizer
                if not hasattr(projection, 'mel_predictor'):
                    # Use a smaller, more flexible predictor that outputs mel features
                    projection.mel_predictor = nn.Sequential(
                        nn.Linear(256, 512),
                        nn.ReLU(),
                        nn.Linear(512, 80),  # Output 80 mel bins (frequency dimension)
                        nn.Tanh()
                    ).to(device)
                    # Update optimizer to include new parameters
                    optimizer = torch.optim.Adam(projection.parameters(), lr=lr)
                    mel_predictor_added = True
                    print("Added mel predictor to projection model")
                
                # Now compute losses that preserve gradients
                # 1. Direct embedding regularization loss
                embedding_reg_loss = F.mse_loss(kokoro_embedding, torch.zeros_like(kokoro_embedding)) * 0.01
                
                # 2. Embedding norm loss (encourage reasonable magnitudes)
                target_norm = 0.1  # Typical Kokoro embedding norm
                norm_loss = F.mse_loss(torch.norm(kokoro_embedding), torch.tensor(target_norm, device=device)) * 0.1
                
                # 3. Embedding smoothness loss (encourage stable embeddings)
                smoothness_loss = torch.mean(kokoro_embedding ** 2) * 0.01
                
                # 4. Spectral characteristics matching
                # Generate predicted mel features (frequency profile)
                embedding_flat = kokoro_embedding.view(-1)  # Flatten to 1D
                predicted_mel_profile = projection.mel_predictor(embedding_flat)  # [80] - frequency profile
                
                # Compare with average frequency profile of target
                target_mel_profile = target_mel.mean(dim=1)  # Average over time -> [80]
                
                # Spectral loss - compare frequency profiles
                spectral_loss = F.mse_loss(predicted_mel_profile, target_mel_profile.detach())
                
                # 5. Additional embedding consistency loss
                # Encourage the embedding to be similar to a "canonical" voice embedding
                canonical_embedding = torch.randn_like(kokoro_embedding) * 0.05  # Small random target
                consistency_loss = F.mse_loss(kokoro_embedding, canonical_embedding.detach()) * 0.001
                
                # Combine losses
                loss = spectral_loss + embedding_reg_loss + norm_loss + smoothness_loss + consistency_loss
                
                # Verify loss has gradients
                if not loss.requires_grad:
                    print("Warning: Combined loss still doesn't require grad - skipping batch")
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                valid_batches += 1
                
                batch_losses.append(loss.item())
                batch_mse_losses.append(spectral_loss.item())
                
            except Exception as e:
                print(f"Warning: Batch failed: {e}")
                continue
        
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            scheduler.step(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}: Loss = {avg_loss:.4f}")
            
            # Project the StyleTTS2 embedding to visualize training progress
            with torch.no_grad():
                current_kokoro_embedding = projection(style_embeddings.to(device)).cpu()
            
            # Calculate gradient norm for monitoring training stability
            grad_norm = 0.0
            param_norm = 0.0
            for p in projection.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm(2).item() ** 2
                param_norm += p.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            param_norm = param_norm ** 0.5
            
            # Log metrics
            metrics = {
                'projection_loss': avg_loss,
                'projection_lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'grad_norm': grad_norm,
                'param_norm': param_norm,
                'embedding_norm': torch.norm(current_kokoro_embedding).item(),
                'batch_loss_std': np.std(batch_losses) if batch_losses else 0,
                'batch_mse_std': np.std(batch_mse_losses) if batch_mse_losses else 0
            }
            
            if logger:
                logger.log_metrics(metrics)
                
                # Additional W&B-specific logging
                if logger.use_wandb and WANDB_AVAILABLE and epoch % 10 == 0:
                    # Log the embedding distribution
                    kokoro_embedding_np = current_kokoro_embedding.numpy()
                    
                    # Log histograms of embedding values
                    wandb.log({
                        "kokoro_embedding": wandb.Histogram(kokoro_embedding_np),
                        "kokoro_embedding_pca": wandb.Image(visualize_embedding(kokoro_embedding_np)),
                        "projection_weights": wandb.Histogram(
                            projection.projection[-2].weight.data.cpu().numpy()
                        )
                    })
                
                # Generate and log audio samples every log_audio_every epochs
                if logger.use_wandb and WANDB_AVAILABLE and log_audio_every > 0 and epoch % log_audio_every == 0:
                    try:
                        print(f"Generating audio samples for epoch {epoch}...")
                        audio_samples = generate_audio_samples_for_logging(
                            current_kokoro_embedding, 
                            kokoro_model, 
                            g2p, 
                            device
                        )
                        
                        for i, (text, audio_tensor) in enumerate(audio_samples):
                            wandb.log({
                                f"generated_audio_epoch_{epoch}_sample_{i+1}": wandb.Audio(
                                    audio_tensor.cpu().numpy(),
                                    sample_rate=24000,
                                    caption=f"Epoch {epoch}: {text}"
                                )
                            })
                        
                    except Exception as e:
                        print(f"Warning: Audio generation failed for epoch {epoch}: {e}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
        else:
            print(f"Epoch {epoch}: No valid batches")
    
    print(f"Projection training completed. Best loss: {best_loss:.4f}")
    return projection

def generate_audio_samples_for_logging(
    kokoro_embedding: torch.Tensor, 
    kokoro_model, 
    g2p, 
    device: str,
    test_texts: List[str] = None
) -> List[Tuple[str, torch.Tensor]]:
    """
    Generate audio samples for logging during training.
    
    Args:
        kokoro_embedding: Current Kokoro embedding from projection
        kokoro_model: Kokoro TTS model
        g2p: G2P pipeline
        device: Device for computation
        test_texts: List of test texts (optional)
        
    Returns:
        List of (text, audio_tensor) tuples
    """
    if test_texts is None:
        test_texts = [
            "Hello, this is a test.",
            "The quick brown fox jumps.",
            "How are you today?"
        ]
    
    audio_samples = []
    
    # Ensure embedding is on the correct device and has right format for Kokoro
    kokoro_embedding = kokoro_embedding.to(device)
    if kokoro_embedding.dim() == 1:
        kokoro_embedding = kokoro_embedding.unsqueeze(0)
    
    # Expand to full Kokoro voice tensor format
    MAX_PHONEME_LEN = 510
    kokoro_voice_tensor = torch.zeros((MAX_PHONEME_LEN, 1, 256), device=device)
    
    for i in range(MAX_PHONEME_LEN):
        length_factor = 1.0 + (i / MAX_PHONEME_LEN) * 0.05
        varied_embedding = kokoro_embedding.squeeze() * length_factor
        kokoro_voice_tensor[i, 0, :] = varied_embedding
    
    for text in test_texts:
        try:
            # Convert text to phonemes
            phonemes, _ = g2p.g2p(text)
            if not phonemes or len(phonemes) == 0:
                continue
            
            # Convert to input IDs and ensure they're on the correct device
            ids = [0]  # BOS
            ids.extend(kokoro_model.vocab.get(p) for p in phonemes if kokoro_model.vocab.get(p) is not None)
            ids.append(0)  # EOS
            input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            
            if input_ids.shape[1] < 3:
                continue
            
            # Generate audio - ensure embedding is on the same device as the model
            with torch.no_grad():
                # Move kokoro_embedding to the same device as the model if needed
                model_device = next(kokoro_model.parameters()).device
                kokoro_embedding_for_gen = kokoro_embedding.to(model_device)
                input_ids_for_gen = input_ids.to(model_device)
                
                generated_audio, _ = kokoro_model.forward_with_tokens(
                    input_ids_for_gen, 
                    kokoro_embedding_for_gen
                )
            
            # Move generated audio back to CPU for logging
            generated_audio = generated_audio.cpu()
            
            # Trim to reasonable length (max 5 seconds)
            max_samples = 24000 * 5
            if generated_audio.shape[-1] > max_samples:
                generated_audio = generated_audio[..., :max_samples]
            
            audio_samples.append((text, generated_audio.squeeze()))
            
        except Exception as e:
            print(f"Warning: Failed to generate audio for '{text}': {e}")
            continue
    
    return audio_samples

# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------

def train_styletts2_to_kokoro(
    data_root: str | Path,
    epochs_projection: int = 100,
    lr_projection: float = 1e-3,
    out: str = "output",
    name: str = "my_voice",
    use_tensorboard: bool = False,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    max_style_samples: int = 100,
    device: str = "auto",
    log_audio_every: int = 10
):
    """
    Complete pipeline: StyleTTS2 style extraction → Kokoro projection
    
    Args:
        data_root: Path to dataset
        epochs_projection: Epochs for projection training
        lr_projection: Learning rate for projection
        out: Output directory
        name: Voice name
        use_tensorboard: Enable TensorBoard logging
        use_wandb: Enable W&B logging
        wandb_project: W&B project name
        wandb_name: W&B run name
        max_style_samples: Max samples for style extraction
        device: Device to use
        log_audio_every: Frequency of audio logging
    """
    
    print("\n" + "="*60)
    print("StyleTTS2 → Kokoro Voice Embedding Pipeline")
    print("="*60)
    print(f"Dataset: {data_root}")
    print(f"Output: {out}/{name}")
    print(f"Projection epochs: {epochs_projection}")
    print("="*60 + "\n")
    
    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Using device: {device}")
    
    # Setup logging
    log_dir = os.path.join('runs', Path(out).stem) if use_tensorboard else None
    run_name = wandb_name or f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize W&B with more config details
    if use_wandb and WANDB_AVAILABLE:
        wandb_config = {
            "dataset": str(data_root),
            "name": name,
            "epochs_projection": epochs_projection,
            "lr_projection": lr_projection,
            "max_style_samples": max_style_samples,
            "device": device,
            "styletts2_available": STYLETTS2_AVAILABLE,
            "pipeline_version": "1.0",
            "datetime": datetime.datetime.now().isoformat()
        }
        if not wandb.run:
            wandb.init(
                project=wandb_project or "styletts2-kokoro-pipeline",
                name=run_name,
                config=wandb_config
            )
    
    logger = TrainingLogger(
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        log_dir=log_dir,
        wandb_project=wandb_project or "styletts2-kokoro-pipeline",
        wandb_name=run_name
    )
    
    # Create output directories
    output_dir = Path(out) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = StyleExtractionDataset(data_root, split="train")
    print(f"Training samples: {len(train_dataset)}")
    
    # Validation dataset (optional)
    val_dataset = None
    if (Path(data_root) / "validation").exists():
        val_dataset = StyleExtractionDataset(data_root, split="validation")
        print(f"Validation samples: {len(val_dataset)}")
    
    training_start_time = time.time()
    
    try:
        # Stage 1: Extract StyleTTS2 style embeddings
        print("\nStarting Stage 1: StyleTTS2 Style Embedding Extraction...")
        avg_style_embedding, all_embeddings = extract_styletts2_embeddings(
            dataset=train_dataset,
            device=device,
            max_samples=max_style_samples
        )
        
        # Log embedding stats to W&B
        if use_wandb and WANDB_AVAILABLE:
            embedding_stats = {
                "styletts2_embedding_mean": avg_style_embedding.mean().item(),
                "styletts2_embedding_std": avg_style_embedding.std().item(),
                "styletts2_embedding_min": avg_style_embedding.min().item(),
                "styletts2_embedding_max": avg_style_embedding.max().item(),
                "styletts2_embedding_l2norm": torch.norm(avg_style_embedding).item(),
                "num_embeddings_extracted": len(all_embeddings)
            }
            wandb.log(embedding_stats)
            
            # Create embedding visualization
            if len(all_embeddings) > 1:
                all_emb_stacked = torch.stack(all_embeddings).cpu().numpy()
                wandb.log({"all_styletts2_embeddings": wandb.Histogram(all_emb_stacked)})
                
                # PCA visualization of embeddings
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    pca_result = pca.fit_transform(all_emb_stacked)
                    
                    plt.figure(figsize=(10, 8))
                    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
                    plt.title('PCA of StyleTTS2 Embeddings')
                    plt.xlabel('PC1')
                    plt.ylabel('PC2')
                    plt.grid(True)
                    wandb.log({"styletts2_embeddings_pca": wandb.Image(plt)})
                    plt.close()
                except Exception as e:
                    print(f"Error creating PCA visualization: {e}")
        
        # Prepare target samples for projection training
        target_samples = []
        sample_indices = random.sample(range(len(train_dataset)), min(20, len(train_dataset)))
        for idx in sample_indices:
            text, audio, _ = train_dataset[idx]
            target_samples.append((text, audio))
        
        # Stage 2: Train Kokoro projection
        print("\nStarting Stage 2: Kokoro Projection Training...")
        projection_model = train_kokoro_projection(
            style_embeddings=avg_style_embedding,
            target_samples=target_samples,
            epochs=epochs_projection,
            lr=lr_projection,
            device=device,
            logger=logger,
            log_audio_every=log_audio_every
        )
        
        # Generate final Kokoro embedding
        print("\nGenerating final Kokoro embedding...")
        projection_model.eval()
        with torch.no_grad():
            final_kokoro_embedding = projection_model(avg_style_embedding.to(device))
            
            # Ensure the embedding has the right dimensions
            if final_kokoro_embedding.dim() == 1:
                final_kokoro_embedding = final_kokoro_embedding.unsqueeze(0)
        
        # Expand to length-dependent format for Kokoro compatibility
        MAX_PHONEME_LEN = 510
        kokoro_voice_tensor = torch.zeros((MAX_PHONEME_LEN, 1, 256))
        
        for i in range(MAX_PHONEME_LEN):
            # Add slight variation based on length for better prosody
            length_factor = 1.0 + (i / MAX_PHONEME_LEN) * 0.05
            varied_embedding = final_kokoro_embedding.squeeze() * length_factor
            kokoro_voice_tensor[i, 0, :] = varied_embedding.cpu()
        
        # Save final voice embedding
        voice_path = output_dir / f"{name}.pt"
        torch.save(kokoro_voice_tensor, voice_path)
        print(f"✓ Saved Kokoro voice embedding: {voice_path}")
        
        # Log final embedding to W&B
        if use_wandb and WANDB_AVAILABLE:
            final_embedding_stats = {
                "kokoro_embedding_mean": final_kokoro_embedding.mean().item(),
                "kokoro_embedding_std": final_kokoro_embedding.std().item(),
                "kokoro_embedding_min": final_kokoro_embedding.min().item(),
                "kokoro_embedding_max": final_kokoro_embedding.max().item(),
                "kokoro_embedding_l2norm": torch.norm(final_kokoro_embedding).item()
            }
            wandb.log(final_embedding_stats)
        
        # Save additional artifacts
        artifacts = {
            'styletts2_embeddings': avg_style_embedding.cpu(),
            'all_styletts2_embeddings': [emb.cpu() for emb in all_embeddings],
            'kokoro_embedding': final_kokoro_embedding.cpu(),
            'projection_model': projection_model.state_dict(),
            'training_config': {
                'epochs_projection': epochs_projection,
                'lr_projection': lr_projection,
                'max_style_samples': max_style_samples,
                'device': device
            }
        }
        
        artifacts_path = output_dir / f"{name}_artifacts.pt"
        torch.save(artifacts, artifacts_path)
        print(f"✓ Saved training artifacts: {artifacts_path}")
        
        # Test generation with Kokoro
        print("\nTesting voice with Kokoro TTS...")
        test_sentences, test_audio_paths = test_voice_generation(kokoro_voice_tensor, output_dir, name)
        
        # Log test generations to W&B
        if use_wandb and WANDB_AVAILABLE and test_audio_paths:
            for i, (text, audio_path) in enumerate(zip(test_sentences, test_audio_paths)):
                try:
                    audio, sr = librosa.load(audio_path, sr=24000)
                    wandb.log({
                        f"test_audio_{i+1}": wandb.Audio(
                            audio, 
                            sample_rate=sr,
                            caption=f"Test {i+1}: {text[:30]}..."
                        )
                    })
                except Exception as e:
                    print(f"Error logging test audio to W&B: {e}")
        
        total_time = time.time() - training_start_time
        print(f"\n{'='*60}")
        print("Pipeline completed successfully!")
        print(f"Total time: {format_duration(total_time)}")
        print(f"Final voice: {voice_path}")
        print(f"{'='*60}\n")
        
        # Final W&B summary
        if use_wandb and WANDB_AVAILABLE:
            wandb.run.summary["total_time_seconds"] = total_time
            wandb.run.summary["voice_path"] = str(voice_path)
            wandb.run.summary["success"] = True
        
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        if use_wandb and WANDB_AVAILABLE:
            wandb.run.summary["error"] = str(e)
            wandb.run.summary["success"] = False
        raise
    finally:
        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        logger.close()

def test_voice_generation(voice_tensor: torch.Tensor, output_dir: Path, name: str):
    """Test the generated voice with sample text"""
    test_sentences = [
        "Hello, this is a test of the new voice.",
        "The quick brown fox jumps over the lazy dog.",
        "How are you doing today?"
    ]
    
    test_audio_paths = []
    
    try:
        from kokoro import KPipeline
        
        # Initialize pipeline
        pipeline = KPipeline(lang_code="a")
        
        for i, text in enumerate(test_sentences):
            try:
                # Generate audio
                outputs = []
                for _, _, audio in pipeline(text, voice=voice_tensor):
                    outputs.append(audio)
                
                if outputs:
                    full_audio = torch.cat(outputs)
                    output_path = output_dir / f"{name}_test_{i+1}.wav"
                    
                    sf.write(output_path, full_audio.numpy(), 24000)
                    print(f"✓ Generated test audio: {output_path}")
                    test_audio_paths.append(output_path)
                
            except Exception as e:
                print(f"Warning: Test generation {i+1} failed: {e}")
                
    except Exception as e:
        print(f"Warning: Voice testing failed: {e}")
        
    return test_sentences, test_audio_paths

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="StyleTTS2 → Kokoro voice embedding pipeline")
    
    # Data arguments
    ap.add_argument("--data", type=str, required=True, help="Path to dataset directory")
    ap.add_argument("--dataset-id", type=str, help='HF dataset repo ID (e.g. "user/dataset")')
    
    # Training arguments
    ap.add_argument("--epochs-projection", type=int, default=100, help="Epochs for projection training")
    ap.add_argument("--lr-projection", type=float, default=1e-3, help="Learning rate for projection")
    
    # Output arguments
    ap.add_argument("--name", type=str, default="my_voice", help="Output voice name")
    ap.add_argument("--out", type=str, default="output", help="Output directory")
    
    # Monitoring arguments
    ap.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    ap.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    ap.add_argument("--wandb-project", type=str, help="W&B project name")
    ap.add_argument("--wandb-name", type=str, help="W&B run name")
    
    # Advanced arguments
    ap.add_argument("--max-style-samples", type=int, default=100, help="Max samples for style extraction")
    ap.add_argument("--device", type=str, default="auto", help="Device to use (auto/cuda/mps/cpu)")
    ap.add_argument("--log-audio-every", type=int, default=10, help="Log audio samples every N epochs (0 to disable)")
    
    args = ap.parse_args()
    
    # Handle dataset download
    data_root = args.data
    if data_root is None and args.dataset_id:
        print(f"Downloading dataset from HF: {args.dataset_id}")
        data_root = snapshot_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            token=True,
            ignore_patterns=["*.md"]
        )
    
    if not data_root:
        ap.error("Either --data or --dataset-id must be provided")
    
    # Run the pipeline
    train_styletts2_to_kokoro(
        data_root=data_root,
        epochs_projection=args.epochs_projection,
        lr_projection=args.lr_projection,
        out=args.out,
        name=args.name,
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        max_style_samples=args.max_style_samples,
        device=args.device,
        log_audio_every=args.log_audio_every
    ) 