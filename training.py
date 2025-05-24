"""training.py
Train a custom Kokoro TTS voice embedding (single 256-dim vector)
----------------------------------------------------------------
This script follows the guidance in the Kokoro research doc:
    * Freeze the 82 M parameter Kokoro model
    * Optimise only a 256-value voice tensor (128 timbre | 128 style)
    * Compare generated speech to reference audio in the log-mel domain
    * Works on Apple-Silicon (MPS) or CUDA / CPU

Expected dataset layout
----------------------
<dataset>/
    train/           # Training data 
        metadata.jsonl   # JSON Lines format: {"file_name": "segment_0000.wav", "text": "<sentence>"}
        segment_*.wav    # Audio files
    validation/      # Validation data (optional)
        metadata.jsonl   # Same format as train
        segment_*.wav    # Audio files
    test/            # Test data (optional)
        metadata.jsonl   # Same format as train
        segment_*.wav    # Audio files

Usage
-----
python training.py --data ./my_dataset --epochs 200 --out my_voice.pt

Tip: set env var to enable MPS GPU on Apple-Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
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
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tempfile
import shutil

from accelerate import Accelerator
from huggingface_hub import snapshot_download, HfApi, upload_folder
from torchaudio.functional import spectrogram

# Import refactored utilities
from utils import VoiceLoss, TrainingLogger, VoiceEmbedding, calculate_voice_drift

from dotenv import load_dotenv
load_dotenv()


# Optional W&B import - will be checked before use
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from kokoro import KModel, KPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Device management is now handled by accelerate
# This function is kept for backward compatibility with other scripts
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def pad_sequence_to_length(sequence, target_length, pad_value=0):
    """Pad a single tensor to a specific length along the last dimension."""
    pad_size = target_length - sequence.shape[-1]
    if pad_size <= 0:
        return sequence
    # Only pad the last dimension on the right side
    padded = torch.nn.functional.pad(sequence, (0, pad_size), value=pad_value)
    return padded

def dynamic_time_truncation(mel_spec1, mel_spec2):
    """Truncate two mel spectrograms to the same length (minimum of both)."""
    min_time = min(mel_spec1.shape[-1], mel_spec2.shape[-1])
    return mel_spec1[..., :min_time], mel_spec2[..., :min_time]

def rms_normalize(audio, target_rms=0.1):
    """Normalize audio to target RMS value.
    
    Args:
        audio: Audio tensor [batch, samples] or [samples]
        target_rms: Target RMS value to normalize to
        
    Returns:
        RMS-normalized audio of same shape
    """
    # Ensure audio is 2D with batch dimension
    is_1d = audio.dim() == 1
    if is_1d:
        audio = audio.unsqueeze(0)
        
    # Calculate current RMS values
    rms = torch.sqrt(torch.mean(audio ** 2, dim=-1, keepdim=True))
    
    # Normalize to target RMS (avoid division by zero)
    eps = 1e-8
    scaling_factor = target_rms / (rms + eps)
    normalized = audio * scaling_factor
    
    # Return same dimensions as input
    if is_1d:
        normalized = normalized.squeeze(0)
        
    return normalized

def estimate_time_remaining(epoch: int, total_epochs: int, epoch_start_time: float) -> str:
    """Estimate time remaining based on current epoch duration."""
    if epoch == 0:
        return "calculating..."
    
    epoch_duration = time.time() - epoch_start_time
    remaining_epochs = total_epochs - epoch
    estimated_seconds = epoch_duration * remaining_epochs
    
    return format_duration(estimated_seconds)

class EarlyStopping:
    """Early stopping helper to prevent overfitting."""
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

class VoiceDataset(Dataset):
    """Loads (<sentence>, log-mel target) pairs for training."""

    def __init__(self, root: str | Path, mel_transform: nn.Module, target_sr: int = 24_000, device=None, split="train"):
        self.device = device or torch.device("cpu")
        self.root = Path(root)
        self.sentences: List[str] = []
        self.wav_paths: List[Path] = []
        self.mel_transform = mel_transform
        self.target_sr = target_sr
        self.split = split

        # Load from the specified split directory (default: train)
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

    def __len__(self) -> int:  # noqa: D401
        return len(self.sentences)

    def _load_wav(self, path: Path) -> torch.Tensor:
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)  # mono
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        return wav

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        text = self.sentences[idx]
        wav = self._load_wav(self.wav_paths[idx])
        
        # Keep mel transforms on CPU during dataset loading to avoid device issues
        # The tensors will be moved to the appropriate device in the training loop
        mel = self.mel_transform(wav.unsqueeze(0))  # (1, n_mels, T)
        log_mel = 20 * torch.log10(mel.clamp(min=1e-5))
        # Keep batch dimension for consistency
        return text, log_mel, wav.unsqueeze(0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def text_to_input_ids(model: KModel, phonemes: str) -> torch.LongTensor:
    ids = [0]  # BOS
    ids.extend(model.vocab.get(p) for p in phonemes if model.vocab.get(p) is not None)
    ids.append(0)  # EOS
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, L)

# 510 is max phoneme length in Kokoro
MAX_PHONEME_LEN = 510

def train(
    data_root: str | Path,
    epochs: int = 200,
    batch_size: int = 4,
    lr: float = 3e-4,  # Reduced default learning rate (5-10x lower than original)
    out: str = "output",
    name: str = "my_voice",
    n_mels: int = 80,
    n_fft: int = 1024,
    hop: int = 256,
    use_tensorboard: bool = False,
    use_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    log_audio_every: int = 10,
    log_dir: Optional[str] = None,
    gradient_accumulation_steps: int = 1,
    memory_efficient: bool = True,
    # Advanced training controls
    timbre_warning_threshold: Optional[float] = 0.3, # Warn if timbre std exceeds this value (no freezing)
    style_regularization: Optional[float] = 1e-5,   # L2 regularization on style part of embedding (default: 1e-5)
    skip_validation: bool = False,                  # Skip validation split
    save_best: bool = True,                        # Save best checkpoint (lowest validation L1 loss after epoch 5)
    upload_to_hf: bool = False,                    # Upload model and artifacts to HuggingFace
    hf_repo_id: Optional[str] = None,              # Repository ID for HuggingFace upload
    checkpoint_every: int = 10,                    # Save checkpoint every N epochs
    patience: int = 15,                            # Early stopping patience
    no_reference_voice: bool = False,              # Disable automatic reference voice selection
    manual_voice_id: Optional[str] = None,         # Manually specify a Kokoro voice ID to use as base
    accent: str = 'auto',                          # Accent preference for voice selection
):
    """Train a Kokoro voice embedding for audiobook narration.
    
    This trains a length-dependent voice embedding (510 × 1 × 256 tensor) that captures
    your unique timbre and speaking style. The pretrained Kokoro model stays frozen;
    only the voice embedding is optimized.
    
    Key features:
    - Length-dependent embeddings for better prosody modeling
    - Multi-scale spectral loss for high-quality voice matching
    - Automatic checkpointing and early stopping
    - Memory-efficient training on Apple Silicon
    """
    
    # Print training configuration
    print("\n" + "="*60)
    print("Kokoro Voice Training - Audiobook Quality")
    print("="*60)
    print(f"Dataset: {data_root}")
    print(f"Output: {out}/{name}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Early stopping patience: {patience}")
    print("="*60 + "\n")
    
    # Initialize accelerator for mixed precision and gradient accumulation
    # Auto-detect the best precision based on device capabilities
    mixed_precision = "no"  # Default to no mixed precision for compatibility
    if torch.cuda.is_available():
        # Check if device supports bf16
        try:
            torch.cuda.is_bf16_supported()
            mixed_precision = "bf16"
        except:
            mixed_precision = "fp16"
    
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        device_placement=True,
    )
    device = accelerator.device
    
    print(f"Using device: {device} with {accelerator.state.mixed_precision} precision")
    
    # Memory optimization settings
    if memory_efficient and device.type == "mps":
        print("Enabling memory optimization settings for MPS")
        # Force garbage collection to run more aggressively
        gc.enable()
        print(f"MPS HIGH_WATERMARK_RATIO set to {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'default')}")
    
    print(f"Gradient accumulation steps: {accelerator.gradient_accumulation_steps}")
    
    # Set up unified logger for TensorBoard and W&B
    log_dir = log_dir or os.path.join('runs', Path(out).stem)
    if use_tensorboard:
        os.makedirs(log_dir, exist_ok=True)
    
    # Initialize logger
    run_name = wandb_name or f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    logger = TrainingLogger(
        use_tensorboard=use_tensorboard, 
        use_wandb=use_wandb,
        log_dir=log_dir,
        wandb_project=wandb_project or "kokoro-voice-training",
        wandb_name=run_name
    )
    
    # Store the config for W&B
    if use_wandb:
        import wandb
        wandb.config.update({
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop_length": hop,
            "output": out,
            "data_root": str(data_root),
        })
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Load pretrained Kokoro model
    print("Loading pretrained Kokoro model...")
    model = KModel()
    
    # Download and load pretrained weights
    try:
        from huggingface_hub import hf_hub_download
        # Download the pretrained model weights
        model_path = hf_hub_download(repo_id='hexgrad/Kokoro-82M', filename='kokoro-v1_0.pth')
        
        # Load the checkpoint which contains the full model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # The checkpoint contains separate components
        if isinstance(checkpoint, dict) and 'bert' in checkpoint:
            # Create a unified state dict from all components
            state_dict = {}
            
            # Process each component
            for component_name in ['bert', 'predictor', 'decoder', 'text_encoder']:
                if component_name in checkpoint:
                    component_dict = checkpoint[component_name]
                    for key, value in component_dict.items():
                        # Remove 'module.' prefix if present (from DataParallel)
                        new_key = key.replace('module.', '')
                        # Add component prefix if not already there
                        if not new_key.startswith(component_name + '.'):
                            new_key = f"{component_name}.{new_key}"
                        state_dict[new_key] = value
            
            # Handle bert_encoder separately as it maps to a different key
            if 'bert_encoder' in checkpoint:
                # bert_encoder is a simple linear layer with weight and bias
                bert_encoder_dict = checkpoint['bert_encoder']
                for key, value in bert_encoder_dict.items():
                    # Map directly without module prefix
                    clean_key = key.replace('module.', '')
                    if clean_key in ['weight', 'bias']:
                        state_dict[f"bert_encoder.{clean_key}"] = value
            
            # Load the processed state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️  Warning: {len(missing_keys)} keys are missing from checkpoint")
                # Only show first few missing keys to avoid clutter
                print(f"   First missing keys: {missing_keys[:5]}")
                
            if unexpected_keys:
                print(f"⚠️  Warning: {len(unexpected_keys)} unexpected keys in checkpoint")
                print(f"   First unexpected keys: {unexpected_keys[:5]}")
                
            print("✓ Loaded pretrained Kokoro weights (non-strict mode)")
        else:
            raise ValueError("Unexpected checkpoint format")
            
    except Exception as e:
        print(f"ERROR: Failed to load pretrained weights: {e}")
        print("The model MUST have pretrained weights to generate speech!")
        raise
    
    # Move model to device AFTER loading weights
    model = model.to(device)
    
    # Freeze the model - we're not fine-tuning it, just the embedding
    for param in model.parameters():
        param.requires_grad = False
        
    # Ensure model is in eval mode for gradient flow to embedding only
    model.eval()

    # Set up mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        center=True,
        power=1,  # amplitude, not power
    ).to(device)
    
    # Keep a separate CPU version for inference
    mel_transform_cpu = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        center=True,
        power=1,
    )
    
    # Initialize loss calculator
    loss_calculator = VoiceLoss(device=device, fft_sizes=[1024, 512, 256])

    # Create dataset
    dataset = VoiceDataset(data_root, mel_transform_cpu, split="train")
    validation_dataset = None
    
    # Dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(dataset)}")
    
    # If validation directory exists and we requested validation data, load it
    if not skip_validation and (Path(data_root) / "validation").exists():
        validation_dataset = VoiceDataset(data_root, mel_transform_cpu, split="validation")
        print(f"Validation samples: {len(validation_dataset)}")
    else:
        print("Validation samples: 0 (using training data for monitoring)")
    
    # Estimate training time
    samples_per_epoch = len(dataset)
    estimated_steps = samples_per_epoch * epochs
    print(f"Estimated total steps: {estimated_steps:,}")
    print()
    
    # Create data loaders - simpler configuration
    loader = DataLoader(
        dataset, 
        batch_size=1,  # Always use batch_size of 1 for stability
        shuffle=True
    )
    validation_loader = None
    if validation_dataset:
        validation_loader = DataLoader(
            validation_dataset, 
            batch_size=1,
            shuffle=False
        )

    # Initialize a full 3D voice tensor [510, 1, 256] to match Kokoro's expected format
    voice_embed = torch.zeros((MAX_PHONEME_LEN, 1, 256), device=device)
    
    # Initialize voice embedding with our utility class
    voice_embedding = VoiceEmbedding(embedding_size=256, max_phoneme_len=MAX_PHONEME_LEN, device=device)
    
    # Try to initialize from a Kokoro base voice for better convergence
    base_voice_loaded = False
    original_reference_voice = None  # Store for final comparison
    
    # Option 1: Try to load from Kokoro base voice (recommended)
    if not no_reference_voice:
        try:
            # Automatic voice selection if no manual voice specified
            if manual_voice_id is None and accent == 'auto':
                print("🎯 Automatically selecting best matching Kokoro voice...")
                try:
                    from auto_voice_selection import select_voice_automatically
                    auto_voice_id, auto_accent = select_voice_automatically(data_root, accent)
                    if auto_voice_id:
                        manual_voice_id = auto_voice_id
                        target_accent = auto_accent
                        print(f"✓ Auto-selected voice: {auto_voice_id} ({auto_accent})")
                    else:
                        target_accent = auto_accent  # Use fallback accent
                        print(f"Using default {auto_accent} voice")
                except Exception as e:
                    print(f"Auto-selection failed: {e}, using default")
                    target_accent = 'american'  # Final fallback
            else:
                # Use manual settings
                target_accent = accent if accent != 'auto' else 'american'
            
            print(f"Attempting to initialize from Kokoro {target_accent} English base voice...")
            base_voice = load_kokoro_base_voice(manual_voice_id, accent=target_accent, device=device)
            
            # Debug: print the actual shape of the loaded voice
            if base_voice is not None:
                print(f"Loaded voice shape: {base_voice.shape}")
            
            # Check if we have a valid voice vector (handle different possible shapes)
            valid_voice = False
            if base_voice is not None:
                if base_voice.dim() == 1 and base_voice.shape[0] == 256:
                    # Single voice vector - need to expand to [510, 1, 256]
                    valid_voice = True
                elif base_voice.dim() == 2 and base_voice.shape[-1] == 256:
                    # Extract the voice vector if it's in a batch format
                    base_voice = base_voice[0] if base_voice.shape[0] == 1 else base_voice.mean(dim=0)
                    valid_voice = True
                elif base_voice.dim() == 3 and base_voice.shape == (510, 1, 256):
                    # Already in the perfect format! [phoneme_len, batch, embed_dim]
                    print("Voice is already in perfect 3D format [510, 1, 256]")
                    valid_voice = True
                elif base_voice.numel() == 256:
                    # Flatten if needed
                    base_voice = base_voice.flatten()
                    valid_voice = True
            
            if valid_voice:
                voice_source = f"auto-selected {manual_voice_id}" if manual_voice_id else "default"
                print(f"Initializing from Kokoro base voice ({voice_source})")
                with torch.no_grad():
                    if base_voice.dim() == 3 and base_voice.shape == (510, 1, 256):
                        # Voice is already in perfect format - use directly with small variations
                        print("Using 3D voice directly with small random variations")
                        for i in range(voice_embedding.max_phoneme_len):
                            # Add slight random variation to each phoneme position
                            noise = torch.randn_like(base_voice[i]) * 0.03
                            voice_embedding.voice_embed[i, 0, :] = base_voice[i, 0, :] + noise.squeeze()
                    else:
                        # For 1D voice vectors, use as base with slight variations for each phoneme length
                        print("Expanding 1D voice to all phoneme positions with variations")
                        for i in range(voice_embedding.max_phoneme_len):
                            # Add slight random variation to avoid exact copy
                            # This helps with training dynamics while starting close to a good voice
                            noise = torch.randn_like(base_voice) * 0.03
                            voice_embedding.voice_embed[i, 0, :] = base_voice + noise
                        
                print("✓ Initialized from Kokoro base voice with small variations")
                base_voice_loaded = True
                original_reference_voice = base_voice  # Store for final comparison
                
        except Exception as e:
            print(f"Could not load Kokoro base voice: {e}")
    
    # Option 2: Fallback to manual HuggingFace download (backup method)
    if not base_voice_loaded:
        try:
            from huggingface_hub import hf_hub_download
            print("Trying manual voice download from HuggingFace...")
            
            # Try to get a reference voice from the official model  
            ref_path = hf_hub_download(repo_id='hexgrad/Kokoro-82M', filename='voices/af_heart.pt')
            ref_voice = torch.load(ref_path, map_location=device)
            
            # If it's a single voice vector, use it as base
            if ref_voice.dim() == 1 and ref_voice.shape[0] == 256:
                print("Initializing from downloaded Kokoro reference voice")
                with torch.no_grad():
                    # Use reference as base but with some variation
                    for i in range(voice_embedding.max_phoneme_len):
                        # Add slight random variation to avoid exact copy
                        noise = torch.randn_like(ref_voice) * 0.05
                        voice_embedding.voice_embed[i, 0, :] = ref_voice + noise
                        
                print("✓ Initialized from downloaded reference voice with variations")
                base_voice_loaded = True
                original_reference_voice = ref_voice  # Store for final comparison
        except Exception as e:
            print(f"Could not download reference voice: {e}")
            
    # Option 3: Try to initialize from local reference if available (final fallback)
    if not base_voice_loaded:
        try:
            # If reference embedding exists, we initialize from it
            ref_path = f"{out}/{name}/reference.pt"
            if not os.path.exists(ref_path):
                ref_path = "reference.pt"
                
            if os.path.exists(ref_path):
                print(f"Initializing from local reference embedding: {ref_path}")
                ref_data = torch.load(ref_path)
                ref_voice = ref_data["base_voice"][0]
                voice_embedding.base_voice.data = ref_data["base_voice"].to(device)
                voice_embedding.update_full_embedding()
                
                # Set target statistics
                voice_embedding.timbre_mean = ref_voice[:128].mean().item()
                voice_embedding.timbre_std = ref_voice[:128].std().item()
                voice_embedding.style_mean = ref_voice[128:].mean().item()
                voice_embedding.style_std = ref_voice[128:].std().item()
                
                print(f"Reference statistics:")
                print(f"  Timbre: mean={voice_embedding.timbre_mean:.4f}, std={voice_embedding.timbre_std:.4f}")
                print(f"  Style: mean={voice_embedding.style_mean:.4f}, std={voice_embedding.style_std:.4f}")
                base_voice_loaded = True
                original_reference_voice = ref_voice  # Store for final comparison
        except Exception as e:
            print(f"Could not load local reference: {e}")
    
    if not base_voice_loaded:
        print("Using default random initialization")

    # For backward compatibility - the rest of the code expects these variables
    # but now we use length-dependent embeddings instead of a single base_voice
    voice_embed = voice_embedding.voice_embed
    
    # Note: voice_embedding.base_voice is now a property that returns the average
    # of all length-dependent embeddings
    print("Using length-dependent voice embeddings!")
    print(f"Total trainable parameters: {voice_embed.numel():,d}")
    

    # Set up optimizer with weight decay for regularization
    # Now we optimize all length-dependent embeddings
    optim = torch.optim.Adam([voice_embedding.voice_embed], lr=lr, weight_decay=1e-6)
    
    # Prepare model, optimizer, and dataloaders with accelerator
    # voice_embedding.voice_embed is already a parameter and will be handled by the optimizer
    model, optim, loader = accelerator.prepare(
        model, optim, loader
    )
    if validation_loader is not None:
        validation_loader = accelerator.prepare(validation_loader)
    
    # Track best validation loss for saving best checkpoint
    best_val_loss = float('inf')
    best_epoch = 0

    # Flat-then-cosine schedule: keep LR constant for first 25 % epochs then decay to eta_min.
    import math
    warmup_epochs = max(1, int(0.25 * epochs))
    def lr_lambda(current_epoch: int):
        if current_epoch < warmup_epochs:
            return 1.0
        # progress ∈ [0,1] for the remaining epochs
        progress = (current_epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    
    # Prepare scheduler with accelerator
    scheduler = accelerator.prepare(scheduler)
    
    # Check for existing checkpoint to resume from
    resume_epoch = 0
    checkpoint_dir = Path(out) / name / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = sorted([f for f in checkpoint_dir.iterdir() if f.suffix == '.pt' and 'epoch' in f.name])
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            resume_epoch = int(latest_checkpoint.stem.split('_')[-1])
            print(f"Found checkpoint at epoch {resume_epoch}, resuming training...")
            checkpoint_data = torch.load(latest_checkpoint, map_location=device)
            voice_embedding.voice_embed.data = checkpoint_data['voice_embed'].to(device)
            if 'optimizer_state' in checkpoint_data:
                optim.load_state_dict(checkpoint_data['optimizer_state'])
            if 'scheduler_state' in checkpoint_data:
                scheduler.load_state_dict(checkpoint_data['scheduler_state'])
            if 'best_val_loss' in checkpoint_data:
                best_val_loss = checkpoint_data['best_val_loss']
                best_epoch = checkpoint_data.get('best_epoch', 0)
    
    # Multiple loss functions for better results
    l1_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()

    # G2P pipeline for text -> phonemes
    # Initialize on CPU to avoid device conflicts, G2P doesn't need GPU
    g2p = KPipeline(lang_code="a", model=False)  # Only G2P functionality needed

    early_stopping = EarlyStopping(patience=patience, min_delta=0.0001)
    training_start_time = time.time()
    
    # Memory usage tracking
    def get_memory_usage():
        if device.type == "cuda":
            return f"GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB/{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB"
        elif device.type == "mps":
            return "MPS: Memory tracking not available"
        else:
            return "CPU: Memory tracking not needed"
    
    for epoch in range(resume_epoch + 1, epochs + 1):
        epoch_loss = 0.0
        batch_count = 0
        
        # Clear cache at the beginning of each epoch
        if memory_efficient and device.type == "mps":
            # Release unused memory
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            gc.collect()
            
        epoch_start_time = time.time()
        
        # Progress bar for better tracking
        progress_bar = tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        for batch in progress_bar:
            batch_count += 1
            
            # Process single sample (we're always using batch_size=1 for stability)
            text, target_log_mel, target_audio = batch
            
            # Extract first element since we're using batch_size=1
            if isinstance(text, (list, tuple)):
                text = text[0]
            target_log_mel = target_log_mel.to(device)
            target_audio = target_audio.to(device)
            
            # Process the text input
            try:
                phonemes, _ = g2p.g2p(text)
            except Exception as e:
                print(f"G2P processing failed for text '{text[:50]}...': {e}")
                continue
                
            if phonemes is None or len(phonemes) == 0:
                print(f"Skipping empty phoneme sequence for text: '{text[:50]}...'")
                continue  # skip un-tokenisable utterance
                
            # Validate phoneme length to prevent memory issues
            if len(phonemes) > MAX_PHONEME_LEN:
                print(f"Skipping too long phoneme sequence ({len(phonemes)} > {MAX_PHONEME_LEN}): '{text[:50]}...'")
                continue
                
            # Convert to input IDs and validate
            ids = text_to_input_ids(model, phonemes).to(device)
            
            # Ensure we have a valid sequence (BOS + at least 1 phoneme + EOS)
            if ids.shape[1] < 3:
                print(f"Skipping too short phoneme sequence: '{phonemes}'")
                continue
            
            # Validate audio length to prevent memory issues
            if target_audio.shape[-1] > 24000 * 30:  # 30 seconds max
                print(f"Skipping too long audio ({target_audio.shape[-1]/24000:.1f}s > 30s): '{text[:50]}...'")
                continue

            # With length-dependent embeddings, we'll select the appropriate embedding
            # based on the phoneme sequence length - this happens later
                
            # Forward pass – call the standard method to retain gradients
            # Ensure all inputs are on the same device as the model
            ids = ids.to(device)
            
            # Get the voice embedding for this specific phoneme length
            phoneme_length = len(phonemes)
            voice_for_input = voice_embedding.get_for_length(phoneme_length).squeeze(1).to(device)
            
            # Clear cache periodically for CUDA devices
            if device.type == "cuda" and batch_count % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            # Use gradient checkpointing for memory efficiency on CUDA
            if device.type == "cuda" and memory_efficient:
                # Enable gradient checkpointing temporarily
                with torch.amp.autocast('cuda', enabled=False):  # Disable autocast to avoid conflicts
                    try:
                        audio_pred, _ = model.forward_with_tokens(ids, voice_for_input)
                    except torch.cuda.OutOfMemoryError:
                        # Emergency memory cleanup
                        torch.cuda.empty_cache()
                        gc.collect()
                        print(f"OOM detected, clearing cache and retrying with smaller sequence...")
                        # Try with truncated sequence
                        if ids.shape[1] > 50:
                            ids = ids[:, :50]  # Truncate to first 50 tokens
                            audio_pred, _ = model.forward_with_tokens(ids, voice_for_input)
                        else:
                            raise  # Re-raise if already short
            else:
                # Normal forward pass for non-CUDA devices
                audio_pred, _ = model.forward_with_tokens(ids, voice_for_input)
            
            # Make sure prediction is on the correct device
            if audio_pred.device != device:
                audio_pred = audio_pred.to(device)
                
            # Normalize the predicted audio to a constant RMS before Mel conversion
            normalized_pred = rms_normalize(audio_pred)
            
            # Convert prediction to log-mel spectrogram
            pred_log_mel = mel_transform(normalized_pred)
            pred_log_mel = 20 * torch.log10(pred_log_mel.clamp(min=1e-5))
            
            # Standardize to 3D tensors [batch, n_mels, time] for consistent processing
            if target_log_mel.dim() == 4:  # [1, 1, n_mels, T]
                target_for_loss = target_log_mel.squeeze(0)  # -> [1, n_mels, T]
            else:
                target_for_loss = target_log_mel
            
            if pred_log_mel.dim() == 2:  # [n_mels, T]
                pred_log_mel = pred_log_mel.unsqueeze(0)  # -> [1, n_mels, T]
            
            # Use the dynamic truncation helper for cleaner code
            pred_for_loss, target_for_loss = dynamic_time_truncation(pred_log_mel, target_for_loss)
            
            # Calculate all losses using our unified loss calculator
            # Get the style vector from the specific length embedding we used
            style_vector = voice_for_input[voice_embedding.embedding_size//2:]
            
            # Calculate total loss and get breakdown of components
            loss, loss_components = loss_calculator(
                pred_for_loss, target_for_loss,  # Mel spectrograms
                audio_pred, target_audio,       # Raw audio
                style_vector=style_vector,      # For style regularization from this length
                epoch=epoch,                    # For conditional regularization
                style_reg_strength=style_regularization
            )
            
            # Extract individual losses for logging
            l1_loss = loss_components['l1_loss']
            mse_loss = loss_components['mse_loss']
            freq_loss = loss_components['freq_loss']
            stft_loss = loss_components['stft_loss']
            style_reg_loss = loss_components['style_reg_loss']
            
            # Get current embedding statistics for health monitoring
            embedding_stats = voice_embedding.get_embedding_stats()
            timbre_mean = embedding_stats["timbre_mean"]
            timbre_std = embedding_stats["timbre_std"]
            style_mean = embedding_stats["style_mean"]
            style_std = embedding_stats["style_std"]
            
            # Add length smoothness regularization to prevent abrupt changes between lengths
            smoothness_loss = voice_embedding.calculate_smoothness_loss(weight=0.005)
            loss = loss + smoothness_loss
            
            # Add smoothness loss to batch metrics for logging
            # loss = 0.7*l1_loss + 0.25*mse_loss + 0.05*freq_loss
            # -------------------- optimize using accelerate for gradient accumulation -------------------------
            # Accelerator handles gradient accumulation automatically
            accelerator.backward(loss)
            
            # Only step optimizer on sync gradients (handled by accelerator)
            if accelerator.sync_gradients:
                # Clip gradients of the *actual* trainable parameters (may change after timbre freeze)
                accelerator.clip_grad_norm_(optim.param_groups[0]['params'], 1.0)
                optim.step()
                optim.zero_grad()
                
                # Explicitly clean memory if needed
                if memory_efficient:
                    if device.type == "mps" and batch_count % 5 == 0:
                        # Release unused memory
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                        gc.collect()
                    elif device.type == "cuda" and batch_count % 3 == 0:
                        # More aggressive memory management for CUDA
                        torch.cuda.empty_cache()
                        gc.collect()

            epoch_loss += loss.item() * gradient_accumulation_steps  # Scale back for reporting
            # print current step statistics
            if batch_count % 10 == 0:  # Print every 10 batches instead of every batch
                avg_batch_loss = epoch_loss / batch_count
                print(f"[Epoch {epoch}/{epochs}] Batch {batch_count}/{len(loader)} | "
                      f"Loss: {loss.item() * gradient_accumulation_steps:.4f} | "
                      f"Avg: {avg_batch_loss:.4f} | "
                      f"Text: {text[:30]}...")
            
            # Log individual losses for current batch using unified logger
            step = (epoch - 1) * len(loader) + batch_count
            
            # Prepare all metrics in a dictionary
            batch_metrics = {
                'batch_l1_loss': l1_loss,
                'batch_mse_loss': mse_loss,
                'batch_freq_loss': freq_loss,
                'batch_stft_loss': stft_loss,
                'batch_loss': loss.item(),
                'step': step,
                'smoothness_loss': smoothness_loss.item(),
                'timbre_mean': timbre_mean,
                'timbre_std': timbre_std,
                'style_mean': style_mean,
                'style_std': style_std,
                'length_variation': embedding_stats["length_variation"]
            }
            
            if style_regularization is not None and style_regularization > 0:
                batch_metrics['batch_style_reg_loss'] = style_reg_loss
                
            # Log all metrics at once
            logger.log_metrics(batch_metrics)

            # Update progress bar with memory usage
            progress_bar.set_postfix({
                'loss': f"{epoch_loss/max(1, batch_count):.4f}",
                'mem': get_memory_usage(),
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        avg_loss = epoch_loss / len(loader)
        
        validation_loss = None
        if validation_loader is not None:
            model.eval()  # Set model to evaluation mode
            val_losses = []
            with torch.no_grad():
                for val_text, val_target_log_mel, val_target_audio in validation_loader:
                    # Process the validation sample
                    val_text = val_text[0]  # batch_size=1 for validation too
                    val_target_log_mel = val_target_log_mel.to(device)
                    # We don't need val_target_audio for validation loss, but unpack it to match dataset format
                    
                    # Process text
                    try:
                        phonemes, _ = g2p.g2p(val_text)
                    except Exception as e:
                        print(f"G2P processing failed for text '{val_text[:50]}...': {e}")
                        continue
                        
                    if phonemes is None or len(phonemes) == 0:
                        print(f"Skipping empty validation phoneme sequence for text: '{val_text[:50]}...'")
                        continue
                    
                    # Validate phoneme length for validation as well
                    if len(phonemes) > MAX_PHONEME_LEN:
                        print(f"Skipping too long validation phoneme sequence ({len(phonemes)} > {MAX_PHONEME_LEN}): '{val_text[:50]}...'")
                        continue
                    
                    # Convert phonemes to input IDs
                    val_ids = text_to_input_ids(model, phonemes).to(device)
                    if val_ids.shape[1] < 3:
                        print(f"Skipping too short validation phoneme sequence: '{phonemes}'")
                        continue
                    
                    # Generate audio - ensure all inputs are on the same device
                    val_ids = val_ids.to(device)
                    
                    # For validation, use the appropriate length-specific voice
                    val_phoneme_length = len(phonemes)
                    voice_input = voice_embedding.get_for_length(val_phoneme_length).squeeze(1).to(device)
                    
                    # Use unwrapped model to avoid accelerator overhead during validation
                    if hasattr(model, 'module'):
                        val_audio_pred, _ = model.module.forward_with_tokens(val_ids, voice_input)
                    else:
                        val_audio_pred, _ = model.forward_with_tokens(val_ids, voice_input)
                    
                    # Normalize prediction before Mel conversion
                    val_normalized_pred = rms_normalize(val_audio_pred)
                    
                    # Convert to mel spectrogram
                    val_pred_mel = mel_transform(val_normalized_pred)
                    val_pred_log_mel = 20 * torch.log10(val_pred_mel.clamp(min=1e-5))
                    
                    # Standardize dimensions
                    if val_target_log_mel.dim() == 4:
                        val_target_for_loss = val_target_log_mel.squeeze(0)
                    else:
                        val_target_for_loss = val_target_log_mel
                    
                    if val_pred_log_mel.dim() == 2:  # [n_mels, T]
                        val_pred_log_mel = val_pred_log_mel.unsqueeze(0)  # -> [1, n_mels, T]
                    
                    # Use dynamic truncation for cleaner code
                    val_pred_for_loss, val_target_for_loss = dynamic_time_truncation(val_pred_log_mel, val_target_for_loss)
                    
                    # Compute loss
                    val_loss = l1_loss_fn(val_pred_for_loss, val_target_for_loss)
                    val_losses.append(val_loss.item())
                    
                    # Explicit cleanup for validation
                    del val_audio_pred, val_normalized_pred, val_pred_mel, val_pred_log_mel
                    del val_pred_for_loss, val_target_for_loss, val_loss
                    
                    # Force memory cleanup every few validation samples
                    if len(val_losses) % 5 == 0:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        elif device.type == "mps":
                            if hasattr(torch.mps, 'empty_cache'):
                                torch.mps.empty_cache()
                        gc.collect()
                    
                if val_losses:
                    validation_loss = sum(val_losses) / len(val_losses)
                    
                    # More informative epoch summary
                    epoch_duration = time.time() - epoch_start_time
                    total_duration = time.time() - training_start_time
                    time_remaining = estimate_time_remaining(epoch, epochs, epoch_start_time)
                    
                    print(f"\n{'='*60}")
                    print(f"Epoch {epoch}/{epochs} Summary:")
                    print(f"  Training Loss:    {avg_loss:.4f}")
                    print(f"  Validation Loss:  {validation_loss:.4f}")
                    print(f"  Best Val Loss:    {best_val_loss:.4f} (epoch {best_epoch})")
                    print(f"  Learning Rate:    {optim.param_groups[0]['lr']:.2e}")
                    print(f"  Epoch Time:       {format_duration(epoch_duration)}")
                    print(f"  Total Time:       {format_duration(total_duration)}")
                    print(f"  Time Remaining:   {time_remaining}")
                    print(f"{'='*60}\n")
                    
                    # Save best model (lowest validation L1 loss after epoch 5)
                    if save_best and epoch >= 5 and validation_loss < best_val_loss:
                        best_val_loss = validation_loss
                        best_epoch = epoch
                        print(f"New best model! Validation loss improved to {best_val_loss:.4f}")
                        # Create output directory if it doesn't exist
                        os.makedirs(f"{out}/{name}", exist_ok=True)
                        voice_embedding.save(f"{out}/{name}/{name}.best.pt")
                else:
                    print(f"Epoch {epoch:>3}/{epochs}: training loss {avg_loss:.4f} (no validation samples processed)")
        else:
            epoch_duration = time.time() - epoch_start_time
            total_duration = time.time() - training_start_time
            time_remaining = estimate_time_remaining(epoch, epochs, epoch_start_time)
            
            print(f"\nEpoch {epoch}/{epochs}: loss {avg_loss:.4f} | "
                  f"Time: {format_duration(epoch_duration)} | "
                  f"Remaining: {time_remaining}")
        
        # Log audio samples and spectrograms every N epochs for monitoring convergence
        if epoch % log_audio_every == 0:
            print(f"\nLogging audio samples for epoch {epoch}...")
            
            # Use a sample from the dataset for comparison
            with torch.no_grad():
                # Get a sample from the training dataset for reference
                sample_idx = epoch % len(dataset)  # Rotate through different samples
                sample_text, sample_target_log_mel, sample_target_audio = dataset[sample_idx]
                
                # Process the sample text
                try:
                    sample_phonemes, _ = g2p.g2p(sample_text)
                except Exception as e:
                    print(f"G2P processing failed for text '{sample_text[:50]}...': {e}")
                    continue
                    
                if sample_phonemes and len(sample_phonemes) > 0:
                    sample_ids = text_to_input_ids(model, sample_phonemes).to(device)
                    
                    if sample_ids.shape[1] >= 3:  # Valid sequence
                        # Generate audio using current voice embedding
                        sample_phoneme_length = len(sample_phonemes)
                        sample_voice_input = voice_embedding.get_for_length(sample_phoneme_length).squeeze(1).to(device)
                        
                        sample_audio_pred, _ = model.forward_with_tokens(sample_ids, sample_voice_input)
                        sample_normalized_pred = rms_normalize(sample_audio_pred)
                        
                        # Convert to mel spectrograms for comparison
                        sample_pred_mel = mel_transform(sample_normalized_pred)
                        sample_pred_log_mel = 20 * torch.log10(sample_pred_mel.clamp(min=1e-5))
                        
                        # Move target audio to device and normalize
                        sample_target_audio_device = sample_target_audio.to(device)
                        sample_target_normalized = rms_normalize(sample_target_audio_device)
                        
                        # Standardize dimensions and truncate for comparison
                        if sample_target_log_mel.dim() == 4:
                            sample_target_for_display = sample_target_log_mel.squeeze(0)
                        else:
                            sample_target_for_display = sample_target_log_mel.to(device)
                        
                        if sample_pred_log_mel.dim() == 2:
                            sample_pred_log_mel = sample_pred_log_mel.unsqueeze(0)
                        
                        # Truncate to same length for fair comparison
                        sample_pred_display, sample_target_display = dynamic_time_truncation(
                            sample_pred_log_mel, sample_target_for_display
                        )
                        
                        # Create comparison spectrogram figure
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                        
                        # Reference spectrogram (top)
                        ref_spec = sample_target_display.squeeze(0).cpu().numpy()
                        im1 = ax1.imshow(ref_spec, aspect='auto', origin='lower', cmap='viridis')
                        ax1.set_title(f"Reference: {sample_text[:50]}...")
                        ax1.set_ylabel("Mel Frequency")
                        plt.colorbar(im1, ax=ax1, shrink=0.6)
                        
                        # Generated spectrogram (bottom)
                        pred_spec = sample_pred_display.squeeze(0).cpu().numpy()
                        im2 = ax2.imshow(pred_spec, aspect='auto', origin='lower', cmap='viridis')
                        ax2.set_title(f"Generated (Epoch {epoch})")
                        ax2.set_ylabel("Mel Frequency")
                        ax2.set_xlabel("Time Frames")
                        plt.colorbar(im2, ax=ax2, shrink=0.6)
                        
                        plt.tight_layout()
                        
                        # Log the comparison spectrogram
                        logger.log_spectrogram(fig, f"Comparison_Epoch_{epoch}", epoch)
                        plt.close(fig)
                        
                        # Log audio samples (move to CPU and convert to numpy for logging)
                        # Reference audio
                        ref_audio_np = sample_target_normalized.squeeze().cpu().numpy()
                        logger.log_audio(
                            ref_audio_np, 
                            24000, 
                            f"Reference: {sample_text[:30]}", 
                            epoch, 
                            is_reference=True
                        )
                        
                        # Generated audio
                        pred_audio_np = sample_normalized_pred.squeeze().cpu().numpy()
                        logger.log_audio(
                            pred_audio_np, 
                            24000, 
                            f"Generated: {sample_text[:30]}", 
                            epoch, 
                            is_reference=False
                        )
                        
                        print(f"Logged audio samples for epoch {epoch}")
                        
                        # Calculate and log voice drift from original reference
                        if original_reference_voice is not None:
                            drift_metrics = calculate_voice_drift(original_reference_voice, voice_embedding, device)
                            logger.log_voice_drift(drift_metrics, epoch)
                            print(f"Voice drift - Cosine similarity: {drift_metrics['cosine_similarity']:.4f}")
                        
                    else:
                        print(f"Skipping too short sample phoneme sequence: '{sample_phonemes}'")
        # Save the full voice tensor every N epochs
        if epoch % checkpoint_every == 0 or epoch == epochs:
            # Create output directory if it doesn't exist
            os.makedirs(f"{out}/{name}", exist_ok=True)
            os.makedirs(f"{out}/{name}/checkpoints", exist_ok=True)
            voice_embedding.save(f"{out}/{name}/{name}.epoch{epoch}.pt")
            print(f"Checkpoint saved: {name}.epoch{epoch}.pt")
            
            # Save checkpoint with optimizer and scheduler states
            checkpoint_path = Path(out) / name / "checkpoints" / f"{name}.epoch{epoch}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
            checkpoint_data = {
                'voice_embed': voice_embedding.voice_embed.data.cpu(),
                'optimizer_state': optim.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'dataset_stats': {}
            }
            torch.save(checkpoint_data, checkpoint_path)
            
            # Visualize the length-dependent embeddings
            # Create embedding visualization - show how they vary by length
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Extract timbre and style components
            timbre_components = voice_embedding.voice_embed[:, 0, :voice_embedding.embedding_size//2].detach().cpu()
            style_components = voice_embedding.voice_embed[:, 0, voice_embedding.embedding_size//2:].detach().cpu()
            
            # Calculate average over the feature dimension to show length trends
            timbre_avg = torch.mean(timbre_components, dim=1).numpy()
            style_avg = torch.mean(style_components, dim=1).numpy()
            
            # Show the first 200 lengths for better visualization
            x = list(range(1, min(201, voice_embedding.max_phoneme_len + 1)))
            ax1.plot(x, timbre_avg[:200])
            ax1.set_title("Average Timbre Component by Phoneme Length")
            ax1.set_xlabel("Phoneme Sequence Length")
            ax1.set_ylabel("Average Activation")
            
            ax2.plot(x, style_avg[:200])
            ax2.set_title("Average Style Component by Phoneme Length")
            ax2.set_xlabel("Phoneme Sequence Length")
            ax2.set_ylabel("Average Activation")
            
            plt.tight_layout()
            logger.log_spectrogram(fig, "Length_Variation", epoch)
            plt.close(fig)

        scheduler.step()
        
        # Check embedding health and apply clamping if needed
        embedding_stats = voice_embedding.check_health(epoch=epoch, warning_threshold=timbre_warning_threshold)
        
        # Every 5 epochs, apply length smoothing to avoid discontinuities
        if epoch % 5 == 0:
            voice_embedding.apply_length_smoothing(weight=0.2)
            print(f"Applied length smoothing at epoch {epoch}")
        
        # Log embedding statistics with length variation
        logger.log_embedding_stats(embedding_stats)
        
        # Check for early stopping
        if validation_loss is not None:
            if early_stopping(validation_loss):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Validation loss hasn't improved for {patience} epochs")
                print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                break
    
    # Print summary info about best checkpoint if available
    if save_best and best_epoch > 0:
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        print(f"Final output: {out}/{name}/{name}.pt")
        print(f"Best model: {out}/{name}/{name}.best.pt")
        print(f"Total training time: {format_duration(time.time() - training_start_time)}")
        print(f"{'='*60}\n")
    
    # Upload to HuggingFace    # HF upload
    if upload_to_hf and hf_repo_id:
        # Collect dataset statistics
        total_samples = len(dataset)
        total_duration = 0
        
        # Calculate approximate duration by loading a few samples
        sample_count = min(10, total_samples)  # Sample up to 10 files to estimate duration
        if sample_count > 0:
            sample_indices = random.sample(range(total_samples), sample_count)
            sample_durations = []
            
            for idx in sample_indices:
                try:
                    path = dataset.wav_paths[idx]
                    info = torchaudio.info(path)
                    duration = info.num_frames / info.sample_rate
                    sample_durations.append(duration)
                except:
                    pass
                    
            if sample_durations:
                avg_duration = sum(sample_durations) / len(sample_durations)
                total_duration = avg_duration * total_samples
        
        # Create descriptive statistics
        dataset_stats = {
            "total_samples": total_samples,
            "total_duration_seconds": total_duration,
            "total_duration_formatted": format_duration(total_duration) if total_duration > 0 else "Unknown",
            "estimated": True
        }
        
        # Add training parameters for README
        training_params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop_length": hop,
            "output": out,
            "data_root": str(data_root),
        }
        
        # Collect audio samples
        audio_samples = []
        if use_wandb and WANDB_AVAILABLE:
            api = wandb.Api()
            runs = api.runs(f"{wandb_project or 'kokoro-voice'}/{wandb.run.id}")
            
            artifacts = []
            for artifact in runs.logged_artifacts():
                if artifact.type == "audio":
                    artifacts.append(artifact)
            
            # Find and download sample audios
            if artifacts:
                samples_dir = os.path.join(out, name, "samples")
                os.makedirs(samples_dir, exist_ok=True)
                
                # Prioritize later epoch samples
                for artifact in reversed(artifacts):
                    artifact_dir = artifact.download()
                    for filename in os.listdir(artifact_dir):
                        if filename.endswith(".wav"):
                            epoch = int(filename.split("_")[0].replace("epoch", ""))
                            sample_text = "_".join(filename.split("_")[1:]).replace(".wav", "")
                            
                            # Add to the samples list
                            if len(audio_samples) < 10:  # Limit to 10 samples
                                audio_samples.append({
                                    "epoch": epoch,
                                    "text": sample_text,
                                    "path": os.path.join("samples", filename)
                                })
                                
                                # Copy to samples dir
                                shutil.copy2(
                                    os.path.join(artifact_dir, filename),
                                    os.path.join(samples_dir, filename)
                                )
                                
        # Upload to HuggingFace
        upload_to_huggingface(
            output_dir=os.path.join(out, name),
            name=name,
            best_epoch=best_epoch,
            dataset_stats=dataset_stats,
            training_params=training_params,
            audio_samples=audio_samples
        )
        
        print(f"Successfully uploaded voice model to HuggingFace: https://huggingface.co/{hf_repo_id}")

    # Close unified logger
    logger.close()

    # Compare original reference voice with trained voice embeddings
    if original_reference_voice is not None:
        compare_voice_embeddings(original_reference_voice, voice_embedding, device=device)


def compare_voice_embeddings(original_voice, trained_voice_embedding, device='cpu'):
    """
    Compare the original reference voice with the final trained voice embedding
    to see how much the training changed the voice characteristics.
    
    Args:
        original_voice: Original reference voice tensor
        trained_voice_embedding: VoiceEmbedding object with trained voice
        device: Device for computation
    """
    if original_voice is None:
        print("No original reference voice available for comparison")
        return
    
    print(f"\n{'='*60}")
    print("VOICE TRAINING COMPARISON")
    print(f"{'='*60}")
    
    with torch.no_grad():
        # Move to same device for comparison
        original_voice = original_voice.to(device)
        
        if original_voice.dim() == 3 and original_voice.shape == (510, 1, 256):
            # 3D format - compare averaged voice
            original_avg = original_voice.mean(dim=0).squeeze()  # [256]
            print("Original voice format: 3D [510, 1, 256] - using average")
        elif original_voice.dim() == 1 and original_voice.shape[0] == 256:
            # 1D format
            original_avg = original_voice
            print("Original voice format: 1D [256]")
        else:
            print(f"Unexpected original voice shape: {original_voice.shape}")
            return
        
        # Get current trained voice (average across phoneme positions)
        trained_avg = trained_voice_embedding.voice_embed.mean(dim=0).squeeze().to(device)  # [256]
        
        # Calculate various similarity metrics
        cosine_sim = F.cosine_similarity(original_avg.unsqueeze(0), trained_avg.unsqueeze(0)).item()
        l2_distance = torch.norm(original_avg - trained_avg).item()
        l1_distance = torch.norm(original_avg - trained_avg, p=1).item()
        max_diff = torch.max(torch.abs(original_avg - trained_avg)).item()
        
        # Calculate relative change percentage
        original_norm = torch.norm(original_avg).item()
        trained_norm = torch.norm(trained_avg).item()
        norm_change_percent = ((trained_norm - original_norm) / original_norm) * 100
        
        # Analyze per-dimension changes
        dim_changes = torch.abs(original_avg - trained_avg)
        top_changed_dims = torch.topk(dim_changes, k=5)
        
        print(f"Cosine similarity:     {cosine_sim:.4f} (1.0 = identical, 0.0 = orthogonal)")
        print(f"L2 distance:           {l2_distance:.4f}")
        print(f"L1 distance:           {l1_distance:.4f}")
        print(f"Max dimension change:  {max_diff:.4f}")
        print(f"Vector norm change:    {norm_change_percent:+.2f}%")
        
        print(f"\nTop 5 most changed dimensions:")
        for i, (change, dim_idx) in enumerate(zip(top_changed_dims.values, top_changed_dims.indices)):
            print(f"  Dim {dim_idx:3d}: {change:.4f}")
        
        # Interpretation
        print(f"\nInterpretation:")
        if cosine_sim > 0.95:
            print("  Very similar - minimal voice change")
        elif cosine_sim > 0.85:
            print("  Moderate similarity - voice adapted while keeping character")
        elif cosine_sim > 0.70:
            print("  Significant change - voice learned new characteristics")
        else:
            print("  Major transformation - voice substantially different")
        
        print(f"{'='*60}\n")
        
        return {
            'cosine_similarity': cosine_sim,
            'l2_distance': l2_distance,
            'l1_distance': l1_distance,
            'max_difference': max_diff,
            'norm_change_percent': norm_change_percent
        }


def save_voice(base_voice, voice_embed, out):
    # ---------------------------------------------------------------------
    # Save voice embedding
    # ---------------------------------------------------------------------
    save_path = Path(out)
    
    with torch.no_grad():
        # Make sure base_voice is on CPU and detached from compute graph
        if hasattr(base_voice, 'device'):
            base_voice_cpu = base_voice.detach().cpu()
        else:
            # Handle accelerator-wrapped models/tensors
            base_voice_cpu = base_voice.detach().cpu() if hasattr(base_voice, 'detach') else base_voice
            
        # Update the full tensor one last time
        for i in range(MAX_PHONEME_LEN):
            voice_embed[i, 0, :] = base_voice_cpu.clone()
        
        torch.save(voice_embed.detach().cpu(), save_path)
        print(f"Saved full voice tensor to {save_path.resolve()} – shape {tuple(voice_embed.shape)}")


def upload_to_huggingface(output_dir, name, best_epoch, dataset_stats, training_params, audio_samples=None):
    """
    Upload trained voice model and artifacts to HuggingFace.
    
    Args:
        output_dir: Local directory containing the voice model and artifacts
        name: Name of the voice model
        best_epoch: Best epoch number (for README)
        dataset_stats: Statistics about the dataset used for training
        training_params: Parameters used for training
        audio_samples: Dictionary mapping epoch numbers to audio sample files
    """
    if not training_params.get("hf_repo_id"):
        print("No HuggingFace repository ID provided. Skipping upload.")
        return
    
    repo_id = training_params["hf_repo_id"]
    print(f"\nPreparing to upload voice model and artifacts to HuggingFace: {repo_id}")
    
    # Create a temporary directory for organizing the HF repo contents
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # 1. Create the main model file (renamed to just name.pt)
        best_model_path = Path(output_dir) / name / f"{name}.best.pt"
        if best_model_path.exists():
            shutil.copy(best_model_path, tmp_path / f"{name}.pt")
        else:
            # Fallback to the final model if best doesn't exist
            final_model_path = Path(output_dir) / name / f"{name}.pt"
            shutil.copy(final_model_path, tmp_path / f"{name}.pt")
        
        # 2. Create 'rejects' directory with all other checkpoints
        rejects_dir = tmp_path / "rejects"
        rejects_dir.mkdir(exist_ok=True)
        
        # Copy all epoch checkpoints to rejects
        for checkpoint in (Path(output_dir) / name).glob(f"{name}.epoch*.pt"):
            shutil.copy(checkpoint, rejects_dir / checkpoint.name)
        
        # 3. Copy audio samples if available
        if audio_samples:
            samples_dir = tmp_path / "samples"
            samples_dir.mkdir(exist_ok=True)
            
            for epoch, sample_path in audio_samples.items():
                if os.path.exists(sample_path):
                    shutil.copy(sample_path, samples_dir / f"epoch_{epoch}.wav")
        
        # 4. Create README.md with model information
        create_readme(tmp_path, name, best_epoch, dataset_stats, training_params)
        
        # 5. Upload to HuggingFace
        print(f"Uploading files to HuggingFace repository: {repo_id}")
        upload_folder(
            folder_path=tmp_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {name} voice model and artifacts"
        )
        
        print(f"Successfully uploaded voice model to HuggingFace: https://huggingface.co/{repo_id}")


def create_readme(output_dir, name, best_epoch, dataset_stats, training_params):
    """
    Create a README.md file with information about the voice model.
    
    Args:
        output_dir: Directory to save the README.md file
        name: Name of the voice model
        best_epoch: Best epoch number
        dataset_stats: Statistics about the dataset used for training
        training_params: Parameters used for training
    """
    # Format the current date and time
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Create README content
    readme_content = f"""---
tags:
- tts
- voice
- kokoro
base_model: "hexgrad/Kokoro-82M"
---
    # {name} - Kokoro Voice Model

## Model Information
- **Created:** {current_date}
- **Name:** {name}
- **Best Epoch:** {best_epoch}

## Dataset Information
- **Audio Samples:** {dataset_stats.get('num_samples', 'N/A')}
- **Total Duration:** {dataset_stats.get('total_duration', 'N/A')} seconds
- **Average Sample Duration:** {dataset_stats.get('avg_duration', 'N/A')} seconds

## Training Parameters
- **Learning Rate:** {training_params.get('lr', 'N/A')}
- **Epochs:** {training_params.get('epochs', 'N/A')}
- **Batch Size:** {training_params.get('batch_size', 'N/A')}
- **Mel Bins:** {training_params.get('n_mels', 'N/A')}
- **FFT Size:** {training_params.get('n_fft', 'N/A')}
- **Hop Length:** {training_params.get('hop', 'N/A')}
- **Style Regularization:** {training_params.get('style_regularization', 'N/A')}
- **Timbre Warning Threshold:** {training_params.get('timbre_warning_threshold', 'N/A')}

## Usage

```python
import torch
from kokoro import KPipeline

# Load the voice model
voice = torch.load("{name}.pt")

# Initialize the pipeline
pipeline = KPipeline(voice=voice)

# Generate speech
audio = pipeline("Hello, world!")
```

## Model Structure
This model consists of a 256-dimensional voice embedding with:
- 128 dimensions for timbre (voice characteristics)
- 128 dimensions for style (expressivity/prosody)

The model was trained using RMS-normalized audio to improve mel-spectrogram loss correlation with intelligibility.
"""
    
    # Write the README.md file
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)


def get_available_voices():
    """Get list of available Kokoro voices by language"""
    voices = {
        'american': [  # lang_code='a'
            '0ab5709b', '6d877149', 'c03bd1a4', '8cb64e02', 'cdfdccb8', 
            '8bfbc512', 'c5561808', 'e0233676', 'e149459b', '49bd364e',
            'c799548a', 'ced7e284', '8bcfdc85', 'ada66f0e', '98e507ec',
            'c8255075', '9a443b79', 'e8452be1', 'dd1d8973', '7f2f7582'
        ],
        'british': [   # lang_code='b'
            'd292651b', 'd0a423de', 'cdd4c370', '6e09c2e4', 'fc3fce4e',
            'd44935f3', 'f1bc8122', 'b5204750'
        ]
    }
    return voices

def load_kokoro_base_voice(voice_file_name, accent='american', device='cpu'):
    """Load a specific Kokoro voice as base initialization"""
    try:
        # If we have a specific voice file name, try to load the voice file
        if voice_file_name is not None:
            try:
                from huggingface_hub import hf_hub_download
                # Download the .pt voice file
                voice_file_path = f'voices/{voice_file_name}.pt'
                
                # Download the .pt voice file
                downloaded_path = hf_hub_download(
                    repo_id='hexgrad/Kokoro-82M',
                    filename=voice_file_path,
                    cache_dir='./voice_cache'
                )
                
                # Load the voice embedding
                voice_embedding = torch.load(downloaded_path, map_location=device)
                print(f"Loaded specific Kokoro voice file: {voice_file_name}")
                return voice_embedding
                
            except Exception as e:
                print(f"Could not load specific voice file {voice_file_name}: {e}")
                # Fall back to default voice from model
        
        # Fallback: Load default voice from Kokoro model
        from kokoro import KPipeline
        
        # Create pipeline with specified language code
        lang_code = 'a' if accent == 'american' else 'b'
        pipeline = KPipeline(lang_code=lang_code)
        
        # The KModel doesn't have a style_encoder attribute directly accessible
        # Instead, we'll use a small random voice vector as fallback
        print(f"Cannot access voice embeddings from KModel, using fallback approach")
        
        # Create a reasonable fallback voice vector (256-dim)
        # Based on typical Kokoro voice embedding ranges
        base_voice = torch.randn(256, device=device) * 0.1  # Small random values
        print(f"Generated fallback voice vector for {accent} accent")
        return base_voice
            
    except Exception as e:
        print(f"Could not load Kokoro base voice: {e}")
        return None

def select_best_base_voice(target_voice_description="", accent='american'):
    """Select the best base voice for initialization based on description"""
    voices = get_available_voices()
    available = voices.get(accent, voices['american'])
    
    print(f"\nAvailable {accent} English voices:")
    for i, voice_id in enumerate(available[:10]):  # Show first 10
        print(f"  {i+1}. {voice_id}")
    
    if len(available) > 10:
        print(f"  ... and {len(available) - 10} more")
    
    # For now, return the first voice as default
    # In the future, this could be enhanced with voice similarity matching
    selected = available[0]
    print(f"Selected base voice: {selected}")
    return selected

def compare_mel_spectrograms(mel1, mel2):
    """Compare two mel spectrograms and return a similarity score"""
    # Calculate the mean squared error between the two spectrograms
    mse = torch.mean((mel1 - mel2) ** 2)
    return mse.item()

def compare_voices(voice1, voice2):
    """Compare two voices and return a similarity score"""
    # Calculate the mean squared error between the two voices
    mse = torch.mean((voice1 - voice2) ** 2)
    return mse.item()

def select_best_voice(target_voice, voices):
    """Select the best voice from a list of voices based on similarity"""
    best_voice = None
    best_similarity = float('inf')
    
    for voice in voices:
        similarity = compare_mel_spectrograms(target_voice, voice)
        if similarity < best_similarity:
            best_similarity = similarity
            best_voice = voice
    
    return best_voice

# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Import at top level to ensure they're available
    import gc
    
    ap = argparse.ArgumentParser(description="Train a Kokoro voice embedding")
    ap.add_argument("--data", type=str, help="Path to dataset directory")
    ap.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size")
    ap.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    ap.add_argument("--name", type=str, default="my_voice", help="Output voice file")
    ap.add_argument("--out", type=str, default="output", help="Output directory")
    ap.add_argument("--dataset-id", type=str,
                help='HF dataset repo ID (e.g. "aimeri/my-voice-demo"). '
                     'Used if --data is not given.')
    # Add memory optimization arguments
    memory_group = ap.add_argument_group('Memory Optimization')
    memory_group.add_argument("--memory-efficient", action="store_true", help="Enable memory efficiency optimizations")
    memory_group.add_argument("--grad-accumulation", type=int, default=1, 
                             help="Number of batches to accumulate gradients over before optimizer step")
    memory_group.add_argument("--mps-watermark", type=float, default=0.7, 
                             help="MPS high watermark ratio (0.0-1.0) - lower values use less memory")
    
    # Add advanced training control arguments
    training_group = ap.add_argument_group('Advanced Training Controls')
    training_group.add_argument("--lr-decay", type=str, choices=['auto', 'step', 'plateau'], default='auto',
                              help="Learning rate decay schedule. 'auto'=every 5 epochs, 'step'=at specific milestones, 'plateau'=when loss plateaus")
    training_group.add_argument("--lr-decay-rate", type=float, default=0.5,
                              help="Factor to multiply learning rate by when decaying (e.g., 0.5 = halve it)")
    training_group.add_argument("--lr-decay-epochs", type=int, nargs='+',
                              help="Epochs at which to decay learning rate (only for 'step' decay)")
    training_group.add_argument("--timbre-warning", type=float, default=0.3,
                              help="Print warning if timbre std exceeds this value (no freezing)")
    training_group.add_argument("--style-reg", type=float, default=1e-5,
                              help="L2 regularization strength for style part of embedding (e.g., 1e-4)")
    training_group.add_argument("--skip-validation", type=bool, default=False,
                              help="Skip validation split")
    
    # Add monitoring arguments
    monitoring_group = ap.add_argument_group('Monitoring')
    monitoring_group.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    monitoring_group.add_argument("--log-dir", type=str, help="Directory for TensorBoard logs (default: runs/voice_name)")
    monitoring_group.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    monitoring_group.add_argument("--wandb-project", type=str, help="W&B project name")
    monitoring_group.add_argument("--wandb-name", type=str, help="W&B run name")
    monitoring_group.add_argument("--log-audio-every", type=int, default=10, help="Log audio samples every N epochs")
    
    # Add voice initialization arguments
    voice_group = ap.add_argument_group('Voice Initialization')
    voice_group.add_argument("--no-reference-voice", action="store_true", 
                           help="Disable automatic reference voice selection and use random initialization")
    voice_group.add_argument("--manual-voice-id", type=str,
                           help="Manually specify a Kokoro voice ID to use as base (e.g., '0ab5709b')")
    voice_group.add_argument("--accent", type=str, choices=['american', 'british', 'auto'], default='auto',
                           help="Accent preference for voice selection. 'auto' will try both and pick best match")
    
    # Add HuggingFace upload arguments
    hf_group = ap.add_argument_group('HuggingFace Upload')
    hf_group.add_argument("--upload-to-hf", action="store_true", help="Upload model and artifacts to HuggingFace")
    hf_group.add_argument("--hf-repo-id", type=str, help="HuggingFace repository ID (e.g., 'username/model-name')")
    args = ap.parse_args()
    
    data_root = args.data
    if data_root is None:
        if not args.dataset_id:
            ap.error("Either --data or --dataset-id must be supplied")
        print(f"Downloading dataset snapshot from hf://datasets/{args.dataset_id} …")
        data_root = snapshot_download(
            repo_id=args.dataset_id,
            repo_type="dataset",
            token=True,              # pick up cached token
            ignore_patterns=["*.md"], # skip large README pre-renders, optional
        )

    # Create output directory structure
    os.makedirs(f"{args.out}/{args.name}", exist_ok=True)
    os.makedirs(f"{args.out}/{args.name}/checkpoints", exist_ok=True)
    
    train(
        data_root=data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out=args.out,
        name=args.name,
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        log_audio_every=args.log_audio_every,
        log_dir=args.log_dir,
        memory_efficient=args.memory_efficient,
        gradient_accumulation_steps=args.grad_accumulation,
        timbre_warning_threshold=args.timbre_warning,
        style_regularization=args.style_reg,
        skip_validation=args.skip_validation,
        upload_to_hf=args.upload_to_hf,
        hf_repo_id=args.hf_repo_id,
        no_reference_voice=args.no_reference_voice,
        manual_voice_id=args.manual_voice_id,
        accent=args.accent,
    )

def select_best_voice(target_voice, voices):
    """Select the best voice from a list of voices based on similarity"""
    best_voice = None
    best_similarity = float('inf')
    
    for voice in voices:
        similarity = compare_mel_spectrograms(target_voice, voice)
        if similarity < best_similarity:
            best_similarity = similarity
            best_voice = voice
    
    return best_voice

def compare_mel_spectrograms(mel1, mel2):
    """Compare two mel spectrograms and return a similarity score"""
    # Calculate the mean squared error between the two spectrograms
    mse = torch.mean((mel1 - mel2) ** 2)
    return mse.item()

def compare_voices(voice1, voice2):
    """Compare two voices and return a similarity score"""
    # Calculate the mean squared error between the two voices
    mse = torch.mean((voice1 - voice2) ** 2)
    return mse.item()

def select_best_base_voice(target_voice_description="", accent='american'):
    """Select the best base voice for initialization based on description"""
    voices = get_available_voices()
    available = voices.get(accent, voices['american'])
    
    print(f"\nAvailable {accent} English voices:")
    for i, voice_id in enumerate(available[:10]):  # Show first 10
        print(f"  {i+1}. {voice_id}")
    
    if len(available) > 10:
        print(f"  ... and {len(available) - 10} more")
    
    # For now, return the first voice as default
    # In the future, this could be enhanced with voice similarity matching
    selected = available[0]
    print(f"Selected base voice: {selected}")
    return selected

