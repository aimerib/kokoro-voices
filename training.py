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

import argparse
import math
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import random
import json
import gc

import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import numpy as np

from huggingface_hub import snapshot_download, HfApi, upload_folder


from dotenv import load_dotenv
load_dotenv()

# Import visualization tools
from torch.utils.tensorboard import SummaryWriter

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

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

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
        return text, log_mel  # (1, n_mels, T)


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
    batch_size: int = 1,
    lr: float = 5e-4,  # Reduced default learning rate (5-10x lower than original)
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
    lr_decay_schedule: Optional[str] = None,       # 'auto', 'step', or 'plateau'
    lr_decay_rate: float = 0.5,                    # Factor to multiply LR when decaying
    lr_decay_epochs: List[int] = None,             # For 'step' schedule, epochs to decay
    timbre_freeze_threshold: Optional[float] = 0.25, # Freeze timbre embedding when std reaches this value (default: 0.25)
    style_regularization: Optional[float] = 1e-3,   # L2 regularization on style part of embedding (default: 1e-3)
    skip_validation: bool = False,                  # Skip validation split
    save_best: bool = True,                        # Save best checkpoint (lowest validation L1 loss after epoch 5)
    upload_to_hf: bool = False,                    # Upload model and artifacts to HuggingFace
    hf_repo_id: Optional[str] = None,              # Repository ID for HuggingFace upload
):
    device = get_device()
    print(f"Using device: {device}")
    
    # Memory optimization settings
    if memory_efficient and device.type == "mps":
        print("Enabling memory optimization settings for MPS")
        # Force garbage collection to run more aggressively
        gc.enable()
        print(f"MPS HIGH_WATERMARK_RATIO set to {os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'default')}")
    
    # Set up monitoring tools
    writer = None
    if use_tensorboard:
        log_dir = log_dir or os.path.join('runs', Path(out).stem)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to {log_dir}")
    
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: W&B not available. Please install with 'pip install wandb'")
        else:
            wandb.init(
                project=wandb_project or "kokoro-voice-training",
                name=wandb_name or Path(out).stem,
                config={
                    "learning_rate": lr,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "n_mels": n_mels,
                    "n_fft": n_fft,
                    "hop_length": hop,
                    "output": out,
                    "data_root": str(data_root),
                }
            )
            print(f"W&B project initialized: {wandb_project or 'kokoro-voice-training'}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    
    # Create two separate mel transforms - one for CPU (dataset) and one for device (training)
    mel_transform_cpu = torchaudio.transforms.MelSpectrogram(
        sample_rate=24_000, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24_000, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    ).to(device)

    # Create dataset
    dataset = VoiceDataset(data_root, mel_transform_cpu)
    
    # Split into training and validation sets if validation set size is specified
    # Simply use the existing train/validation splits directly
    dataset = VoiceDataset(data_root, mel_transform_cpu, split="train")
    validation_dataset = None
    
    # If validation directory exists and we requested validation data, load it
    if not skip_validation and (Path(data_root) / "validation").exists():
        validation_dataset = VoiceDataset(data_root, mel_transform_cpu, split="validation")
        print(f"Using dataset splits: {len(dataset)} training samples, {len(validation_dataset)} validation samples")
    
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
        
    # Effective batch size through gradient accumulation
    # This provides the benefits of larger batches without memory issues
    effective_batch_size = batch_size
    if effective_batch_size > 1:
        print(f"Using effective batch size of {effective_batch_size} through gradient accumulation")
        # Update gradient_accumulation_steps to match the requested batch size
        gradient_accumulation_steps = effective_batch_size

    # ---------------------------------------------------------------------
    # Model + voice embedding
    # ---------------------------------------------------------------------
    # model = KModel().to(device).eval()
    model = KModel().to(device).train()
    for p in model.parameters():
        p.requires_grad_(False)

    # Initialize a full 3D voice tensor [510, 1, 256] to match Kokoro's expected format
    voice_embed = torch.zeros((MAX_PHONEME_LEN, 1, 256), device=device)
    
    # Initialize with proper distribution similar to Kokoro voices
    # Get reference statistics from a known good voice (optional)
    try:
        from huggingface_hub import hf_hub_download
        ref_path = hf_hub_download(repo_id='hexgrad/Kokoro-82M', filename='voices/af_heart.pt')
        ref_voice = torch.load(ref_path, map_location=device)
        
        # Extract reference distribution statistics
        if ref_voice.dim() == 3:
            ref_voice = ref_voice[250, 0]  # Get middle-ish voice
        elif ref_voice.dim() == 2 and ref_voice.shape[0] > 1:
            ref_voice = ref_voice[0]

        # Split into timbre and style for better initialization
        timbre_mean = ref_voice[:128].mean().item()
        timbre_std = ref_voice[:128].std().item()
        style_mean = ref_voice[128:].mean().item()
        style_std = ref_voice[128:].std().item()
        
        print(f"Initializing from reference distribution:")
        print(f"  Timbre: mean={timbre_mean:.4f}, std={timbre_std:.4f}")
        print(f"  Style: mean={style_mean:.4f}, std={style_std:.4f}")
    except Exception as e:
        print(f"Using default distribution, could not load reference: {e}")
        timbre_mean, timbre_std = 0.0, 0.14
        style_mean, style_std = 0.0, 0.14
    
    # Initialize base voice with proper distribution
    base_voice = torch.zeros((1, 256), device=device)
    base_voice[0, :128] = torch.randn(128, device=device) * timbre_std + timbre_mean  # Timbre
    base_voice[0, 128:] = torch.randn(128, device=device) * style_std + style_mean  # Style
    base_voice.requires_grad_(True)
    
    # Initialize the voice tensor with the base voice
    for i in range(MAX_PHONEME_LEN):
        # Start with the same vector, but we'll add small variations based on phoneme position
        # This follows the pattern in train_full_voice.py
        style_factor = max(0.5, 1.0 - (i / MAX_PHONEME_LEN) * 0.5)  # Decay expressivity for longer sequences
        voice_embed[i, 0, :] = base_voice.clone()  # Copy the base voice to each position

    # Set up optimizer with weight decay for regularization
    optim = torch.optim.Adam([base_voice], lr=lr, weight_decay=1e-6)
    
    # Track best validation loss for saving best checkpoint
    best_val_loss = float('inf')
    best_epoch = 0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=epochs, eta_min=1e-5
    )
    
    # Multiple loss functions for better results
    l1_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()

    # G2P pipeline for text -> phonemes
    g2p = KPipeline(lang_code="a", model=False)  # Only G2P functionality needed

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batch_count = 0
        
        # Clear cache at the beginning of each epoch
        if memory_efficient and device.type == "mps":
            # Release unused memory
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            gc.collect()
            
        for text, target_log_mel in loader:
            batch_count += 1
            
            # Process single sample (we're always using batch_size=1 for stability)
            text = text[0]  # batch_size=1
            target_log_mel = target_log_mel.to(device)
            
            # Process the text input
            phonemes, _ = g2p.g2p(text)
            if phonemes is None or len(phonemes) == 0:
                continue  # skip un-tokenisable utterance
                
            # Convert to input IDs and validate
            ids = text_to_input_ids(model, phonemes).to(device)
            
            # Ensure we have a valid sequence (BOS + at least 1 phoneme + EOS)
            if ids.shape[1] < 3:
                print(f"Skipping too short phoneme sequence: '{phonemes}'")
                continue

            # Here's the problem - when accessing a specific voice based on phoneme length,
            # we need to make sure base_voice matches the expected shape for the model
            voice_for_input = base_voice  # Already [1, 256] and requires_grad=True
                
            # Forward pass – call the *undecorated* version to retain gradients
            with torch.autocast(device_type=device.type if device.type != 'mps' else 'cpu', enabled=memory_efficient):
                audio_pred, _ = model.forward_with_tokens.__wrapped__(  # type: ignore[attr-defined]
                    model, ids, voice_for_input
                )
            
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
            
            # Multiple loss components for better learning
            # 1. L1 loss on mel-spectrogram (primary loss)
            l1_loss = l1_loss_fn(pred_for_loss, target_for_loss)
            
            # 2. MSE loss for spectral contour
            mse_loss = mse_loss_fn(pred_for_loss, target_for_loss)
            
            # 3. High-frequency emphasis loss - focus on important details
            # Calculate gradients in frequency domain (approximate gradients across mel bands)
            if pred_for_loss.shape[1] > 1:  # Need at least 2 mel bands
                pred_diff = pred_for_loss[:, 1:] - pred_for_loss[:, :-1]
                target_diff = target_for_loss[:, 1:] - target_for_loss[:, :-1]
                freq_loss = l1_loss_fn(pred_diff, target_diff) * 0.5
            else:
                freq_loss = torch.tensor(0.0, device=device)
            
            # 4. Optional style regularization (KL/L2) to control variance
            style_reg_loss = torch.tensor(0.0, device=device)
            if style_regularization is not None and style_regularization > 0:
                # L2 regularization on the style part of the embedding (second 128 values)
                style_reg_loss = torch.norm(base_voice[0, 128:]) ** 2 * style_regularization
            
            # Combine losses with appropriate weights
            # Higher weight on L1 ensures primary convergence
            loss = l1_loss * 0.6 + mse_loss * 0.3 + freq_loss * 0.1 + style_reg_loss

            # -------------------- optimize using gradient accumulation -------------------------
            # Scale the loss by the accumulation steps to maintain the same gradients
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Clip gradients of the *actual* trainable parameters (may change after timbre freeze)
            torch.nn.utils.clip_grad_norm_(optim.param_groups[0]['params'], 1.0)

            # Only step optimizer after accumulating gradients from multiple samples
            if batch_count % gradient_accumulation_steps == 0 or batch_count == len(loader):
                optim.step()
                optim.zero_grad()
                
                # Explicitly clean memory if needed
                if memory_efficient and device.type == "mps" and batch_count % 5 == 0:
                    # Release unused memory
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    gc.collect()

            epoch_loss += loss.item() * gradient_accumulation_steps  # Scale back for reporting
            # print current step statistics
            print(f"{text[:20]}: loss {loss.item() * gradient_accumulation_steps:.4f} - {len(phonemes)} phonemes - epoch loss {epoch_loss:.4f}")
            
            # Log individual losses for current batch
            step = (epoch - 1) * len(loader) + loader.batch_size
            if writer is not None:
                writer.add_scalar('Batch/L1_Loss', l1_loss.item(), step)
                writer.add_scalar('Batch/MSE_Loss', mse_loss.item(), step)
                writer.add_scalar('Batch/Freq_Loss', freq_loss.item(), step)
                if style_regularization is not None and style_regularization > 0:
                    writer.add_scalar('Batch/Style_Reg_Loss', style_reg_loss.item(), step)
                writer.add_scalar('Batch/Combined_Loss', loss.item(), step)
                
            if use_wandb and WANDB_AVAILABLE:
                log_data = {
                    'batch_l1_loss': l1_loss.item(),
                    'batch_mse_loss': mse_loss.item(),
                    'batch_freq_loss': freq_loss.item(),
                    'batch_loss': loss.item(),
                    'step': step
                }
                if style_regularization is not None and style_regularization > 0:
                    log_data['batch_style_reg_loss'] = style_reg_loss.item()
                wandb.log(log_data)

        avg_loss = epoch_loss / len(loader)
        
        # Run validation if validation dataset is available
        validation_loss = None
        if validation_loader is not None:
            model.eval()  # Set model to evaluation mode
            val_losses = []
            with torch.no_grad():
                for val_text, val_target_log_mel in validation_loader:
                    # Process the validation sample
                    val_text = val_text[0]  # batch_size=1 for validation too
                    val_target_log_mel = val_target_log_mel.to(device)
                    
                    # Process text
                    phonemes, _ = g2p.g2p(val_text)
                    if phonemes is None or len(phonemes) == 0:
                        continue
                    
                    # Convert phonemes to input IDs
                    val_ids = text_to_input_ids(model, phonemes).to(device)
                    if val_ids.shape[1] < 3:
                        continue
                    
                    # Generate audio
                    val_audio_pred, _ = model.forward_with_tokens.__wrapped__(model, val_ids, base_voice)
                    
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
                    
                if val_losses:
                    validation_loss = sum(val_losses) / len(val_losses)
                    print(f"Epoch {epoch:>3}/{epochs}: training loss {avg_loss:.4f}, validation loss {validation_loss:.4f}")
                    
                    # Save best model (lowest validation L1 loss after epoch 5)
                    if save_best and epoch >= 5 and validation_loss < best_val_loss:
                        best_val_loss = validation_loss
                        best_epoch = epoch
                        print(f"New best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                        # Create output directory if it doesn't exist
                        os.makedirs(f"{out}/{name}", exist_ok=True)
                        save_voice(base_voice, voice_embed, f"{out}/{name}/{name}.best.pt")
                else:
                    print(f"Epoch {epoch:>3}/{epochs}: training loss {avg_loss:.4f} (no validation samples processed)")
            model.train()  # Set model back to training mode
        else:
            print(f"Epoch {epoch:>3}/{epochs}: loss {avg_loss:.4f}")
        
        # Log epoch metrics
        if writer is not None:
            writer.add_scalar('Epoch/Training_Loss', avg_loss, epoch)
            if validation_loss is not None:
                writer.add_scalar('Epoch/Validation_Loss', validation_loss, epoch)
            writer.add_scalar('Epoch/Learning_Rate', optim.param_groups[0]['lr'], epoch)
            
            # Log voice embedding stats
            timbre_data = base_voice[0, :128].detach().cpu().numpy()
            style_data = base_voice[0, 128:].detach().cpu().numpy()
            
            # Create histograms for timbre and style components
            writer.add_histogram('Voice/Timbre', timbre_data, epoch)
            writer.add_histogram('Voice/Style', style_data, epoch)
            
            # Log voice statistics
            writer.add_scalar('Voice/Timbre_Mean', timbre_data.mean(), epoch)
            writer.add_scalar('Voice/Timbre_Std', timbre_data.std(), epoch)
            writer.add_scalar('Voice/Style_Mean', style_data.mean(), epoch)
            writer.add_scalar('Voice/Style_Std', style_data.std(), epoch)
        
        if use_wandb and WANDB_AVAILABLE:
            log_dict = {
                'epoch': epoch,
                'epoch_loss': avg_loss,
                'learning_rate': optim.param_groups[0]['lr'],
                'timbre_mean': base_voice[0, :128].mean().item(),
                'timbre_std': base_voice[0, :128].std().item(),
                'style_mean': base_voice[0, 128:].mean().item(),
                'style_std': base_voice[0, 128:].std().item(),
            }
            
            if validation_loss is not None:
                log_dict['validation_loss'] = validation_loss
                
            wandb.log(log_dict)
            
            # Log histograms in W&B
            wandb.log({
                'timbre_hist': wandb.Histogram(base_voice[0, :128].detach().cpu().numpy()),
                'style_hist': wandb.Histogram(base_voice[0, 128:].detach().cpu().numpy())
            })
        
        # Log audio samples periodically
        if (epoch % log_audio_every == 0 or epoch == epochs) and (writer is not None or use_wandb):
            # Generate a sample with current voice embedding
            with torch.no_grad():
                # Use a different validation sample each epoch for better monitoring of phoneme drift
                if validation_dataset and len(validation_dataset) > 0:
                    # Pick a validation sample based on epoch number for variety
                    sample_idx = epoch % len(validation_dataset)
                    sample_text = validation_dataset.sentences[sample_idx]
                else:
                    # Fall back to training data with rotating selection if no validation set
                    sample_idx = epoch % len(dataset)
                    sample_text = dataset.sentences[sample_idx]
                phonemes, _ = g2p.g2p(sample_text)
                
                if phonemes:
                    # Generate audio
                    ids = text_to_input_ids(model, phonemes).to(device)
                    audio_sample, _ = model.forward_with_tokens.__wrapped__(model, ids, base_voice)
                    audio_sample = audio_sample.squeeze().cpu().numpy()
                    
                    # Generate mel spectrogram for visualization
                    with torch.no_grad():
                        mel = mel_transform_cpu(torch.tensor(audio_sample).unsqueeze(0))
                        log_mel_sample = 20 * torch.log10(mel.clamp(min=1e-5))[0].cpu().numpy()
                    
                    # Log to TensorBoard
                    if writer is not None:
                        writer.add_audio(f'Sample/{sample_text[:30]}', 
                                         audio_sample.reshape(1, -1), 
                                         epoch, 
                                         sample_rate=24000)
                        
                        # Create and log mel spectrogram figure
                        fig, ax = plt.subplots(figsize=(10, 4))
                        im = ax.imshow(log_mel_sample, aspect='auto', origin='lower')
                        plt.colorbar(im, ax=ax)
                        plt.title(f'Epoch {epoch}: {sample_text[:30]}')
                        plt.tight_layout()
                        writer.add_figure('Spectrogram/Sample', fig, epoch)
                        plt.close(fig)
                    
                    # Log to W&B
                    if use_wandb and WANDB_AVAILABLE:
                        wandb.log({f"audio_sample_epoch_{epoch}": wandb.Audio(
                            audio_sample, 
                            sample_rate=24000,
                            caption=f"Epoch {epoch}: {sample_text[:30]}"
                        )})
                        
                        # Log spectrogram to W&B
                        fig, ax = plt.subplots(figsize=(10, 4))
                        im = ax.imshow(log_mel_sample, aspect='auto', origin='lower')
                        plt.colorbar(im, ax=ax)
                        plt.title(f'Epoch {epoch}: {sample_text[:30]}')
                        plt.tight_layout()
                        wandb.log({f"spectrogram_epoch_{epoch}": wandb.Image(fig)})
                        plt.close(fig)
        
        scheduler.step()
        
        # Monitor embedding statistics to potentially freeze timbre
        current_timbre = base_voice[0, :128]
        current_style = base_voice[0, 128:]
        current_timbre_std = current_timbre.std().item()
        current_style_std = current_style.std().item()
        
        if (timbre_freeze_threshold is not None
                and current_timbre_std >= timbre_freeze_threshold
                and base_voice.requires_grad):

            print(f"Freezing timbre at std={current_timbre_std:.3f}")

            # 1.  Grab constant timbre part
            frozen_timbre = base_voice[0, :128].detach()     # (128,)
            # 2.  Create a *new* leaf Parameter for style
            style_param   = nn.Parameter(base_voice[0, 128:].detach())  # (128,)
            # 3.  Re-assemble 256-dim voice each forward pass
            def get_voice():
                return torch.cat([frozen_timbre, style_param]).unsqueeze(0)  # (1,256)

            base_voice = get_voice()           # first call
            voice_getter = get_voice           # keep for later reuse

            # 4.  Re-initialise optimizer to track only style_param
            optim = torch.optim.Adam([style_param], lr=optim.param_groups[0]['lr'],
                                    weight_decay=1e-6)
            voice_for_input = voice_getter()
        
        # Periodically normalize the voice tensor for better quality
        # This prevents drift and keeps voice characteristics appropriate
        if epoch % 25 == 0 and epoch > 0:
            with torch.no_grad():
                # Only normalize parts that aren't frozen
                if base_voice[0, :128].requires_grad:  # If timbre isn't frozen
                    current_timbre_mean = current_timbre.mean()
                    # Adjust toward target statistics
                    base_voice[0, :128] = ((current_timbre - current_timbre_mean) / current_timbre_std) * timbre_std + timbre_mean
                
                if base_voice[0, 128:].requires_grad:  # Style should always be trainable
                    current_style_mean = current_style.mean()
                    # Adjust toward target statistics
                    base_voice[0, 128:] = ((current_style - current_style_mean) / current_style_std) * style_std + style_mean
                
                print(f"Normalized voice at epoch {epoch}:")
                print(f"  Timbre: mean={base_voice[0, :128].mean().item():.4f}, std={base_voice[0, :128].std().item():.4f}")
                print(f"  Style: mean={base_voice[0, 128:].mean().item():.4f}, std={base_voice[0, 128:].std().item():.4f}")

        # Update the full voice tensor
        for i in range(MAX_PHONEME_LEN):
            voice_embed[i, 0, :] = base_voice.clone()
        
        # Save the full voice tensor every 10 epochs
        if epoch % 5 == 0:
            # Create output directory if it doesn't exist
            os.makedirs(f"{out}/{name}", exist_ok=True)
            save_voice(base_voice, voice_embed, f"{out}/{name}/{name}.epoch{epoch}.pt")

    
    # Save the final voice tensor
    os.makedirs(f"{out}/{name}", exist_ok=True)
    save_voice(base_voice, voice_embed, f"{out}/{name}/{name}.pt")
    
    # Print summary info about best checkpoint if available
    if save_best and best_epoch > 0:
        print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        print(f"Best model saved to: {out}/{name}/{name}.best.pt")
    
    # Upload to HuggingFace if requested
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
                wav_path = dataset.wav_paths[idx]
                info = torchaudio.info(wav_path)
                sample_durations.append(info.num_frames / info.sample_rate)
            
            avg_duration = sum(sample_durations) / len(sample_durations)
            total_duration = avg_duration * total_samples
        else:
            avg_duration = "Unknown"
            total_duration = "Unknown"
        
        dataset_stats = {
            "num_samples": total_samples,
            "total_duration": f"{total_duration:.2f}" if isinstance(total_duration, float) else total_duration,
            "avg_duration": f"{avg_duration:.2f}" if isinstance(avg_duration, float) else avg_duration,
        }
        
        # Collect audio samples
        audio_samples = {}
        if use_wandb and WANDB_AVAILABLE:
            # Find any saved audio files from wandb runs
            for epoch in range(1, epochs + 1, log_audio_every):
                sample_path = f"wandb/latest-run/files/media/audio/audio_sample_epoch_{epoch}_*.wav"
                matches = list(Path().glob(sample_path))
                if matches:
                    audio_samples[epoch] = str(matches[0])
        
        # Collect training parameters
        training_params = {
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop": hop,
            "style_regularization": style_regularization,
            "timbre_freeze_threshold": timbre_freeze_threshold,
            "hf_repo_id": hf_repo_id,
        }
        
        # Upload to HuggingFace
        print(f"Uploading files to HuggingFace repository: {hf_repo_id}")
        upload_to_huggingface(
            output_dir=out,
            name=name,
            best_epoch=best_epoch,
            dataset_stats=dataset_stats,
            training_params=training_params,
            audio_samples=audio_samples
        )
        
        print(f"Successfully uploaded voice model to HuggingFace: https://huggingface.co/{hf_repo_id}")

    # Close monitoring tools (always run)
    if writer is not None:
        writer.close()
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()


def save_voice(base_voice, voice_embed, out):
    # ---------------------------------------------------------------------
    # Save voice embedding
    # ---------------------------------------------------------------------
    save_path = Path(out)
    
    with torch.no_grad():
        # Update the full tensor one last time
        for i in range(MAX_PHONEME_LEN):
            voice_embed[i, 0, :] = base_voice.clone()
        
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
    readme_content = f"""# {name} - Kokoro Voice Model

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
- **Timbre Freeze Threshold:** {training_params.get('timbre_freeze_threshold', 'N/A')}

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
        

# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Import at top level to ensure they're available
    import gc
    
    ap = argparse.ArgumentParser(description="Train a Kokoro voice embedding")
    ap.add_argument("--data", type=str, help="Path to dataset directory")
    ap.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size")
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
    training_group.add_argument("--timbre-freeze", type=float, default=0.25,
                              help="Freeze timbre part of embedding when its std reaches this value (e.g., 0.5)")
    training_group.add_argument("--style-reg", type=float, default=1e-3,
                              help="L2 regularization strength for style part of embedding (e.g., 1e-4)")
    training_group.add_argument("--skip-validation", type=bool, default=False,
                              help="Skip validation split (default: False)")
    
    # Add monitoring arguments
    monitoring_group = ap.add_argument_group('Monitoring')
    monitoring_group.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    monitoring_group.add_argument("--log-dir", type=str, help="Directory for TensorBoard logs (default: runs/voice_name)")
    monitoring_group.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    monitoring_group.add_argument("--wandb-project", type=str, help="W&B project name")
    monitoring_group.add_argument("--wandb-name", type=str, help="W&B run name")
    monitoring_group.add_argument("--log-audio-every", type=int, default=10, help="Log audio samples every N epochs")
    
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
        # Advanced training parameters
        lr_decay_schedule=args.lr_decay,
        lr_decay_rate=args.lr_decay_rate,
        lr_decay_epochs=args.lr_decay_epochs,
        timbre_freeze_threshold=args.timbre_freeze,
        style_regularization=args.style_reg,
        skip_validation=args.skip_validation,
        # HuggingFace upload
        upload_to_hf=args.upload_to_hf,
        hf_repo_id=args.hf_repo_id,
    )
