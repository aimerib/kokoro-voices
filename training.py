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
from utils import VoiceLoss, TrainingLogger, VoiceEmbedding

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
):
    # Initialize accelerator for mixed precision and gradient accumulation
    accelerator = Accelerator(
        mixed_precision="bf16",
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
    
    # # For backward compatibility
    # writer = logger.writer if use_tensorboard else None
    # WANDB_AVAILABLE = use_wandb
    
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
    model = KModel()
    
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
        
    # Effective batch size is handled by accelerator's gradient_accumulation_steps
    effective_batch_size = batch_size
    if effective_batch_size > 1:
        print(f"Using effective batch size of {effective_batch_size} through gradient accumulation")
        # Note: accelerator was already initialized with this value

    # ---------------------------------------------------------------------
    # Model + voice embedding
    # ---------------------------------------------------------------------
    # model = KModel().to(device).eval()
    model = KModel().train()  # Accelerator will handle device placement
    for p in model.parameters():
        p.requires_grad_(False)

    # Initialize a full 3D voice tensor [510, 1, 256] to match Kokoro's expected format
    voice_embed = torch.zeros((MAX_PHONEME_LEN, 1, 256), device=device)
    
    # Initialize with proper distribution similar to Kokoro voices
    # Get reference statistics from a known good voice (optional)
    ref_voice = None
    try:
        from huggingface_hub import hf_hub_download
        try:
            ref_path = hf_hub_download(repo_id='hexgrad/Kokoro-82M', filename='voices/af_heart.pt')
            ref_voice = torch.load(ref_path, map_location=device)
        except Exception as e:
            print(f"Could not download reference voice: {e}")
    except Exception as e:
        print(f"Could not import hf_hub_download: {e}")
        
        # Initialize voice embedding with our utility class
    voice_embedding = VoiceEmbedding(embedding_size=256, max_phoneme_len=MAX_PHONEME_LEN, device=device)
    
    # Try to initialize from reference if available
    try:
        # If reference embedding exists, we initialize from it
        ref_path = f"{out}/{name}/reference.pt"
        if not os.path.exists(ref_path):
            ref_path = "reference.pt"
            
        if os.path.exists(ref_path):
            print(f"Initializing from reference embedding: {ref_path}")
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
    except Exception as e:
        print(f"Using default distribution, could not load reference: {e}")
    
    # For backward compatibility - the rest of the code expects these variables
    # but now we use length-dependent embeddings instead of a single base_voice
    voice_embed = voice_embedding.voice_embed
    
    # Note: voice_embedding.base_voice is now a property that returns the average
    # of all length-dependent embeddings
    print("Using length-dependent voice embeddings!")
    print(f"Total trainable parameters: {voice_embed.numel():,d}")
    
    # Target statistics for health checks
    timbre_mean = voice_embedding.timbre_mean
    timbre_std = voice_embedding.timbre_std
    style_mean = voice_embedding.style_mean
    style_std = voice_embedding.style_std

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
            
        for text, target_log_mel, target_audio in loader:
            batch_count += 1
            
            # Process single sample (we're always using batch_size=1 for stability)
            text = text[0]  # batch_size=1
            target_log_mel = target_log_mel.to(device)
            target_audio = target_audio.to(device)
            
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

            # With length-dependent embeddings, we'll select the appropriate embedding
            # based on the phoneme sequence length - this happens later
                
            # Forward pass – call the *undecorated* version to retain gradients
            # Ensure all inputs are on the same device as the model
            ids = ids.to(device)
            
            # Get the voice embedding for this specific phoneme length
            phoneme_length = len(phonemes)
            voice_for_input = voice_embedding.get_for_length(phoneme_length).squeeze(1).to(device)
            
            # No need for autocast with accelerate - it handles this automatically
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
                if memory_efficient and device.type == "mps" and batch_count % 5 == 0:
                    # Release unused memory
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                    gc.collect()

            epoch_loss += loss.item() * gradient_accumulation_steps  # Scale back for reporting
            # print current step statistics
            print(f"{text[:20]}: loss {loss.item() * gradient_accumulation_steps:.4f} - {len(phonemes)} phonemes - epoch loss {epoch_loss:.4f}")
            
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
                'smoothness_loss': smoothness_loss.item()   
            }
            
            if style_regularization is not None and style_regularization > 0:
                batch_metrics['batch_style_reg_loss'] = style_reg_loss
                
            # Log all metrics at once
            logger.log_metrics(batch_metrics, step)

        avg_loss = epoch_loss / len(loader)
        
        # Run validation if validation dataset is available
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
                    phonemes, _ = g2p.g2p(val_text)
                    if phonemes is None or len(phonemes) == 0:
                        continue
                    
                    # Convert phonemes to input IDs
                    val_ids = text_to_input_ids(model, phonemes).to(device)
                    if val_ids.shape[1] < 3:
                        continue
                    
                    # Generate audio - ensure all inputs are on the same device
                    val_ids = val_ids.to(device)
                    
                    # For validation, use the appropriate length-specific voice
                    val_phoneme_length = len(phonemes)
                    voice_input = voice_embedding.get_for_length(val_phoneme_length).squeeze(1).to(device)
                    
                    val_audio_pred, _ = model.forward_with_tokens.__wrapped__(model, val_ids, voice_input)
                    
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
                        voice_embedding.save(f"{out}/{name}/{name}.best.pt")
                else:
                    print(f"Epoch {epoch:>3}/{epochs}: training loss {avg_loss:.4f} (no validation samples processed)")
            model.train()  # Set model back to training mode
        else:
            print(f"Epoch {epoch:>3}/{epochs}: loss {avg_loss:.4f}")
        
        # # Log epoch metrics
        # if writer is not None:
        #     writer.add_scalar('Epoch/Training_Loss', avg_loss, epoch)
        #     if validation_loss is not None:
        #         writer.add_scalar('Epoch/Validation_Loss', validation_loss, epoch)
        #     writer.add_scalar('Epoch/Learning_Rate', optim.param_groups[0]['lr'], epoch)
            
        #     # Log voice embedding stats
        #     timbre_data = voice_embedding.base_voice[0, :128].detach().cpu().numpy()
        #     style_data = voice_embedding.base_voice[0, 128:].detach().cpu().numpy()
            
        #     # Create histograms for timbre and style components
        #     writer.add_histogram('Voice/Timbre', timbre_data, epoch)
        #     writer.add_histogram('Voice/Style', style_data, epoch)
            
        #     # Log voice statistics
        #     writer.add_scalar('Voice/Timbre_Mean', timbre_data.mean(), epoch)
        #     writer.add_scalar('Voice/Timbre_Std', timbre_data.std(), epoch)
        #     writer.add_scalar('Voice/Style_Mean', style_data.mean(), epoch)
        #     writer.add_scalar('Voice/Style_Std', style_data.std(), epoch)
        
        # if use_wandb and WANDB_AVAILABLE:
            # Log voice embedding stats
        timbre_data = voice_embedding.base_voice[0, :128].detach().cpu().numpy()
        style_data = voice_embedding.base_voice[0, 128:].detach().cpu().numpy()
        log_dict = {
            'epoch': epoch,
            'epoch_loss': avg_loss,
            'learning_rate': optim.param_groups[0]['lr'],
            'timbre_mean': timbre_data.mean().item(),
            'timbre_std': timbre_data.std().item(),
            'style_mean': style_data.mean().item(),
            'style_std': style_data.std().item()
        }
        
        if validation_loss is not None:
            log_dict['validation_loss'] = validation_loss
            
        # wandb.log(log_dict)
        
        # # Log histograms in W&B
        # wandb.log({
        #     'timbre_hist': wandb.Histogram(voice_embedding.base_voice[0, :128].detach().cpu().numpy()),
        #     'style_hist': wandb.Histogram(voice_embedding.base_voice[0, 128:].detach().cpu().numpy())
        # })
        logger.log_metrics(log_dict, epoch)
        logger.log_metrics({
            'timbre_hist': wandb.Histogram(voice_embedding.base_voice[0, :128].detach().cpu().numpy()),
            'style_hist': wandb.Histogram(voice_embedding.base_voice[0, 128:].detach().cpu().numpy())
        })
    
        # Log audio samples periodically
        if (epoch % log_audio_every == 0 or epoch == epochs):
            # Generate a sample with current voice embedding
            with torch.no_grad():
                # Use a different validation sample each epoch for better monitoring of phoneme drift
                if validation_dataset and len(validation_dataset) > 0:
                    # Pick a validation sample based on epoch number for variety
                    sample_idx = epoch % len(validation_dataset)
                    sample_text = validation_dataset.sentences[sample_idx]
                    reference_audio = validation_dataset._load_wav(validation_dataset.wav_paths[sample_idx])
                else:
                    # Fall back to training data with rotating selection if no validation set
                    sample_idx = epoch % len(dataset)
                    sample_text = dataset.sentences[sample_idx]
                    reference_audio = dataset._load_wav(dataset.wav_paths[sample_idx])
                phonemes, _ = g2p.g2p(sample_text)
                
                if phonemes:
                    # Generate audio
                    ids = text_to_input_ids(model, phonemes).to(device)
                    voice_input = voice_embedding.base_voice.to(device)
                    audio_sample, _ = model.forward_with_tokens.__wrapped__(model, ids, voice_input)
                    audio_sample = audio_sample.squeeze().cpu().numpy()
                    
                    # Generate mel spectrogram for visualization
                    with torch.no_grad():
                        mel = mel_transform_cpu(torch.tensor(audio_sample).unsqueeze(0))
                        reference_mel = mel_transform_cpu(torch.tensor(reference_audio).unsqueeze(0))
                        log_mel_sample = 20 * torch.log10(mel.clamp(min=1e-5))[0].cpu().numpy()
                        log_reference_mel = 20 * torch.log10(reference_mel.clamp(min=1e-5))[0].cpu().numpy()
                    
                    # Use unified logger to log audio samples and spectrograms
                    logger.log_audio(
                        audio=reference_audio.squeeze().cpu().numpy(),
                        sample_rate=24000,
                        caption=f"Reference: {sample_text[:30]}",
                        step=epoch
                    )
                    logger.log_audio(
                        audio=audio_sample,
                        sample_rate=24000,
                        caption=f"Epoch {epoch}: {sample_text[:30]}",
                        step=epoch
                    )
                    
                    # Log spectrogram
                    logger.log_spectrogram(
                        spec_img=log_reference_mel,
                        caption=f"Reference: {sample_text[:30]}",
                        step=0
                    )
                    logger.log_spectrogram(
                        spec_img=log_mel_sample,
                        caption=f"Epoch {epoch}: {sample_text[:30]}",
                        step=epoch
                    )
        
        scheduler.step()
        
        # Check embedding health and apply clamping if needed
        embedding_stats = voice_embedding.check_health(epoch=epoch, warning_threshold=timbre_warning_threshold)
        
        # Every 5 epochs, apply length smoothing to avoid discontinuities
        if epoch % 5 == 0:
            voice_embedding.apply_length_smoothing(weight=0.2)
            print(f"Applied length smoothing at epoch {epoch}")
        
        # Log embedding statistics with length variation
        logger.log_embedding_stats(embedding_stats, step=epoch)
        
        # Save the full voice tensor every 5 epochs
        if epoch % 5 == 0:
            # Create output directory if it doesn't exist
            os.makedirs(f"{out}/{name}", exist_ok=True)
            voice_embedding.save(f"{out}/{name}/{name}.epoch{epoch}.pt")
            
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

    
    # Save the final voice embedding
    os.makedirs(f"{out}/{name}", exist_ok=True)
    
    # # Add special flag for length-dependent training to the name
    # name_suffix = "_length_dependent"
    # if name and not name.endswith(name_suffix):
    #     name = f"{name}{name_suffix}"
        
    if wandb_name is None:
        wandb_name = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
        
    # Use the VoiceEmbedding save method to save length-dependent embeddings
    voice_embedding.save(f"{out}/{name}/{name}.pt")
    
    # Also save a backwards compatibility version
    print("Saving backwards-compatible version for older inference code")
    base_voice_avg = voice_embedding.base_voice.detach().cpu()
    save_voice(base_voice_avg, voice_embedding.voice_embed.detach().cpu(), f"{out}/{name}/{name}.compat.pt")
    
    # Print summary info about best checkpoint if available
    if save_best and best_epoch > 0:
        print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        print(f"Best model saved to: {out}/{name}/{name}.best.pt")
    
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
            "hop": hop,
            "style_regularization": style_regularization,
            "timbre_warning_threshold": timbre_warning_threshold,
            "hf_repo_id": hf_repo_id,
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
        upload_model_to_hf(
            local_dir=os.path.join(out, name),
            repo_id=hf_repo_id,
            voice_name=name,
            best_epoch=best_epoch,
            dataset_stats=dataset_stats,
            training_params=training_params,
            audio_samples=audio_samples
        )
        
        print(f"Successfully uploaded voice model to HuggingFace: https://huggingface.co/{hf_repo_id}")

    # Close unified logger
    logger.close()


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
        timbre_warning_threshold=args.timbre_warning,
        style_regularization=args.style_reg,
        skip_validation=args.skip_validation,
        upload_to_hf=args.upload_to_hf,
        hf_repo_id=args.hf_repo_id,
    )
