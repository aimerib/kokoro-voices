"""
Utility classes for Kokoro Voices training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchaudio.functional import spectrogram
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def apply_audio_augmentations(audio: torch.Tensor, sample_rate: int = 24000, training: bool = True) -> torch.Tensor:
    """Apply subtle augmentations to audio for better generalization.
    
    Args:
        audio: Audio tensor [batch, samples] or [samples]
        sample_rate: Audio sample rate
        training: Whether to apply augmentations (disabled during validation)
    
    Returns:
        Augmented audio tensor
    """
    if not training:
        return audio
        
    # Ensure 2D
    is_1d = audio.dim() == 1
    if is_1d:
        audio = audio.unsqueeze(0)
    
    # 1. Subtle pitch shift (±2%)
    if torch.rand(1).item() < 0.3:
        shift_factor = 0.98 + torch.rand(1).item() * 0.04
        audio = F.interpolate(audio.unsqueeze(1), scale_factor=shift_factor, mode='linear', align_corners=False).squeeze(1)
    
    # 2. Very subtle time stretching (±1%)
    if torch.rand(1).item() < 0.2:
        stretch_factor = 0.99 + torch.rand(1).item() * 0.02
        audio = F.interpolate(audio.unsqueeze(1), scale_factor=stretch_factor, mode='linear', align_corners=False).squeeze(1)
    
    # 3. Subtle volume variation (±5%)
    if torch.rand(1).item() < 0.5:
        volume_factor = 0.95 + torch.rand(1).item() * 0.1
        audio = audio * volume_factor
    
    if is_1d:
        audio = audio.squeeze(0)
        
    return audio


class VoiceLoss(nn.Module):
    """
    Unified loss function for voice cloning that combines multiple loss components:
    - L1 and MSE on mel spectrograms 
    - Frequency gradient loss for detail preservation
    - Multi-resolution STFT loss on raw audio
    - Style regularization to prevent embedding explosion
    """
    def __init__(self, device="cpu", fft_sizes=[1024, 512, 256]):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.device = device
        
        # STFT windows setup
        self.fft_sizes = fft_sizes
        self.windows = {n: torch.hann_window(n).to(device) for n in fft_sizes}
        
    def mrstft(self, x):
        """Compute multi-resolution STFT spectrograms"""
        specs = []
        for n in self.fft_sizes:
            spec = spectrogram(
                x, pad=0, window=self.windows[n], n_fft=n,
                hop_length=n//4, win_length=n, power=1, normalized=True
            )
            specs.append(torch.log(spec.clamp(1e-5)))
        return specs
        
    def forward(self, pred_mel, target_mel, pred_audio, target_audio=None, 
                style_vector=None, epoch=0, style_reg_strength=0):
        """
        Compute combined loss
        
        Args:
            pred_mel: Predicted mel spectrogram [B, F, T]
            target_mel: Target mel spectrogram [B, F, T]
            pred_audio: Predicted raw audio waveform
            target_audio: Target raw audio waveform (optional)
            style_vector: Voice embedding style vector (for regularization)
            epoch: Current training epoch (for conditional regularization)
            style_reg_strength: Strength of style regularization
            
        Returns:
            tuple: (total_loss, loss_components_dict)
        """
        # Compute mel-based losses
        l1_loss = self.l1(pred_mel, target_mel)
        mse_loss = self.mse(pred_mel, target_mel)
        
        # Frequency gradient loss
        if pred_mel.shape[1] > 1:
            pred_diff = pred_mel[:, 1:] - pred_mel[:, :-1]
            target_diff = target_mel[:, 1:] - target_mel[:, :-1]
            freq_loss = self.l1(pred_diff, target_diff) * 0.5
        else:
            freq_loss = torch.tensor(0.0, device=self.device)
            
        # Style regularization
        style_reg_loss = torch.tensor(0.0, device=self.device)
        if epoch > 10 and style_reg_strength > 0 and style_vector is not None:
            style_std = style_vector.std().item()
            if style_std > 0.18:
                style_reg_loss = torch.norm(style_vector) ** 2 * style_reg_strength
        
        # STFT loss if raw audio is available
        stft_loss = torch.tensor(0.0, device=self.device)
        if target_audio is not None:
            # Process audio and truncate to matching lengths
            wave_pred = pred_audio.flatten().unsqueeze(0)
            wave_tgt = target_audio.squeeze(1)
            min_length = min(wave_pred.size(-1), wave_tgt.size(-1))
            wave_pred = wave_pred[..., :min_length]
            wave_tgt = wave_tgt[..., :min_length]
            
            # Compute MR-STFT loss
            mr_pred = self.mrstft(wave_pred)
            mr_tgt = self.mrstft(wave_tgt)
            stft_loss = sum(self.l1(p, t) for p, t in zip(mr_pred, mr_tgt)) / len(self.fft_sizes)
        
        # Combine with appropriate weights
        total_loss = (
            0.55 * l1_loss + 
            0.25 * mse_loss + 
            0.05 * freq_loss + 
            0.15 * stft_loss + 
            style_reg_loss
        )
        
        # Return both the total loss and individual components
        return total_loss, {
            "l1_loss": l1_loss.item(),
            "mse_loss": mse_loss.item(),
            "freq_loss": freq_loss.item(),
            "stft_loss": stft_loss.item(),
            "style_reg_loss": style_reg_loss.item(),
            "total_loss": total_loss.item()
        }


class TrainingLogger:
    """
    Unified logging for both TensorBoard and Weights & Biases
    """
    def __init__(self, use_tensorboard=False, use_wandb=False, 
                 log_dir=None, wandb_project=None, wandb_name=None):
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.writer = None
        
        # Initialize TensorBoard
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            
        # Initialize W&B
        if use_wandb:
            import wandb
            if not wandb.run:
                wandb.init(project=wandb_project, name=wandb_name)
    
    def log_metrics(self, metrics, step=None):
        """Log scalar metrics to both TensorBoard and W&B"""
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f'Metrics/{k}', v, step)
                
        if self.use_wandb:
            import wandb
            wandb.log(metrics)
    
    def log_audio(self, audio, sample_rate, caption, step, is_reference=False):
        """Log audio sample to both platforms"""
        if self.writer:
            self.writer.add_audio(f'Sample/{caption[:30]}', 
                                 audio.reshape(1, -1), 
                                 step, 
                                 sample_rate=sample_rate)
        
        if self.use_wandb:
            import wandb
            if is_reference:
                wandb.log({f"Reference Audio Epoch {step}": wandb.Audio(audio, sample_rate=sample_rate, caption=caption[:30])})
            else:
                wandb.log({f"Audio Epoch {step}": wandb.Audio(audio, sample_rate=sample_rate, caption=caption[:30])})
    
    def log_spectrogram(self, spec_img, caption, step, is_reference=False):
        """Log spectrogram figure to both platforms"""
        import matplotlib.pyplot as plt
        
        # Allow passing in either a numpy image (spectrogram matrix) **or** a pre-made Matplotlib Figure.
        if isinstance(spec_img, plt.Figure):
            # Caller provided a ready-made figure – use as-is
            fig = spec_img
        else:
            # spec_img is an ndarray; build a figure around it
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.imshow(spec_img, aspect='auto', origin='lower')
            plt.colorbar(im, ax=ax)
            plt.title(caption)
            plt.tight_layout()
        
        if self.writer:
            self.writer.add_figure('Spectrogram/Sample', fig, step)
            
        if self.use_wandb:
            import wandb
            if is_reference:
                wandb.log({f"Reference Spectrogram Epoch {step}": wandb.Image(fig)})
            else:
                wandb.log({f"Spectrogram Epoch {step}": wandb.Image(fig)})
            
        plt.close(fig)
    
    def log_embedding_stats(self, stats, step=None):
        """Log embedding statistics"""
        if self.writer:
            for k, v in stats.items():
                self.writer.add_scalar(f'Embedding/{k}', v, step)
                
        if self.use_wandb:
            import wandb
            wandb.log({f"embedding_{k}": v for k, v in stats.items()})
    
    def close(self):
        """Close logger resources"""
        if self.writer:
            self.writer.close()


class VoiceEmbedding:
    """
    Voice embedding class that handles initialization, normalization,
    and health checks for length-dependent embeddings during training.
    
    This version supports training different voice embeddings for different phoneme
    sequence lengths, as used by the Kokoro model with pack[len(ps)-1] indexing.
    """
    def __init__(self, embedding_size=256, max_phoneme_len=510, device="cpu"):
        self.embedding_size = embedding_size
        self.max_phoneme_len = max_phoneme_len
        self.device = device
        
        # Initialize length-dependent voice embeddings (all trainable)
        # Shape: (max_phoneme_len, 1, embedding_size)
        self.voice_embed = nn.Parameter(
            torch.zeros(max_phoneme_len, 1, embedding_size, device=device)
        )
        
        # Initialize with small variations based on length
        with torch.no_grad():
            # Start with a base voice with appropriate scale for Kokoro
            # Kokoro voices typically have values in the range [-0.5, 0.5]
            base = torch.randn(1, embedding_size, device=device) * 0.1
            
            # Create length-dependent variations
            for i in range(max_phoneme_len):
                # Shorter sequences get more expressive style
                style_factor = max(0.5, 1.0 - (i / max_phoneme_len) * 0.5)
                
                # Apply style factor to the style portion (last 128 dims)
                voice_vec = base.clone()
                voice_vec[0, embedding_size//2:] *= style_factor
                
                # Set as initial value
                self.voice_embed[i, 0, :] = voice_vec
        
        # For backward compatibility and health checks
        self._base_voice = None  # Will be computed on-demand as average
        
        # Target statistics for normalization
        self.timbre_mean = 0.0
        self.timbre_std = 0.25
        self.style_mean = 0.0
        self.style_std = 0.25
        
        # Health parameters
        self.target_std = 0.30
        self.max_std = 0.35
    
    @property
    def base_voice(self):
        """Get average voice for compatibility"""
        if self._base_voice is None:
            self._base_voice = self.voice_embed.mean(dim=0)
        return self._base_voice
    
    def get_for_length(self, length):
        """Get the appropriate voice embedding for a specific length"""
        idx = min(length - 1, self.max_phoneme_len - 1)
        idx = max(0, idx)  # Ensure non-negative
        return self.voice_embed[idx:idx+1]  # Keep the batch dimension
    
    def get_timbre(self, length=None):
        """Get the timbre part of the embedding (first half)
        
        Args:
            length: Phoneme sequence length, or None to use average
        """
        if length is None:
            return self.base_voice[0, :self.embedding_size//2]
        else:
            voice = self.get_for_length(length)
            return voice[0, 0, :self.embedding_size//2]
        
    def get_style(self, length=None):
        """Get the style part of the embedding (second half)
        
        Args:
            length: Phoneme sequence length, or None to use average
        """
        if length is None:
            return self.base_voice[0, self.embedding_size//2:]
        else:
            voice = self.get_for_length(length)
            return voice[0, 0, self.embedding_size//2:]
    
    def get_embedding_stats(self):
        """Get statistics across all length embeddings"""
        # Reset cached base voice
        self._base_voice = None
        
        # Calculate statistics across all embeddings
        all_timbre = self.voice_embed[:, 0, :self.embedding_size//2]
        all_style = self.voice_embed[:, 0, self.embedding_size//2:]
        
        timbre_mean = all_timbre.mean().item()
        timbre_std = all_timbre.std().item()
        style_mean = all_style.mean().item()
        style_std = all_style.std().item()
        
        # Also compute variation between adjacent embeddings
        embedding_diffs = self.voice_embed[1:] - self.voice_embed[:-1]
        length_variation = torch.norm(embedding_diffs, dim=(1, 2)).mean().item()
        
        return {
            "timbre_mean": timbre_mean,
            "timbre_std": timbre_std,
            "style_mean": style_mean,
            "style_std": style_std,
            "length_variation": length_variation
        }
    
    def check_health(self, epoch=0, warning_threshold=0.3):
        """Check embedding health and clamp if needed"""
        with torch.no_grad():
            # Get statistics across all embeddings
            stats = self.get_embedding_stats()
            timbre_std = stats["timbre_std"]
            style_std = stats["style_std"]
            
            clamped = False
            
            # Check if any embedding's timbre exceeds max_std
            timbre_norms = torch.norm(self.voice_embed[:, 0, :self.embedding_size//2], dim=1)
            max_timbre_norm = timbre_norms.max().item()
            
            if timbre_std > self.max_std or max_timbre_norm > self.max_std * 5:
                # Normalize all timbre parts
                for i in range(self.max_phoneme_len):
                    timbre = self.voice_embed[i, 0, :self.embedding_size//2]
                    if torch.norm(timbre) > self.max_std * 3:
                        scale = self.target_std / (timbre.std() + 1e-8)
                        self.voice_embed[i, 0, :self.embedding_size//2] = (
                            (timbre - timbre.mean()) * scale + self.timbre_mean
                        )
                print(f"[Clamp] Timbre std {timbre_std:.3f} → ~{self.target_std}")
                clamped = True
                
            # Check if any embedding's style exceeds max_std
            style_norms = torch.norm(self.voice_embed[:, 0, self.embedding_size//2:], dim=1)
            max_style_norm = style_norms.max().item()
            
            if style_std > self.max_std or max_style_norm > self.max_std * 5:
                # Normalize all style parts
                for i in range(self.max_phoneme_len):
                    style = self.voice_embed[i, 0, self.embedding_size//2:]
                    if torch.norm(style) > self.max_std * 3:
                        scale = self.target_std / (style.std() + 1e-8)
                        self.voice_embed[i, 0, self.embedding_size//2:] = (
                            (style - style.mean()) * scale + self.style_mean
                        )
                print(f"[Clamp] Style std {style_std:.3f} → ~{self.target_std}")
                clamped = True
            
            # Print warning if above threshold
            if timbre_std >= warning_threshold and epoch % 5 == 0:
                print(f"[Warn] Timbre std high ({timbre_std:.3f}) at epoch {epoch}")
                
            if clamped:
                # Recalculate stats after clamping
                stats = self.get_embedding_stats()
                
            return stats
    
    def apply_length_smoothing(self, weight=0.2):
        """Apply smoothing to reduce abrupt changes between adjacent length embeddings"""
        with torch.no_grad():
            # Simple exponential moving average
            smoothed = self.voice_embed.clone()
            
            for i in range(1, self.max_phoneme_len):
                smoothed[i] = weight * self.voice_embed[i-1] + (1 - weight) * self.voice_embed[i]
            
            self.voice_embed.copy_(smoothed)
    
    def calculate_smoothness_loss(self, weight=0.01):
        """Calculate a smoothness regularization loss between adjacent embeddings"""
        # Calculate differences between adjacent embeddings
        diffs = self.voice_embed[1:] - self.voice_embed[:-1]
        
        # Calculate squared L2 norm of differences
        squared_diffs = torch.sum(diffs**2, dim=(1, 2))
        
        # The loss is the mean of squared differences
        return weight * torch.mean(squared_diffs)
            
    def save(self, path):
        """Save voice embedding to disk"""
        state_dict = {
            "voice_embed": self.voice_embed.detach().cpu(),
            "embedding_size": self.embedding_size,
            "max_phoneme_len": self.max_phoneme_len,
            "timbre_mean": self.timbre_mean,
            "timbre_std": self.timbre_std,
            "style_mean": self.style_mean,
            "style_std": self.style_std
        }
        torch.save(state_dict, path)
        
        # Also save a simplified base_voice for backwards compatibility
        base_voice = self.base_voice
        torch.save({"base_voice": base_voice, "voice_embed": self.voice_embed}, 
                   path.replace(".pt", ".backward_compat.pt"))
        
    @classmethod
    def load(cls, path, device="cpu"):
        """Load voice embedding from disk"""
        state_dict = torch.load(path, map_location=device)
        embedding = cls(
            embedding_size=state_dict.get("embedding_size", 256),
            max_phoneme_len=state_dict.get("max_phoneme_len", 510),
            device=device
        )
        
        # Handle both new and old format
        if "voice_embed" in state_dict:
            # New format with length-dependent embeddings
            embedding.voice_embed = nn.Parameter(state_dict["voice_embed"].to(device))
            
            # Load statistics if available
            if "timbre_mean" in state_dict:
                embedding.timbre_mean = state_dict["timbre_mean"]
                embedding.timbre_std = state_dict["timbre_std"]
                embedding.style_mean = state_dict["style_mean"]
                embedding.style_std = state_dict["style_std"]
        elif "base_voice" in state_dict:
            # Old format with single base_voice - expand to all lengths
            base_voice = state_dict["base_voice"].to(device)
            with torch.no_grad():
                for i in range(embedding.max_phoneme_len):
                    embedding.voice_embed[i, 0, :] = base_voice[0]
        
        return embedding
