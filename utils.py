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

import soundfile as sf
from kokoro import KPipeline

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
    - Perceptual similarity loss for better voice matching
    """
    def __init__(self, device="cpu", fft_sizes=[1024, 512, 256], use_perceptual_loss=False):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.device = device
        self.use_perceptual_loss = use_perceptual_loss
        
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
        
        # Perceptual similarity loss
        perceptual_loss = torch.tensor(0.0, device=self.device)
        perceptual_components = {}
        if self.use_perceptual_loss and epoch > 5:  # Start after initial training
            perceptual_loss, perceptual_components = calculate_training_similarity_loss(
                pred_mel, target_mel, self.device
            )
            # Scale down perceptual loss to avoid dominating
            perceptual_loss = perceptual_loss * 0.1
        
        # Combine with appropriate weights
        # Adjust weights when using perceptual loss
        if self.use_perceptual_loss and epoch > 5:
            total_loss = (
                0.45 * l1_loss + 
                0.20 * mse_loss + 
                0.05 * freq_loss + 
                0.15 * stft_loss + 
                0.15 * perceptual_loss +  # Add perceptual loss
                style_reg_loss
            )
        else:
            # Original weights
            total_loss = (
                0.55 * l1_loss + 
                0.25 * mse_loss + 
                0.05 * freq_loss + 
                0.15 * stft_loss + 
                style_reg_loss
            )
        
        # Prepare loss components dictionary
        loss_components = {
            "l1_loss": l1_loss.item(),
            "mse_loss": mse_loss.item(),
            "freq_loss": freq_loss.item(),
            "stft_loss": stft_loss.item(),
            "style_reg_loss": style_reg_loss.item(),
            "perceptual_loss": perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else 0.0,
            "total_loss": total_loss.item()
        }
        
        # Add perceptual sub-components if available
        if perceptual_components:
            for key, value in perceptual_components.items():
                loss_components[f"perceptual_{key}"] = value.item() if isinstance(value, torch.Tensor) else value
        
        # Return both the total loss and individual components
        return total_loss, loss_components


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
    
    def log_voice_drift(self, drift_metrics, step=None):
        """Log voice drift metrics from original reference voice"""
        if not drift_metrics:
            return
            
        if self.writer:
            for k, v in drift_metrics.items():
                if k != 'top_changed_dims':  # Skip the topk tensor
                    self.writer.add_scalar(f'VoiceDrift/{k}', v, step)
                
        if self.use_wandb:
            import wandb
            # Log main drift metrics
            drift_log = {}
            for k, v in drift_metrics.items():
                if k != 'top_changed_dims':  # Skip the topk tensor
                    drift_log[f"voice_drift_{k}"] = v
            
            # Log top changed dimensions if available
            if 'top_changed_dims' in drift_metrics:
                top_dims = drift_metrics['top_changed_dims']
                for i, (change, dim_idx) in enumerate(zip(top_dims.values, top_dims.indices)):
                    drift_log[f"voice_drift_top_change_{i+1}"] = change.item()
                    drift_log[f"voice_drift_top_dim_{i+1}"] = dim_idx.item()
            
            wandb.log(drift_log)
    
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
        voice_embedding = self.voice_embed.detach().cpu()
        state_dict = {
            "voice_embed": voice_embedding,
            "embedding_size": self.embedding_size,
            "max_phoneme_len": self.max_phoneme_len,
            "timbre_mean": self.timbre_mean,
            "timbre_std": self.timbre_std,
            "style_mean": self.style_mean,
            "style_std": self.style_std
        }
        torch.save(state_dict, path.replace(".pt", ".embedding.pt"))
        
        # Save the real voice tensor for actual use with Kokoro
        torch.save(voice_embedding, path)
        
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

def calculate_voice_drift(original_voice, voice_embedding, device):
    """
    Calculate voice drift metrics between the original reference voice and the current voice embedding.
    
    Args:
        original_voice: Original reference voice tensor
        voice_embedding: VoiceEmbedding object with current voice
        device: Device for computation
    
    Returns:
        Dictionary with voice drift metrics
    """
    if original_voice is None:
        return {}
    
    with torch.no_grad():
        # Move to same device for comparison
        original_voice = original_voice.to(device)
        
        if original_voice.dim() == 3 and original_voice.shape == (510, 1, 256):
            # 3D format - compare averaged voice
            original_avg = original_voice.mean(dim=0).squeeze()  # [256]
        elif original_voice.dim() == 1 and original_voice.shape[0] == 256:
            # 1D format
            original_avg = original_voice
        else:
            return {}
        
        # Get current trained voice (average across phoneme positions)
        trained_avg = voice_embedding.voice_embed.mean(dim=0).squeeze().to(device)  # [256]
        
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
        
        return {
            'cosine_similarity': cosine_sim,
            'l2_distance': l2_distance,
            'l1_distance': l1_distance,
            'max_difference': max_diff,
            'norm_change_percent': norm_change_percent,
            'top_changed_dims': top_changed_dims
        }


def generate_with_custom_voice(text, voice_path, output_file="output.wav"):
    # Initialize the pipeline
    pipeline = KPipeline(lang_code='a')
    
    # Load and expand the voice tensor
    try:
        voice_tensor = torch.load(voice_path, map_location='cpu')
# # Register under a name, e.g. 'my_voice'
# pipeline.voices['my_voice'] = voice_tensor.squeeze(0)  
#         voice_tensor = torch.load(voice_path)
        print(f"Loaded voice tensor with shape: {voice_tensor.shape}")
        
        # Make sure tensor has the right format for Kokoro
        # expanded_voice = expand_voice_tensor(voice_tensor)
        
        # Generate audio
        outputs = []
        for _, _, audio in pipeline(text, voice=voice_tensor):
            print(f"Generated {audio.shape[0]} samples")
            outputs.append(audio)
        
        # Combine and save
        if outputs:
            full_audio = torch.cat(outputs)
            sf.write(output_file, full_audio.numpy(), 24000)
            print(f"Saved audio to {output_file}")
            return True
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to stock voice")
        
        # Fallback to a stock voice
        outputs = []
        for _, _, audio in pipeline(text, voice="af_heart"):
            outputs.append(audio)
        
        if outputs:
            full_audio = torch.cat(outputs)
            sf.write(output_file, full_audio.numpy(), 24000)
            print(f"Saved audio with fallback voice to {output_file}")
            return True
    
    return False

def generate_with_standard_voice(text, output_file="output.wav"):
    # Initialize the pipeline
    pipeline = KPipeline(lang_code='a')
    
    # Generate audio
    outputs = []
    for _, _, audio in pipeline(text, voice="af_heart"):
        outputs.append(audio)
    
    # Combine and save
    if outputs:
        full_audio = torch.cat(outputs)
        sf.write(output_file, full_audio.numpy(), 24000)
        print(f"Saved audio to {output_file}")
        return True
    
    return False

def calculate_audio_similarity(voice_embedding, target_audio_path, text, kokoro_pipeline, device, sr=24000):
    """
    Calculate audio similarity between generated speech from current voice and target audio.
    
    This provides perceptual similarity metrics by generating speech with the current voice
    and comparing it to target audio samples using mel spectrograms and audio features.
    
    Args:
        voice_embedding: VoiceEmbedding object with current trained voice
        target_audio_path: Path to target wav file
        text: Text that was used to generate the target audio
        kokoro_pipeline: Kokoro pipeline for speech generation
        device: Device for computation
        sr: Sample rate (default 24000 for Kokoro)
    
    Returns:
        Dictionary with audio similarity metrics
    """
    try:
        import librosa
        import numpy as np
        from scipy.spatial.distance import cosine
        from scipy.stats import pearsonr
        from kokoro import KPipeline
    except ImportError:
        print("Warning: librosa and scipy required for audio similarity calculation")
        return {}
    
    try:
        pipeline = KPipeline(lang_code='a')
        # Generate audio with current voice
        audio_generator = pipeline(text, voice=voice_embedding.voice_embed.squeeze().cpu().numpy())
        
        generated_audio = torch.empty(0)
        for _, _, audio in audio_generator:
            generated_audio = torch.cat((generated_audio, torch.from_numpy(audio)))

        # Load target audio
        target_audio, _ = librosa.load(target_audio_path, sr=sr)
        
        # Ensure both audios are same length (pad shorter or trim longer)
        min_len = min(len(generated_audio), len(target_audio))
        generated_audio = generated_audio[:min_len]
        target_audio = target_audio[:min_len]
        
        # Skip if audio too short
        if min_len < sr * 0.5:  # Less than 0.5 seconds
            return {}
        
        # Extract mel spectrograms
        n_mels = 80
        hop_length = 256
        win_length = 1024
        n_fft = 1024
        
        generated_mel = librosa.feature.melspectrogram(
            y=generated_audio, sr=sr, n_mels=n_mels, 
            hop_length=hop_length, win_length=win_length, n_fft=n_fft
        )
        target_mel = librosa.feature.melspectrogram(
            y=target_audio, sr=sr, n_mels=n_mels,
            hop_length=hop_length, win_length=win_length, n_fft=n_fft
        )
        
        # Convert to log scale
        generated_mel_db = librosa.power_to_db(generated_mel)
        target_mel_db = librosa.power_to_db(target_mel)
        
        # Align spectrograms (pad shorter one)
        min_frames = min(generated_mel_db.shape[1], target_mel_db.shape[1])
        generated_mel_db = generated_mel_db[:, :min_frames]
        target_mel_db = target_mel_db[:, :min_frames]
        
        # Calculate similarity metrics
        metrics = {}
        
        # 1. Mel Cepstral Distortion (MCD) - lower is better
        generated_mfcc = librosa.feature.mfcc(S=generated_mel_db, n_mfcc=13)
        target_mfcc = librosa.feature.mfcc(S=target_mel_db, n_mfcc=13)
        mcd = np.mean(np.sqrt(np.sum((generated_mfcc - target_mfcc) ** 2, axis=0)))
        metrics['mel_cepstral_distortion'] = mcd
        
        # 2. Spectral convergence - lower is better
        spectral_conv = np.linalg.norm(generated_mel_db - target_mel_db, 'fro') / np.linalg.norm(target_mel_db, 'fro')
        metrics['spectral_convergence'] = spectral_conv
        
        # 3. Cosine similarity on flattened mel spectrograms - higher is better
        generated_flat = generated_mel_db.flatten()
        target_flat = target_mel_db.flatten()
        mel_cosine_sim = 1 - cosine(generated_flat, target_flat)
        metrics['mel_cosine_similarity'] = mel_cosine_sim
        
        # 4. Pearson correlation on mel spectrograms - higher is better
        mel_correlation, _ = pearsonr(generated_flat, target_flat)
        metrics['mel_correlation'] = mel_correlation
        
        # 5. Fundamental frequency similarity (F0)
        try:
            generated_f0, _ = librosa.piptrack(y=generated_audio, sr=sr)
            target_f0, _ = librosa.piptrack(y=target_audio, sr=sr)
            
            # Get dominant F0
            generated_f0_dom = np.max(generated_f0, axis=0)
            target_f0_dom = np.max(target_f0, axis=0)
            
            # Remove zero values
            generated_f0_nonzero = generated_f0_dom[generated_f0_dom > 0]
            target_f0_nonzero = target_f0_dom[target_f0_dom > 0]
            
            if len(generated_f0_nonzero) > 0 and len(target_f0_nonzero) > 0:
                f0_diff = abs(np.mean(generated_f0_nonzero) - np.mean(target_f0_nonzero))
                metrics['f0_difference_hz'] = f0_diff
                
                # F0 similarity (normalized by target F0)
                f0_similarity = 1 - (f0_diff / np.mean(target_f0_nonzero))
                metrics['f0_similarity'] = max(0, f0_similarity)  # Clamp to 0-1
        except:
            metrics['f0_difference_hz'] = float('inf')
            metrics['f0_similarity'] = 0.0
        
        # 6. Spectral centroid similarity
        generated_centroid = librosa.feature.spectral_centroid(y=generated_audio, sr=sr)[0]
        target_centroid = librosa.feature.spectral_centroid(y=target_audio, sr=sr)[0]
        
        min_centroid_len = min(len(generated_centroid), len(target_centroid))
        generated_centroid = generated_centroid[:min_centroid_len]
        target_centroid = target_centroid[:min_centroid_len]
        
        centroid_correlation, _ = pearsonr(generated_centroid, target_centroid)
        metrics['spectral_centroid_correlation'] = centroid_correlation
        
        # 7. Overall perceptual similarity score (weighted combination)
        # Higher is better, range 0-1
        perceptual_score = (
            mel_cosine_sim * 0.3 +
            mel_correlation * 0.3 +
            (1 - min(spectral_conv, 1.0)) * 0.2 +  # Invert spectral convergence
            metrics['f0_similarity'] * 0.1 +
            (centroid_correlation if not np.isnan(centroid_correlation) else 0) * 0.1
        )
        metrics['perceptual_similarity_score'] = max(0, min(1, perceptual_score))
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating audio similarity: {e}")
        return {}

def calculate_training_similarity_loss(pred_mel, target_mel, device):
    """
    Calculate differentiable similarity losses between predicted and target mel spectrograms.
    This is a simplified version of calculate_audio_similarity designed for use during training.
    
    Args:
        pred_mel: Predicted mel spectrogram [B, n_mels, T]
        target_mel: Target mel spectrogram [B, n_mels, T]
        device: Device for computation
        
    Returns:
        Dictionary with differentiable loss components
    """
    import torch.nn.functional as F
    
    # Ensure same shape
    min_t = min(pred_mel.shape[-1], target_mel.shape[-1])
    pred_mel = pred_mel[..., :min_t]
    target_mel = target_mel[..., :min_t]
    
    losses = {}
    
    # 1. Cosine similarity loss (maximize similarity = minimize negative cosine)
    # Flatten spectrograms for cosine similarity
    pred_flat = pred_mel.view(pred_mel.shape[0], -1)
    target_flat = target_mel.view(target_mel.shape[0], -1)
    
    # Normalize for cosine similarity
    pred_norm = F.normalize(pred_flat, p=2, dim=1)
    target_norm = F.normalize(target_flat, p=2, dim=1)
    
    # Cosine similarity (batch-wise)
    cosine_sim = (pred_norm * target_norm).sum(dim=1)
    cosine_loss = 1.0 - cosine_sim.mean()  # Convert to loss (0 = identical)
    losses['cosine_loss'] = cosine_loss
    
    # 2. Spectral convergence loss (normalized L2)
    spectral_conv = torch.norm(pred_mel - target_mel, p='fro') / (torch.norm(target_mel, p='fro') + 1e-8)
    losses['spectral_convergence'] = spectral_conv
    
    # 3. Log-magnitude L1 loss (perceptually weighted)
    # Apply log scaling for perceptual weighting
    pred_log = torch.log(torch.clamp(pred_mel, min=1e-5))
    target_log = torch.log(torch.clamp(target_mel, min=1e-5))
    log_l1_loss = F.l1_loss(pred_log, target_log)
    losses['log_l1_loss'] = log_l1_loss
    
    # 4. Frequency-wise correlation loss
    # Calculate correlation along time axis for each frequency bin
    # This helps match the spectral envelope
    freq_corr_loss = 0
    for i in range(pred_mel.shape[1]):  # For each frequency bin
        pred_freq = pred_mel[:, i, :]
        target_freq = target_mel[:, i, :]
        
        # Normalize each frequency bin
        pred_freq_norm = (pred_freq - pred_freq.mean(dim=1, keepdim=True)) / (pred_freq.std(dim=1, keepdim=True) + 1e-8)
        target_freq_norm = (target_freq - target_freq.mean(dim=1, keepdim=True)) / (target_freq.std(dim=1, keepdim=True) + 1e-8)
        
        # Correlation as dot product of normalized vectors
        corr = (pred_freq_norm * target_freq_norm).mean(dim=1)
        freq_corr_loss += (1.0 - corr.mean())
    
    freq_corr_loss = freq_corr_loss / pred_mel.shape[1]
    losses['freq_correlation_loss'] = freq_corr_loss
    
    # 5. Spectral centroid matching loss
    # Compute spectral centroid (center of mass of spectrum)
    freq_bins = torch.arange(pred_mel.shape[1], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    
    pred_centroid = (freq_bins * pred_mel).sum(dim=1) / (pred_mel.sum(dim=1) + 1e-8)
    target_centroid = (freq_bins * target_mel).sum(dim=1) / (target_mel.sum(dim=1) + 1e-8)
    
    centroid_loss = F.l1_loss(pred_centroid, target_centroid)
    losses['centroid_loss'] = centroid_loss
    
    # Combined perceptual loss
    perceptual_loss = (
        0.3 * cosine_loss +
        0.2 * spectral_conv +
        0.2 * log_l1_loss +
        0.2 * freq_corr_loss +
        0.1 * centroid_loss
    )
    
    losses['perceptual_loss'] = perceptual_loss
    
    return perceptual_loss, losses


def interpret_audio_similarity(metrics):
    """
    Provide human-readable interpretation of audio similarity metrics.
    
    Args:
        metrics: Dictionary from calculate_audio_similarity
    
    Returns:
        String with interpretation
    """
    if not metrics:
        return "No audio similarity metrics available"
    
    interpretation = []
    
    # Perceptual similarity score
    perceptual = metrics.get('perceptual_similarity_score', 0)
    if perceptual > 0.85:
        interpretation.append(" Excellent perceptual similarity - voices sound very similar")
    elif perceptual > 0.70:
        interpretation.append(" Good perceptual similarity - voices are recognizably similar")
    elif perceptual > 0.50:
        interpretation.append(" Moderate similarity - some voice characteristics match")
    else:
        interpretation.append(" Low similarity - significant voice differences")
    
    # Mel Cepstral Distortion
    mcd = metrics.get('mel_cepstral_distortion', float('inf'))
    if mcd < 2.0:
        interpretation.append(" Very low MCD - excellent spectral match")
    elif mcd < 4.0:
        interpretation.append(" Good MCD - solid spectral similarity")
    elif mcd < 6.0:
        interpretation.append(" Moderate MCD - some spectral differences")
    else:
        interpretation.append(" High MCD - significant spectral differences")
    
    # F0 similarity
    f0_sim = metrics.get('f0_similarity', 0)
    if f0_sim > 0.90:
        interpretation.append(" Excellent pitch matching")
    elif f0_sim > 0.70:
        interpretation.append(" Good pitch similarity")
    elif f0_sim > 0.50:
        interpretation.append(" Moderate pitch differences")
    else:
        interpretation.append(" Significant pitch differences")
    
    return " | ".join(interpretation)