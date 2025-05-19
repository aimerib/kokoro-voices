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
    prompts.txt       # tab-separated: <wav_filename> \t <sentence>
    *.wav             # mono wavs (~24 kHz). Any sample-rate is accepted and resampled.

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
from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader

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


class VoiceDataset(Dataset):
    """Loads (<sentence>, log-mel target) pairs for training."""

    def __init__(self, root: str | Path, mel_transform: nn.Module, target_sr: int = 24_000, device=None):
        self.device = device or torch.device("cpu")
        self.root = Path(root)
        self.sentences: List[str] = []
        self.wav_paths: List[Path] = []
        self.mel_transform = mel_transform
        self.target_sr = target_sr

        prompts = self.root / "prompts.txt"
        if not prompts.exists():
            raise FileNotFoundError(f"{prompts} not found – expected prompts.txt in dataset directory")

        with prompts.open() as f:
            for i, line in enumerate(f):
                if i >= 60:
                    break
                line = line.strip()
                if not line:
                    continue
                wav_name, text = line.split("\t", 1)
                wav_path = self.root / wav_name
                if not wav_path.exists():
                    raise FileNotFoundError(f"Missing WAV file: {wav_path}")
                self.wav_paths.append(wav_path)
                self.sentences.append(text)

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
    lr: float = 1e-2,
    out: str = "my_voice.pt",
    n_mels: int = 80,
    n_fft: int = 1024,
    hop: int = 256,
):
    device = get_device()
    print(f"Using device: {device}")

    # Create two separate mel transforms - one for CPU (dataset) and one for device (training)
    mel_transform_cpu = torchaudio.transforms.MelSpectrogram(
        sample_rate=24_000, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24_000, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    ).to(device)

    dataset = VoiceDataset(data_root, mel_transform_cpu)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ---------------------------------------------------------------------
    # Model + voice embedding
    # ---------------------------------------------------------------------
    model = KModel().to(device).eval()
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
    
    # Scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Multiple loss functions for better results
    l1_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()

    # G2P pipeline for text -> phonemes
    g2p = KPipeline(lang_code="a", model=False)  # Only G2P functionality needed

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for text, target_log_mel in loader:
            text = text[0]  # batch_size=1
            target_log_mel = target_log_mel.to(device)
            # Dataset now consistently returns (1, n_mels, T) shape

            # -------------------- prepare inputs --------------------
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
            
            # We need to ensure voice_for_input is [B, 256], not [1, 1, 256]
            # In this case, B=1 (batch size)
            voice_for_input = base_voice  # Already [1, 256] and requires_grad=True
                
            # Forward pass – call the *undecorated* version to retain gradients
            audio_pred, _ = model.forward_with_tokens.__wrapped__(  # type: ignore[attr-defined]
                model, ids, voice_for_input
            )
            
            # Make sure prediction is on the correct device
            if audio_pred.device != device:
                audio_pred = audio_pred.to(device)
                
            # audio_pred is (1, samples) - transform it to spectrogram using device-specific transform
            pred_log_mel = mel_transform(audio_pred)
            pred_log_mel = 20 * torch.log10(pred_log_mel.clamp(min=1e-5))
            # Shape will be [1, n_mels, T]

            # Make sure dimensions match - DataLoader might add an extra batch dim
            # Standardize both to 3D tensors [batch, n_mels, time]
            if target_log_mel.dim() == 4:  # [1, 1, n_mels, T]
                target_log_mel = target_log_mel.squeeze(0)  # -> [1, n_mels, T]
            
            if pred_log_mel.dim() == 2:  # [n_mels, T]
                pred_log_mel = pred_log_mel.unsqueeze(0)  # -> [1, n_mels, T]
            
            # Align time dimension
            T = min(pred_log_mel.shape[-1], target_log_mel.shape[-1])
            
            # Multiple loss components for better learning
            # 1. L1 loss on mel-spectrogram (primary loss)
            l1_loss = l1_loss_fn(pred_log_mel[..., :T], target_log_mel[..., :T])
            
            # 2. MSE loss for spectral contour
            mse_loss = mse_loss_fn(pred_log_mel[..., :T], target_log_mel[..., :T])
            
            # 3. High-frequency emphasis loss - focus on important details
            # Calculate gradients in frequency domain (approximate gradients across mel bands)
            if pred_log_mel.shape[1] > 1:  # Need at least 2 mel bands
                pred_diff = pred_log_mel[:, 1:, :T] - pred_log_mel[:, :-1, :T]
                target_diff = target_log_mel[:, 1:, :T] - target_log_mel[:, :-1, :T]
                freq_loss = l1_loss_fn(pred_diff, target_diff) * 0.5
            else:
                freq_loss = torch.tensor(0.0, device=device)
            
            # Combine losses with appropriate weights
            # Higher weight on L1 ensures primary convergence
            loss = l1_loss * 0.6 + mse_loss * 0.3 + freq_loss * 0.1

            # -------------------- optimise -------------------------
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            # print current step statistics
            print(f"{text[:20]}: loss {loss.item():.4f} - {len(phonemes)} phonemes - epoch loss {epoch_loss:.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch:>3}/{epochs}: loss {avg_loss:.4f}")
        
        # Update scheduler based on validation loss
        scheduler.step(avg_loss)
        
        # Periodically normalize the voice tensor for better quality
        # This prevents drift and keeps voice characteristics appropriate
        if epoch % 25 == 0 and epoch > 0:
            with torch.no_grad():
                # Normalize timbre and style components separately
                current_timbre = base_voice[0, :128]
                current_style = base_voice[0, 128:]
                
                # Calculate current statistics
                current_timbre_mean = current_timbre.mean()
                current_timbre_std = current_timbre.std()
                current_style_mean = current_style.mean()
                current_style_std = current_style.std()
                
                # Adjust toward target statistics
                base_voice[0, :128] = ((current_timbre - current_timbre_mean) / current_timbre_std) * timbre_std + timbre_mean
                base_voice[0, 128:] = ((current_style - current_style_mean) / current_style_std) * style_std + style_mean
                
                print(f"Normalized voice at epoch {epoch}:")
                print(f"  Timbre: mean={base_voice[0, :128].mean().item():.4f}, std={base_voice[0, :128].std().item():.4f}")
                print(f"  Style: mean={base_voice[0, 128:].mean().item():.4f}, std={base_voice[0, 128:].std().item():.4f}")

        # Update the full voice tensor
        for i in range(MAX_PHONEME_LEN):
            voice_embed[i, 0, :] = base_voice.clone()
        
        # Save the full voice tensor every 10 epochs
        if epoch % 10 == 0:
            save_voice(base_voice, voice_embed, f"{out.strip(".pt")}.epoch{epoch}.pt")
        

        # Optionally decay LR
        if epoch % 50 == 0:
            for g in optim.param_groups:
                g["lr"] *= 0.5
                print(f"Reducing LR to {g['lr']}")
    
    # Save the final voice tensor
    save_voice(base_voice, voice_embed, out)

def save_voice(base_voice, voice_embed, out):
    # ---------------------------------------------------------------------
    # Save voice embedding
    # ---------------------------------------------------------------------
    save_path = Path(out)
    
    # Save both the full voice tensor and the compact version
    with torch.no_grad():
        # Update the full tensor one last time
        for i in range(MAX_PHONEME_LEN):
            voice_embed[i, 0, :] = base_voice.clone()
        
        # Save the full 3D tensor version
        torch.save({"voice": voice_embed.detach().cpu()}, save_path)
        print(f"Saved full voice tensor to {save_path.resolve()} – shape {tuple(voice_embed.shape)}")
        

# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train a Kokoro voice embedding")
    ap.add_argument("--data", required=True, help="Directory containing WAVs and prompts.txt")
    ap.add_argument("--epochs", type=int, default=200, help="Training epochs (default: 200)")
    ap.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate (default: 1e-2)")
    ap.add_argument("--batch", type=int, default=1, help="Batch size (default: 1)")
    ap.add_argument("--out", default="my_voice.pt", help="Output .pt filename")
    args = ap.parse_args()

    train(
        data_root=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        out=args.out,
    )
