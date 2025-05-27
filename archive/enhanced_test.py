#!/usr/bin/env python3
"""
Enhanced audio processing script using DeepFilter Python API with MPS support.
This version provides better device control and more robust multiprocessing.
"""

import subprocess
import pathlib
import multiprocessing as mp
from sympy import tensor
import torch
import torchaudio
from df import enhance, init_df
from df.enhance import load_audio, save_audio
import argparse
from typing import Optional
import traceback
from pathlib import Path
import math
import librosa
from tempfile import NamedTemporaryFile
import soundfile as sf

# Configuration
SRC_DIR = pathlib.Path("/Users/aimeri/Downloads/processed/unt")
STEM_DIR = pathlib.Path("stems")      # speech/music split
CLEAN_DIR = pathlib.Path("clean")     # denoised speech
SEGMENTS_DIR = Path("segments")       # segmented utterances

# Auto-detect best available device (used for informative printouts only)
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# ---------------------------------------------------------------------------
# Global enhancer handles (lazy-loaded once)
# ---------------------------------------------------------------------------
VF = None               # VoiceFixer instance (high quality)
SB_ENHANCER = None      # SpeechBrain MetricGAN instance (fast)
DF_MODEL = None         # DeepFilter model (optional)
DF_STATE = None

# ---------------------------  DeepFilter stage  ----------------------------

def apply_deepfilter(audio: torch.Tensor) -> torch.Tensor:
    """Run DeepFilterNet denoising on a mono tensor [1, N] (expects 48 kHz)."""
    global DF_MODEL, DF_STATE
    if DF_MODEL is None or DF_STATE is None:
        DF_MODEL, DF_STATE, _ = init_df(default_model="DeepFilterNet3")
    return enhance(DF_MODEL, DF_STATE, audio.squeeze(0), pad=True).unsqueeze(0)

# -------------------------  VoiceFixer (HQ)  ------------------------------

def apply_voicefixer(audio: torch.Tensor, sr: int) -> torch.Tensor:
    """High-quality dereverb/noise using VoiceFixer (file-based API)."""
    global VF
    if VF is None:
        from voicefixer import VoiceFixer
        VF = VoiceFixer()

    # Ensure 2-D [C, N] tensor and float32
    audio_write = audio.clone().detach()
    if audio_write.dim() == 1:
        audio_write = audio_write.unsqueeze(0)
    audio_write = audio_write.to(torch.float32).cpu()

    with NamedTemporaryFile(suffix=".wav", delete=False) as f_in, NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        torchaudio.save(f_in.name, audio_write, sample_rate=int(sr))
        VF.restore(f_in.name, f_out.name, cuda=torch.cuda.is_available())
        enhanced, _sr = torchaudio.load(f_out.name)
    return enhanced

# ----------------------  MetricGAN+ (fast)  -------------------------------

def load_metricgan():
    global SB_ENHANCER
    if SB_ENHANCER is None:
        from speechbrain.inference import SpectralMaskEnhancement
        SB_ENHANCER = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir="sb_metricgan",
        )
    return SB_ENHANCER

def apply_metricgan(audio: torch.Tensor) -> torch.Tensor:
    """Apply SpeechBrain MetricGAN+ enhancer. Expects mono audio."""
    try:
        enhancer = load_metricgan()

        # Ensure mono [1, N]
        if audio.dim() == 2 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # 1. Resample to 16 kHz and move to the model’s device
        wav16 = torchaudio.transforms.Resample(48_000, 16_000)(audio)
        wav16 = wav16.to(enhancer.device)          # (1, T)

        # 2. Batch-wise lengths tensor *on the same device*
        lengths = torch.tensor([wav16.shape[-1]], device=enhancer.device)

        # 3. Run MetricGAN+
        cleaned16 = enhancer.enhance_batch(wav16, lengths)

        # Back to 48 kHz
        # resampler_back = torchaudio.transforms.Resample(sr_target, sr_orig)
        return cleaned16
    except Exception as e:
        traceback.print_exc()
        return audio

# ---------------------------------------------------------------------------
# Core processing per file
# ---------------------------------------------------------------------------

def process_stem(
    stem_path: Path,
    use_deepfilter: bool,
    enhancer_type: str,
    stretch_factor: float,
) -> str:
    try:
        wave, audio_info = load_audio(stem_path)

        # Stage A: optional DeepFilter (after separation)
        if use_deepfilter:
            wave = apply_deepfilter(wave)

        # Stage B: main enhancer (VoiceFixer for HQ, MetricGAN for fast)
        if enhancer_type == "voicefixer":
            wave = apply_voicefixer(wave, audio_info.sample_rate)
        else:
            # audio (Tensor): Audio tensor of shape [C, T], if channels_first=True (default).
            # metricgan expects (1, T)
            if wave.dim() == 1:
                wave = wave.unsqueeze(0)
            wave = apply_metricgan(wave)

        # Stage C: optional time-stretch
        if stretch_factor != 1.0:
            wave = apply_time_stretch(wave, audio_info.sample_rate, stretch_factor)

        # Save result
        CLEAN_DIR.mkdir(parents=True, exist_ok=True)
        out_path = CLEAN_DIR / stem_path.name
        save_audio(out_path, wave, audio_info.sample_rate)
        return f"✓ Enhanced {stem_path.name}"

    except Exception as e:
        traceback.print_exc()
        return f"✗ Failed {stem_path.name}: {e}"

def separate_audio(path: pathlib.Path) -> None:
    """Separate vocals from music using Demucs"""
    try:
        subprocess.run([
            "demucs", "-d", "mps", "-j", "10", 
            "--two-stems=vocals", "-o", str(STEM_DIR), str(path)
        ], check=True)
        print(f"✓ Separated: {path.name}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error separating {path}: {e}")

def apply_time_stretch(wave: torch.Tensor, sr: int, factor: float) -> torch.Tensor:
    # wave: (1, N) or (N,)
    y = wave.squeeze().cpu().numpy()
    y_stretch = librosa.effects.time_stretch(y, rate=factor)
    return torch.from_numpy(y_stretch)

def denoise_audio_sequential(stem_files: list, use_deepfilter: bool, enhancer_type: str, stretch_factor: float = 1.0) -> None:
    """Sequential processing - shares one model instance"""
    for stem in stem_files:
        print(process_stem(stem, use_deepfilter, enhancer_type, stretch_factor))

def denoise_audio_parallel(stem_files: list, num_workers: int = None, use_deepfilter: bool = False, enhancer_type: str = "voicefixer", stretch_factor: float = 1.0) -> None:
    """Parallel processing - each worker loads its own model"""
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(stem_files))
    
    print(f"Processing {len(stem_files)} files with {num_workers} workers...")
    
    # Prepare arguments for workers
    args = [(stem, use_deepfilter, enhancer_type, stretch_factor) for stem in stem_files]
    
    with mp.Pool(num_workers) as pool:
        results = pool.starmap(process_stem, args)
    
    # Print all results
    for result in results:
        print(result)

# ---------------------------------------------------------------------------
# New CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Audio processing with DeepFilter")
    parser.add_argument("--fast", action="store_true", help="Use fast MetricGAN enhancer (default HQ VoiceFixer)")
    parser.add_argument("--deepfilter", action="store_true", help="Add DeepFilterNet stage after separation (experimental)")
    parser.add_argument("--stretch-factor", type=float, default=1.0,
                    help="Time-stretch factor (<1 = slower, >1 = faster)")
    parser.add_argument("--segment", action="store_true", help="Run VAD segmentation after denoising")
    parser.add_argument("--min-silence-ms", type=int, default=300, help="Minimum silence gap (ms) to split")
    parser.add_argument("--min-utt-ms", type=int, default=700, help="Minimum utterance length to keep (ms)")
    
    args = parser.parse_args()
    
    # Determine enhancer choice
    enhancer_type = "metricgan" if args.fast else "voicefixer"
    
    # Ensure directories exist
    STEM_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    
    # # Step 1: Source separation (if requested)
    # print("Starting source separation...")
    # src_files = list(SRC_DIR.glob("*.mp3"))
    # if not src_files:
    #     print(f"No MP3 files found in {SRC_DIR}")
    # else:
    #     for src_file in src_files:
    #         separate_audio(src_file)
    
    # Step 2: Denoising
    stem_files = list(STEM_DIR.rglob("vocals.wav"))
    if not stem_files:
        print(f"No vocal stems found in {STEM_DIR}")
        return
    
    print(f"Found {len(stem_files)} vocal files to denoise")
    
    denoise_audio_sequential(stem_files, args.deepfilter, enhancer_type, stretch_factor=args.stretch_factor)
    
    # After denoising complete, optionally segment
    if args.segment:
        print("\n▶ Segmenting utterances with Silero-VAD…")
        vad_model, get_speech_timestamps = load_vad_model()
        total_utts = 0
        for wav_path in CLEAN_DIR.glob("*.wav"):
            exported = segment_audio_file(
                wav_path,
                vad_model,
                get_speech_timestamps,
                SEGMENTS_DIR,
                min_silence_ms=args.min_silence_ms,
                min_utt_ms=args.min_utt_ms,
            )
            total_utts += exported
        print(f"✓ Segmentation complete → {total_utts} utterances saved to {SEGMENTS_DIR}")
    
    print(f"✓ Processing complete! Cleaned files saved to {CLEAN_DIR}")

def load_vad_model():
    """Load Silero VAD model via torch.hub (cached after first call)."""
    print("Loading Silero VAD model (torch.hub: snakers4/silero-vad)…")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    (
        get_speech_timestamps,
        save_audio_vad,
        read_audio_vad,
        VADIterator,
        collect_chunks,
    ) = utils
    return model, get_speech_timestamps

def segment_audio_file(
    wav_path: Path,
    vad_model,
    get_speech_timestamps,
    out_dir: Path,
    min_silence_ms: int = 300,
    min_utt_ms: int = 700,
    target_sr: int = 48000,
):
    """Split `wav_path` into speech utterances using VAD and export to `out_dir`."""
    try:
        audio, sr = torchaudio.load(wav_path)
        # mono for VAD
        audio_mono = audio.mean(dim=0, keepdim=True)

        # Resample to 16 kHz for VAD
        if sr != 16000:
            resampler16 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio16 = resampler16(audio_mono)
        else:
            audio16 = audio_mono

        # Get speech timestamps (list of dicts with "start" and "end" in samples @16 kHz)
        speech_ts = get_speech_timestamps(
            audio16.squeeze(), vad_model, sampling_rate=16000, min_silence_duration_ms=min_silence_ms
        )

        if not speech_ts:
            print(f"No speech detected in {wav_path.name}")
            return 0

        # Prepare output directory
        out_dir.mkdir(parents=True, exist_ok=True)
        exported = 0
        for idx, ts in enumerate(speech_ts):
            start16, end16 = ts["start"], ts["end"]
            duration_ms = (end16 - start16) / 16
            if duration_ms < min_utt_ms:
                continue  # skip very short utterances

            # Map indices back to original sample rate
            start_orig = int(start16 * sr / 16000)
            end_orig = int(end16 * sr / 16000)
            clip_audio = audio[:, start_orig:end_orig]

            # Optional: ensure output SR (keep original 48k)
            if sr != target_sr:
                resampler_out = torchaudio.transforms.Resample(sr, target_sr)
                clip_audio = resampler_out(clip_audio)
                out_sr = target_sr
            else:
                out_sr = sr

            out_path = out_dir / f"{wav_path.stem}_utt_{exported:03d}.wav"
            torchaudio.save(str(out_path), clip_audio, out_sr)
            exported += 1
        return exported
    except Exception as e:
        print(f"✗ Error segmenting {wav_path.name}: {e}")
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    mp.freeze_support()
    main()
