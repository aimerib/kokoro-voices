# utils.py
"""Utility functions for Kokoro Pipeline"""

import logging
import importlib
from typing import List, Generator
import torch


def setup_logging(level: int = logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True  # Override any existing configuration
    )

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Make sure our loggers are at the right level
    for module in ["dataprep", "__main__"]:
        logger = logging.getLogger(module)
        logger.setLevel(level)

    # Suppress some noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_module_logger(module_name: str) -> logging.Logger:
    """Get a logger for a specific module"""
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    return logger


def get_device_info() -> str:
    """Get information about available compute device"""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def check_dependencies() -> List[str]:
    """Check if all required dependencies are installed"""
    required_packages = [
        "torch",
        "torchaudio",
        "numpy",
        "scipy",
        "librosa",
        "yt_dlp",
        "rich",
        "tqdm",
        "inflect",
        "huggingface_hub",
    ]

    missing = []
    for package in required_packages:
        try:
            importlib.import_module(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    # Check for optional but recommended packages
    optional_packages = {
        "demucs": "Speaker isolation",
        "df": "DeepFilter enhancement",
        "voicefixer": "VoiceFixer enhancement",
        "speechbrain": "MetricGAN+ enhancement",
        "whisper": "Whisper transcription",
        "phonemizer": "Phoneme analysis",
    }

    for package, feature in optional_packages.items():
        try:
            importlib.import_module(package)
        except ImportError:
            logging.warning(
                "Optional package %s not found - %s will be unavailable",
                package,
                feature,
            )

    return missing


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def estimate_processing_time(
    num_files: int, total_duration: float, mode: str = "balanced"
) -> float:
    """Estimate processing time based on mode and input"""
    # Rough estimates based on testing
    time_factors = {
        "fast": 0.5,  # 0.5x real-time
        "balanced": 1.0,  # 1x real-time
        "quality": 2.0,  # 2x real-time
    }

    factor = time_factors.get(mode, 1.0)

    # Add overhead for other steps
    overhead = num_files * 5  # 5 seconds per file for I/O, transcription, etc.

    return (total_duration * factor) + overhead


def chunk_generator(
    wav: torch.Tensor,
    sr: int,
    chunk_sec: float,
    overlap_sec: float = 0.0,
) -> Generator[torch.Tensor, None, None]:
    """Yield overlapping chunks of ``wav`` (C, N) in a memory-friendly way.

    The function slices *wav* into consecutive segments of ``chunk_sec`` seconds
    with an optional ``overlap_sec`` seconds overlap between neighbouring
    chunks.  It is designed to work with audio tensors shaped ``(C, N)`` or
    ``(N,)`` and avoids allocating additional buffers beyond the current chunk.

    Parameters
    ----------
    wav:
        Input waveform of shape ``(C, N)`` or ``(N,)`` (`torch.float32`).
    sr:
        Sample-rate of *wav* in Hertz.
    chunk_sec:
        Target duration per chunk **in seconds**.
    overlap_sec:
        Optional overlap between chunks **in seconds** (default: ``0.0``).

    Yields
    ------
    torch.Tensor
        A view of the current chunk (shape ``(C, M)``).
    """

    # Ensure 2-D ``(C, N)`` layout for slicing convenience
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    max_len = int(chunk_sec * sr)
    overlap = int(overlap_sec * sr)

    if max_len <= 0:
        raise ValueError("chunk_sec must be > 0")
    if overlap >= max_len:
        raise ValueError("overlap_sec must be < chunk_sec")

    start = 0
    total = wav.shape[-1]

    while start < total:
        end = min(start + max_len, total)
        # Yield a slice – no additional copy required
        yield wav[..., start:end]
        if end == total:
            break
        start = end - overlap
