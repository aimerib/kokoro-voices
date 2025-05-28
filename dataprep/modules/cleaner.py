# modules/cleaner.py
"""Dataset cleaning module"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torchaudio
import numpy as np
from tqdm import tqdm
import scipy.signal

from utilities import get_module_logger


class DatasetCleaner:
    """Clean and validate dataset quality"""

    def __init__(
        self,
        min_snr_db: float = 15.0,
        min_duration: float = 0.5,
        max_duration: float = 10.0,
        min_peak: float = 0.01,
    ):
        self.min_snr_db = min_snr_db
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_peak = min_peak
        self.logger = get_module_logger(__name__)

    def clean(self, dataset_dir: Path) -> Dict:
        """Clean dataset by removing low-quality samples"""
        self.logger.info("Cleaning dataset: %s", dataset_dir)

        quality_report = {
            "total_analyzed": 0,
            "total_kept": 0,
            "total_rejected": 0,
            "rejection_reasons": defaultdict(int),
            "quality_metrics": [],
            "average_quality": 0,
        }

        # Process each split
        for split in ["train", "validation", "test"]:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue

            self.logger.info("Processing %s split", split)

            metadata_file = split_dir / "metadata.jsonl"
            if not metadata_file.exists():
                continue

            # Load metadata
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = [json.loads(line) for line in f]

            # Analyze each file
            kept_metadata = []

            for item in tqdm(metadata, desc=f"Analyzing {split}"):
                audio_path = split_dir / item["file_name"]

                if not audio_path.exists():
                    self.logger.warning("File not found: %s", audio_path)
                    continue

                # Analyze quality
                is_good, metrics, issues = self._analyze_audio_quality(
                    audio_path)

                quality_report["total_analyzed"] += 1

                if is_good:
                    kept_metadata.append(item)
                    quality_report["total_kept"] += 1
                    quality_report["quality_metrics"].append(
                        metrics["overall_score"])
                else:
                    quality_report["total_rejected"] += 1
                    for issue in issues:
                        quality_report["rejection_reasons"][issue] += 1

                    # Optionally remove bad files
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            "Rejected %s: %s", audio_path.name, ", ".join(
                                issues)
                        )

            # Save cleaned metadata
            if kept_metadata:
                with open(metadata_file, "w", encoding="utf-8") as f:
                    for item in kept_metadata:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

                self.logger.info(
                    "Kept %s/%s files in %s",
                    len(kept_metadata),
                    len(metadata),
                    split,
                )

        # Calculate average quality
        if quality_report["quality_metrics"]:
            quality_report["average_quality"] = np.mean(
                quality_report["quality_metrics"]
            )

        # Generate summary
        self.logger.info("\nCleaning Summary:")
        self.logger.info("  Total analyzed: %s",
                         quality_report["total_analyzed"])
        self.logger.info(
            "  Kept: %s (%.1f%%)",
            quality_report["total_kept"],
            quality_report["total_kept"] /
            max(1, quality_report["total_analyzed"]) * 100,
        )
        self.logger.info("  Rejected: %s", quality_report["total_rejected"])
        self.logger.info("  Average quality: %.2f",
                         quality_report["average_quality"])

        if quality_report["rejection_reasons"]:
            self.logger.info("\nRejection reasons:")
            for reason, count in quality_report["rejection_reasons"].items():
                self.logger.info("  %s: %s", reason, count)

        return quality_report

    def _analyze_audio_quality(self, audio_path: Path) -> Tuple[bool, Dict, List[str]]:
        """Analyze audio quality and determine if it meets standards"""
        try:
            # Load and prepare audio
            audio, sr = torchaudio.load(audio_path)

            # Basic validation
            if audio.numel() == 0:
                return False, {"error": "empty_file"}, ["empty_file"]

            # Convert to mono if needed
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            if audio.dim() > 1:
                audio = audio.squeeze(0)

            # Resample to 16kHz for consistent analysis
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)
                sr = 16000

            # Convert to numpy
            audio_np = audio.numpy().astype('float32')
            if audio_np.ndim > 1:
                audio_np = audio_np.flatten()

            # Calculate metrics
            metrics = {}
            issues = []

            # Duration check
            duration = len(audio_np) / sr
            metrics["duration"] = duration

            if duration < self.min_duration:
                issues.append(f"too_short_{duration:.1f}s")
            elif duration > self.max_duration:
                issues.append(f"too_long_{duration:.1f}s")

            # Skip spectral analysis for very short files
            if duration < 0.05:  # 50ms minimum for analysis
                return False, metrics, issues + ["too_short_for_analysis"]

            # Peak amplitude check
            peak = np.max(np.abs(audio_np))
            metrics["peak"] = peak

            if peak < self.min_peak:
                issues.append("too_quiet")
            elif peak > 0.99:
                issues.append("clipping")

            # SNR estimation
            signal_power = np.mean(audio_np**2)

            # Estimate noise from quiet parts
            frame_size = max(64, int(0.025 * sr))  # Minimum 64 samples
            frame_powers = []

            if len(audio_np) > frame_size:
                for i in range(0, len(audio_np) - frame_size, frame_size):
                    frame = audio_np[i:i + frame_size]
                    frame_powers.append(np.mean(frame**2))

                if frame_powers:
                    noise_power = np.percentile(frame_powers, 10) + 1e-10
                    snr_db = 10 * np.log10(signal_power / noise_power)
                    metrics["snr_db"] = snr_db

                    if snr_db < self.min_snr_db:
                        issues.append(f"low_snr_{snr_db:.1f}dB")

            # DC offset check
            dc_offset = np.mean(audio_np)
            metrics["dc_offset"] = dc_offset

            if abs(dc_offset) > 0.05:
                issues.append("dc_offset")

            # Low frequency rumble check (only if long enough)
            if len(audio_np) >= 512:  # Minimum samples for Welch
                try:
                    f, psd = scipy.signal.welch(
                        audio_np,
                        sr,
                        nperseg=min(1024, len(audio_np) // 4)
                    )
                    low_freq_mask = f < 80
                    if np.any(low_freq_mask):
                        low_freq_ratio = np.sum(
                            psd[low_freq_mask]) / (np.sum(psd) + 1e-10)
                        metrics["low_freq_ratio"] = low_freq_ratio

                        if low_freq_ratio > 0.1:
                            issues.append("low_freq_rumble")
                except ValueError as e:
                    self.logger.warning(
                        "Spectral analysis failed for %s: %s", audio_path.name, str(e))
                    issues.append("spectral_analysis_failed")

            # Calculate overall quality score
            quality_score = 1.0
            if "snr_db" in metrics:
                quality_score *= min(1.0, metrics["snr_db"] / 20)
            if "low_freq_ratio" in metrics:
                quality_score *= 1.0 - metrics["low_freq_ratio"]
            quality_score *= (
                min(1.0, peak / 0.5) if peak < 0.5 else (1.0 - (peak - 0.5) / 0.5)
            )

            metrics["overall_score"] = quality_score

            # Determine if audio is good
            is_good = len(issues) == 0

            return is_good, metrics, issues

        except Exception as e:
            self.logger.error("Error analyzing %s: %s",
                              audio_path.name, str(e), exc_info=True)
            return False, {"error": str(e)}, ["analysis_error"]
