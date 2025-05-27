# modules/preparer.py
"""Dataset preparation module"""

import json
import random
from pathlib import Path
from typing import List, Dict, Union
import torch
import torchaudio
from tqdm import tqdm

from utilities import get_module_logger


class DatasetPreparer:
    """Prepare structured dataset for Kokoro training"""

    def __init__(
        self,
        output_dir: Path,
        target_sr: int = 24000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        self.output_dir = Path(output_dir)
        self.target_sr = target_sr
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.logger = get_module_logger(__name__)

        # Create split directories
        self.train_dir = self.output_dir / "train"
        self.val_dir = self.output_dir / "validation"
        self.test_dir = self.output_dir / "test"

        for split_dir in [self.train_dir, self.val_dir, self.test_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self, data: Union[List[Dict], List[Path]]) -> Dict:
        """Prepare dataset from transcribed segments or audio files"""
        if not data:
            raise ValueError("No data provided for dataset preparation")

        # Handle different input types
        if isinstance(data[0], dict):
            # Transcribed data
            return self._prepare_transcribed(data)
        else:
            # Just audio files (no transcription)
            return self._prepare_audio_only(data)

    def _prepare_transcribed(self, transcriptions: List[Dict]) -> Dict:
        """Prepare dataset from transcribed segments"""
        self.logger.info(
            "Preparing dataset from %s transcribed segments", len(transcriptions)
        )

        # Filter out failed transcriptions
        valid_data = [t for t in transcriptions if t.get("text") and not t.get("error")]

        if not valid_data:
            raise ValueError("No valid transcriptions found")

        # Shuffle and split
        random.shuffle(valid_data)

        total = len(valid_data)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)

        train_data = valid_data[:train_end]
        val_data = valid_data[train_end:val_end]
        test_data = valid_data[val_end:]

        # Process each split
        stats = {"total_segments": 0, "total_duration_seconds": 0, "splits": {}}

        for split_name, split_data, split_dir in [
            ("train", train_data, self.train_dir),
            ("validation", val_data, self.val_dir),
            ("test", test_data, self.test_dir),
        ]:
            split_stats = self._process_split(split_name, split_data, split_dir)
            stats["splits"][split_name] = split_stats
            stats["total_segments"] += split_stats["segments"]
            stats["total_duration_seconds"] += split_stats["duration_seconds"]

        stats["total_duration_hours"] = stats["total_duration_seconds"] / 3600
        stats["avg_duration"] = (
            stats["total_duration_seconds"] / stats["total_segments"]
            if stats["total_segments"] > 0
            else 0
        )

        self.logger.info(
            "Dataset prepared: %s segments, %.1f hours",
            stats["total_segments"],
            stats["total_duration_hours"],
        )

        return stats

    def _process_split(
        self, split_name: str, data: List[Dict], output_dir: Path
    ) -> Dict:
        """Process a single data split"""
        self.logger.info("Processing %s split: %s segments", split_name, len(data))

        metadata_path = output_dir / "metadata.jsonl"

        total_duration = 0
        processed = 0

        with open(metadata_path, "w", encoding="utf-8") as f:
            for idx, item in enumerate(tqdm(data, desc=f"Processing {split_name}")):
                try:
                    # Load audio
                    audio, sr = torchaudio.load(item["file"])

                    # Convert to mono
                    if audio.shape[0] > 1:
                        audio = audio.mean(dim=0, keepdim=True)

                    # Resample if needed
                    if sr != self.target_sr:
                        resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                        audio = resampler(audio)

                    # Normalize
                    audio = self._normalize_audio(audio)

                    # Save audio
                    filename = f"segment_{idx:05d}.wav"
                    output_path = output_dir / filename
                    torchaudio.save(output_path, audio, self.target_sr)

                    # Write metadata
                    metadata = {
                        "file_name": filename,
                        "text": item["text"],
                        "split": split_name,
                        "duration": float(audio.shape[-1] / self.target_sr),
                        "confidence": item.get("confidence", 1.0),
                    }
                    f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

                    total_duration += metadata["duration"]
                    processed += 1

                except Exception as e:  # pylint: disable=broad-except
                    self.logger.error(
                        "Error processing %s: %s", item.get("file", "unknown"), e
                    )

        return {
            "segments": processed,
            "duration_seconds": total_duration,
            "duration_hours": total_duration / 3600,
        }

    def _prepare_audio_only(self, audio_files: List[Path]) -> Dict:
        """Prepare dataset without transcriptions (for pre-transcribed data)"""
        self.logger.info(
            "Preparing dataset from %s audio files (no transcription)", len(audio_files)
        )

        # For audio-only, put everything in train split
        stats = {
            "total_segments": 0,
            "total_duration_seconds": 0,
            "splits": {"train": {"segments": 0, "duration_seconds": 0}},
        }

        metadata_path = self.train_dir / "metadata.jsonl"

        with open(metadata_path, "w", encoding="utf-8") as f:
            for idx, audio_file in enumerate(
                tqdm(audio_files, desc="Processing audio")
            ):
                try:
                    # Copy audio file
                    filename = f"segment_{idx:05d}.wav"
                    output_path = self.train_dir / filename

                    # Load, normalize, and save
                    audio, sr = torchaudio.load(audio_file)

                    if audio.shape[0] > 1:
                        audio = audio.mean(dim=0, keepdim=True)

                    if sr != self.target_sr:
                        resampler = torchaudio.transforms.Resample(sr, self.target_sr)
                        audio = resampler(audio)

                    audio = self._normalize_audio(audio)
                    torchaudio.save(output_path, audio, self.target_sr)

                    duration = float(audio.shape[-1] / self.target_sr)

                    # Simple metadata without text
                    metadata = {
                        "file_name": filename,
                        "text": "",  # Empty for now
                        "split": "train",
                        "duration": duration,
                    }
                    f.write(json.dumps(metadata, ensure_ascii=False) + "\n")

                    stats["total_segments"] += 1
                    stats["total_duration_seconds"] += duration

                except Exception as e:  # pylint: disable=broad-except
                    self.logger.error("Error processing %s: %s", audio_file, e)

        stats["total_duration_hours"] = stats["total_duration_seconds"] / 3600
        stats["splits"]["train"] = {
            "segments": stats["total_segments"],
            "duration_seconds": stats["total_duration_seconds"],
            "duration_hours": stats["total_duration_hours"],
        }

        return stats

    def _normalize_audio(
        self, audio: torch.Tensor, target_db: float = -3.0
    ) -> torch.Tensor:
        """Normalize audio to target peak level"""
        # Remove DC offset
        audio = audio - audio.mean()

        # Normalize to target peak
        peak = audio.abs().max()
        if peak > 0:
            target_amp = 10 ** (target_db / 20)
            audio = audio * (target_amp / peak)

        return audio
