# modules/segmenter.py
"""Audio segmentation module using Silero VAD"""

from pathlib import Path
from typing import List

import torch
import torchaudio

from utilities import get_module_logger


class AudioSegmenter:
    """Segment audio into utterances using Voice Activity Detection"""

    def __init__(
        self, output_dir: Path, min_silence_ms: int = 300, min_utt_ms: int = 700
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_module_logger(__name__)

        self.min_silence_ms = min_silence_ms
        self.min_utt_ms = min_utt_ms

        # Load VAD model
        self.logger.info("Loading Silero VAD model...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        (
            self.get_speech_timestamps,
            self.save_audio_vad,
            self.read_audio_vad,
            self.VADIterator,  # pylint: disable=invalid-name
            self.collect_chunks,
        ) = self.utils

    def process(self, input_path: Path) -> List[Path]:
        """Segment audio file into utterances"""
        try:
            self.logger.info("Segmenting: %s", input_path.name)

            # Load audio
            audio, sr = torchaudio.load(input_path)

            # Convert to mono if needed
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            # Resample to 16kHz for VAD
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio_16k = resampler(audio)
            else:
                audio_16k = audio

            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_16k.squeeze(),
                self.model,
                sampling_rate=16000,
                min_silence_duration_ms=self.min_silence_ms,
            )

            if not speech_timestamps:
                self.logger.warning("No speech detected in %s", input_path.name)
                return []

            # Extract segments
            segments = []
            for idx, timestamp in enumerate(speech_timestamps):
                start_16k = timestamp["start"]
                end_16k = timestamp["end"]

                # Check minimum duration
                duration_ms = (end_16k - start_16k) * 1000 / 16000
                if duration_ms < self.min_utt_ms:
                    continue

                # Map back to original sample rate
                start_orig = int(start_16k * sr / 16000)
                end_orig = int(end_16k * sr / 16000)

                # Extract segment
                segment_audio = audio[:, start_orig:end_orig]

                # Save segment
                output_path = (
                    self.output_dir / f"{input_path.stem}_segment_{idx:04d}.wav"
                )
                torchaudio.save(output_path, segment_audio, sr)
                segments.append(output_path)

                self.logger.debug("Saved segment %d: %dms", idx, duration_ms)

            self.logger.info("Created %d segments from %s", len(segments), input_path.name)
            return segments

        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error segmenting %s: %s", input_path, e)
            return []

    def process_batch(self, input_paths: List[Path]) -> List[Path]:
        """Process multiple files"""
        all_segments = []
        for path in input_paths:
            segments = self.process(path)
            all_segments.extend(segments)
        return all_segments
