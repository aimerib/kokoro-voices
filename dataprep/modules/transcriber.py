# modules/transcriber.py
"""Audio transcription module using Whisper"""

import unicodedata
from pathlib import Path
from typing import Dict, List
import re
import traceback

import torch
import torchaudio
import inflect
from whisper_mps import whisper as whisper_mps
import whisper

from utilities import get_module_logger

# Initialize inflect engine for number to word conversion
eng = inflect.engine()


class AudioTranscriber:
    """Transcribe audio using OpenAI Whisper"""

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self.logger = get_module_logger(__name__)
        self.model = None
        self.transcribe_fn = None

    def _lazy_load_model(self):
        """Load Whisper model on first use"""
        if self.model is not None:
            return

        self.logger.info("Loading Whisper %s model...", self.model_name)

        if torch.backends.mps.is_available():
            # Use MPS-optimized version if available
            self._load_mps_whisper()
        else:
            self._load_standard_whisper()

    def _load_standard_whisper(self):
        """Load standard Whisper"""
        self.model = whisper.load_model(self.model_name)
        self.transcribe_fn = lambda audio: self.model.transcribe(
            audio,
            word_timestamps=False,
            verbose=False,
            language="en",
        )

    def _load_mps_whisper(self):
        """Load Whisper with MPS support"""
        try:
            self.transcribe_fn = lambda audio: whisper_mps.transcribe(
                audio,
                model=self.model_name,
                without_timestamps=True,
                verbose=False,
                language="en",
        )
        except ImportError:
            self.logger.warning("whisper_mps not found, falling back to standard Whisper")
            self._load_standard_whisper()

    def normalize_text(self, text: str) -> str:
        """Normalize transcription text"""
        # Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Convert numbers to words
        text = re.sub(r"\d+", lambda m: eng.number_to_words(m.group(0)), text)

        # Remove special characters except basic punctuation
        text = re.sub(r"[^a-zA-Z0-9\.,\?!\' ]", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        return text

    def transcribe(self, audio_path: Path) -> Dict:
        """Transcribe a single audio file"""
        self._lazy_load_model()

        try:
            self.logger.info("Transcribing: %s", audio_path.name)

            # Load and prepare audio
            audio, sr = torchaudio.load(audio_path)

            # Convert to mono if needed
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0)

            # Resample to 16kHz for Whisper
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(audio)

            # Convert to numpy
            audio_np = audio.numpy()

            # Transcribe
            result = self.transcribe_fn(audio_np)

            # Normalize text
            normalized_text = self.normalize_text(result["text"])

            # Calculate confidence if available
            segments = result.get("segments", [])
            if segments:
                avg_logprob = sum(s.get("avg_logprob", 0) for s in segments) / len(
                    segments
                )
                confidence = min(
                    1.0, max(0.0, 1.0 + avg_logprob / 10)
                )  # Rough confidence estimate
            else:
                confidence = 1.0

            return {
                "text": normalized_text,
                "raw_text": result["text"],
                "confidence": confidence,
                "language": result.get("language", "en"),
                "segments": segments,
            }

        except Exception as e:  # pylint: disable=broad-except
            traceback.print_exc()
            self.logger.error("Error transcribing %s: %s", audio_path, e)
            return {"text": "", "error": str(e), "confidence": 0.0}

    def transcribe_batch(self, audio_paths: List[Path]) -> List[Dict]:
        """Transcribe multiple files"""
        results = []
        for path in audio_paths:
            result = self.transcribe(path)
            result["file"] = path
            results.append(result)
        return results
