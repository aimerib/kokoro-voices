# modules/enhancer.py
"""Audio enhancement module combining DeepFilter, VoiceFixer, and MetricGAN+"""

from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import torch
import torchaudio
from df import init_df
from df.enhance import enhance
# from voicefixer import VoiceFixer
from speechbrain.inference import SpectralMaskEnhancement

# Utilities
from utilities import chunk_generator, get_module_logger


class AudioEnhancer:
    """Multi-stage audio enhancement with configurable quality/speed tradeoffs"""

    def __init__(
        self,
        output_dir: Path,
        mode: str = "balanced",
        enable_deepfilter: bool | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.logger = get_module_logger(__name__)
        self.df_model = None
        self.df_state = None
        self.vf = None
        self.metricgan = None

        # Enhancement configurations
        self.configs = {
            "fast": {
                "use_deepfilter": False,
                "use_voicefixer": False,
                "use_metricgan": True,
                "chunk_duration": 30.0,
            },
            "balanced": {
                "use_deepfilter": True,
                "use_voicefixer": False,
                "use_metricgan": True,
                "chunk_duration": 20.0,
            },
            "quality": {
                "use_deepfilter": True,
                "use_voicefixer": True,
                "use_metricgan": False,
                "chunk_duration": 10.0,
            },
        }

        self.config = self.configs[mode]

        # Allow caller to override DeepFilter usage (add-on step)
        if enable_deepfilter is not None:
            self.config["use_deepfilter"] = bool(enable_deepfilter)

        self._models_loaded = False

    def _lazy_load_models(self):
        """Load models only when needed"""
        if self._models_loaded:
            return

        if self.config["use_deepfilter"]:
            self.logger.info("Loading DeepFilter model...")

            self.df_model, self.df_state, _ = init_df(default_model="DeepFilterNet3")

        if self.config["use_voicefixer"]:
            self.logger.info("Loading VoiceFixer model...")

            # self.vf = VoiceFixer()

        if self.config["use_metricgan"]:
            self.logger.info("Loading MetricGAN+ model...")

            self.metricgan = SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank", savedir="sb_metricgan"
            )

        self._models_loaded = True

    def process(self, input_path: Path) -> Path:
        """Enhance audio file"""
        self._lazy_load_models()

        # Load audio
        audio, sr = torchaudio.load(input_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Apply enhancement stages
        if self.config["use_deepfilter"]:
            audio = self._apply_deepfilter(audio, sr)

        if self.config["use_voicefixer"]:
            audio = self._apply_voicefixer(audio, sr)

        if self.config["use_metricgan"]:
            audio = self._apply_metricgan(audio, sr)

        # Normalize and save
        audio = self._normalize_audio(audio)

        output_path = (
            self.output_dir / f"{input_path.parent.name.strip()}_enhanced_{input_path.name}"
        )
        torchaudio.save(output_path, audio, sr)

        return output_path

    def _apply_deepfilter(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply DeepFilter denoising with chunking"""

        # Resample to 48kHz if needed (DeepFilter expects 48 kHz)
        if sr != 48000:
            audio = torchaudio.transforms.Resample(sr, 48000)(audio)
            sr = 48000

        # Fast path: shorter than one chunk
        if audio.shape[-1] <= int(self.config["chunk_duration"] * sr):
            return enhance(self.df_model, self.df_state, audio, pad=True)

        overlap_sec = 0.5
        overlap_samples = int(overlap_sec * sr)

        enhanced_chunks = []
        for i, chunk in enumerate(
            chunk_generator(
                audio,
                sr,
                chunk_sec=self.config["chunk_duration"],
                overlap_sec=overlap_sec,
            )
        ):
            enhanced_chunk = enhance(self.df_model, self.df_state, chunk, pad=True)

            # Skip first ``overlap`` samples on all but first chunk to avoid duplication
            if i > 0 and overlap_samples > 0:
                enhanced_chunk = enhanced_chunk[:, overlap_samples:]

            enhanced_chunks.append(enhanced_chunk)

        return torch.cat(enhanced_chunks, dim=-1)

    def _apply_voicefixer(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply VoiceFixer enhancement with chunking"""
        start_time = time.time()
        self.logger.info(
            "Starting VoiceFixer processing on audio with %.2fs duration",
            audio.shape[-1] / sr,
        )

        from resemble_enhance.enhancer.inference import enhance

        # dwav = audio.mean(0)
        # hwav, _ = enhance(dwav, sr, device="cpu", nfe=64)

        # if sr != 44100:
        #     audio = torchaudio.transforms.Resample(sr, 44100)(audio)
        #     sr = 44100

        processed = torch.zeros_like(audio)
        chunk_sec = self.config.get("voicefixer_chunk_duration", 10)
        overlap_sec = 1.0
        total_chunks = 0

        for i, chunk in enumerate(
            chunk_generator(audio, sr, chunk_sec=chunk_sec, overlap_sec=overlap_sec)
        ):
            self.logger.info("Processing VoiceFixer chunk %d", i + 1)
            # Convert to integers for slicing
            start = int(i * (chunk_sec - overlap_sec) * sr)
            end = int(start + chunk.shape[-1])

            dwav = chunk.mean(0)
            hwav, _ = enhance(dwav, sr, device="cpu", nfe=64)

            processed[..., start:end] = hwav[None]
            total_chunks += 1

        self.logger.info(
            "Completed VoiceFixer processing in %.2fs (%d chunks)",
            time.time() - start_time,
            total_chunks,
        )
        return processed


        #     with NamedTemporaryFile(suffix=".wav", delete=False) as f_in, \
        #          NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
        #         torchaudio.save(f_in.name, chunk, sample_rate=int(sr))
        #         self.vf.restore(f_in.name, f_out.name, cuda=torch.mps.is_available())
        #         enhanced, _sr = torchaudio.load(f_out.name)

        #     chunk_processed = enhanced.unsqueeze(0)

        #     if i > 0:
        #         crossfade_samples = int(overlap_sec * sr)
        #         prev_chunk = processed[..., start : start + crossfade_samples]
        #         blended = (
        #             torch.linspace(0, 1, crossfade_samples)
        #             * chunk_processed[..., :crossfade_samples]
        #             + torch.linspace(1, 0, crossfade_samples) * prev_chunk
        #         )
        #         processed[..., start : start + crossfade_samples] = blended
        #         processed[..., start + crossfade_samples : end] = chunk_processed[
        #             ..., crossfade_samples:
        #         ]
        #     else:
        #         processed[..., :end] = chunk_processed

        #     total_chunks += 1

        # self.logger.info(
        #     "Completed VoiceFixer processing in %.2fs (%d chunks)",
        #     time.time() - start_time,
        #     total_chunks,
        # )
        # return processed
        return hwav[None]

    def _apply_metricgan(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply MetricGAN+ enhancement"""
        # Resample to 16kHz
        if sr != 16000:
            self.logger.info("Resampling audio to 16kHz")
            audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
        else:
            audio_16k = audio

        # Process
        self.logger.info("Processing MetricGAN+")
        audio_16k = audio_16k.to(self.metricgan.device)
        lengths = torch.tensor([1.0], device=self.metricgan.device)
        enhanced_16k = self.metricgan.enhance_batch(audio_16k, lengths)
        self.logger.info("Completed MetricGAN+ processing")

        # Resample back
        if sr != 16000:
            self.logger.info("Resampling audio back to %dHz", sr)
            enhanced = torchaudio.transforms.Resample(16000, sr)(enhanced_16k)
        else:
            enhanced = enhanced_16k

        return enhanced.cpu()

    def _normalize_audio(
        self, audio: torch.Tensor, target_db: float = -3.0
    ) -> torch.Tensor:
        """Normalize audio to target peak level"""
        peak = audio.abs().max()
        if peak > 0:
            target_amp = 10 ** (target_db / 20)
            self.logger.info("Normalizing audio to %f dB", target_db)
            return audio * (target_amp / peak)
        return audio
