# modules/enhancer.py
"""Audio enhancement module combining DeepFilter, VoiceFixer, and MetricGAN+"""

from pathlib import Path

# from tempfile import NamedTemporaryFile
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
            "auto": {
                "use_adaptive_enhancement": True,
                "available_methods": [
                    "deepfilter", "resemble_enhance", "metricgan",
                    "spectral_subtraction", "pitch_correction"
                ],
                "chunk_duration": 15.0,
                "quality_threshold": 0.7,  # SNR threshold for method selection
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

            self.df_model, self.df_state, _ = init_df(
                default_model="DeepFilterNet3")

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
        if self.config.get("use_adaptive_enhancement"):
            # Auto mode - analyze and select methods
            quality_metrics = self._analyze_audio_quality(audio, sr)
            selected_methods = self._select_enhancement_methods(
                quality_metrics)
            for method in selected_methods:
                if method == "deepfilter":
                    audio = self._apply_deepfilter(audio, sr)
                elif method == "resemble_enhance":
                    audio = self._apply_resemble_enhance(audio, sr)
                elif method == "metricgan":
                    audio = self._apply_metricgan(audio, sr)
                elif method == "spectral_subtraction":
                    audio = self._apply_spectral_subtraction(audio, sr)
                elif method == "pitch_correction":
                    audio = self._apply_pitch_correction(audio, sr)
        else:
            # Manual mode - use configured methods
            if self.config["use_deepfilter"]:
                audio = self._apply_deepfilter(audio, sr)

            if self.config["use_resemble_enhance"]:
                audio = self._apply_resemble_enhance(audio, sr)

            if self.config["use_metricgan"]:
                audio = self._apply_metricgan(audio, sr)

        # Normalize and save
        audio = self._normalize_audio(audio)

        output_path = (
            self.output_dir
            / f"{input_path.parent.name.strip()}_enhanced_{input_path.name}"
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
            enhanced_chunk = enhance(
                self.df_model, self.df_state, chunk, pad=True)

            # Skip first ``overlap`` samples on all but first chunk to avoid duplication
            if i > 0 and overlap_samples > 0:
                enhanced_chunk = enhanced_chunk[:, overlap_samples:]

            enhanced_chunks.append(enhanced_chunk)

        return torch.cat(enhanced_chunks, dim=-1)

    def _apply_resemble_enhance(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply Resemble Enhance enhancement with chunking"""
        start_time = time.time()
        self.logger.info(
            "Starting Resemble Enhance processing on audio with %.2fs duration",
            audio.shape[-1] / sr,
        )

        from resemble_enhance.enhancer.inference import enhance
        from utilities import get_device_info

        device = get_device_info()
        device = "cpu" if device == "mps" else device

        processed = torch.zeros_like(audio)
        chunk_sec = self.config.get("voicefixer_chunk_duration", 10)
        overlap_sec = 1.0
        total_chunks = 0

        for i, chunk in enumerate(
            chunk_generator(audio, sr, chunk_sec=chunk_sec,
                            overlap_sec=overlap_sec)
        ):
            self.logger.info("Processing VoiceFixer chunk %d", i + 1)
            # Convert to integers for slicing
            start = int(i * (chunk_sec - overlap_sec) * sr)
            end = int(start + chunk.shape[-1])

            dwav = chunk.mean(0)
            hwav, _ = enhance(dwav, sr, device=device, nfe=64)

            processed[..., start:end] = hwav[None]
            total_chunks += 1

        self.logger.info(
            "Completed VoiceFixer processing in %.2fs (%d chunks)",
            time.time() - start_time,
            total_chunks,
        )
        return processed

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

    def _apply_spectral_subtraction(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply spectral subtraction for noise reduction"""
        self.logger.info("Applying spectral subtraction")

        # Convert to magnitude and phase
        stft = torch.stft(audio.squeeze(), n_fft=1024,
                          hop_length=256, return_complex=True)
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)

        # Estimate noise from first few frames (assuming initial silence/noise)
        noise_frames = min(10, magnitude.shape[1] // 4)
        noise_estimate = torch.mean(
            magnitude[:, :noise_frames], dim=1, keepdim=True)

        # Spectral subtraction with over-subtraction factor
        alpha = 2.0  # Over-subtraction factor
        beta = 0.01  # Spectral floor

        # Subtract noise estimate
        enhanced_magnitude = magnitude - alpha * noise_estimate

        # Apply spectral floor
        enhanced_magnitude = torch.maximum(
            enhanced_magnitude,
            beta * magnitude
        )

        # Reconstruct signal
        enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)
        enhanced_audio = torch.istft(enhanced_stft, n_fft=1024, hop_length=256)

        # Ensure same length as input
        if len(enhanced_audio) > len(audio.squeeze()):
            enhanced_audio = enhanced_audio[:len(audio.squeeze())]
        elif len(enhanced_audio) < len(audio.squeeze()):
            enhanced_audio = torch.nn.functional.pad(
                enhanced_audio,
                (0, len(audio.squeeze()) - len(enhanced_audio))
            )

        return enhanced_audio.unsqueeze(0) if audio.dim() > 1 else enhanced_audio

    def _apply_pitch_correction(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Apply light pitch correction and prosody smoothing"""
        self.logger.info("Applying pitch correction")

        # Convert to numpy for processing
        audio_np = audio.squeeze().cpu().numpy()

        # Simple pitch stabilization using moving average
        # This smooths out small pitch variations without major changes

        # Frame-based processing
        frame_size = int(0.025 * sr)  # 25ms frames
        hop_size = int(0.010 * sr)    # 10ms hop

        enhanced_frames = []

        for i in range(0, len(audio_np) - frame_size, hop_size):
            frame = audio_np[i:i + frame_size]

            # Simple pitch smoothing using autocorrelation
            # Find dominant period
            autocorr = torch.nn.functional.conv1d(
                torch.tensor(frame).unsqueeze(0).unsqueeze(0),
                torch.tensor(frame).flip(0).unsqueeze(0).unsqueeze(0),
                padding=frame_size-1
            ).squeeze()

            # Find peak (excluding DC component)
            autocorr = autocorr[frame_size//2:]
            if len(autocorr) > 20:
                peak_idx = torch.argmax(autocorr[20:]) + 20
                period = peak_idx.item()

                # Light pitch stabilization
                if 50 < period < 400:  # Reasonable voice pitch range
                    # Apply gentle smoothing
                    window = torch.hann_window(min(period, len(frame)))
                    if len(window) <= len(frame):
                        frame[:len(window)] *= window.numpy()

            enhanced_frames.append(frame)

        # Overlap-add reconstruction
        enhanced_audio = torch.zeros_like(torch.tensor(audio_np))
        for i, frame in enumerate(enhanced_frames):
            start_idx = i * hop_size
            end_idx = min(start_idx + len(frame), len(enhanced_audio))
            frame_end = end_idx - start_idx
            enhanced_audio[start_idx:end_idx] += torch.tensor(
                frame[:frame_end])

        # Normalize to prevent clipping
        max_val = torch.max(torch.abs(enhanced_audio))
        if max_val > 1.0:
            enhanced_audio = enhanced_audio / max_val

        return enhanced_audio.unsqueeze(0) if audio.dim() > 1 else enhanced_audio

    def _analyze_audio_quality(self, audio: torch.Tensor, sr: int) -> dict:
        """Analyze audio quality metrics to determine optimal enhancement methods"""

        # Calculate SNR (Signal-to-Noise Ratio)
        energy = torch.mean(audio ** 2)
        snr_db = 10 * torch.log10(energy + 1e-10)

        # Calculate spectral centroid (brightness measure)
        stft = torch.stft(audio.squeeze(), n_fft=512, return_complex=True)
        magnitude = torch.abs(stft)
        freqs = torch.linspace(0, sr/2, magnitude.shape[0])
        spectral_centroid = torch.sum(
            magnitude * freqs.unsqueeze(1), dim=0) / (torch.sum(magnitude, dim=0) + 1e-10)
        avg_spectral_centroid = torch.mean(spectral_centroid)

        # Calculate zero-crossing rate (roughness measure)
        diff = torch.diff(torch.sign(audio.squeeze()))
        zcr = torch.sum(diff != 0).float() / len(audio.squeeze())

        # Calculate RMS energy (loudness measure)
        rms_energy = torch.sqrt(torch.mean(audio ** 2))

        # Detect silence ratio
        silence_threshold = 0.01 * torch.max(torch.abs(audio))
        silence_ratio = torch.sum(
            torch.abs(audio) < silence_threshold).float() / len(audio.squeeze())

        quality_metrics = {
            "snr_db": float(snr_db),
            "spectral_centroid": float(avg_spectral_centroid),
            "zero_crossing_rate": float(zcr),
            "rms_energy": float(rms_energy),
            "silence_ratio": float(silence_ratio),
            "needs_noise_reduction": float(snr_db) < 15.0,
            "needs_spectral_enhancement": float(avg_spectral_centroid) < sr * 0.1,
            "needs_dynamics_processing": float(rms_energy) < 0.1 or float(silence_ratio) > 0.3,
        }

        self.logger.info(f"Audio quality analysis: SNR={snr_db:.1f}dB, "
                         f"Spectral centroid={avg_spectral_centroid:.0f}Hz, "
                         f"Silence ratio={silence_ratio:.2f}")

        return quality_metrics

    def _select_enhancement_methods(self, quality_metrics: dict) -> list:
        """Select optimal enhancement methods based on audio quality analysis"""
        selected_methods = []

        # Always start with basic processing
        if quality_metrics["needs_noise_reduction"]:
            if quality_metrics["snr_db"] < 5.0:
                selected_methods.append("deepfilter")  # Heavy noise reduction
                # Additional cleaning
                selected_methods.append("spectral_subtraction")
            elif quality_metrics["snr_db"] < 12.0:
                # Moderate noise reduction
                selected_methods.append("deepfilter")
            else:
                selected_methods.append("voice_isolation")  # Light cleaning

        # Spectral enhancement for dull audio
        if quality_metrics["needs_spectral_enhancement"]:
            selected_methods.append("resemble_enhance")

        # Dynamic processing for quiet or inconsistent audio
        if quality_metrics["needs_dynamics_processing"]:
            selected_methods.append("pitch_correction")
            if quality_metrics["rms_energy"] < 0.05:
                selected_methods.append("metricgan")  # For very quiet audio

        # If audio is already good quality, minimal processing
        if (not quality_metrics["needs_noise_reduction"] and
            not quality_metrics["needs_spectral_enhancement"] and
                not quality_metrics["needs_dynamics_processing"]):
            selected_methods.append("voice_isolation")  # Light touch-up only

        self.logger.info(f"Selected enhancement methods: {selected_methods}")
        return selected_methods

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
