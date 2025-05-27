#!/usr/bin/env python3
"""
Kokoro Voice Dataset Pipeline - End-to-End Dataset Preparation

This tool provides a complete pipeline for preparing voice datasets for Kokoro TTS training:
1. Download audio from YouTube links or process local files
2. Isolate speakers using source separation
3. Enhance audio quality (denoise, dereverb, etc.)
4. Segment into utterances
5. Transcribe with Whisper ASR
6. Clean and validate dataset
7. Upload to HuggingFace (optional)

Usage:
    # From YouTube
    dataset-pipeline https://youtube.com/watch?v=... --output my-voice-dataset

    # From local files
    dataset-pipeline ./audio-files/ --output my-voice-dataset

    # With HuggingFace upload
    dataset-pipeline ./audio-files/ --output my-voice-dataset --upload-hf --hf-repo username/dataset

    # Resume from checkpoint
    dataset-pipeline --resume --output my-voice-dataset

    # Resume with force restart
    dataset-pipeline --resume --force-restart --output my-voice-dataset

Examples:
  # From YouTube video
  kokoro-pipeline "https://youtube.com/watch?v=..." --output my-voice

  # From local files
  kokoro-pipeline ./audio-files/ --output my-voice

  # Multiple sources
  kokoro-pipeline "https://youtube.com/..." ./more-audio/ --output my-voice

  # Fast mode with upload
  kokoro-pipeline ./audio/ --output my-voice --enhancement-mode fast --upload-hf --hf-repo username/my-voice

  # High quality mode
  kokoro-pipeline ./audio/ --output my-voice --enhancement-mode quality --whisper-model large

  # Resume from checkpoint
  kokoro-pipeline --resume --output my-voice

  # Resume with force restart
  kokoro-pipeline --resume --force-restart --output my-voice
"""

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any
import shutil
from dataclasses import dataclass, asdict
import yaml
import time
import os
import signal
import atexit


# Core modules
from modules import (
    AudioDownloader,
    SpeakerIsolator,
    AudioEnhancer,
    AudioSegmenter,
    AudioTranscriber,
    DatasetPreparer,
    DatasetCleaner,
    HuggingFaceUploader,
)
from utilities import (
    setup_logging,
    get_module_logger,
    check_dependencies,
    get_device_info,
)


@dataclass
class PipelineConfig:
    """Configuration for the entire pipeline"""

    input_paths: List[str]
    output_dir: Path
    temp_dir: Optional[Path] = None

    # Pipeline stages
    download_audio: bool = True
    isolate_speakers: bool = True
    enhance_audio: bool = True
    segment_audio: bool = True
    transcribe_audio: bool = True
    clean_dataset: bool = True

    # Model options
    whisper_model: str = "base"
    enhancement_mode: str = "balanced"
    disable_deepfilter: bool = False

    # Quality settings
    min_snr_db: float = 15.0
    min_duration: float = 0.5
    max_duration: float = 10.0
    target_sample_rate: int = 24000

    # HuggingFace upload
    upload_to_hf: bool = False
    hf_repo: Optional[str] = None
    hf_private: bool = True

    # Cleanup
    cleanup_temp: bool = True

    # Resume functionality
    resume: bool = False
    force_restart: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        """Create config from command line arguments"""
        return cls(
            input_paths=args.input,
            output_dir=Path(args.output),
            temp_dir=Path(args.temp_dir) if args.temp_dir else None,
            download_audio=not args.skip_download,
            isolate_speakers=not args.skip_isolation,
            enhance_audio=not args.skip_enhancement,
            segment_audio=not args.skip_segmentation,
            transcribe_audio=not args.skip_transcription,
            clean_dataset=not args.skip_cleaning,
            whisper_model=args.whisper_model,
            enhancement_mode=args.enhancement_mode,
            disable_deepfilter=args.no_deepfilter,
            min_snr_db=args.min_snr_db,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            target_sample_rate=args.target_sample_rate,
            upload_to_hf=args.upload_hf,
            hf_repo=args.hf_repo,
            hf_private=not args.hf_public,
            cleanup_temp=not args.keep_temp,
            resume=args.resume,
            force_restart=args.force_restart,
        )

    def save(self, path: Path):
        """Save config to YAML file"""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert the config to a dict and handle Path objects
        config_dict = asdict(self)

        # Convert Path objects to strings for YAML serialization
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            else:
                return obj

        config_dict = convert_paths(config_dict)

        with open(path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        """Load config from YAML file"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Convert string paths back to Path objects
        if "output_dir" in data:
            data["output_dir"] = Path(data["output_dir"])
        if "temp_dir" in data and data["temp_dir"]:
            data["temp_dir"] = Path(data["temp_dir"])

        return cls(**data)


class DatasetPipeline:
    """Main pipeline orchestrator with checkpoint/resume functionality"""

    # Constants for checkpoint files
    STATE_FILE = "pipeline_state.yaml"
    CONFIG_FILE = "pipeline_config.yaml"
    LOCK_FILE = ".pipeline_running"

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = get_module_logger(__name__)

        # Setup directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.temp_dir is None:
            self.config.temp_dir = self.config.output_dir / ".temp"
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint file paths
        self.state_file = self.config.output_dir / self.STATE_FILE
        self.config_file = self.config.output_dir / self.CONFIG_FILE
        self.lock_file = self.config.output_dir / self.LOCK_FILE

        # Handle resume/restart logic
        self._handle_resume_logic()

        # Initialize modules
        self._init_modules()

        # Pipeline state
        self.state = {
            "pipeline_version": "1.0",
            "start_time": time.time(),
            "last_checkpoint": None,
            "completed_stages": [],
            "downloaded_files": [],
            "isolated_files": [],
            "enhanced_files": [],
            "segments": [],
            "transcriptions": [],
            "dataset_stats": {},
            "quality_report": {},
            "stage_timings": {},
        }

        # Load existing state if resuming
        if self.config.resume:
            self._load_state()

        # Setup lock file and cleanup handlers
        self._setup_lock_file()

        # Save initial config
        self.config.save(self.config_file)

    def _handle_resume_logic(self):
        """Handle resume vs restart logic with safety checks"""
        if self.config.force_restart:
            if self.state_file.exists():
                self.logger.info(
                    "🔄 Force restart requested - removing existing checkpoint")
                self.state_file.unlink()
            if self.lock_file.exists():
                self.lock_file.unlink()
            return

        if self.config.resume:
            if not self.state_file.exists():
                self.logger.warning(
                    "⚠️ Resume requested but no checkpoint found - starting fresh")
                self.config.resume = False
            elif self.lock_file.exists():
                self.logger.warning(
                    "⚠️ Lock file exists - previous run may have crashed")
                response = input("Continue anyway? (y/N): ").lower().strip()
                if response != 'y':
                    self.logger.info("Aborted by user")
                    sys.exit(1)
                self.lock_file.unlink()
        else:
            # Auto-detect resume opportunity
            if self.state_file.exists() and not self.lock_file.exists():
                self.logger.info("📂 Found existing checkpoint")
                response = input(
                    "Resume from checkpoint? (Y/n): ").lower().strip()
                if response != 'n':
                    self.config.resume = True

    def _setup_lock_file(self):
        """Setup lock file and cleanup handlers"""
        def cleanup_lock():
            if self.lock_file.exists():
                try:
                    self.lock_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to remove lock file: {e}")
                    pass

        def signal_handler(signum, frame):
            self.logger.info("🛑 Received interrupt signal - cleaning up...")
            cleanup_lock()
            sys.exit(1)

        # Create lock file
        with open(self.lock_file, 'w') as f:
            f.write(f"PID: {os.getpid()}\nStarted: {time.ctime()}\n")

        # Register cleanup handlers
        atexit.register(cleanup_lock)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _save_state(self):
        """Save current pipeline state to disk"""
        try:
            self.state["last_checkpoint"] = time.time()

            # Convert Path objects to strings for YAML serialization
            def convert_paths(obj):
                if isinstance(obj, Path):
                    return str(obj)
                elif isinstance(obj, dict):
                    return {k: convert_paths(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_paths(item) for item in obj]
                else:
                    return obj

            state_to_save = convert_paths(self.state)

            with open(self.state_file, 'w') as f:
                yaml.safe_dump(state_to_save, f, default_flow_style=False)
            self.logger.debug("💾 Checkpoint saved")
        except Exception as e:
            self.logger.warning(f"Failed to save checkpoint: {e}")

    def _load_state(self):
        """Load pipeline state from disk"""
        try:
            with open(self.state_file) as f:
                saved_state = yaml.safe_load(f)
            if saved_state:
                # Convert string paths back to Path objects for file lists
                for file_list_key in ["downloaded_files", "isolated_files", "enhanced_files", "segments"]:
                    if file_list_key in saved_state and saved_state[file_list_key]:
                        saved_state[file_list_key] = [Path(f) if isinstance(f, str) else f
                                                      for f in saved_state[file_list_key]]

                # Handle transcriptions which have nested file paths
                if "transcriptions" in saved_state:
                    for transcription in saved_state["transcriptions"]:
                        if "file" in transcription and isinstance(transcription["file"], str):
                            transcription["file"] = Path(transcription["file"])

                self.state.update(saved_state)
                self.logger.info(
                    f"📂 Loaded checkpoint from {time.ctime(self.state.get('last_checkpoint', 0))}")
                self.logger.info(
                    f"✅ Completed stages: {', '.join(self.state.get('completed_stages', []))}")
                self._validate_existing_files()
        except Exception as e:
            self.logger.warning(f"Failed to load checkpoint: {e}")
            self.config.resume = False

    def _validate_existing_files(self):
        """Validate that files from previous run still exist and are valid"""
        for stage_name, file_list in [
            ("downloaded_files", self.state.get("downloaded_files", [])),
            ("isolated_files", self.state.get("isolated_files", [])),
            ("enhanced_files", self.state.get("enhanced_files", [])),
            ("segments", self.state.get("segments", [])),
        ]:
            if not file_list:
                continue

            valid_files = []
            removed_count = 0

            for file_path in file_list:
                path = Path(file_path)
                if path.exists() and path.stat().st_size > 0:
                    valid_files.append(file_path)
                else:
                    removed_count += 1

            if removed_count > 0:
                self.logger.warning(
                    f"🗑️ Removed {removed_count} missing/corrupted files from {stage_name}")
                self.state[stage_name] = valid_files

    def _is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage was already completed"""
        return stage_name in self.state.get("completed_stages", [])

    def _mark_stage_completed(self, stage_name: str):
        """Mark a stage as completed and save checkpoint"""
        if "completed_stages" not in self.state:
            self.state["completed_stages"] = []
        if stage_name not in self.state["completed_stages"]:
            self.state["completed_stages"].append(stage_name)
        self._save_state()

    def _stage_wrapper(self, stage_name: str, stage_func, *args, **kwargs):
        """Wrapper for pipeline stages with timing and checkpointing"""
        if self._is_stage_completed(stage_name):
            self.logger.info(f"⏭️ Skipping {stage_name} (already completed)")
            return None

        start_time = time.time()
        self.logger.info(f"🚀 Starting {stage_name}...")

        try:
            result = stage_func(*args, **kwargs)
            duration = time.time() - start_time
            self.state["stage_timings"][stage_name] = duration
            self._mark_stage_completed(stage_name)
            self.logger.info(f"✅ Completed {stage_name} in {duration:.1f}s")
            return result
        except Exception as e:
            self.logger.error(f"❌ Failed {stage_name}: {e}")
            raise

    def _init_modules(self):
        """Initialize pipeline modules"""
        self.downloader = AudioDownloader(
            output_dir=self.config.temp_dir / "downloads")
        self.isolator = SpeakerIsolator(
            output_dir=self.config.temp_dir / "isolated")
        self.enhancer = AudioEnhancer(
            output_dir=self.config.temp_dir / "enhanced",
            mode=self.config.enhancement_mode,
            enable_deepfilter=not self.config.disable_deepfilter,
        )
        self.segmenter = AudioSegmenter(
            output_dir=self.config.temp_dir / "segments")
        self.transcriber = AudioTranscriber(
            model_name=self.config.whisper_model)
        self.preparer = DatasetPreparer(
            output_dir=self.config.output_dir, target_sr=self.config.target_sample_rate
        )
        self.cleaner = DatasetCleaner(
            min_snr_db=self.config.min_snr_db,
            min_duration=self.config.min_duration,
            max_duration=self.config.max_duration,
        )
        self.uploader = HuggingFaceUploader()

    def run(self) -> Dict[str, Any]:
        """Run the complete pipeline with checkpoint/resume support"""
        try:
            self.logger.info("🚀 Starting Kokoro Dataset Pipeline")
            if self.config.resume:
                self.logger.info("📂 Resuming from checkpoint")
            self._log_config()

            # Step 1: Download/collect audio files
            if self.config.download_audio:
                result = self._stage_wrapper(
                    "download_audio", self._download_audio)
                if result is not None:
                    self.state["downloaded_files"] = result
            else:
                result = self._stage_wrapper(
                    "collect_local_files", self._collect_local_files)
                if result is not None:
                    self.state["downloaded_files"] = result

            # Step 2: Isolate speakers
            if self.config.isolate_speakers:
                result = self._stage_wrapper(
                    "isolate_speakers", self._isolate_speakers)
                if result is not None:
                    self.state["isolated_files"] = result
            else:
                if not self._is_stage_completed("isolate_speakers"):
                    self.state["isolated_files"] = self.state["downloaded_files"]
                    self._mark_stage_completed("isolate_speakers")

            # Step 3: Enhance audio
            if self.config.enhance_audio:
                result = self._stage_wrapper(
                    "enhance_audio", self._enhance_audio)
                if result is not None:
                    self.state["enhanced_files"] = result
            else:
                if not self._is_stage_completed("enhance_audio"):
                    self.state["enhanced_files"] = self.state["isolated_files"]
                    self._mark_stage_completed("enhance_audio")

            # Step 4: Segment audio
            if self.config.segment_audio:
                result = self._stage_wrapper(
                    "segment_audio", self._segment_audio)
                if result is not None:
                    self.state["segments"] = result
            else:
                if not self._is_stage_completed("segment_audio"):
                    self.state["segments"] = self.state["enhanced_files"]
                    self._mark_stage_completed("segment_audio")

            # Step 5: Transcribe
            if self.config.transcribe_audio:
                result = self._stage_wrapper(
                    "transcribe_audio", self._transcribe_audio)
                if result is not None:
                    self.state["transcriptions"] = result

            # Step 6: Prepare dataset
            result = self._stage_wrapper(
                "prepare_dataset", self._prepare_dataset)
            if result is not None:
                self.state["dataset_stats"] = result

            # Step 7: Clean dataset
            if self.config.clean_dataset:
                result = self._stage_wrapper(
                    "clean_dataset", self._clean_dataset)
                if result is not None:
                    self.state["quality_report"] = result

            # Optional: Upload to HuggingFace
            if self.config.upload_to_hf:
                self._stage_wrapper("upload_to_hf", self._upload_to_hf)

            # Cleanup
            if self.config.cleanup_temp:
                self.logger.info("🗑️ Cleaning up temporary files...")
                if self.config.temp_dir.exists():
                    shutil.rmtree(self.config.temp_dir)

            # Final cleanup
            self._cleanup_completion()

            total_time = time.time() - self.state["start_time"]
            self.logger.info(
                f"✅ Pipeline completed successfully in {total_time:.1f}s!")
            self._print_stage_summary()

            return self.state

        except Exception as e:
            self.logger.error("❌ Pipeline failed: %s", e)
            self.logger.info(
                "💾 State saved - use --resume to continue from checkpoint")
            raise

    def _cleanup_completion(self):
        """Clean up files after successful completion"""
        try:
            # Remove lock file
            if self.lock_file.exists():
                self.lock_file.unlink()

            # Optionally remove state file after successful completion
            # (Keep it for now in case user wants to inspect)
            # if self.state_file.exists():
            #     self.state_file.unlink()

            self.logger.debug("🧹 Cleanup completed")
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")

    def _print_stage_summary(self):
        """Print a summary of stage timings"""
        if not self.state.get("stage_timings"):
            return

        self.logger.info("\n📊 Stage Summary:")
        total_time = sum(self.state["stage_timings"].values())
        for stage, duration in self.state["stage_timings"].items():
            percentage = (duration / total_time) * 100 if total_time > 0 else 0
            self.logger.info(f"  {stage}: {duration:.1f}s ({percentage:.1f}%)")

    def _log_config(self):
        """Log pipeline configuration"""
        self.logger.info("Configuration:")
        self.logger.info("  Input: %s", self.config.input_paths)
        self.logger.info("  Output: %s", self.config.output_dir)
        self.logger.info("  Whisper model: %s", self.config.whisper_model)
        self.logger.info("  Enhancement mode: %s",
                         self.config.enhancement_mode)
        self.logger.info("  Device: %s", get_device_info())

    def _download_audio(self) -> List[Path]:
        """Download audio from YouTube links with per-file checkpointing"""
        downloaded = self.state.get("downloaded_files", [])

        for input_path in self.config.input_paths:
            if input_path.startswith(("http://", "https://", "www.")):
                # Check if already downloaded
                existing = [f for f in downloaded if str(
                    f).endswith(input_path.split('/')[-1])]
                if existing:
                    self.logger.info("⏭️ Already downloaded: %s", input_path)
                    continue

                self.logger.info("Downloading: %s", input_path)
                try:
                    result = self.downloader.download(input_path)
                    if result:
                        downloaded.append(result)
                        self.state["downloaded_files"] = downloaded
                        self._save_state()  # Save after each file
                except Exception as e:
                    self.logger.warning(
                        f"Failed to download {input_path}: {e}")
            else:
                # Local file
                local_path = Path(input_path)
                if local_path not in downloaded:
                    downloaded.append(local_path)
                    self.state["downloaded_files"] = downloaded
                    self._save_state()
        return downloaded

    def _collect_local_files(self) -> List[Path]:
        """Collect local audio files with checkpointing"""
        files = self.state.get("downloaded_files", [])

        for input_path in self.config.input_paths:
            path = Path(input_path)
            if path.is_dir():
                for pattern in ["*.wav", "*.mp3", "*.m4a", "*.flac"]:
                    for file_path in path.glob(pattern):
                        if file_path not in files:
                            files.append(file_path)
                            self.state["downloaded_files"] = files
                            self._save_state()
            elif path.is_file() and path not in files:
                files.append(path)
                self.state["downloaded_files"] = files
                self._save_state()
        return files

    def _isolate_speakers(self) -> List[Path]:
        """Isolate speakers from audio with per-file checkpointing"""
        isolated = self.state.get("isolated_files", [])
        processed_files = {Path(f).stem for f in isolated}

        for audio_file in self.state["downloaded_files"]:
            # Skip if already processed
            if Path(audio_file).stem in processed_files:
                self.logger.info("⏭️ Already isolated: %s",
                                 Path(audio_file).name)
                continue

            self.logger.info("Isolating speakers: %s", Path(audio_file).name)
            try:
                result = self.isolator.process(audio_file)
                isolated.append(result)
                self.state["isolated_files"] = isolated
                self._save_state()  # Save after each file
            except Exception as e:
                self.logger.warning(f"Failed to isolate {audio_file}: {e}")
        return isolated

    def _enhance_audio(self) -> List[Path]:
        """Enhance audio quality with per-file checkpointing"""
        enhanced = self.state.get("enhanced_files", [])
        processed_files = {Path(f).stem for f in enhanced}

        for audio_file in self.state["isolated_files"]:
            # Skip if already processed
            if Path(audio_file).stem in processed_files:
                self.logger.info("⏭️ Already enhanced: %s",
                                 Path(audio_file).name)
                continue

            self.logger.info("Enhancing: %s", Path(audio_file).name)
            try:
                result = self.enhancer.process(audio_file)
                enhanced.append(result)
                self.state["enhanced_files"] = enhanced
                self._save_state()  # Save after each file
            except Exception as e:
                self.logger.warning(f"Failed to enhance {audio_file}: {e}")
        return enhanced

    def _segment_audio(self) -> List[Path]:
        """Segment audio into utterances with per-file checkpointing"""
        segments = self.state.get("segments", [])
        processed_files = {Path(f).stem.split('_')[0] for f in segments}

        for audio_file in self.state["enhanced_files"]:
            # Skip if already processed (check base filename)
            if Path(audio_file).stem in processed_files:
                self.logger.info("⏭️ Already segmented: %s",
                                 Path(audio_file).name)
                continue

            self.logger.info("Segmenting: %s", Path(audio_file).name)
            try:
                results = self.segmenter.process(audio_file)
                segments.extend(results)
                self.state["segments"] = segments
                self._save_state()  # Save after each file
            except Exception as e:
                self.logger.warning(f"Failed to segment {audio_file}: {e}")
        return segments

    def _transcribe_audio(self) -> List[Dict]:
        """Transcribe audio segments with per-file checkpointing"""
        transcriptions = self.state.get("transcriptions", [])
        processed_files = {t["file"] if isinstance(t["file"], str) else str(t["file"])
                           for t in transcriptions}

        for segment in self.state["segments"]:
            # Skip if already processed
            segment_str = str(segment)
            if segment_str in processed_files:
                self.logger.info("⏭️ Already transcribed: %s",
                                 Path(segment).name)
                continue

            self.logger.info("Transcribing: %s", Path(segment).name)
            try:
                result = self.transcriber.transcribe(segment)
                transcription = {
                    "file": segment_str,
                    "text": result["text"],
                    "confidence": result.get("confidence", 1.0),
                }
                transcriptions.append(transcription)
                self.state["transcriptions"] = transcriptions
                self._save_state()  # Save after each file
            except Exception as e:
                self.logger.warning(f"Failed to transcribe {segment}: {e}")
        return transcriptions

    def _prepare_dataset(self) -> Dict:
        """Prepare final dataset structure"""
        return self.preparer.prepare(
            self.state["transcriptions"]
            if self.config.transcribe_audio
            else self.state["segments"]
        )

    def _clean_dataset(self) -> Dict:
        """Clean and validate dataset"""
        return self.cleaner.clean(self.config.output_dir)

    def _upload_to_hf(self):
        """Upload dataset to HuggingFace"""
        self.uploader.upload(
            dataset_dir=self.config.output_dir,
            repo_id=self.config.hf_repo,
            private=self.config.hf_private,
            stats=self.state["dataset_stats"],
            quality_report=self.state["quality_report"],
        )


def create_cli():
    """Create command line interface"""
    parser = argparse.ArgumentParser(
        description="Kokoro Voice Dataset Pipeline - End-to-End Dataset Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From YouTube video
  kokoro-pipeline "https://youtube.com/watch?v=..." --output my-voice

  # From local files
  kokoro-pipeline ./audio-files/ --output my-voice

  # Multiple sources
  kokoro-pipeline "https://youtube.com/..." ./more-audio/ --output my-voice

  # Fast mode with upload
  kokoro-pipeline ./audio/ --output my-voice --enhancement-mode fast --upload-hf --hf-repo username/my-voice

  # High quality mode
  kokoro-pipeline ./audio/ --output my-voice --enhancement-mode quality --whisper-model large

  # Resume from checkpoint
  kokoro-pipeline --resume --output my-voice

  # Resume with force restart
  kokoro-pipeline --resume --force-restart --output my-voice
        """,
    )

    # Input/Output
    parser.add_argument(
        "input",
        nargs="+",
        help="Input sources (YouTube URLs, audio files, or directories)",
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output dataset directory"
    )
    parser.add_argument(
        "--temp-dir", help="Temporary directory (default: OUTPUT/.temp)"
    )

    # Pipeline stages
    stages = parser.add_argument_group("Pipeline Stages")
    stages.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading (use local files)",
    )
    stages.add_argument(
        "--skip-isolation", action="store_true", help="Skip speaker isolation"
    )
    stages.add_argument(
        "--skip-enhancement", action="store_true", help="Skip audio enhancement"
    )
    stages.add_argument(
        "--skip-segmentation", action="store_true", help="Skip segmentation"
    )
    stages.add_argument(
        "--skip-transcription", action="store_true", help="Skip transcription"
    )
    stages.add_argument(
        "--skip-cleaning", action="store_true", help="Skip dataset cleaning"
    )

    # Model options
    models = parser.add_argument_group("Model Options")
    models.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        default="base",
        help="Whisper model size (default: base)",
    )
    models.add_argument(
        "--enhancement-mode",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Audio enhancement mode (default: balanced)",
    )
    models.add_argument(
        "--no-deepfilter",
        action="store_true",
        help="Disable DeepFilter even when selected mode would use it",
    )

    # Quality settings
    quality = parser.add_argument_group("Quality Settings")
    quality.add_argument(
        "--min-snr-db", type=float, default=15.0, help="Minimum SNR in dB"
    )
    quality.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum segment duration (seconds)",
    )
    quality.add_argument(
        "--max-duration",
        type=float,
        default=10.0,
        help="Maximum segment duration (seconds)",
    )
    quality.add_argument(
        "--target-sample-rate",
        type=int,
        default=24000,
        help="Target sample rate (default: 24000)",
    )

    # HuggingFace upload
    hf = parser.add_argument_group("HuggingFace Upload")
    hf.add_argument(
        "--upload-hf", action="store_true", help="Upload to HuggingFace Hub"
    )
    hf.add_argument(
        "--hf-repo", help="HuggingFace repo (e.g., username/dataset-name)"
    )
    hf.add_argument(
        "--hf-public", action="store_true", help="Make repo public (default: private)"
    )

    # Other options
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary files")
    parser.add_argument("--config", help="Load configuration from YAML file")
    parser.add_argument(
        "--save-config", help="Save configuration to YAML file")
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Increase verbosity"
    )
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimal output")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--force-restart", action="store_true",
                        help="Force restart from scratch")

    return parser


def main():
    """Main entry point"""
    parser = create_cli()
    args = parser.parse_args()

    # Setup logging
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose >= 3:
        log_level = logging.DEBUG
    elif args.verbose >= 2:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    setup_logging(log_level)

    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Please install required packages:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

    # Load or create config
    if args.config:
        config = PipelineConfig.load(Path(args.config))
    else:
        config = PipelineConfig.from_args(args)

    # Save config if requested
    if args.save_config:
        config.save(Path(args.save_config))
        print(f"Configuration saved to: {args.save_config}")

    # Validate config
    if config.upload_to_hf and not config.hf_repo:
        parser.error("--upload-hf requires --hf-repo")

    # Run pipeline
    try:
        pipeline = DatasetPipeline(config)
        results = pipeline.run()

        # Print summary
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Dataset location: {config.output_dir}")
        print(
            f"📊 Total segments: {results['dataset_stats'].get('total_segments', 0)}")
        print(
            f"⏱️  Total duration: {results['dataset_stats'].get('total_duration_hours', 0):.1f} hours"
        )
        if results.get("quality_report"):
            print(
                f"✨ Quality score: {results['quality_report'].get('average_quality', 0):.1%}"
            )
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        print(f"\n❌ Pipeline failed: {e}")
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
