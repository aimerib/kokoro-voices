# modules/isolator.py
"""Speaker isolation module using Demucs"""

import subprocess
from typing import Optional, List
from pathlib import Path
import os

from utilities import get_module_logger

import torch

class SpeakerIsolator:
    """Isolate speakers from audio using source separation"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_module_logger(__name__)

        # Detect device
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    def process(self, input_path: Path) -> Optional[Path]:
        """Separate vocals from audio file"""
        try:
            self.logger.info("Separating vocals: %s", input_path.name)
            num_processes = os.cpu_count() or 4

            # Run Demucs
            cmd = [
                "demucs",
                "-d",
                self.device,
                "-j",
                str(num_processes),  # number of jobs
                "--two-stems=vocals",
                "-o",
                str(self.output_dir),
                str(input_path),
            ]

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Find output file
            stem_name = input_path.stem
            vocals_path = self.output_dir / "htdemucs" / stem_name / "vocals.wav"

            if vocals_path.exists():
                self.logger.info("Separated vocals: %s", vocals_path)
                return vocals_path
            else:
                self.logger.error("Vocals file not found: %s", vocals_path)
                return None

        except subprocess.CalledProcessError as e:
            self.logger.error("Demucs failed: %s", e)
            self.logger.error("stdout: %s", e.stdout)
            self.logger.error("stderr: %s", e.stderr)
            return None
        except Exception as e:  # pylint: disable=broad-except
            self.logger.error("Error isolating speakers: %s", e)
            return None

    def process_batch(self, input_paths: List[Path]) -> List[Path]:
        """Process multiple files"""
        results = []
        for path in input_paths:
            result = self.process(path)
            if result:
                results.append(result)
        return results
