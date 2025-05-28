# modules/uploader.py
"""HuggingFace dataset uploader module"""

from pathlib import Path
from typing import Dict, Optional, Set
from huggingface_hub import HfApi, upload_folder
import logging
import shutil
import tempfile
import json
from utilities import get_module_logger


class HuggingFaceUploader:
    """Upload datasets to HuggingFace Hub"""

    def __init__(self):
        self.logger = get_module_logger(__name__)
        self.api = HfApi()
        self.logger.setLevel(logging.DEBUG)

    def _get_referenced_files(self, dataset_dir: Path) -> Dict[str, Set[str]]:
        """Get set of audio files actually referenced in metadata for each split"""
        referenced_files = {}
        
        for split in ["train", "validation", "test"]:
            split_dir = dataset_dir / split
            metadata_file = split_dir / "metadata.jsonl"
            
            if not split_dir.exists() or not metadata_file.exists():
                continue
                
            files_in_metadata = set()
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        entry = json.loads(line)
                        if "file_name" in entry:
                            files_in_metadata.add(entry["file_name"])
                            
                referenced_files[split] = files_in_metadata
                self.logger.info("Found %d files referenced in %s metadata", 
                               len(files_in_metadata), split)
                               
            except Exception as e:
                self.logger.warning("Error reading metadata for %s: %s", split, e)
                referenced_files[split] = set()
                
        return referenced_files

    def _copy_split_selectively(self, source_split_dir: Path, dest_split_dir: Path, 
                               referenced_files: Set[str]):
        """Copy only the files that are referenced in metadata"""
        dest_split_dir.mkdir(parents=True, exist_ok=True)
        
        # Always copy metadata.jsonl
        metadata_file = source_split_dir / "metadata.jsonl"
        if metadata_file.exists():
            shutil.copy2(metadata_file, dest_split_dir / "metadata.jsonl")
            
        # Copy only referenced audio files
        copied_count = 0
        missing_files = []
        
        for file_name in referenced_files:
            source_file = source_split_dir / file_name
            if source_file.exists():
                shutil.copy2(source_file, dest_split_dir / file_name)
                copied_count += 1
            else:
                missing_files.append(file_name)
                
        self.logger.info("Copied %d audio files to %s", copied_count, dest_split_dir.name)
        
        if missing_files:
            self.logger.warning("Missing %d files referenced in metadata: %s", 
                              len(missing_files), missing_files[:5])  # Show first 5

    def upload(
        self,
        dataset_dir: Path,
        repo_id: str,
        private: bool = True,
        stats: Optional[Dict] = None,
        quality_report: Optional[Dict] = None,
    ):
        """Upload dataset to HuggingFace Hub"""
        self.logger.info("Uploading dataset to HuggingFace: %s", repo_id)

        # Check if repo exists, create if needed
        try:
            if not self.api.repo_exists(repo_id, repo_type="dataset"):
                self.logger.info("Creating repository: %s", repo_id)
                self.api.create_repo(
                    repo_id,
                    private=private,
                    repo_type="dataset",
                )
        except Exception as e:
            self.logger.error("Error checking/creating repo: %s", e)
            raise

        # Create dataset card
        self._create_dataset_card(dataset_dir, stats, quality_report)

        # Upload dataset
        try:
            self.logger.info("Uploading files...")
            
            # Get files actually referenced in metadata
            referenced_files = self._get_referenced_files(dataset_dir)
            
            # Create temporary directory for selective upload
            temp_dir = Path(tempfile.gettempdir()) / "kokoro-dataset"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True)
            
            # Copy only referenced files for each split
            for split in ["train", "validation", "test"]:
                source_split_dir = dataset_dir / split
                if source_split_dir.exists() and split in referenced_files:
                    dest_split_dir = temp_dir / split
                    self._copy_split_selectively(
                        source_split_dir, 
                        dest_split_dir, 
                        referenced_files[split]
                    )
            
            # Copy README
            readme_path = dataset_dir / "README.md"
            if readme_path.exists():
                shutil.copy2(readme_path, temp_dir / "README.md")

            # Verify what we're actually uploading
            total_audio_files = 0
            for split_dir in temp_dir.iterdir():
                if split_dir.is_dir() and split_dir.name in ["train", "validation", "test"]:
                    audio_files = list(split_dir.glob("*.wav"))
                    total_audio_files += len(audio_files)
                    self.logger.info("Uploading %d audio files from %s", 
                                   len(audio_files), split_dir.name)

            self.logger.info("Total audio files to upload: %d", total_audio_files)

            upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(temp_dir),
                commit_message="Add Kokoro voice dataset",
                ignore_patterns=["*.pyc", "__pycache__", ".DS_Store", "*.log", ".temp"],
            )
            self.logger.info(
                "✅ Upload complete: https://huggingface.co/datasets/%s",
                repo_id,
            )
            shutil.rmtree(temp_dir)

        except Exception as e:
            self.logger.error("Error uploading dataset: %s", e)
            raise

    def _create_dataset_card(
        self,
        dataset_dir: Path,
        stats: Optional[Dict] = None,
        quality_report: Optional[Dict] = None,
    ):
        """Create README.md dataset card"""

        # Count files and calculate size
        total_files = 0
        total_size = 0

        for split in ["train", "validation", "test"]:
            split_dir = dataset_dir / split
            if split_dir.exists():
                wav_files = list(split_dir.glob("*.wav"))
                total_files += len(wav_files)
                total_size += sum(f.stat().st_size for f in wav_files)

        size_mb = total_size / (1024 * 1024)

        # Determine size category
        if size_mb < 100:
            size_category = "n<100M"
        elif size_mb < 1000:
            size_category = "100M<n<1G"
        elif size_mb < 10000:
            size_category = "1G<n<10G"
        else:
            size_category = "n>10G"

        # Create README content
        readme_content = f"""---
dataset_info:
  features:
  - name: file_name
    dtype: string
  - name: text
    dtype: string
  - name: split
    dtype: string
  - name: duration
    dtype: float32
  - name: confidence
    dtype: float32
  configs:
  - config_name: default
    data_files:
    - split: train
      path: "train/*"
    - split: validation
      path: "validation/*"
    - split: test
      path: "test/*"
task_categories:
- text-to-speech
language:
- en
tags:
- kokoro-tts
- voice-cloning
- speech-synthesis
size_categories:
- {size_category}
license: cc-by-nc-nd-4.0
---

# Kokoro Voice Dataset

This dataset was created using the Kokoro Pipeline for voice cloning with Kokoro TTS.

## Dataset Statistics

- **Total files**: {total_files:,}
- **Total size**: {size_mb:.1f} MB
"""

        if stats:
            readme_content += f"""
- **Total segments**: {stats.get('total_segments', 0):,}
- **Total duration**: {stats.get('total_duration_hours', 0):.1f} hours
- **Average segment duration**: {stats.get('avg_duration', 0):.1f} seconds

### Split Distribution
"""
            if "splits" in stats:
                for split_name, split_stats in stats["splits"].items():
                    readme_content += (
                        f"- **{split_name}**: {split_stats.get('segments', 0)} segments "
                        f"({split_stats.get('duration_hours', 0):.1f} hours)\n"
                    )

        if quality_report:
            # Calculate percentage kept separately to keep line length manageable
            percentage_kept = quality_report.get('total_kept', 0) / max(1, quality_report.get('total_analyzed', 1)) * 100

            readme_content += f"""

## Quality Report

- **Files analyzed**: {quality_report.get('total_analyzed', 0):,}
- **Files kept**: {quality_report.get('total_kept', 0):,} ({percentage_kept:.1f}%)
- **Files rejected**: {quality_report.get('total_rejected', 0):,}
- **Average quality score**: {quality_report.get('average_quality', 0):.2f}/1.0
"""

            if quality_report.get("rejection_reasons"):
                readme_content += "\n### Rejection Reasons\n"
                for reason, count in quality_report["rejection_reasons"].items():
                    readme_content += f"- {reason.replace('_', ' ').title()}: {count}\n"

        readme_content += """

## Dataset Structure
```
dataset/
├── train/
│   ├── metadata.jsonl
│   └── segment_.wav
├── validation/
│   ├── metadata.jsonl
│   └── segment_.wav
└── test/
    ├── segment_*.wav
    └── metadata.jsonl
```

Each `metadata.jsonl` contains:
```json
{
  "file_name": "segment_00000.wav",
  "text": "Transcribed text here.",
  "split": "train",
  "duration": 2.5,
  "confidence": 0.95
}
```
Usage with Kokoro

This dataset is formatted for direct use with Kokoro TTS training.

License

This dataset is provided under CC-BY-NC-ND-4.0 license.


Generated by Kokoro Dataset Preparation Pipeline
"""
        # Save README
        readme_path = dataset_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)

        self.logger.info("Created dataset card: %s", readme_path)
