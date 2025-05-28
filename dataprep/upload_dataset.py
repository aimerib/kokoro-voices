#!/usr/bin/env python3
"""
Standalone dataset uploader for Kokoro TTS prepared datasets.
Upload an already-prepared dataset to HuggingFace Hub.
"""

import argparse
import sys
from pathlib import Path
import json
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel

from modules.uploader import HuggingFaceUploader
from utilities import get_module_logger


class DatasetUploader:
    """Standalone dataset uploader"""
    
    def __init__(self):
        self.logger = get_module_logger(__name__)
        self.console = Console()
        self.uploader = HuggingFaceUploader()
    
    def validate_dataset(self, dataset_dir: Path) -> dict:
        """Validate that dataset is properly prepared"""
        errors = []
        warnings = []
        stats = {}
        
        # Check directory exists
        if not dataset_dir.exists():
            errors.append(f"Dataset directory does not exist: {dataset_dir}")
            return {"valid": False, "errors": errors, "warnings": warnings, "stats": stats}
        
        # Check required structure
        required_dirs = ["train", "validation", "test"]
        found_dirs = []
        
        for split_dir in required_dirs:
            split_path = dataset_dir / split_dir
            if split_path.exists():
                found_dirs.append(split_dir)
                
                # Check metadata file
                metadata_file = split_path / "metadata.jsonl"
                if not metadata_file.exists():
                    errors.append(f"Missing metadata.jsonl in {split_dir}")
                else:
                    # Count entries
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            entries = [json.loads(line) for line in f]
                            stats[f"{split_dir}_entries"] = len(entries)
                            
                            # Validate entries have required fields
                            sample_entry = entries[0] if entries else {}
                            required_fields = ["file_name", "text"]
                            missing_fields = [field for field in required_fields if field not in sample_entry]
                            if missing_fields:
                                errors.append(f"Missing required fields in {split_dir}/metadata.jsonl: {missing_fields}")
                            
                            # Check if audio files exist
                            missing_audio = []
                            for entry in entries[:5]:  # Check first 5
                                audio_path = split_path / entry["file_name"]
                                if not audio_path.exists():
                                    missing_audio.append(entry["file_name"])
                            
                            if missing_audio:
                                warnings.append(f"Some audio files missing in {split_dir}: {missing_audio}")
                                
                    except Exception as e:
                        errors.append(f"Error reading {split_dir}/metadata.jsonl: {e}")
        
        if not found_dirs:
            errors.append("No train/validation/test directories found")
        elif len(found_dirs) < len(required_dirs):
            missing = set(required_dirs) - set(found_dirs)
            warnings.append(f"Missing optional split directories: {missing}")
        
        # Check for README
        readme_path = dataset_dir / "README.md"
        if not readme_path.exists():
            warnings.append("No README.md found (will be generated)")
        
        # Calculate total entries
        total_entries = sum(stats.get(f"{split}_entries", 0) for split in found_dirs)
        stats["total_entries"] = total_entries
        stats["splits"] = found_dirs
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": stats
        }
    
    def display_validation_results(self, validation: dict):
        """Display validation results in a nice format"""
        
        if validation["valid"]:
            self.console.print("✅ [green]Dataset validation passed![/green]")
        else:
            self.console.print("❌ [red]Dataset validation failed![/red]")
        
        # Show stats
        if validation["stats"]:
            stats_table = Table(title="Dataset Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="magenta")
            
            stats = validation["stats"]
            for split in stats.get("splits", []):
                entries = stats.get(f"{split}_entries", 0)
                stats_table.add_row(f"{split.title()} entries", str(entries))
            
            if "total_entries" in stats:
                stats_table.add_row("Total entries", str(stats["total_entries"]))
            
            self.console.print(stats_table)
        
        # Show errors
        if validation["errors"]:
            self.console.print("\n[red]Errors:[/red]")
            for error in validation["errors"]:
                self.console.print(f"  • {error}")
        
        # Show warnings
        if validation["warnings"]:
            self.console.print("\n[yellow]Warnings:[/yellow]")
            for warning in validation["warnings"]:
                self.console.print(f"  • {warning}")
    
    def get_upload_config(self, args):
        """Get upload configuration interactively if needed"""
        config = {}
        
        # Repository ID
        if args.repo_id:
            config["repo_id"] = args.repo_id
        else:
            config["repo_id"] = Prompt.ask("Enter HuggingFace repository ID (username/dataset-name)")
        
        # Privacy
        if args.private is not None:
            config["private"] = args.private
        else:
            config["private"] = Confirm.ask("Make repository private?", default=True)
        
        return config
    
    def upload_dataset(self, dataset_dir: Path, repo_id: str, private: bool = True):
        """Upload dataset to HuggingFace"""
        try:
            self.console.print(f"\n📤 [cyan]Uploading dataset to HuggingFace...[/cyan]")
            self.console.print(f"Repository: {repo_id}")
            self.console.print(f"Privacy: {'Private' if private else 'Public'}")
            
            # Load any existing stats/reports
            stats = None
            quality_report = None
            
            # Try to load stats from common locations
            stats_file = dataset_dir / "dataset_stats.json"
            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                    self.console.print("✓ Found dataset stats")
                except Exception as e:
                    self.logger.warning("Failed to load dataset stats: %s", e)
            
            quality_file = dataset_dir / "quality_report.json"
            if quality_file.exists():
                try:
                    with open(quality_file, 'r') as f:
                        quality_report = json.load(f)
                    self.console.print("✓ Found quality report")
                except Exception as e:
                    self.logger.warning("Failed to load quality report: %s", e)
            
            # Upload
            self.uploader.upload(
                dataset_dir=dataset_dir,
                repo_id=repo_id,
                private=private,
                stats=stats,
                quality_report=quality_report
            )
            
            self.console.print(f"\n✅ [green]Dataset uploaded successfully![/green]")
            self.console.print(f"View at: https://huggingface.co/datasets/{repo_id}")
            
        except Exception as e:
            self.console.print(f"\n❌ [red]Upload failed: {e}[/red]")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload prepared Kokoro TTS dataset to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive upload
  python upload_dataset.py ./my-voice-dataset
  
  # Upload to specific repo
  python upload_dataset.py ./my-voice-dataset --repo-id username/my-voice
  
  # Public repository
  python upload_dataset.py ./my-voice-dataset --repo-id username/my-voice --public
  
  # Skip validation (not recommended)
  python upload_dataset.py ./my-voice-dataset --repo-id username/my-voice --skip-validation
        """
    )
    
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to prepared dataset directory"
    )
    
    parser.add_argument(
        "--repo-id",
        help="HuggingFace repository ID (username/dataset-name)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private (default if not specified)"
    )
    
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make repository public"
    )
    
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip dataset validation (not recommended)"
    )
    
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts"
    )
    
    args = parser.parse_args()
    
    # Resolve privacy setting
    if args.public:
        args.private = False
    elif not args.private:
        args.private = None  # Will prompt interactively
    
    uploader = DatasetUploader()
    console = Console()
    
    # Header
    console.print(Panel(
        "[bold blue]Kokoro TTS Dataset Uploader[/bold blue]\n"
        "Upload prepared datasets to HuggingFace Hub",
        title="🎤 Dataset Upload",
        border_style="blue"
    ))
    
    try:
        # Validate dataset
        if not args.skip_validation:
            console.print(f"\n🔍 [cyan]Validating dataset: {args.dataset_dir}[/cyan]")
            validation = uploader.validate_dataset(args.dataset_dir)
            uploader.display_validation_results(validation)
            
            if not validation["valid"]:
                console.print("\n❌ [red]Cannot upload invalid dataset. Fix errors first or use --skip-validation[/red]")
                sys.exit(1)
            
            if validation["warnings"] and not args.yes:
                if not Confirm.ask("\nProceed despite warnings?"):
                    console.print("Upload cancelled.")
                    sys.exit(0)
        
        # Get upload configuration
        config = uploader.get_upload_config(args)
        
        # Final confirmation
        if not args.yes:
            console.print(f"\nAbout to upload:")
            console.print(f"  Dataset: {args.dataset_dir}")
            console.print(f"  Repository: {config['repo_id']}")
            console.print(f"  Privacy: {'Private' if config['private'] else 'Public'}")
            
            if not Confirm.ask("\nProceed with upload?"):
                console.print("Upload cancelled.")
                sys.exit(0)
        
        # Upload
        uploader.upload_dataset(
            dataset_dir=args.dataset_dir,
            repo_id=config["repo_id"],
            private=config["private"]
        )
        
    except KeyboardInterrupt:
        console.print("\n\n⚠️ Upload cancelled by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ [red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
