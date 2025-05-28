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
from modules.cleaner import DatasetCleaner
from utilities import get_module_logger


class DatasetUploader:
    """Standalone dataset uploader"""

    def __init__(self):
        self.logger = get_module_logger(__name__)
        self.console = Console()
        self.uploader = HuggingFaceUploader()
        self.cleaner = DatasetCleaner()

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
                            missing_fields = [
                                field for field in required_fields if field not in sample_entry]
                            if missing_fields:
                                errors.append(
                                    f"Missing required fields in {split_dir}/metadata.jsonl: {missing_fields}")

                            # Check if audio files exist
                            missing_audio = []
                            for entry in entries[:5]:  # Check first 5
                                audio_path = split_path / entry["file_name"]
                                if not audio_path.exists():
                                    missing_audio.append(entry["file_name"])

                            if missing_audio:
                                warnings.append(
                                    f"Some audio files missing in {split_dir}: {missing_audio}")

                    except Exception as e:
                        errors.append(
                            f"Error reading {split_dir}/metadata.jsonl: {e}")

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
        total_entries = sum(stats.get(f"{split}_entries", 0)
                            for split in found_dirs)
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
                stats_table.add_row(
                    "Total entries", str(stats["total_entries"]))

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
            config["repo_id"] = Prompt.ask(
                "Enter HuggingFace repository ID (username/dataset-name)")

        # Privacy
        if args.private is not None:
            config["private"] = args.private
        else:
            config["private"] = Confirm.ask(
                "Make repository private?", default=True)

        return config

    def upload_dataset(self, dataset_dir: Path, repo_id: str, private: bool = True):
        """Upload dataset to HuggingFace"""
        try:
            self.console.print(
                "\n📤 [cyan]Uploading dataset to HuggingFace...[/cyan]")
            self.console.print(f"Repository: {repo_id}")
            self.console.print(
                f"Privacy: {'Private' if private else 'Public'}")

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

            self.console.print(
                "\n✅ [green]Dataset uploaded successfully![/green]")
            self.console.print(
                f"View at: https://huggingface.co/datasets/{repo_id}")

        except Exception as e:
            self.console.print(f"\n❌ [red]Upload failed: {e}[/red]")
            raise

    def analyze_confidence_distribution(self, dataset_dir: Path) -> dict:
        """Analyze confidence score distribution in the dataset"""
        confidence_scores = []
        stats = {"splits": {}}

        for split in ["train", "validation", "test"]:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue

            metadata_file = split_dir / "metadata.jsonl"
            if not metadata_file.exists():
                continue

            split_confidences = []
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    confidence = entry.get("confidence", 1.0)
                    confidence_scores.append(confidence)
                    split_confidences.append(confidence)

            if split_confidences:
                stats["splits"][split] = {
                    "count": len(split_confidences),
                    "mean": sum(split_confidences) / len(split_confidences),
                    "min": min(split_confidences),
                    "max": max(split_confidences),
                    "below_0_8": sum(1 for c in split_confidences if c < 0.8),
                    "below_0_9": sum(1 for c in split_confidences if c < 0.9),
                }

        if confidence_scores:
            stats["overall"] = {
                "total": len(confidence_scores),
                "mean": sum(confidence_scores) / len(confidence_scores),
                "min": min(confidence_scores),
                "max": max(confidence_scores),
                "below_0_8": sum(1 for c in confidence_scores if c < 0.8),
                "below_0_9": sum(1 for c in confidence_scores if c < 0.9),
            }

        return stats

    def check_for_orphaned_files(self, dataset_dir: Path) -> dict:
        """Check for audio files that exist but aren't referenced in metadata"""
        orphan_report = {"splits": {}, "total_orphaned": 0, "total_referenced": 0}

        for split in ["train", "validation", "test"]:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue

            # Get all audio files in directory
            audio_files = set(f.name for f in split_dir.glob("*.wav"))
            
            # Get files referenced in metadata
            referenced_files = set()
            metadata_file = split_dir / "metadata.jsonl"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            entry = json.loads(line)
                            if "file_name" in entry:
                                referenced_files.add(entry["file_name"])
                except Exception as e:
                    self.logger.warning("Error reading %s metadata: %s", split, e)

            # Find orphaned files
            orphaned_files = audio_files - referenced_files
            missing_files = referenced_files - audio_files

            orphan_report["splits"][split] = {
                "total_audio_files": len(audio_files),
                "referenced_files": len(referenced_files),
                "orphaned_files": len(orphaned_files),
                "missing_files": len(missing_files),
                "orphaned_list": list(orphaned_files)[:10],  # First 10 for display
                "missing_list": list(missing_files)[:10]     # First 10 for display
            }

            orphan_report["total_orphaned"] += len(orphaned_files)
            orphan_report["total_referenced"] += len(referenced_files)

        return orphan_report

    def display_confidence_analysis(self, stats: dict):
        """Display confidence score analysis"""
        if not stats.get("overall"):
            self.console.print("❌ [red]No confidence data found in dataset[/red]")
            return

        self.console.print("\n📊 [cyan]Confidence Score Analysis[/cyan]")
        
        # Overall stats table
        overall_table = Table(title="Overall Statistics")
        overall_table.add_column("Metric", style="cyan")
        overall_table.add_column("Value", style="magenta")
        
        overall = stats["overall"]
        overall_table.add_row("Total entries", str(overall["total"]))
        overall_table.add_row("Mean confidence", f"{overall['mean']:.3f}")
        overall_table.add_row("Min confidence", f"{overall['min']:.3f}")
        overall_table.add_row("Max confidence", f"{overall['max']:.3f}")
        overall_table.add_row("Entries < 0.8", f"{overall['below_0_8']} ({overall['below_0_8']/overall['total']*100:.1f}%)")
        overall_table.add_row("Entries < 0.9", f"{overall['below_0_9']} ({overall['below_0_9']/overall['total']*100:.1f}%)")
        
        self.console.print(overall_table)

        # Per-split stats
        if stats.get("splits"):
            splits_table = Table(title="Per-Split Statistics")
            splits_table.add_column("Split", style="cyan")
            splits_table.add_column("Count", style="magenta")
            splits_table.add_column("Mean", style="magenta")
            splits_table.add_column("< 0.8", style="red")
            splits_table.add_column("< 0.9", style="yellow")

            for split_name, split_stats in stats["splits"].items():
                splits_table.add_row(
                    split_name,
                    str(split_stats["count"]),
                    f"{split_stats['mean']:.3f}",
                    f"{split_stats['below_0_8']} ({split_stats['below_0_8']/split_stats['count']*100:.1f}%)",
                    f"{split_stats['below_0_9']} ({split_stats['below_0_9']/split_stats['count']*100:.1f}%)"
                )

            self.console.print(splits_table)

    def display_orphan_analysis(self, orphan_report: dict):
        """Display orphaned files analysis"""
        if orphan_report["total_orphaned"] == 0:
            self.console.print("✅ [green]No orphaned files found - all audio files are referenced in metadata[/green]")
            return

        self.console.print(f"\n⚠️ [yellow]Found {orphan_report['total_orphaned']} orphaned audio files[/yellow]")
        self.console.print("[dim]These files exist on disk but aren't referenced in metadata and won't be uploaded.[/dim]")

        # Create table for each split
        for split_name, split_data in orphan_report["splits"].items():
            if split_data["orphaned_files"] > 0 or split_data["missing_files"] > 0:
                self.console.print(f"\n[cyan]{split_name.title()} Split:[/cyan]")
                
                split_table = Table(show_header=False, box=None)
                split_table.add_column("Type", style="cyan", width=20)
                split_table.add_column("Count", style="magenta")

                split_table.add_row("Total audio files", str(split_data["total_audio_files"]))
                split_table.add_row("Referenced in metadata", str(split_data["referenced_files"]))
                
                if split_data["orphaned_files"] > 0:
                    split_table.add_row("Orphaned files", f"[red]{split_data['orphaned_files']}[/red]")
                    
                if split_data["missing_files"] > 0:
                    split_table.add_row("Missing files", f"[red]{split_data['missing_files']}[/red]")

                self.console.print(split_table)

                # Show some example orphaned files
                if split_data["orphaned_list"]:
                    self.console.print(f"[dim]Example orphaned files: {', '.join(split_data['orphaned_list'][:5])}[/dim]")

        self.console.print(f"\n💡 [blue]Only {orphan_report['total_referenced']} files will be uploaded to HuggingFace[/blue]")

    def clean_dataset_interactive(self, dataset_dir: Path) -> bool:
        """Interactive dataset cleaning with confidence and quality filters"""
        self.console.print("\n🧹 [cyan]Dataset Cleaning Options[/cyan]")
        
        # Analyze current confidence distribution
        confidence_stats = self.analyze_confidence_distribution(dataset_dir)
        self.display_confidence_analysis(confidence_stats)

        if not confidence_stats.get("overall"):
            self.console.print("⚠️ [yellow]No confidence scores found, skipping confidence filtering[/yellow]")
        else:
            # Ask about confidence filtering
            if Confirm.ask("\nFilter by confidence score?"):
                min_confidence = float(Prompt.ask(
                    "Minimum confidence score (0.0-1.0)",
                    default="0.8"
                ))
                
                self.console.print(f"🔍 [cyan]Filtering entries with confidence < {min_confidence}...[/cyan]")
                confidence_report = self.cleaner.filter_by_confidence(dataset_dir, min_confidence)
                
                self.console.print(f"✅ Confidence filtering complete:")
                self.console.print(f"  • Kept: {confidence_report['total_kept']}/{confidence_report['total_analyzed']} entries")
                self.console.print(f"  • Removed: {confidence_report['total_rejected']} low-confidence entries")

        # Ask about quality cleaning
        if Confirm.ask("\nRun quality-based cleaning? (removes audio with poor SNR, clipping, etc.)"):
            self.console.print("🔍 [cyan]Running quality analysis and cleaning...[/cyan]")
            quality_report = self.cleaner.clean(dataset_dir)
            
            self.console.print(f"✅ Quality cleaning complete:")
            self.console.print(f"  • Kept: {quality_report['total_kept']}/{quality_report['total_analyzed']} files")
            self.console.print(f"  • Removed: {quality_report['total_rejected']} poor-quality files")
            
            if quality_report["rejection_reasons"]:
                self.console.print("  • Rejection reasons:")
                for reason, count in quality_report["rejection_reasons"].items():
                    self.console.print(f"    - {reason.replace('_', ' ').title()}: {count}")

        return True


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

  # Clean dataset before upload
  python upload_dataset.py ./my-voice-dataset --clean --min-confidence 0.8

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
        "--clean",
        action="store_true",
        help="Clean dataset before upload (confidence + quality filtering)"
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.8,
        help="Minimum confidence score for filtering (default: 0.8)"
    )

    parser.add_argument(
        "--quality-clean",
        action="store_true",
        help="Run quality-based cleaning (SNR, clipping, etc.)"
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
        # Optional cleaning step
        if args.clean:
            console.print("\n🧹 [cyan]Cleaning dataset before upload...[/cyan]")
            
            # Analyze confidence distribution
            confidence_stats = uploader.analyze_confidence_distribution(args.dataset_dir)
            uploader.display_confidence_analysis(confidence_stats)
            
            if confidence_stats.get("overall"):
                # Confidence filtering
                console.print(f"\n🔍 [cyan]Applying confidence filter (>= {args.min_confidence})...[/cyan]")
                confidence_report = uploader.cleaner.filter_by_confidence(
                    args.dataset_dir, args.min_confidence
                )
                console.print(f"✅ Kept {confidence_report['total_kept']}/{confidence_report['total_analyzed']} entries")
            
            # Quality cleaning
            if args.quality_clean:
                console.print("\n🔍 [cyan]Running quality-based cleaning...[/cyan]")
                quality_report = uploader.cleaner.clean(args.dataset_dir)
                console.print(f"✅ Kept {quality_report['total_kept']}/{quality_report['total_analyzed']} files")

        # Validate dataset
        if not args.skip_validation:
            console.print(
                f"\n🔍 [cyan]Validating dataset: {args.dataset_dir}[/cyan]")
            validation = uploader.validate_dataset(args.dataset_dir)
            uploader.display_validation_results(validation)

            # Check for orphaned files
            console.print("\n🔍 [cyan]Checking for orphaned files...[/cyan]")
            orphan_report = uploader.check_for_orphaned_files(args.dataset_dir)
            uploader.display_orphan_analysis(orphan_report)

            if not validation["valid"]:
                console.print(
                    "\n❌ [red]Cannot upload invalid dataset. Fix errors first or use --skip-validation[/red]")
                sys.exit(1)

            if validation["warnings"] and not args.yes:
                if not Confirm.ask("\nProceed despite warnings?"):
                    console.print("Upload cancelled.")
                    sys.exit(0)

            # Warn about orphaned files
            if orphan_report["total_orphaned"] > 0 and not args.yes:
                if not Confirm.ask(f"\nProceed with upload? ({orphan_report['total_orphaned']} orphaned files will NOT be uploaded)"):
                    console.print("Upload cancelled.")
                    sys.exit(0)

        # Get upload configuration
        config = uploader.get_upload_config(args)

        # Final confirmation
        if not args.yes:
            console.print("\nAbout to upload:")
            console.print(f"  Dataset: {args.dataset_dir}")
            console.print(f"  Repository: {config['repo_id']}")
            console.print(
                f"  Privacy: {'Private' if config['private'] else 'Public'}")

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
