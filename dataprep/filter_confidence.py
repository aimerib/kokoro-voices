#!/usr/bin/env python3
"""
Standalone confidence score filter for Kokoro TTS datasets.
Filter dataset entries by confidence score threshold.
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.prompt import Confirm

from modules.cleaner import DatasetCleaner
from upload_dataset import DatasetUploader
from utilities import get_module_logger


def main():
    parser = argparse.ArgumentParser(
        description="Filter Kokoro dataset by confidence score",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive confidence analysis and filtering
  python filter_confidence.py ./my-voice-dataset

  # Filter with specific threshold
  python filter_confidence.py ./my-voice-dataset --min-confidence 0.85

  # Dry run - analyze without making changes
  python filter_confidence.py ./my-voice-dataset --dry-run
        """
    )

    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to prepared dataset directory"
    )

    parser.add_argument(
        "--min-confidence",
        type=float,
        help="Minimum confidence score threshold (0.0-1.0)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze confidence distribution without making changes"
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts"
    )

    args = parser.parse_args()

    console = Console()
    uploader = DatasetUploader()
    cleaner = DatasetCleaner()

    console.print("[bold blue]Kokoro Dataset Confidence Filter[/bold blue]\n")

    try:
        # Analyze current confidence distribution
        console.print("🔍 [cyan]Analyzing confidence scores...[/cyan]")
        confidence_stats = uploader.analyze_confidence_distribution(args.dataset_dir)
        uploader.display_confidence_analysis(confidence_stats)

        if not confidence_stats.get("overall"):
            console.print("❌ [red]No confidence scores found in dataset[/red]")
            sys.exit(1)

        if args.dry_run:
            console.print("\n📊 [yellow]Dry run complete - no changes made[/yellow]")
            return

        # Get threshold
        if args.min_confidence is not None:
            min_confidence = args.min_confidence
        else:
            # Interactive threshold selection
            console.print("\n🎯 [cyan]Select Confidence Threshold[/cyan]")
            overall = confidence_stats["overall"]
            console.print(f"Current dataset has {overall['below_0_8']} entries below 0.8 confidence")
            console.print(f"Current dataset has {overall['below_0_9']} entries below 0.9 confidence")
            
            while True:
                try:
                    min_confidence = float(input("\nEnter minimum confidence threshold (0.0-1.0): ").strip())
                    if 0.0 <= min_confidence <= 1.0:
                        break
                    else:
                        console.print("❌ [red]Please enter a value between 0.0 and 1.0[/red]")
                except ValueError:
                    console.print("❌ [red]Please enter a valid number[/red]")

        # Calculate what would be removed
        total = confidence_stats["overall"]["total"]
        would_remove = sum(1 for split_stats in confidence_stats["splits"].values() 
                          for confidence in [0] * split_stats["count"] 
                          if confidence < min_confidence)

        # This is a simplified calculation - let's get the actual numbers
        entries_below_threshold = 0
        for split_name, split_stats in confidence_stats["splits"].items():
            split_dir = args.dataset_dir / split_name
            metadata_file = split_dir / "metadata.jsonl"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        entry = json.loads(line)
                        if entry.get("confidence", 1.0) < min_confidence:
                            entries_below_threshold += 1

        console.print(f"\n📊 [yellow]Impact Analysis[/yellow]")
        console.print(f"  • Threshold: {min_confidence:.2f}")
        console.print(f"  • Entries to remove: {entries_below_threshold}")
        console.print(f"  • Entries to keep: {total - entries_below_threshold}")
        console.print(f"  • Removal percentage: {entries_below_threshold/total*100:.1f}%")

        # Confirm filtering
        if not args.yes:
            if not Confirm.ask(f"\nProceed with filtering (remove {entries_below_threshold} entries)?"):
                console.print("Filtering cancelled.")
                return

        # Apply filtering
        console.print(f"\n🔍 [cyan]Applying confidence filter (>= {min_confidence:.2f})...[/cyan]")
        confidence_report = cleaner.filter_by_confidence(args.dataset_dir, min_confidence)

        console.print("\n✅ [green]Confidence filtering complete![/green]")
        console.print(f"  • Analyzed: {confidence_report['total_analyzed']} entries")
        console.print(f"  • Kept: {confidence_report['total_kept']} entries")
        console.print(f"  • Removed: {confidence_report['total_rejected']} entries")
        console.print(f"  • New mean confidence: {confidence_report['confidence_stats']['mean_after']:.3f}")

    except KeyboardInterrupt:
        console.print("\n\n⚠️ Filtering cancelled by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ [red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main() 