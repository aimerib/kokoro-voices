#!/usr/bin/env python3
"""
Cleanup orphaned audio files in Kokoro TTS datasets.
Find and optionally remove audio files that exist but aren't referenced in metadata.
"""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.prompt import Confirm

from upload_dataset import DatasetUploader
from utilities import get_module_logger


def main():
    parser = argparse.ArgumentParser(
        description="Find and cleanup orphaned audio files in Kokoro dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze orphaned files without making changes
  python cleanup_orphans.py ./my-voice-dataset

  # Remove orphaned files
  python cleanup_orphans.py ./my-voice-dataset --remove

  # Remove orphaned files without confirmation
  python cleanup_orphans.py ./my-voice-dataset --remove --yes
        """
    )

    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to prepared dataset directory"
    )

    parser.add_argument(
        "--remove",
        action="store_true",
        help="Remove orphaned files (default: analyze only)"
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompts"
    )

    args = parser.parse_args()

    console = Console()
    uploader = DatasetUploader()

    console.print("[bold blue]Kokoro Dataset Orphan Cleanup[/bold blue]\n")

    try:
        # Analyze orphaned files
        console.print("🔍 [cyan]Analyzing orphaned files...[/cyan]")
        orphan_report = uploader.check_for_orphaned_files(args.dataset_dir)
        uploader.display_orphan_analysis(orphan_report)

        if orphan_report["total_orphaned"] == 0:
            console.print("\n✅ [green]No cleanup needed - no orphaned files found![/green]")
            return

        if not args.remove:
            console.print("\n💡 [blue]Run with --remove to delete orphaned files[/blue]")
            return

        # Confirm removal
        if not args.yes:
            console.print(f"\n⚠️ [yellow]About to delete {orphan_report['total_orphaned']} orphaned files[/yellow]")
            if not Confirm.ask("Are you sure you want to proceed?"):
                console.print("Cleanup cancelled.")
                return

        # Remove orphaned files
        total_removed = 0
        total_size_freed = 0

        for split_name, split_data in orphan_report["splits"].items():
            if split_data["orphaned_files"] == 0:
                continue

            split_dir = args.dataset_dir / split_name
            console.print(f"\n🧹 [cyan]Cleaning {split_name} split...[/cyan]")

            # Get full list of orphaned files (not just the first 10)
            audio_files = set(f.name for f in split_dir.glob("*.wav"))
            referenced_files = set()
            
            metadata_file = split_dir / "metadata.jsonl"
            if metadata_file.exists():
                import json
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        entry = json.loads(line)
                        if "file_name" in entry:
                            referenced_files.add(entry["file_name"])

            orphaned_files = audio_files - referenced_files

            # Remove orphaned files
            removed_count = 0
            for orphan_file in orphaned_files:
                file_path = split_dir / orphan_file
                if file_path.exists():
                    try:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        removed_count += 1
                        total_size_freed += file_size
                    except Exception as e:
                        console.print(f"❌ [red]Failed to remove {orphan_file}: {e}[/red]")

            console.print(f"  • Removed {removed_count} orphaned files from {split_name}")
            total_removed += removed_count

        # Summary
        size_mb = total_size_freed / (1024 * 1024)
        console.print(f"\n✅ [green]Cleanup complete![/green]")
        console.print(f"  • Removed {total_removed} orphaned files")
        console.print(f"  • Freed {size_mb:.1f} MB of disk space")

        # Re-analyze to confirm
        console.print("\n🔍 [cyan]Verifying cleanup...[/cyan]")
        final_orphan_report = uploader.check_for_orphaned_files(args.dataset_dir)
        if final_orphan_report["total_orphaned"] == 0:
            console.print("✅ [green]All orphaned files removed successfully![/green]")
        else:
            console.print(f"⚠️ [yellow]{final_orphan_report['total_orphaned']} orphaned files remain[/yellow]")

    except KeyboardInterrupt:
        console.print("\n\n⚠️ Cleanup cancelled by user")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n❌ [red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main() 