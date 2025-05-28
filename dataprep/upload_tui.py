#!/usr/bin/env python3
"""
TUI for uploading prepared datasets to HuggingFace Hub
"""

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from upload_dataset import DatasetUploader


class DatasetUploadTUI:
    """TUI for dataset upload"""

    def __init__(self):
        self.console = Console()
        self.uploader = DatasetUploader()

    def show_header(self):
        """Show application header"""
        header = Panel(
            "[bold blue]Kokoro TTS Dataset Uploader[/bold blue]\n"
            "Upload prepared datasets to HuggingFace Hub",
            title="🎤 Dataset Upload",
            border_style="blue"
        )
        self.console.print(header)

    def select_dataset_directory(self) -> Optional[Path]:
        """Let user select dataset directory"""
        while True:
            self.console.print("\n📁 [cyan]Select Dataset Directory[/cyan]")

            # Get directory path
            path_input = Prompt.ask(
                "Enter path to prepared dataset directory",
                default="."
            )

            dataset_dir = Path(path_input).expanduser().resolve()

            if not dataset_dir.exists():
                self.console.print(
                    f"❌ [red]Directory does not exist: {dataset_dir}[/red]")
                if not Confirm.ask("Try again?"):
                    return None
                continue

            if not dataset_dir.is_dir():
                self.console.print(
                    f"❌ [red]Path is not a directory: {dataset_dir}[/red]")
                if not Confirm.ask("Try again?"):
                    return None
                continue

            # Quick check for dataset structure
            has_splits = any((dataset_dir / split).exists()
                             for split in ["train", "validation", "test"])

            if not has_splits:
                self.console.print(
                    f"⚠️ [yellow]Warning: No train/validation/test directories found in {dataset_dir}[/yellow]")
                if not Confirm.ask("This doesn't look like a prepared dataset. Continue anyway?"):
                    if not Confirm.ask("Try different directory?"):
                        return None
                    continue

            return dataset_dir

    def get_repository_config(self) -> dict:
        """Get HuggingFace repository configuration"""
        self.console.print(
            "\n🚀 [cyan]HuggingFace Repository Configuration[/cyan]")

        # Repository ID
        repo_id = Prompt.ask(
            "Enter repository ID (username/dataset-name)",
            default="my-username/my-voice-dataset"
        )

        # Privacy
        private = Confirm.ask("Make repository private?", default=True)

        return {
            "repo_id": repo_id,
            "private": private
        }

    def show_upload_summary(self, dataset_dir: Path, config: dict, validation: dict):
        """Show upload summary before confirmation"""
        self.console.print("\n📋 [cyan]Upload Summary[/cyan]")

        # Create summary table
        table = Table(show_header=False, box=None)
        table.add_column("Item", style="cyan", width=20)
        table.add_column("Value", style="white")

        table.add_row("Dataset Directory", str(dataset_dir))
        table.add_row("Repository ID", config["repo_id"])
        table.add_row("Privacy", "Private" if config["private"] else "Public")

        if validation.get("stats"):
            stats = validation["stats"]
            if "total_entries" in stats:
                table.add_row("Total Entries", str(stats["total_entries"]))
            if "splits" in stats:
                table.add_row("Splits", ", ".join(stats["splits"]))

        self.console.print(table)

        # Show validation status
        if validation["valid"]:
            self.console.print("✅ [green]Dataset validation: PASSED[/green]")
        else:
            self.console.print("❌ [red]Dataset validation: FAILED[/red]")

        if validation["warnings"]:
            self.console.print(
                f"⚠️ [yellow]{len(validation['warnings'])} warnings[/yellow]")

        if validation["errors"]:
            self.console.print(
                f"❌ [red]{len(validation['errors'])} errors[/red]")

    def run(self):
        """Main TUI flow"""
        try:
            self.show_header()

            # Step 1: Select dataset directory
            dataset_dir = self.select_dataset_directory()
            if dataset_dir is None:
                self.console.print("Upload cancelled.")
                return

            # Step 2: Validate dataset
            self.console.print("\n🔍 [cyan]Validating dataset...[/cyan]")
            validation = self.uploader.validate_dataset(dataset_dir)
            self.uploader.display_validation_results(validation)

            if not validation["valid"]:
                self.console.print("\n❌ [red]Dataset validation failed![/red]")
                if not Confirm.ask("Upload anyway? (not recommended)"):
                    self.console.print("Upload cancelled.")
                    return

            if validation["warnings"]:
                if not Confirm.ask("\nProceed despite warnings?"):
                    self.console.print("Upload cancelled.")
                    return

            # Step 3: Get repository configuration
            config = self.get_repository_config()

            # Step 4: Show summary and confirm
            self.show_upload_summary(dataset_dir, config, validation)

            if not Confirm.ask("\n🚀 Proceed with upload?"):
                self.console.print("Upload cancelled.")
                return

            # Step 5: Upload
            self.uploader.upload_dataset(
                dataset_dir=dataset_dir,
                repo_id=config["repo_id"],
                private=config["private"]
            )

        except KeyboardInterrupt:
            self.console.print("\n\n⚠️ Upload cancelled by user")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"\n❌ [red]Error: {e}[/red]")
            sys.exit(1)


def main():
    """Main entry point"""
    tui = DatasetUploadTUI()
    tui.run()


if __name__ == "__main__":
    main()
