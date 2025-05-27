# tui.py
"""Terminal UI for Kokoro Pipeline using Rich"""

from os.path import isfile
import sys
from pathlib import Path
import yaml
import time

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich.prompt import Prompt, Confirm, TextType
from rich.panel import Panel
from rich.table import Table
from dataset_pipeline import main as run_pipeline
from typing import List, Optional, TextIO


console = Console()


class YoutubePrompt(Prompt):
    response_type = [str]

    def process_response(self, value: str) -> List[str]:
        """Convert choices to a bool."""
        value = value.strip().split("\n")
        return value

    @classmethod
    def get_input(
        cls,
        console: Console,
        prompt: TextType,
        password: bool,
        stream: Optional[TextIO] = None,
    ) -> str:
        """Get input from user.

        Args:
            console (Console): Console instance.
            prompt (TextType): Prompt text.
            password (bool): Enable password entry.

        Returns:
            str: String from user.
        """
        # a while loop to get multiple lines of input while the return is not an empty string
        response = []
        while True:
            line = console.input(prompt)
            if line.strip() == "":
                break
            response.append(line)
        response = "\n".join(response)
        return response


class DatasetPipelineTUI:
    """Interactive Terminal UI for Kokoro Pipeline"""

    def __init__(self):
        self.console = console

    def welcome(self):
        """Show welcome screen"""
        self.console.clear()
        welcome_text = """
[bold cyan]Kokoro Voice Dataset Pipeline[/bold cyan]
[dim]End-to-end dataset preparation for TTS training[/dim]

This tool will guide you through:
1. 📥 Downloading/loading audio
2. 🎤 Isolating speakers
3. ✨ Enhancing quality
4. ✂️  Segmenting audio
5. 📝 Transcribing speech
6. 🧹 Cleaning dataset
7. ☁️  Uploading to HuggingFace

[bold yellow]💾 Checkpoint/Resume Support:[/bold yellow]
• Automatic progress saving
• Resume from interruptions
• Per-file fault tolerance
        """
        self.console.print(
            Panel(welcome_text, title="Welcome", border_style="cyan"))

    def get_input_sources(self):
        """Interactive input source selection"""
        sources = []

        self.console.print("\n[bold]Step 2: Select Input Sources[/bold]")

        while True:
            choice = Prompt.ask(
                "\nWhat would you like to add?",
                choices=["youtube", "file", "folder", "done"],
                default="done" if sources else "folder",
            )

            if choice == "done":
                if not sources:
                    self.console.print(
                        "[red]Please add at least one source![/red]")
                    continue
                break

            elif choice == "youtube":
                urls = YoutubePrompt.ask("Enter YouTube URL")
                expanded_urls = []
                for url in urls:
                    if isfile(url):
                        with open(url, "r", encoding="utf-8") as f:
                            expanded_urls.extend([line.strip() for line in f])
                    else:
                        expanded_urls.append(url)
                sources.extend(expanded_urls)
                self.console.print("[green]✓ Added YouTube video[/green]")

            elif choice == "file":
                path = Prompt.ask("Enter file path")
                if Path(path).exists():
                    sources.append(path)
                    self.console.print(f"[green]✓ Added {path}[/green]")
                else:
                    self.console.print(f"[red]File not found: {path}[/red]")

            elif choice == "folder":
                path = Prompt.ask("Enter folder path", default="./")
                if Path(path).is_dir():
                    sources.append(path)
                    self.console.print(f"[green]✓ Added folder {path}[/green]")
                else:
                    self.console.print(f"[red]Folder not found: {path}[/red]")

        # Show summary
        table = Table(title="Input Sources")
        table.add_column("Type", style="cyan")
        table.add_column("Source", style="white")

        for source in sources:
            if source.startswith(("http://", "https://", "www.")):
                table.add_row("YouTube", source)
            elif Path(source).is_file():
                table.add_row("File", source)
            else:
                table.add_row("Folder", source)

        self.console.print(table)

        return sources

    def get_processing_options(self):
        """Get processing configuration"""
        self.console.print("\n[bold]Step 3: Processing Options[/bold]")

        # Enhancement mode
        mode = Prompt.ask(
            "\nSelect enhancement mode",
            choices=["fast", "balanced", "quality"],
            default="balanced",
        )

        mode_info = {
            "fast": "⚡ Fast: MetricGAN+ only (good for quick processing)",
            "balanced": "⚖️  Balanced: MetricGAN+ (recommended)",
            "quality": "💎 Quality: VoiceFixer (best quality, slower)",
        }
        self.console.print(mode_info[mode])

        # DeepFilter setting (only show if balanced or quality mode selected)
        enable_deepfilter = False
        if mode in ["balanced", "quality"]:
            enable_deepfilter = Confirm.ask(
                "\nEnable DeepFilter enhancement?",
                default=False
            )

        # Whisper model
        whisper_model = Prompt.ask(
            "\nSelect Whisper model",
            choices=["tiny", "base", "small", "medium", "large", "turbo"],
            default="base",
        )

        # Optional stages
        skip_isolation = not Confirm.ask("\nIsolate speakers?", default=True)
        skip_cleaning = not Confirm.ask(
            "Clean dataset (remove bad samples)?", default=True
        )

        return {
            "enhancement_mode": mode,
            "whisper_model": whisper_model,
            "skip_isolation": skip_isolation,
            "skip_cleaning": skip_cleaning,
            "enable_deepfilter": enable_deepfilter,
        }

    def get_output_options(self):
        """Get output configuration"""
        self.console.print("\n[bold]Step 1: Output Options[/bold]")

        output_dir = Prompt.ask("\nOutput directory",
                                default="./output/kokoro-dataset")

        return {
            "output_dir": output_dir,
        }

    def get_upload_options(self):
        """Get upload configuration"""
        self.console.print("\n[bold]Step 4: Upload Options[/bold]")

        upload_hf = Confirm.ask("\nUpload to HuggingFace?", default=False)
        hf_repo = None
        hf_private = True

        if upload_hf:
            hf_repo = Prompt.ask("HuggingFace repo (username/dataset-name)")
            hf_private = Confirm.ask("Make repo private?", default=True)

        return {
            "upload_hf": upload_hf,
            "hf_repo": hf_repo,
            "hf_private": hf_private,
        }

    def show_progress(self):
        """Show real-time progress"""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:

            # Main task
            main_task = progress.add_task(
                "[cyan]Processing dataset...", total=7)

            # Step tasks
            steps = [
                "Downloading audio",
                "Isolating speakers",
                "Enhancing audio",
                "Segmenting audio",
                "Transcribing speech",
                "Preparing dataset",
                "Cleaning dataset",
            ]

            for step in steps:
                progress.update(
                    main_task, description=f"[cyan]{step}...", advance=1)
                # Here you would yield to the actual pipeline processing

    def show_results(self, results):
        """Show final results"""
        self.console.print("\n[bold green]✅ Pipeline Complete![/bold green]\n")

        # Results table
        table = Table(title="Dataset Statistics", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        stats = results.get("dataset_stats", {})
        table.add_row("Total Segments", f"{stats.get('total_segments', 0):,}")
        table.add_row(
            "Total Duration", f"{stats.get('total_duration_hours', 0):.1f} hours"
        )
        table.add_row("Average Duration",
                      f"{stats.get('avg_duration', 0):.1f} seconds")

        if "quality_report" in results:
            report = results["quality_report"]
            table.add_row("Quality Score",
                          f"{report.get('average_quality', 0):.1%}")
            table.add_row("Rejected Files",
                          f"{report.get('rejected_count', 0)}")

        self.console.print(table)

    def check_for_checkpoints(self, output_dir: str) -> Optional[dict]:
        """Check for existing checkpoints in the output directory"""
        output_path = Path(output_dir)
        state_file = output_path / "pipeline_state.yaml"
        config_file = output_path / "pipeline_config.yaml"
        lock_file = output_path / ".pipeline_running"

        if not state_file.exists():
            return None

        try:
            # Load checkpoint info
            with open(state_file, 'r') as f:
                state = yaml.safe_load(f)

            config = None
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

            # Check for lock file
            lock_exists = lock_file.exists()
            lock_info = None
            if lock_exists:
                try:
                    lock_info = lock_file.read_text().strip()
                except Exception as e:
                    self.console.print(
                        f"[red]Error reading lock file: {e}[/red]")
                    lock_info = "Unknown"

            return {
                "state": state,
                "config": config,
                "lock_exists": lock_exists,
                "lock_info": lock_info,
                "state_file": state_file,
                "config_file": config_file,
                "lock_file": lock_file
            }
        except Exception as e:
            self.console.print(f"[red]Error reading checkpoint: {e}[/red]")
            return None

    def show_checkpoint_status(self, checkpoint_info: dict) -> str:
        """Display checkpoint status and get user choice"""
        state = checkpoint_info["state"]
        config = checkpoint_info["config"]

        # Create checkpoint summary
        self.console.print(
            "\n[bold yellow]🔍 Existing Checkpoint Found![/bold yellow]")

        # Basic info panel
        last_checkpoint = state.get("last_checkpoint", 0)
        start_time = state.get("start_time", 0)
        completed_stages = state.get("completed_stages", [])

        info_text = f"""
[bold]Last saved:[/bold] {time.ctime(last_checkpoint) if last_checkpoint else 'Unknown'}
[bold]Started:[/bold] {time.ctime(start_time) if start_time else 'Unknown'}
[bold]Completed stages:[/bold] {len(completed_stages)}/7
[bold]Pipeline version:[/bold] {state.get('pipeline_version', 'Unknown')}
        """

        if checkpoint_info["lock_exists"]:
            info_text += f"\n[bold red]⚠️ Lock file exists:[/bold red] {checkpoint_info['lock_info']}"

        self.console.print(
            Panel(info_text, title="Checkpoint Info", border_style="yellow"))

        # Progress table
        if completed_stages:
            progress_table = Table(title="Completed Stages", show_header=True)
            progress_table.add_column("Stage", style="cyan")
            progress_table.add_column("Status", style="green")
            progress_table.add_column("Files", style="white")

            stage_files = {
                "download_audio": len(state.get("downloaded_files", [])),
                "collect_local_files": len(state.get("downloaded_files", [])),
                "isolate_speakers": len(state.get("isolated_files", [])),
                "enhance_audio": len(state.get("enhanced_files", [])),
                "segment_audio": len(state.get("segments", [])),
                "transcribe_audio": len(state.get("transcriptions", [])),
            }

            for stage in completed_stages:
                file_count = stage_files.get(stage, 0)
                progress_table.add_row(
                    stage.replace("_", " ").title(),
                    "✅ Complete",
                    str(file_count) if file_count > 0 else "-"
                )

            self.console.print(progress_table)

        # Show original config if available
        if config:
            config_text = f"""
[bold]Input sources:[/bold] {len(config.get('input_paths', []))} paths
[bold]Enhancement mode:[/bold] {config.get('enhancement_mode', 'Unknown')}
[bold]Whisper model:[/bold] {config.get('whisper_model', 'Unknown')}
[bold]Upload to HF:[/bold] {'Yes' if config.get('upload_to_hf') else 'No'}
            """
            self.console.print(
                Panel(config_text, title="Original Configuration", border_style="blue"))

        # Get user choice
        choices = ["resume", "restart", "cancel"]
        choice_descriptions = {
            "resume": "📂 Resume from checkpoint (continue where you left off)",
            "restart": "🔄 Start fresh (delete checkpoint and begin again)",
            "cancel": "❌ Cancel and exit"
        }

        self.console.print("\n[bold]What would you like to do?[/bold]")
        for choice, desc in choice_descriptions.items():
            self.console.print(f"  [cyan]{choice}[/cyan]: {desc}")

        return Prompt.ask(
            "\nChoose action",
            choices=choices,
            default="resume"
        )

    def handle_checkpoint_choice(self, choice: str, checkpoint_info: dict, output_dir: str) -> dict:
        """Handle the user's checkpoint choice and return appropriate config"""
        if choice == "cancel":
            self.console.print("[yellow]Operation cancelled.[/yellow]")
            sys.exit(0)

        elif choice == "restart":
            # Confirm restart
            confirm = Confirm.ask(
                "\n[bold red]⚠️ This will delete all checkpoint data. Are you sure?[/bold red]",
                default=False
            )

            if not confirm:
                self.console.print("[yellow]Restart cancelled.[/yellow]")
                return self.handle_checkpoint_choice(
                    self.show_checkpoint_status(checkpoint_info),
                    checkpoint_info,
                    output_dir
                )

            # Delete checkpoint files
            try:
                checkpoint_info["state_file"].unlink(missing_ok=True)
                checkpoint_info["config_file"].unlink(missing_ok=True)
                checkpoint_info["lock_file"].unlink(missing_ok=True)
                self.console.print(
                    "[green]✅ Checkpoint cleared. Starting fresh...[/green]")
                return {"force_restart": True, "resume": False}
            except Exception as e:
                self.console.print(
                    f"[red]Error clearing checkpoint: {e}[/red]")
                return {"force_restart": True, "resume": False}

        elif choice == "resume":
            self.console.print("[green]📂 Resuming from checkpoint...[/green]")
            return {"resume": True, "force_restart": False}

        return {"resume": False, "force_restart": False}

    def run_interactive(self):
        """Run the full interactive pipeline"""
        self.welcome()

        if not Confirm.ask("\nContinue?", default=True):
            return

        # Step 1: Get output directory first
        out_opts = self.get_output_options()

        # Check for checkpoints
        checkpoint_info = self.check_for_checkpoints(out_opts["output_dir"])

        # Initialize config with output directory
        config = {
            "output_dir": out_opts["output_dir"],
            "resume": False,
            "force_restart": False
        }

        if checkpoint_info:
            # Show checkpoint status and get user choice
            choice = self.show_checkpoint_status(checkpoint_info)
            checkpoint_choice = self.handle_checkpoint_choice(
                choice, checkpoint_info, out_opts["output_dir"])

            config.update(checkpoint_choice)

            # If resuming, use the saved config from checkpoint
            if checkpoint_choice.get("resume") and checkpoint_info["config"]:
                saved_config = checkpoint_info["config"]
                config.update({
                    "input_paths": saved_config.get("input_paths", []),
                    "enhancement_mode": saved_config.get("enhancement_mode", "balanced"),
                    "whisper_model": saved_config.get("whisper_model", "base"),
                    "skip_isolation": saved_config.get("skip_isolation", False),
                    "skip_cleaning": saved_config.get("skip_cleaning", False),
                    "enable_deepfilter": saved_config.get("enable_deepfilter", False),
                    "upload_hf": saved_config.get("upload_hf", False),
                    "hf_repo": saved_config.get("hf_repo"),
                    "hf_private": saved_config.get("hf_private", True)
                })

                # Skip questions since we're resuming
                self.console.print(
                    "[green]📂 Using saved configuration from checkpoint...[/green]")
                return self._build_and_run_command(config)

            elif checkpoint_choice.get("force_restart"):
                self.console.print(
                    "[green]🔄 Starting fresh with new configuration...[/green]")

        # If not resuming, get all configuration interactively
        config["input_paths"] = self.get_input_sources()
        proc_opts = self.get_processing_options()
        config.update(proc_opts)

        upload_opts = self.get_upload_options()
        config.update(upload_opts)

        return self._build_and_run_command(config)

    def _build_and_run_command(self, config: dict):
        """Build and run the pipeline command from config"""
        cmd_parts = [
            "dataset-pipeline",
            *config["input_paths"],
            "--output",
            config["output_dir"],
            "--enhancement-mode",
            config["enhancement_mode"],
            "--whisper-model",
            config["whisper_model"],
        ]

        if config["skip_isolation"]:
            cmd_parts.append("--skip-isolation")
        if config["skip_cleaning"]:
            cmd_parts.append("--skip-cleaning")
        if not config["enable_deepfilter"]:
            cmd_parts.append("--no-deepfilter")

        if config["upload_hf"]:
            cmd_parts.extend(["--upload-hf", "--hf-repo", config["hf_repo"]])
            if not config["hf_private"]:
                cmd_parts.append("--hf-public")

        if config.get("force_restart"):
            cmd_parts.append("--force-restart")
        elif config.get("resume"):
            cmd_parts.append("--resume")

        # Show command
        self.console.print("\n[bold]Command to run:[/bold]")
        self.console.print(f"[dim]{' '.join(cmd_parts)}[/dim]\n")

        if Confirm.ask("Start processing?", default=True):
            # Import and run the actual pipeline
            sys.argv = cmd_parts
            run_pipeline()


if __name__ == "__main__":
    tui = DatasetPipelineTUI()
    tui.run_interactive()
