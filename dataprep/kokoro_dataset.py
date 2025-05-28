#!/usr/bin/env python3
"""
Kokoro TTS Dataset Tools
Main CLI entry point with subcommands for dataset operations
"""

import argparse
import sys


def main():
    """Main CLI entry point with subcommands"""
    parser = argparse.ArgumentParser(
        prog="kokoro-dataset",
        description="Kokoro TTS Dataset Tools - Prepare and upload voice datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  prepare    Prepare a new dataset from audio files
  upload     Upload a prepared dataset to HuggingFace Hub
  tui        Interactive Text User Interface

Examples:
  # Prepare dataset with TUI
  kokoro-dataset tui

  # Prepare dataset from command line
  kokoro-dataset prepare ./audio-files/ --output my-voice-dataset

  # Upload prepared dataset
  kokoro-dataset upload ./my-voice-dataset --repo-id username/my-voice

  # Upload with TUI
  kokoro-dataset upload-tui
        """
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    # Prepare subcommand
    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Prepare dataset from audio files",
        description="Run the full dataset preparation pipeline"
    )
    prepare_parser.add_argument(
        "input_dir", nargs="?", help="Input audio directory")
    prepare_parser.add_argument(
        "--output", "-o", help="Output dataset directory")
    prepare_parser.add_argument(
        "--upload-hf", action="store_true", help="Upload to HuggingFace after preparation")
    prepare_parser.add_argument("--hf-repo", help="HuggingFace repository ID")
    prepare_parser.add_argument(
        "--hf-public", action="store_true", help="Make repository public")
    prepare_parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint")
    prepare_parser.add_argument(
        "--force-restart", action="store_true", help="Force restart (clear checkpoint)")

    # Upload subcommand
    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload prepared dataset to HuggingFace Hub",
        description="Upload an already-prepared dataset"
    )
    upload_parser.add_argument(
        "dataset_dir", help="Path to prepared dataset directory")
    upload_parser.add_argument(
        "--repo-id", required=True, help="HuggingFace repository ID")
    upload_parser.add_argument(
        "--private", action="store_true", help="Make repository private")
    upload_parser.add_argument(
        "--public", action="store_true", help="Make repository public")
    upload_parser.add_argument(
        "--skip-validation", action="store_true", help="Skip dataset validation")
    upload_parser.add_argument(
        "--yes", action="store_true", help="Skip confirmation prompts")

    # TUI subcommands
    subparsers.add_parser(
        "tui",
        help="Interactive Text User Interface for dataset preparation",
        description="Run the interactive TUI for dataset preparation"
    )

    subparsers.add_parser(
        "upload-tui",
        help="Interactive TUI for dataset upload",
        description="Run the interactive TUI for uploading prepared datasets"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "prepare":
            # Import and run dataset pipeline
            from dataset_pipeline import main as pipeline_main

            # Convert args to pipeline format
            sys.argv = ["dataset_pipeline.py"]
            if args.input_dir:
                sys.argv.append(args.input_dir)
            if args.output:
                sys.argv.extend(["--output", args.output])
            if args.upload_hf:
                sys.argv.append("--upload-hf")
            if args.hf_repo:
                sys.argv.extend(["--hf-repo", args.hf_repo])
            if args.hf_public:
                sys.argv.append("--hf-public")
            if args.resume:
                sys.argv.append("--resume")
            if args.force_restart:
                sys.argv.append("--force-restart")

            pipeline_main()

        elif args.command == "upload":
            # Import and run upload script
            from upload_dataset import main as upload_main

            # Convert args to upload format
            sys.argv = ["upload_dataset.py", args.dataset_dir]
            sys.argv.extend(["--repo-id", args.repo_id])
            if args.private:
                sys.argv.append("--private")
            elif args.public:
                sys.argv.append("--public")
            if args.skip_validation:
                sys.argv.append("--skip-validation")
            if args.yes:
                sys.argv.append("--yes")

            upload_main()

        elif args.command == "tui":
            # Import and run TUI
            from tui import DatasetPipelineTUI
            tui = DatasetPipelineTUI()
            tui.run_interactive()

        elif args.command == "upload-tui":
            # Import and run upload TUI
            from upload_tui import main as upload_tui_main
            upload_tui_main()

    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
