# dataset-pipeline
#!/usr/bin/env python3
"""
Kokoro Dataset Pipeline - Main entry point

Usage:
    dataset-pipeline                     # Interactive TUI mode
    dataset-pipeline <args...>          # CLI mode with arguments
    dataset-pipeline --help            # Show help
"""

import sys

from tui import DatasetPipelineTUI
from dataset_pipeline import main as cli_main


def main():
    """Main entry point for the dataset pipeline.
    
    Determines whether to start in TUI mode (interactive) or CLI mode based on
    whether command-line arguments are provided.
    """
    # If no arguments, launch TUI
    if len(sys.argv) == 1:
        tui = DatasetPipelineTUI()
        tui.run_interactive()
    else:
        # CLI mode
        cli_main()

if __name__ == "__main__":
    main()
