#!/usr/bin/env python3
"""
Real-time training monitor for Kokoro voice training.
Shows live metrics and estimates without needing W&B.
"""

import json
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

class TrainingMonitor:
    def __init__(self, log_file="training.log"):
        self.log_file = Path(log_file)
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        plt.ion()
        
    def parse_log_line(self, line):
        """Extract metrics from a log line."""
        metrics = {}
        
        # Parse epoch summary lines
        if "Epoch" in line and "Summary:" in line:
            metrics['type'] = 'epoch'
            if "Training Loss:" in line:
                metrics['train_loss'] = float(line.split("Training Loss:")[1].split()[0])
            if "Validation Loss:" in line:
                metrics['val_loss'] = float(line.split("Validation Loss:")[1].split()[0])
                
        # Parse batch lines
        elif "[Epoch" in line and "Batch" in line:
            metrics['type'] = 'batch'
            parts = line.split("|")
            for part in parts:
                if "Loss:" in part and "Avg:" not in part:
                    metrics['batch_loss'] = float(part.split("Loss:")[1].split()[0])
                elif "Avg:" in part:
                    metrics['avg_loss'] = float(part.split("Avg:")[1].split()[0])
                    
        return metrics
    
    def update_plots(self, epochs, train_losses, val_losses, batch_losses):
        """Update the monitoring plots."""
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
            
        # Plot 1: Training vs Validation Loss
        ax1 = self.axes[0, 0]
        if train_losses:
            ax1.plot(epochs, train_losses, 'b-', label='Training', linewidth=2)
        if val_losses:
            ax1.plot(epochs, val_losses, 'r-', label='Validation', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Recent batch losses
        ax2 = self.axes[0, 1]
        if batch_losses:
            recent_batches = batch_losses[-100:]  # Last 100 batches
            ax2.plot(recent_batches, 'g-', alpha=0.7)
            ax2.set_xlabel('Batch')
            ax2.set_ylabel('Loss')
            ax2.set_title('Recent Batch Losses')
            ax2.grid(True, alpha=0.3)
            
        # Plot 3: Loss improvement rate
        ax3 = self.axes[1, 0]
        if len(val_losses) > 1:
            improvements = np.diff(val_losses)
            ax3.bar(epochs[1:], improvements, color=['red' if x > 0 else 'green' for x in improvements])
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Δ Validation Loss')
            ax3.set_title('Epoch-to-Epoch Improvement')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.grid(True, alpha=0.3)
            
        # Plot 4: Statistics
        ax4 = self.axes[1, 1]
        ax4.axis('off')
        stats_text = "Training Statistics\n" + "="*25 + "\n"
        
        if epochs:
            stats_text += f"Current Epoch: {epochs[-1]}\n"
        if train_losses:
            stats_text += f"Latest Train Loss: {train_losses[-1]:.4f}\n"
            stats_text += f"Best Train Loss: {min(train_losses):.4f}\n"
        if val_losses:
            stats_text += f"Latest Val Loss: {val_losses[-1]:.4f}\n"
            stats_text += f"Best Val Loss: {min(val_losses):.4f}\n"
            best_epoch = epochs[np.argmin(val_losses)]
            stats_text += f"Best Epoch: {best_epoch}\n"
            
        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
    def monitor(self, update_interval=5):
        """Monitor the training log file."""
        print("Monitoring training progress...")
        print(f"Reading from: {self.log_file}")
        print("Press Ctrl+C to stop\n")
        
        epochs = []
        train_losses = []
        val_losses = []
        batch_losses = []
        
        last_position = 0
        
        try:
            while True:
                if self.log_file.exists():
                    with open(self.log_file, 'r') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()
                        
                    for line in new_lines:
                        metrics = self.parse_log_line(line)
                        
                        if metrics.get('type') == 'epoch':
                            if 'train_loss' in metrics:
                                # Extract epoch number from line
                                try:
                                    epoch_num = int(line.split("Epoch")[1].split("/")[0])
                                    if epoch_num not in epochs:
                                        epochs.append(epoch_num)
                                        train_losses.append(metrics['train_loss'])
                                        if 'val_loss' in metrics:
                                            val_losses.append(metrics['val_loss'])
                                except:
                                    pass
                                    
                        elif metrics.get('type') == 'batch':
                            if 'batch_loss' in metrics:
                                batch_losses.append(metrics['batch_loss'])
                                
                    if epochs:
                        self.update_plots(epochs, train_losses, val_losses, batch_losses)
                        
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor Kokoro voice training")
    parser.add_argument("--log", default="training.log", help="Path to training log file")
    parser.add_argument("--interval", type=int, default=5, help="Update interval in seconds")
    
    args = parser.parse_args()
    monitor = TrainingMonitor(args.log)
    monitor.monitor(args.interval)
