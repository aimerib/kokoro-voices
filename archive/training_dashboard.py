#!/usr/bin/env python3
"""
Real-time Training Dashboard for Kokoro Voice Cloning
Monitors training progress with live updates and alerts
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from collections import deque
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("TensorBoard not installed. Install with: pip install tensorboard")
    exit(1)

class TrainingDashboard:
    def __init__(self, log_dir: Path, update_interval: int = 5):
        self.log_dir = Path(log_dir)
        self.update_interval = update_interval
        
        # Find tensorboard logs
        self.find_event_files()
        
        # Data storage with history
        self.history_length = 500
        self.epochs = deque(maxlen=self.history_length)
        self.train_losses = deque(maxlen=self.history_length)
        self.val_losses = deque(maxlen=self.history_length)
        self.learning_rates = deque(maxlen=self.history_length)
        self.timbre_means = deque(maxlen=self.history_length)
        self.style_means = deque(maxlen=self.history_length)
        self.smoothness_losses = deque(maxlen=self.history_length)
        self.batch_stft_losses = deque(maxlen=self.history_length)
        
        # Setup the figure and subplots
        self.fig = plt.figure(figsize=(15, 10))
        self.fig.suptitle('Kokoro Voice Training Dashboard', fontsize=16)
        
        gs = gridspec.GridSpec(3, 3, figure=self.fig)
        self.ax_loss = self.fig.add_subplot(gs[0, :2])
        self.ax_embedding = self.fig.add_subplot(gs[0, 2])
        self.ax_lr = self.fig.add_subplot(gs[1, 0])
        self.ax_smoothness = self.fig.add_subplot(gs[1, 1])
        self.ax_metrics = self.fig.add_subplot(gs[1, 2])
        self.ax_status = self.fig.add_subplot(gs[2, :])
        
        # Initialize plots
        self.setup_plots()
        
        # Status tracking
        self.last_update = None
        self.training_active = False
    
    def find_event_files(self):
        """Find TensorBoard event files in the log directory"""
        # Look for runs directory (default TensorBoard structure)
        runs_dir = self.log_dir / "runs"
        if runs_dir.exists():
            event_dirs = list(runs_dir.glob("*"))
            if event_dirs:
                # Use the most recent run
                self.event_dir = max(event_dirs, key=lambda p: p.stat().st_mtime)
                print(f"Found TensorBoard logs in: {self.event_dir}")
                return
        
        # Look for event files directly in log_dir
        event_files = list(self.log_dir.glob("**/events.out.tfevents.*"))
        if event_files:
            # Use the most recent event file's directory
            self.event_dir = max(event_files, key=lambda p: p.stat().st_mtime).parent
            print(f"Found TensorBoard logs in: {self.event_dir}")
            return
            
        # Also check wandb directory
        wandb_dir = self.log_dir / "wandb"
        if wandb_dir.exists():
            print(f"Found WandB logs in: {wandb_dir}")
            self.wandb_dir = wandb_dir
        else:
            self.wandb_dir = None
            
        # If no event files found, create a placeholder
        self.event_dir = self.log_dir
        print(f"No TensorBoard event files found yet in {self.log_dir}")
        print("Waiting for training to start...")
    
    def setup_plots(self):
        """Initialize all plot configurations"""
        # Main loss plot
        self.ax_loss.set_title('Training & Validation Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Embedding visualization
        self.ax_embedding.set_title('Voice Embedding Stats')
        self.ax_embedding.set_xlabel('Epoch')
        self.ax_embedding.set_ylabel('Mean Value')
        self.ax_embedding.grid(True, alpha=0.3)
        
        # Learning rate
        self.ax_lr.set_title('Learning Rate')
        self.ax_lr.set_xlabel('Epoch')
        self.ax_lr.set_ylabel('LR')
        self.ax_lr.grid(True, alpha=0.3)
        
        # Smoothness loss
        self.ax_smoothness.set_title('Smoothness & STFT Loss')
        self.ax_smoothness.set_xlabel('Epoch')
        self.ax_smoothness.set_ylabel('Loss')
        self.ax_smoothness.grid(True, alpha=0.3)
        
        # Current metrics
        self.ax_metrics.axis('off')
        
        # Status text
        self.ax_status.axis('off')
        
        plt.tight_layout()
    
    def read_tensorboard_data(self):
        """Read data from TensorBoard event files"""
        try:
            # Re-check for event files if we don't have any
            if not hasattr(self, 'event_dir') or not any(self.event_dir.glob("events.out.tfevents.*")):
                self.find_event_files()
                
            event_files = list(self.event_dir.glob("events.out.tfevents.*"))
            if not event_files:
                return None
                
            # Use the most recent event file
            event_file = max(event_files, key=lambda p: p.stat().st_mtime)
            
            # Load events
            ea = EventAccumulator(str(event_file.parent))
            ea.Reload()
            
            # Get available scalar tags
            scalar_tags = ea.Tags()['scalars']
            
            data = {}
            
            # Map TensorBoard tags to our expected metrics
            tag_mapping = {
                'Metrics/batch_loss': 'train_loss',
                'Metrics/validation_loss': 'val_loss',
                'Metrics/learning_rate': 'learning_rate',
                'Metrics/timbre_mean': 'timbre_mean',
                'Metrics/style_mean': 'style_mean',
                'Metrics/smoothness_loss': 'smoothness_loss',
                'Metrics/batch_stft_loss': 'batch_stft_loss',
            }
            
            # Extract data for each metric
            for tb_tag, metric_name in tag_mapping.items():
                if tb_tag in scalar_tags:
                    events = ea.Scalars(tb_tag)
                    if events:
                        # Get the latest value
                        latest = events[-1]
                        data[metric_name] = latest.value
                        data['step'] = latest.step
                        data['epoch'] = latest.step  # Assuming step is epoch
                        data['wall_time'] = latest.wall_time
            
            return data if data else None
            
        except Exception as e:
            print(f"Error reading TensorBoard data: {e}")
            return None
    
    def update_data(self, frame):
        """Update data from logs and refresh plots"""
        # Read latest metrics
        metrics = self.read_tensorboard_data()
        
        if metrics is None:
            # Update status to show waiting
            self.ax_status.clear()
            self.ax_status.text(0.5, 0.5, 'Waiting for training data...\nMake sure training is running with --tensorboard flag', 
                              ha='center', va='center', fontsize=12, color='orange')
            self.ax_status.axis('off')
            return
        
        # Update data arrays
        if 'epoch' in metrics:
            self.epochs.append(metrics['epoch'])
            
        if 'train_loss' in metrics:
            self.train_losses.append(metrics['train_loss'])
            
        if 'val_loss' in metrics:
            self.val_losses.append(metrics['val_loss'])
            
        if 'learning_rate' in metrics:
            self.learning_rates.append(metrics['learning_rate'])
            
        if 'timbre_mean' in metrics:
            self.timbre_means.append(metrics['timbre_mean'])
            
        if 'style_mean' in metrics:
            self.style_means.append(metrics['style_mean'])
            
        if 'smoothness_loss' in metrics:
            self.smoothness_losses.append(metrics['smoothness_loss'])
            
        if 'batch_stft_loss' in metrics:
            self.batch_stft_losses.append(metrics['batch_stft_loss'])
        
        # Update plots
        self.update_plots(metrics)
        
        # Update status
        self.update_status(metrics)
        
        self.last_update = time.time()
    
    def update_plots(self, current_metrics):
        """Update all plots with current data"""
        # Clear all plots
        self.ax_loss.clear()
        self.ax_embedding.clear()
        self.ax_lr.clear()
        self.ax_smoothness.clear()
        self.ax_metrics.clear()
        
        epochs_array = np.array(self.epochs)
        
        # Loss plot
        if len(self.train_losses) > 0:
            self.ax_loss.plot(epochs_array, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        if len(self.val_losses) > 0:
            self.ax_loss.plot(epochs_array, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        self.ax_loss.set_title('Training & Validation Loss')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.legend()
        self.ax_loss.grid(True, alpha=0.3)
        
        # Embedding stats
        if len(self.timbre_means) > 0:
            self.ax_embedding.plot(epochs_array, self.timbre_means, 'g-', label='Timbre Mean', linewidth=2)
        if len(self.style_means) > 0:
            self.ax_embedding.plot(epochs_array, self.style_means, 'm-', label='Style Mean', linewidth=2)
        self.ax_embedding.set_title('Voice Embedding Stats')
        self.ax_embedding.set_xlabel('Epoch')
        self.ax_embedding.set_ylabel('Mean Value')
        self.ax_embedding.legend()
        self.ax_embedding.grid(True, alpha=0.3)
        
        # Learning rate
        if len(self.learning_rates) > 0:
            self.ax_lr.plot(epochs_array, self.learning_rates, 'orange', linewidth=2)
        self.ax_lr.set_title('Learning Rate')
        self.ax_lr.set_xlabel('Epoch')
        self.ax_lr.set_ylabel('LR')
        self.ax_lr.set_yscale('log')
        self.ax_lr.grid(True, alpha=0.3)
        
        # Smoothness loss
        if len(self.smoothness_losses) > 0:
            self.ax_smoothness.plot(epochs_array, self.smoothness_losses, 'c-', label='Smoothness Loss', linewidth=2)
        if len(self.batch_stft_losses) > 0:
            self.ax_smoothness.plot(epochs_array, self.batch_stft_losses, 'y-', label='STFT Loss', linewidth=2)
        self.ax_smoothness.set_title('Smoothness & STFT Loss')
        self.ax_smoothness.set_xlabel('Epoch')
        self.ax_smoothness.set_ylabel('Loss')
        self.ax_smoothness.legend()
        self.ax_smoothness.grid(True, alpha=0.3)
        
        # Current metrics display
        if current_metrics:
            metrics_text = "Current Metrics:\n\n"
            metrics_text += f"Epoch: {current_metrics.get('epoch', 'N/A')}\n"
            metrics_text += f"Train Loss: {current_metrics.get('train_loss', 'N/A'):.4f}\n" if 'train_loss' in current_metrics else ""
            metrics_text += f"Val Loss: {current_metrics.get('val_loss', 'N/A'):.4f}\n" if 'val_loss' in current_metrics else ""
            metrics_text += f"Learning Rate: {current_metrics.get('learning_rate', 'N/A'):.2e}\n" if 'learning_rate' in current_metrics else ""
            metrics_text += f"Timbre Mean: {current_metrics.get('timbre_mean', 'N/A'):.4f}\n" if 'timbre_mean' in current_metrics else ""
            
            self.ax_metrics.text(0.1, 0.5, metrics_text, fontsize=11, va='center')
        self.ax_metrics.axis('off')
        
        plt.tight_layout()
    
    def update_status(self, metrics):
        """Update status bar with training information"""
        self.ax_status.clear()
        
        if metrics:
            # Check if training is active
            if 'wall_time' in metrics:
                time_since_update = time.time() - self.last_update if self.last_update else 0
                self.training_active = time_since_update < 60  # Consider inactive after 60s
            
            status_color = 'green' if self.training_active else 'orange'
            status_text = 'Training Active' if self.training_active else 'Training Paused/Complete'
            
            # Add warnings if needed
            warnings = []
            if 'timbre_mean' in metrics and abs(metrics['timbre_mean']) > 0.35:
                warnings.append("⚠️  High timbre variance detected")
            if 'train_loss' in metrics and metrics['train_loss'] > 10:
                warnings.append("⚠️  High training loss")
                
            status_msg = f"Status: {status_text}"
            if warnings:
                status_msg += " | " + " | ".join(warnings)
                
            self.ax_status.text(0.5, 0.5, status_msg, ha='center', va='center', 
                              fontsize=12, color=status_color, weight='bold')
        else:
            self.ax_status.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                              fontsize=12, color='red')
        
        self.ax_status.axis('off')

def main():
    parser = argparse.ArgumentParser(description='Monitor Kokoro voice training in real-time')
    parser.add_argument('--log_dir', type=str, default='output/my_voice',
                      help='Directory containing training logs')
    parser.add_argument('--update_interval', type=int, default=5,
                      help='Update interval in seconds')
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory {log_dir} does not exist")
        return
    
    print("Starting training dashboard...")
    print(f"Monitoring: {log_dir}")
    print(f"Update interval: {args.update_interval} seconds")
    print("Press Ctrl+C to stop")
    
    # Create dashboard
    dashboard = TrainingDashboard(log_dir, args.update_interval)
    
    # Start animation
    ani = FuncAnimation(dashboard.fig, dashboard.update_data, 
                       interval=args.update_interval * 1000,
                       cache_frame_data=False)
    
    plt.show()

if __name__ == "__main__":
    main()
