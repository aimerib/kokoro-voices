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

class TrainingDashboard:
    def __init__(self, log_dir: Path, update_interval: int = 5):
        self.log_dir = Path(log_dir)
        self.update_interval = update_interval
        self.metrics_file = self.log_dir / "metrics.json"
        
        # Data storage with history
        self.history_length = 500
        self.epochs = deque(maxlen=self.history_length)
        self.train_losses = deque(maxlen=self.history_length)
        self.val_losses = deque(maxlen=self.history_length)
        self.learning_rates = deque(maxlen=self.history_length)
        self.timbre_means = deque(maxlen=self.history_length)
        self.style_means = deque(maxlen=self.history_length)
        self.smoothness_losses = deque(maxlen=self.history_length)
        
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
        self.training_start = None
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
    def setup_plots(self):
        """Initialize all subplot configurations"""
        # Loss plot
        self.ax_loss.set_title('Training Progress')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Embedding stats
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
        self.ax_smoothness.set_title('Smoothness Loss')
        self.ax_smoothness.set_xlabel('Epoch')
        self.ax_smoothness.set_ylabel('Loss')
        self.ax_smoothness.grid(True, alpha=0.3)
        
        # Key metrics text
        self.ax_metrics.set_title('Key Metrics')
        self.ax_metrics.axis('off')
        
        # Status bar
        self.ax_status.set_title('Training Status')
        self.ax_status.axis('off')
        
        plt.tight_layout()
        
    def load_metrics(self):
        """Load latest metrics from file"""
        if not self.metrics_file.exists():
            return False
            
        try:
            with open(self.metrics_file, 'r') as f:
                # Read all lines and parse the latest metrics
                lines = f.readlines()
                if not lines:
                    return False
                    
                # Process last few lines to get latest data
                for line in lines[-10:]:
                    try:
                        data = json.loads(line.strip())
                        
                        # Extract metrics
                        if 'epoch' in data:
                            epoch = data['epoch']
                            if not self.epochs or epoch > self.epochs[-1]:
                                self.epochs.append(epoch)
                                
                                # Training metrics
                                if 'train_loss' in data:
                                    self.train_losses.append(data['train_loss'])
                                if 'val_loss' in data:
                                    self.val_losses.append(data['val_loss'])
                                    if data['val_loss'] < self.best_val_loss:
                                        self.best_val_loss = data['val_loss']
                                        self.best_epoch = epoch
                                        
                                # Embedding stats
                                if 'timbre_mean' in data:
                                    self.timbre_means.append(data['timbre_mean'])
                                if 'style_mean' in data:
                                    self.style_means.append(data['style_mean'])
                                    
                                # Other metrics
                                if 'learning_rate' in data:
                                    self.learning_rates.append(data['learning_rate'])
                                if 'smoothness_loss' in data:
                                    self.smoothness_losses.append(data['smoothness_loss'])
                                    
                                self.last_update = datetime.now()
                                if self.training_start is None:
                                    self.training_start = self.last_update
                                    
                    except json.JSONDecodeError:
                        continue
                        
            return True
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return False
            
    def update_plots(self, frame):
        """Update all plots with latest data"""
        if not self.load_metrics():
            return
            
        # Clear all axes
        for ax in [self.ax_loss, self.ax_embedding, self.ax_lr, self.ax_smoothness]:
            ax.clear()
            
        if len(self.epochs) > 0:
            # Loss plot
            if self.train_losses:
                self.ax_loss.plot(list(self.epochs)[-len(self.train_losses):], 
                                 list(self.train_losses), 'b-', label='Train Loss', linewidth=2)
            if self.val_losses:
                self.ax_loss.plot(list(self.epochs)[-len(self.val_losses):], 
                                 list(self.val_losses), 'r-', label='Val Loss', linewidth=2)
                self.ax_loss.axhline(y=self.best_val_loss, color='g', linestyle='--', 
                                    label=f'Best Val ({self.best_val_loss:.4f})')
            self.ax_loss.set_title('Training Progress')
            self.ax_loss.set_xlabel('Epoch')
            self.ax_loss.set_ylabel('Loss')
            self.ax_loss.legend()
            self.ax_loss.grid(True, alpha=0.3)
            
            # Embedding stats
            if self.timbre_means:
                self.ax_embedding.plot(list(self.epochs)[-len(self.timbre_means):], 
                                      list(self.timbre_means), 'g-', label='Timbre Mean', linewidth=2)
            if self.style_means:
                self.ax_embedding.plot(list(self.epochs)[-len(self.style_means):], 
                                      list(self.style_means), 'm-', label='Style Mean', linewidth=2)
            self.ax_embedding.set_title('Voice Embedding Stats')
            self.ax_embedding.set_xlabel('Epoch')
            self.ax_embedding.set_ylabel('Mean Value')
            self.ax_embedding.legend()
            self.ax_embedding.grid(True, alpha=0.3)
            
            # Learning rate
            if self.learning_rates:
                self.ax_lr.semilogy(list(self.epochs)[-len(self.learning_rates):], 
                                   list(self.learning_rates), 'b-', linewidth=2)
            self.ax_lr.set_title('Learning Rate')
            self.ax_lr.set_xlabel('Epoch')
            self.ax_lr.set_ylabel('LR')
            self.ax_lr.grid(True, alpha=0.3)
            
            # Smoothness loss
            if self.smoothness_losses:
                self.ax_smoothness.plot(list(self.epochs)[-len(self.smoothness_losses):], 
                                       list(self.smoothness_losses), 'r-', linewidth=2)
                # Add warning threshold
                if max(self.smoothness_losses) > 0.1:
                    self.ax_smoothness.axhline(y=0.1, color='orange', linestyle='--', 
                                              label='Warning Threshold')
            self.ax_smoothness.set_title('Smoothness Loss')
            self.ax_smoothness.set_xlabel('Epoch')
            self.ax_smoothness.set_ylabel('Loss')
            self.ax_smoothness.grid(True, alpha=0.3)
            
            # Update metrics display
            self.update_metrics_display()
            
            # Update status
            self.update_status()
            
    def update_metrics_display(self):
        """Update the key metrics text display"""
        self.ax_metrics.clear()
        self.ax_metrics.set_title('Key Metrics')
        self.ax_metrics.axis('off')
        
        metrics_text = []
        
        if self.epochs:
            metrics_text.append(f"Current Epoch: {self.epochs[-1]}")
            
        if self.train_losses:
            metrics_text.append(f"Train Loss: {self.train_losses[-1]:.4f}")
            
        if self.val_losses:
            metrics_text.append(f"Val Loss: {self.val_losses[-1]:.4f}")
            
        metrics_text.append(f"Best Val Loss: {self.best_val_loss:.4f} (Epoch {self.best_epoch})")
        
        if self.learning_rates:
            metrics_text.append(f"Learning Rate: {self.learning_rates[-1]:.2e}")
            
        if self.smoothness_losses:
            smoothness = self.smoothness_losses[-1]
            metrics_text.append(f"Smoothness: {smoothness:.4f}")
            if smoothness > 0.1:
                metrics_text.append("⚠️ HIGH SMOOTHNESS LOSS")
                
        # Display metrics
        y_pos = 0.9
        for text in metrics_text:
            self.ax_metrics.text(0.1, y_pos, text, fontsize=12, 
                               transform=self.ax_metrics.transAxes)
            y_pos -= 0.15
            
    def update_status(self):
        """Update the status bar"""
        self.ax_status.clear()
        self.ax_status.set_title('Training Status')
        self.ax_status.axis('off')
        
        status_text = []
        
        # Time information
        if self.last_update:
            time_since_update = (datetime.now() - self.last_update).total_seconds()
            if time_since_update > 300:  # 5 minutes
                status_text.append(f"⚠️ No updates for {int(time_since_update/60)} minutes")
            else:
                status_text.append(f"✓ Last update: {self.last_update.strftime('%H:%M:%S')}")
                
        # Training duration
        if self.training_start:
            duration = datetime.now() - self.training_start
            status_text.append(f"Training duration: {str(duration).split('.')[0]}")
            
        # Estimated time remaining
        if len(self.epochs) > 1 and hasattr(self, 'total_epochs'):
            epochs_per_minute = len(self.epochs) / (duration.total_seconds() / 60)
            remaining_epochs = self.total_epochs - self.epochs[-1]
            eta_minutes = remaining_epochs / epochs_per_minute
            eta = datetime.now() + timedelta(minutes=eta_minutes)
            status_text.append(f"ETA: {eta.strftime('%H:%M:%S')} ({int(eta_minutes)} minutes)")
            
        # Display status
        status_line = " | ".join(status_text)
        self.ax_status.text(0.5, 0.5, status_line, fontsize=14, 
                           transform=self.ax_status.transAxes,
                           ha='center', va='center')
        
    def run(self):
        """Start the dashboard animation"""
        print(f"Starting training dashboard...")
        print(f"Monitoring: {self.log_dir}")
        print(f"Update interval: {self.update_interval} seconds")
        print("Press Ctrl+C to stop")
        
        # Create animation
        ani = FuncAnimation(self.fig, self.update_plots, interval=self.update_interval*1000,
                          blit=False, cache_frame_data=False)
        
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nDashboard stopped")
            

def main():
    parser = argparse.ArgumentParser(description='Kokoro Voice Training Dashboard')
    parser.add_argument('--log_dir', default='output/audiobook_voice', 
                       help='Directory containing training logs')
    parser.add_argument('--update_interval', type=int, default=5,
                       help='Update interval in seconds')
    
    args = parser.parse_args()
    
    dashboard = TrainingDashboard(args.log_dir, args.update_interval)
    dashboard.run()
    

if __name__ == '__main__':
    main()
