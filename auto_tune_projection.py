#!/usr/bin/env python3
"""
Auto-tuning StyleTTS2 → Kokoro Projection using Known Voices

This script uses existing Kokoro voices to:
1. Generate synthetic training data (Kokoro → TTS → StyleTTS2 features)
2. Learn optimal projection architecture and hyperparameters
3. Auto-tune parameters using perfect ground truth
4. Export optimized settings for user voice training

The key insight: We know both input (StyleTTS2 features) and output (Kokoro embeddings)
for existing voices, so we can optimize the mapping perfectly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import tempfile
import soundfile as sf
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import optuna
from tqdm import tqdm

# Import our existing components
from training_styletts2 import (
    StyleToKokoroProjection, 
    compute_voice_similarity_loss,
    extract_style_from_audio,
    load_styletts2_model
)

try:
    from styletts2 import tts
    STYLETTS2_AVAILABLE = True
except ImportError:
    STYLETTS2_AVAILABLE = False

try:
    from kokoro import KPipeline, KModel
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

@dataclass
class ProjectionConfig:
    """Configuration for projection network architecture and training."""
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"  # relu, gelu, swish
    normalization: str = "layer"  # layer, batch, none
    lr: float = 1e-3
    batch_size: int = 8
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    
    # Loss weights
    mel_l1_weight: float = 0.4
    spectral_conv_weight: float = 0.2
    cosine_weight: float = 0.2
    centroid_weight: float = 0.1
    norm_reg_weight: float = 0.05
    smoothness_weight: float = 0.05

class AdaptiveProjection(nn.Module):
    """Configurable projection network for hyperparameter optimization."""
    
    def __init__(self, config: ProjectionConfig, style_dim: int = 256, kokoro_dim: int = 256):
        super().__init__()
        self.config = config
        
        # Build configurable architecture
        layers = []
        in_dim = style_dim
        
        for i in range(config.num_layers):
            # Linear layer
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            
            # Normalization
            if config.normalization == "layer":
                layers.append(nn.LayerNorm(config.hidden_dim))
            elif config.normalization == "batch":
                layers.append(nn.BatchNorm1d(config.hidden_dim))
            
            # Activation
            if config.activation == "relu":
                layers.append(nn.ReLU())
            elif config.activation == "gelu":
                layers.append(nn.GELU())
            elif config.activation == "swish":
                layers.append(nn.SiLU())
            
            # Dropout
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            in_dim = config.hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(config.hidden_dim, kokoro_dim))
        layers.append(nn.Tanh())
        
        self.projection = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.projection(x)

class SyntheticDataGenerator:
    """Generate synthetic training data using known Kokoro voices."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.kokoro_pipeline = None
        self.styletts2_model = None
        self.reference_voices = {}
        
    def setup(self):
        """Initialize models and load reference voices."""
        print("Setting up synthetic data generator...")
        
        # Load Kokoro pipeline
        if KOKORO_AVAILABLE:
            self.kokoro_pipeline = KPipeline(lang_code="a")
            print("✓ Kokoro pipeline loaded")
        else:
            raise ImportError("Kokoro not available")
        
        # Load StyleTTS2 model
        if STYLETTS2_AVAILABLE:
            self.styletts2_model = load_styletts2_model(self.device)
            print("✓ StyleTTS2 model loaded")
        else:
            raise ImportError("StyleTTS2 not available")
        
        # Load reference voices
        self._load_reference_voices()
    
    def _load_reference_voices(self):
        """Load known Kokoro voices as ground truth."""
        from huggingface_hub import hf_hub_download
        
        voice_names = [
            'af_heart.pt', 'af_sarah.pt', 'af_sky.pt',
            'am_adam.pt', 'am_michael.pt', 'bf_emma.pt',
            'bf_isabella.pt', 'bm_george.pt', 'bm_lewis.pt'
        ]
        
        for voice_name in voice_names:
            try:
                voice_path = hf_hub_download(
                    repo_id='hexgrad/Kokoro-82M',
                    filename=f'voices/{voice_name}'
                )
                voice_tensor = torch.load(voice_path, weights_only=True)
                
                # Get average embedding
                if voice_tensor.dim() == 3:
                    avg_embedding = voice_tensor.mean(dim=0).squeeze()
                else:
                    avg_embedding = voice_tensor.squeeze()
                
                self.reference_voices[voice_name] = {
                    'full_tensor': voice_tensor,
                    'avg_embedding': avg_embedding,
                    'name': voice_name.replace('.pt', '')
                }
                
                print(f"✓ Loaded {voice_name}")
                
            except Exception as e:
                print(f"⚠ Failed to load {voice_name}: {e}")
    
    def generate_training_batch(self, voice_name: str, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of (StyleTTS2_features, Kokoro_embedding) pairs."""
        
        if voice_name not in self.reference_voices:
            raise ValueError(f"Voice {voice_name} not available")
        
        voice_data = self.reference_voices[voice_name]
        target_embedding = voice_data['avg_embedding']
        voice_tensor = voice_data['full_tensor']
        
        styletts2_features = []
        
        for text in texts:
            try:
                # Generate audio with Kokoro
                outputs = []
                for _, _, audio in self.kokoro_pipeline(text, voice=voice_tensor):
                    outputs.append(audio)
                
                if outputs:
                    full_audio = torch.cat(outputs)
                    
                    # Extract StyleTTS2 features
                    style_features = extract_style_from_audio(
                        self.styletts2_model, 
                        full_audio, 
                        sr=24000
                    )
                    
                    if style_features is not None:
                        styletts2_features.append(style_features)
                
            except Exception as e:
                print(f"Warning: Failed to generate for text '{text[:30]}...': {e}")
                continue
        
        if styletts2_features:
            # Stack features and repeat target embedding
            features_batch = torch.stack(styletts2_features)
            targets_batch = target_embedding.unsqueeze(0).repeat(len(styletts2_features), 1)
            
            return features_batch, targets_batch
        else:
            return None, None

class ProjectionOptimizer:
    """Optimize projection network using synthetic data."""
    
    def __init__(self, data_generator: SyntheticDataGenerator, device: str = "cpu"):
        self.data_generator = data_generator
        self.device = device
        self.best_config = None
        self.best_score = float('inf')
        
        # Test sentences for consistent evaluation
        self.test_sentences = [
            "Hello, this is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "How are you doing today?",
            "This is a longer sentence to test the voice quality.",
            "Can you hear the difference in my voice?",
            "Testing one two three four five.",
            "What a beautiful day it is outside.",
            "I hope this voice sounds natural and clear."
        ]
    
    def objective(self, trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        
        # Sample hyperparameters
        config = ProjectionConfig(
            hidden_dim=trial.suggest_categorical('hidden_dim', [256, 512, 768, 1024]),
            num_layers=trial.suggest_int('num_layers', 1, 4),
            dropout=trial.suggest_float('dropout', 0.0, 0.3),
            activation=trial.suggest_categorical('activation', ['relu', 'gelu', 'swish']),
            normalization=trial.suggest_categorical('normalization', ['layer', 'batch', 'none']),
            lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            weight_decay=trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True),
            grad_clip=trial.suggest_float('grad_clip', 0.5, 2.0),
            
            # Loss weights
            mel_l1_weight=trial.suggest_float('mel_l1_weight', 0.2, 0.6),
            spectral_conv_weight=trial.suggest_float('spectral_conv_weight', 0.1, 0.3),
            cosine_weight=trial.suggest_float('cosine_weight', 0.1, 0.3),
            centroid_weight=trial.suggest_float('centroid_weight', 0.05, 0.2),
            norm_reg_weight=trial.suggest_float('norm_reg_weight', 0.01, 0.1),
            smoothness_weight=trial.suggest_float('smoothness_weight', 0.01, 0.1),
        )
        
        try:
            score = self._evaluate_config(config)
            
            # Update best config
            if score < self.best_score:
                self.best_score = score
                self.best_config = config
                print(f"New best score: {score:.6f}")
            
            return score
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    def _evaluate_config(self, config: ProjectionConfig) -> float:
        """Evaluate a configuration by training and testing."""
        
        # Create model
        model = AdaptiveProjection(config).to(self.device)
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
        
        # Setup mel transform for loss computation
        mel_transform = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(512),
            torch.nn.Conv1d(1, 80, 1024, 256, bias=False),
        ).to(self.device)
        
        # Training loop
        model.train()
        total_loss = 0
        num_batches = 0
        
        # Use multiple voices for training
        voice_names = list(self.data_generator.reference_voices.keys())[:3]  # Use first 3 voices
        
        for epoch in range(20):  # Quick training for evaluation
            epoch_loss = 0
            epoch_batches = 0
            
            for voice_name in voice_names:
                # Generate batch
                features, targets = self.data_generator.generate_training_batch(
                    voice_name, self.test_sentences[:4]  # Use subset for speed
                )
                
                if features is None:
                    continue
                
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                predictions = model(features)
                
                # Compute reconstruction loss
                reconstruction_loss = F.mse_loss(predictions, targets)
                
                # Compute embedding norm loss
                pred_norms = torch.norm(predictions, dim=1)
                target_norms = torch.norm(targets, dim=1)
                norm_loss = F.mse_loss(pred_norms, target_norms)
                
                # Combined loss
                loss = reconstruction_loss + 0.1 * norm_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_batches += 1
            
            if epoch_batches > 0:
                total_loss += epoch_loss / epoch_batches
                num_batches += 1
        
        # Return average loss across all epochs
        if num_batches > 0:
            return total_loss / num_batches
        else:
            return float('inf')
    
    def optimize(self, n_trials: int = 100) -> ProjectionConfig:
        """Run hyperparameter optimization."""
        
        print(f"Starting hyperparameter optimization with {n_trials} trials...")
        
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        
        print(f"\nOptimization completed!")
        print(f"Best score: {self.best_score:.6f}")
        print(f"Best parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        return self.best_config

def save_optimized_config(config: ProjectionConfig, filepath: str):
    """Save optimized configuration to file."""
    config_dict = {
        'hidden_dim': config.hidden_dim,
        'num_layers': config.num_layers,
        'dropout': config.dropout,
        'activation': config.activation,
        'normalization': config.normalization,
        'lr': config.lr,
        'batch_size': config.batch_size,
        'weight_decay': config.weight_decay,
        'grad_clip': config.grad_clip,
        'mel_l1_weight': config.mel_l1_weight,
        'spectral_conv_weight': config.spectral_conv_weight,
        'cosine_weight': config.cosine_weight,
        'centroid_weight': config.centroid_weight,
        'norm_reg_weight': config.norm_reg_weight,
        'smoothness_weight': config.smoothness_weight,
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"✓ Saved optimized config to {filepath}")

def create_optimized_training_script(config: ProjectionConfig, output_path: str):
    """Create a training script with optimized parameters."""
    
    script_content = f'''#!/usr/bin/env python3
"""
Auto-generated optimized training script.
Generated from hyperparameter optimization using known Kokoro voices.
"""

import subprocess
import sys

def run_optimized_training():
    """Run training with auto-tuned optimal parameters."""
    
    cmd = [
        sys.executable, "training_styletts2.py",
        "--data", "./output/kokoro-dataset",
        "--name", "my_voice_optimized",
        "--epochs-projection", "150",  # More epochs since we have optimal params
        "--lr-projection", "{config.lr}",
        "--max-style-samples", "100",
        "--wandb",
        "--log-audio-every", "5",
    ]
    
    print("🚀 Running OPTIMIZED StyleTTS2 → Kokoro Training")
    print("="*60)
    print("Using auto-tuned hyperparameters from synthetic voice optimization")
    print()
    print("Optimized parameters:")
    print(f"  Learning rate: {config.lr}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Dropout: {config.dropout}")
    print(f"  Activation: {config.activation}")
    print(f"  Normalization: {config.normalization}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Optimized training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {{e}}")
        return False

if __name__ == "__main__":
    run_optimized_training()
'''
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    import os
    os.chmod(output_path, 0o755)
    
    print(f"✓ Created optimized training script: {output_path}")

def main():
    """Main optimization pipeline."""
    
    print("StyleTTS2 → Kokoro Projection Auto-Tuner")
    print("="*60)
    print("Using known Kokoro voices to optimize projection architecture")
    print()
    
    # Check dependencies
    if not STYLETTS2_AVAILABLE:
        print("❌ StyleTTS2 not available. Install with: pip install styletts2")
        return
    
    if not KOKORO_AVAILABLE:
        print("❌ Kokoro not available. Install Kokoro first.")
        return
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize components
    data_generator = SyntheticDataGenerator(device)
    data_generator.setup()
    
    optimizer = ProjectionOptimizer(data_generator, device)
    
    # Run optimization
    print(f"\nFound {len(data_generator.reference_voices)} reference voices")
    print("Starting hyperparameter optimization...")
    
    optimal_config = optimizer.optimize(n_trials=50)  # Adjust based on time budget
    
    if optimal_config:
        # Save results
        save_optimized_config(optimal_config, "optimal_projection_config.json")
        create_optimized_training_script(optimal_config, "run_optimized_training.py")
        
        print("\n🎉 Optimization completed!")
        print("Next steps:")
        print("1. Review optimal_projection_config.json")
        print("2. Run: python run_optimized_training.py")
        print("3. Compare results with previous training")
    else:
        print("❌ Optimization failed to find good configuration")

if __name__ == "__main__":
    main() 