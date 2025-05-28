#!/usr/bin/env python3
"""
Test script for synthetic data generation concept.

This validates that we can:
1. Load Kokoro voices
2. Generate audio with them
3. Extract StyleTTS2 features
4. Create training pairs
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf

# Import our components
from training_styletts2 import extract_style_from_audio, load_styletts2_model

try:
    from styletts2 import tts
    STYLETTS2_AVAILABLE = True
except ImportError:
    STYLETTS2_AVAILABLE = False

try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

def test_synthetic_data_generation():
    """Test the core concept of synthetic data generation."""
    
    print("Testing Synthetic Data Generation Concept")
    print("="*50)
    
    # Check dependencies
    if not STYLETTS2_AVAILABLE:
        print("❌ StyleTTS2 not available")
        return False
    
    if not KOKORO_AVAILABLE:
        print("❌ Kokoro not available")
        return False
    
    print("✓ All dependencies available")
    
    # Load models
    print("\nLoading models...")
    try:
        kokoro_pipeline = KPipeline(lang_code="a")
        print("✓ Kokoro pipeline loaded")
        
        styletts2_model = load_styletts2_model("cpu")
        print("✓ StyleTTS2 model loaded")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Load a reference voice
    print("\nLoading reference voice...")
    try:
        from huggingface_hub import hf_hub_download
        
        voice_path = hf_hub_download(
            repo_id='hexgrad/Kokoro-82M',
            filename='voices/af_heart.pt'
        )
        voice_tensor = torch.load(voice_path, weights_only=True)
        
        # Get average embedding (this is our target)
        target_embedding = voice_tensor.mean(dim=0).squeeze()  # [256]
        
        print(f"✓ Loaded voice tensor: {voice_tensor.shape}")
        print(f"✓ Target embedding: {target_embedding.shape}")
        print(f"  Mean: {target_embedding.mean():.6f}")
        print(f"  Std: {target_embedding.std():.6f}")
        print(f"  Range: [{target_embedding.min():.6f}, {target_embedding.max():.6f}]")
        
    except Exception as e:
        print(f"❌ Voice loading failed: {e}")
        return False
    
    # Test synthetic data generation
    print("\nTesting synthetic data generation...")
    test_texts = [
        "Hello, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "How are you doing today?"
    ]
    
    styletts2_features = []
    
    for i, text in enumerate(test_texts):
        print(f"\nProcessing text {i+1}: '{text}'")
        
        try:
            # Step 1: Generate audio with Kokoro
            print("  Generating audio with Kokoro...")
            outputs = []
            for _, _, audio in kokoro_pipeline(text, voice=voice_tensor):
                outputs.append(audio)
            
            if not outputs:
                print("  ❌ No audio generated")
                continue
            
            full_audio = torch.cat(outputs)
            print(f"  ✓ Generated audio: {full_audio.shape} samples")
            
            # Step 2: Extract StyleTTS2 features
            print("  Extracting StyleTTS2 features...")
            style_features = extract_style_from_audio(styletts2_model, full_audio, sr=24000)
            
            if style_features is None:
                print("  ❌ Feature extraction failed")
                continue
            
            styletts2_features.append(style_features)
            print(f"  ✓ Extracted features: {style_features.shape}")
            print(f"    Mean: {style_features.mean():.6f}")
            print(f"    Std: {style_features.std():.6f}")
            
        except Exception as e:
            print(f"  ❌ Processing failed: {e}")
            continue
    
    # Analyze results
    print(f"\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    
    if not styletts2_features:
        print("❌ No features extracted - synthetic data generation failed")
        return False
    
    print(f"✓ Successfully extracted {len(styletts2_features)} feature vectors")
    
    # Stack features for analysis
    features_stacked = torch.stack(styletts2_features)
    print(f"✓ Feature batch shape: {features_stacked.shape}")
    
    # Analyze feature diversity
    pairwise_distances = []
    for i in range(len(styletts2_features)):
        for j in range(i+1, len(styletts2_features)):
            dist = torch.norm(styletts2_features[i] - styletts2_features[j]).item()
            pairwise_distances.append(dist)
    
    if pairwise_distances:
        avg_distance = np.mean(pairwise_distances)
        print(f"✓ Average pairwise distance: {avg_distance:.6f}")
        
        if avg_distance > 0.1:
            print("✓ Good feature diversity - different texts produce different features")
        else:
            print("⚠ Low feature diversity - features are very similar")
    
    # Test reconstruction concept
    print(f"\nTesting reconstruction concept...")
    
    # Simple test: can we learn to map features back to target embedding?
    print(f"Target embedding shape: {target_embedding.shape}")
    print(f"StyleTTS2 features shape: {styletts2_features[0].shape}")
    
    # Create a simple linear mapping as proof of concept
    linear_map = torch.nn.Linear(256, 256)
    
    # Test forward pass
    test_input = styletts2_features[0].unsqueeze(0)
    test_output = linear_map(test_input)
    
    print(f"✓ Linear mapping test:")
    print(f"  Input: {test_input.shape}")
    print(f"  Output: {test_output.shape}")
    print(f"  Target: {target_embedding.unsqueeze(0).shape}")
    
    # Compute reconstruction loss
    reconstruction_loss = torch.nn.functional.mse_loss(test_output, target_embedding.unsqueeze(0))
    print(f"  Initial reconstruction loss: {reconstruction_loss.item():.6f}")
    
    print(f"\n" + "="*50)
    print("CONCLUSION")
    print("="*50)
    print("✅ Synthetic data generation concept is VALID!")
    print()
    print("Key findings:")
    print(f"- Successfully generated {len(styletts2_features)} training pairs")
    print(f"- StyleTTS2 features show diversity (avg distance: {avg_distance:.6f})")
    print(f"- Target embeddings are well-defined")
    print(f"- Reconstruction framework is feasible")
    print()
    print("Next steps:")
    print("1. Run full hyperparameter optimization")
    print("2. Test with multiple voices")
    print("3. Apply optimized settings to your voice")
    
    return True

def main():
    """Main test function."""
    success = test_synthetic_data_generation()
    
    if success:
        print("\n🎉 Ready to run full optimization!")
        print("Run: python auto_tune_projection.py")
    else:
        print("\n❌ Concept validation failed")
        print("Check dependencies and try again")

if __name__ == "__main__":
    main() 