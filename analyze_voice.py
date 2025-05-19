import torch
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
import argparse
import os

def load_voice(path, device="cpu"):
    """Load a voice tensor from file or HF hub"""
    if path.startswith("hf://"):
        # Format: hf://voice_name (without .pt)
        voice_name = path.replace("hf://", "")
        f = hf_hub_download(repo_id='hexgrad/Kokoro-82M', filename=f'voices/{voice_name}.pt')
        voice = torch.load(f, map_location=device)
    else:
        # Local file
        voice = torch.load(path, map_location=device)["voice"]
    return voice

def analyze_voice(voice_tensor):
    """Analyze statistical properties of a voice tensor"""
    # Convert to CPU and detach from graph if needed
    if hasattr(voice_tensor, 'detach'):
        voice_tensor = voice_tensor.detach().cpu().numpy()
    else:
        voice_tensor = voice_tensor.cpu().numpy()
        
    results = {}
    
    # Handle different tensor shapes
    if voice_tensor.ndim == 1:
        # 1D [256]
        timbre = voice_tensor[:128]
        style = voice_tensor[128:]
        shape = "1D [256]"
    elif voice_tensor.ndim == 2:
        if voice_tensor.shape[0] == 1:
            # 2D [1, 256]
            timbre = voice_tensor[0, :128]
            style = voice_tensor[0, 128:]
            shape = "2D [1, 256]"
        else:
            # 2D [N, 256] - take middle
            mid = voice_tensor.shape[0] // 2
            timbre = voice_tensor[mid, :128]
            style = voice_tensor[mid, 128:]
            shape = f"2D [{voice_tensor.shape[0]}, 256] (showing middle)"
    elif voice_tensor.ndim == 3:
        # 3D [N, 1, 256] - take middle
        mid = voice_tensor.shape[0] // 2
        timbre = voice_tensor[mid, 0, :128]
        style = voice_tensor[mid, 0, 128:]
        shape = f"3D [{voice_tensor.shape[0]}, 1, 256] (showing middle)"
    else:
        raise ValueError(f"Unexpected tensor shape: {voice_tensor.shape}")
    
    # Statistics
    results["shape"] = shape
    results["timbre_mean"] = float(np.mean(timbre))
    results["timbre_std"] = float(np.std(timbre))
    results["timbre_min"] = float(np.min(timbre))
    results["timbre_max"] = float(np.max(timbre))
    results["timbre_range"] = float(np.max(timbre) - np.min(timbre))
    
    results["style_mean"] = float(np.mean(style))
    results["style_std"] = float(np.std(style))
    results["style_min"] = float(np.min(style))
    results["style_max"] = float(np.max(style))
    results["style_range"] = float(np.max(style) - np.min(style))
    
    # Check for NaN/Inf values
    results["has_nan"] = bool(np.isnan(voice_tensor).any())
    results["has_inf"] = bool(np.isinf(voice_tensor).any())
    
    # Check if values are too small or too large
    results["extreme_values"] = bool(np.abs(voice_tensor).max() > 10)
    
    return results, timbre, style

def plot_comparison(voices, labels):
    """Plot comparison of voice tensors"""
    # Set up the figure with two rows (timbre and style)
    fig, axs = plt.subplots(2, len(voices), figsize=(len(voices)*5, 10))
    
    # Plot heatmaps for each voice
    for i, (_, timbre, style) in enumerate(voices):
        # Timbre plot
        if len(voices) > 1:
            ax = axs[0, i]
        else:
            ax = axs[0]
        im = ax.imshow(timbre.reshape(1, -1), aspect='auto', cmap='viridis')
        ax.set_title(f"{labels[i]} - Timbre")
        fig.colorbar(im, ax=ax, orientation='vertical')
        
        # Style plot
        if len(voices) > 1:
            ax = axs[1, i]
        else:
            ax = axs[1]
        im = ax.imshow(style.reshape(1, -1), aspect='auto', cmap='viridis')
        ax.set_title(f"{labels[i]} - Style")
        fig.colorbar(im, ax=ax, orientation='vertical')
    
    plt.tight_layout()
    plt.savefig('voice_comparison.png')
    print(f"Plot saved to {os.getcwd()}/voice_comparison.png")
    
    # Now create a distribution plot
    plt.figure(figsize=(12, 6))
    for i, (_, timbre, style) in enumerate(voices):
        plt.subplot(1, 2, 1)
        plt.hist(timbre, alpha=0.5, label=f"{labels[i]} timbre", bins=30)
        plt.subplot(1, 2, 2)
        plt.hist(style, alpha=0.5, label=f"{labels[i]} style", bins=30)
    
    plt.subplot(1, 2, 1)
    plt.title("Timbre Distribution")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.title("Style Distribution")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('voice_distribution.png')
    print(f"Distribution plot saved to {os.getcwd()}/voice_distribution.png")

def display_comparison(results_list, labels):
    """Display comparison table of voice tensor properties"""
    print("\n" + "="*80)
    print("VOICE TENSOR COMPARISON".center(80))
    print("="*80)
    
    headers = ["Property"] + labels
    row_format = "{:<20}" + "{:<20}" * len(labels)
    
    print(row_format.format(*headers))
    print("-"*20 * (len(labels) + 1))
    
    # Shape
    print(row_format.format("Shape", *[r["shape"] for r in results_list]))
    
    # Timbre stats
    print("\nTIMBRE STATISTICS:")
    print(row_format.format("Mean", *[f"{r['timbre_mean']:.6f}" for r in results_list]))
    print(row_format.format("Std Dev", *[f"{r['timbre_std']:.6f}" for r in results_list]))
    print(row_format.format("Min", *[f"{r['timbre_min']:.6f}" for r in results_list]))
    print(row_format.format("Max", *[f"{r['timbre_max']:.6f}" for r in results_list]))
    print(row_format.format("Range", *[f"{r['timbre_range']:.6f}" for r in results_list]))
    
    # Style stats
    print("\nSTYLE STATISTICS:")
    print(row_format.format("Mean", *[f"{r['style_mean']:.6f}" for r in results_list]))
    print(row_format.format("Std Dev", *[f"{r['style_std']:.6f}" for r in results_list]))
    print(row_format.format("Min", *[f"{r['style_min']:.6f}" for r in results_list]))
    print(row_format.format("Max", *[f"{r['style_max']:.6f}" for r in results_list]))
    print(row_format.format("Range", *[f"{r['style_range']:.6f}" for r in results_list]))
    
    # Issues
    print("\nPOTENTIAL ISSUES:")
    print(row_format.format("Has NaN values", *[str(r["has_nan"]) for r in results_list]))
    print(row_format.format("Has Inf values", *[str(r["has_inf"]) for r in results_list]))
    print(row_format.format("Has extreme values", *[str(r["extreme_values"]) for r in results_list]))
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    for i, r in enumerate(results_list):
        if r["has_nan"] or r["has_inf"]:
            print(f"- {labels[i]}: CRITICAL: Contains NaN/Inf values. Voice will not work properly.")
        elif r["extreme_values"]:
            print(f"- {labels[i]}: WARNING: Contains extreme values. Voice may sound distorted.")
        elif r["timbre_std"] < 0.01:
            print(f"- {labels[i]}: WARNING: Very low timbre variation. Voice may sound flat/noisy.")
        elif r["timbre_std"] > 1.0:
            print(f"- {labels[i]}: WARNING: Very high timbre variation. Voice may sound unstable.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and compare voice tensors")
    parser.add_argument("--voice1", required=True, help="Path to first voice tensor (or hf://voice_name)")
    parser.add_argument("--label1", default="Voice 1", help="Label for first voice")
    parser.add_argument("--voice2", help="Path to second voice tensor (optional)")
    parser.add_argument("--label2", default="Voice 2", help="Label for second voice")
    parser.add_argument("--voice3", help="Path to third voice tensor (optional)")
    parser.add_argument("--label3", default="Voice 3", help="Label for third voice")
    args = parser.parse_args()
    
    voices = []
    labels = []
    
    # Load voice 1
    print(f"Loading {args.label1} from {args.voice1}")
    voice1 = load_voice(args.voice1)
    results1, timbre1, style1 = analyze_voice(voice1)
    voices.append((results1, timbre1, style1))
    labels.append(args.label1)
    
    # Load voice 2 if provided
    if args.voice2:
        print(f"Loading {args.label2} from {args.voice2}")
        voice2 = load_voice(args.voice2)
        results2, timbre2, style2 = analyze_voice(voice2)
        voices.append((results2, timbre2, style2))
        labels.append(args.label2)
    
    # Load voice 3 if provided
    if args.voice3:
        print(f"Loading {args.label3} from {args.voice3}")
        voice3 = load_voice(args.voice3)
        results3, timbre3, style3 = analyze_voice(voice3)
        voices.append((results3, timbre3, style3))
        labels.append(args.label3)
    
    # Display comparison
    display_comparison([v[0] for v in voices], labels)
    
    # Create plots
    plot_comparison(voices, labels)
