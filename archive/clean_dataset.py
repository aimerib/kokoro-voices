"""
Clean and improve audio dataset quality for Kokoro voice training

This script automatically cleans up a dataset by:
1. Analyzing audio quality metrics (SNR, dynamic range, clipping, etc.)
2. Removing problematic samples (too noisy, clipped, artifacts)
3. Normalizing audio levels consistently
4. Detecting and removing low-frequency artifacts
5. Filtering out samples that are too short/long
6. Creating a cleaned dataset with quality reports
7. Analyzing phoneme distribution for Kokoro training compatibility

Usage:
    python clean_dataset.py --input ./raw_dataset --output ./cleaned_dataset [options]
"""

import argparse
import os
import torch
import torchaudio
import numpy as np
import json
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import scipy.signal
from collections import defaultdict, Counter
import traceback

from dotenv import load_dotenv
from huggingface_hub import HfApi, upload_folder

load_dotenv()

# Check if phonemizer is available
try:
    from phonemizer import phonemize
    from phonemizer.backend import EspeakBackend
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    import espeakng_loader

    # Set espeak-ng library path and espeak-ng-data
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    # Change data_path as needed when editing espeak-ng phonemes
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False
    print("⚠️  Phonemizer not available - install espeak-ng and phonemizer for phoneme analysis")


class AudioQualityAnalyzer:
    """Analyze and assess audio quality metrics"""
    
    def __init__(self, sample_rate: int = 24000):
        self.sr = sample_rate
        
    def analyze_audio(self, audio: torch.Tensor) -> Dict:
        """Comprehensive audio quality analysis"""
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
        
        # Basic statistics
        rms = np.sqrt(np.mean(audio_np ** 2))
        peak = np.max(np.abs(audio_np))
        
        # Dynamic range
        dynamic_range_db = 20 * np.log10(peak / (rms + 1e-8))
        
        # Signal-to-noise ratio estimation (using quiet regions)
        signal_power = np.mean(audio_np ** 2)
        # Estimate noise from the quietest 10% of frames
        frame_powers = []
        frame_size = int(0.025 * self.sr)  # 25ms frames
        for i in range(0, len(audio_np) - frame_size, frame_size):
            frame = audio_np[i:i + frame_size]
            frame_powers.append(np.mean(frame ** 2))
        
        noise_power = np.percentile(frame_powers, 10) + 1e-10
        snr_db = 10 * np.log10(signal_power / noise_power)
        
        # Clipping detection
        clipping_threshold = 0.95
        clipped_samples = np.sum(np.abs(audio_np) > clipping_threshold)
        clipping_ratio = clipped_samples / len(audio_np)
        
        # DC offset
        dc_offset = np.mean(audio_np)
        
        # Spectral analysis
        f, psd = scipy.signal.welch(audio_np, self.sr, nperseg=1024)
        
        # Low frequency energy (below 80 Hz - often artifacts/rumble)
        low_freq_mask = f < 80
        low_freq_energy = np.sum(psd[low_freq_mask])
        total_energy = np.sum(psd)
        low_freq_ratio = low_freq_energy / (total_energy + 1e-10)
        
        # High frequency rolloff (above 8kHz)
        high_freq_mask = f > 8000
        high_freq_energy = np.sum(psd[high_freq_mask])
        high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
        
        # Spectral centroid (brightness measure)
        spectral_centroid = np.sum(f * psd) / (np.sum(psd) + 1e-10)
        
        return {
            'duration': len(audio_np) / self.sr,
            'rms': float(rms),
            'peak': float(peak),
            'dynamic_range_db': float(dynamic_range_db),
            'snr_db': float(snr_db),
            'clipping_ratio': float(clipping_ratio),
            'dc_offset': float(dc_offset),
            'low_freq_ratio': float(low_freq_ratio),
            'high_freq_ratio': float(high_freq_ratio),
            'spectral_centroid': float(spectral_centroid)
        }
    
    def is_good_quality(self, metrics: Dict, thresholds: Dict) -> Tuple[bool, List[str]]:
        """Determine if audio meets quality thresholds"""
        issues = []
        
        # Check each quality metric
        if metrics['snr_db'] < thresholds['min_snr_db']:
            issues.append(f"Low SNR: {metrics['snr_db']:.1f}dB < {thresholds['min_snr_db']}dB")
            
        if metrics['clipping_ratio'] > thresholds['max_clipping_ratio']:
            issues.append(f"Clipping: {metrics['clipping_ratio']:.3f} > {thresholds['max_clipping_ratio']}")
            
        if abs(metrics['dc_offset']) > thresholds['max_dc_offset']:
            issues.append(f"DC offset: {abs(metrics['dc_offset']):.3f} > {thresholds['max_dc_offset']}")
            
        if metrics['low_freq_ratio'] > thresholds['max_low_freq_ratio']:
            issues.append(f"Low-freq artifacts: {metrics['low_freq_ratio']:.3f} > {thresholds['max_low_freq_ratio']}")
            
        if metrics['duration'] < thresholds['min_duration']:
            issues.append(f"Too short: {metrics['duration']:.1f}s < {thresholds['min_duration']}s")
            
        if metrics['duration'] > thresholds['max_duration']:
            issues.append(f"Too long: {metrics['duration']:.1f}s > {thresholds['max_duration']}s")
            
        if metrics['peak'] < thresholds['min_peak']:
            issues.append(f"Too quiet: peak {metrics['peak']:.3f} < {thresholds['min_peak']}")
        
        return len(issues) == 0, issues


class AudioCleaner:
    """Clean and process audio files"""
    
    def __init__(self, sample_rate: int = 24000):
        self.sr = sample_rate
        
    def remove_dc_offset(self, audio: torch.Tensor) -> torch.Tensor:
        """Remove DC offset"""
        return audio - torch.mean(audio)
    
    def normalize_loudness(self, audio: torch.Tensor, target_peak_db: float = -3.0) -> torch.Tensor:
        """Normalize to target peak level"""
        peak = torch.max(torch.abs(audio))
        if peak > 0:
            target_amp = 10 ** (target_peak_db / 20)
            return audio * (target_amp / peak)
        return audio
    
    def high_pass_filter(self, audio: torch.Tensor, cutoff_hz: float = 80.0) -> torch.Tensor:
        """Remove low frequency rumble/artifacts"""
        # Design high-pass filter
        nyquist = self.sr / 2
        normalized_cutoff = cutoff_hz / nyquist
        
        # Use scipy for filter design
        audio_np = audio.numpy()
        b, a = scipy.signal.butter(4, normalized_cutoff, btype='high')
        filtered = scipy.signal.filtfilt(b, a, audio_np)
        
        return torch.from_numpy(filtered.copy()).float()
    
    def trim_silence(self, audio: torch.Tensor, threshold_db: float = -40.0) -> torch.Tensor:
        """Trim silence from beginning and end"""
        # Convert to dB
        audio_db = 20 * torch.log10(torch.abs(audio) + 1e-8)
        
        # Find start and end of speech
        above_threshold = audio_db > threshold_db
        if not torch.any(above_threshold):
            return audio  # All silence, return as-is
            
        start_idx = torch.where(above_threshold)[0][0].item()
        end_idx = torch.where(above_threshold)[0][-1].item() + 1
        
        return audio[start_idx:end_idx]
    
    def clean_audio(self, audio: torch.Tensor, aggressive: bool = False) -> torch.Tensor:
        """Apply all cleaning steps"""
        # Remove DC offset
        audio = self.remove_dc_offset(audio)
        
        # High-pass filter to remove low-frequency artifacts
        audio = self.high_pass_filter(audio, cutoff_hz=80.0 if not aggressive else 100.0)
        
        # Trim silence
        audio = self.trim_silence(audio, threshold_db=-40.0 if not aggressive else -35.0)
        
        # Normalize loudness
        audio = self.normalize_loudness(audio, target_peak_db=-3.0)
        
        return audio


def load_metadata(dataset_dir: Path) -> List[Dict]:
    """Load metadata from dataset"""
    metadata_file = dataset_dir / "train" / "metadata.jsonl"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    metadata = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            metadata.append(json.loads(line.strip()))
    
    return metadata


def save_metadata(metadata: List[Dict], output_dir: Path):
    """Save cleaned metadata"""
    output_file = output_dir / "train" / "metadata.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def generate_quality_report(all_metrics: List[Dict], kept_metrics: List[Dict], 
                          rejected_files: Dict, output_dir: Path):
    """Generate comprehensive quality report"""
    report = {
        'summary': {
            'total_files': len(all_metrics),
            'kept_files': len(kept_metrics),
            'rejected_files': len(rejected_files),
            'rejection_rate': len(rejected_files) / len(all_metrics) if all_metrics else 0
        },
        'quality_stats': {
            'original': calculate_stats(all_metrics),
            'cleaned': calculate_stats(kept_metrics)
        },
        'rejection_reasons': analyze_rejection_reasons(rejected_files)
    }
    
    # Save detailed report
    with open(output_dir / "quality_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create visualizations
    create_quality_plots(all_metrics, kept_metrics, output_dir)
    
    return report


def calculate_stats(metrics_list: List[Dict]) -> Dict:
    """Calculate summary statistics"""
    if not metrics_list:
        return {}
    
    stats = {}
    for key in ['snr_db', 'dynamic_range_db', 'clipping_ratio', 'low_freq_ratio', 'duration']:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
    
    return stats


def analyze_rejection_reasons(rejected_files: Dict) -> Dict:
    """Analyze why files were rejected"""
    reason_counts = defaultdict(int)
    
    for file_path, issues in rejected_files.items():
        for issue in issues:
            # Extract the main reason
            if "Low SNR" in issue:
                reason_counts["low_snr"] += 1
            elif "Clipping" in issue:
                reason_counts["clipping"] += 1
            elif "DC offset" in issue:
                reason_counts["dc_offset"] += 1
            elif "Low-freq artifacts" in issue:
                reason_counts["low_freq_artifacts"] += 1
            elif "Too short" in issue:
                reason_counts["too_short"] += 1
            elif "Too long" in issue:
                reason_counts["too_long"] += 1
            elif "Too quiet" in issue:
                reason_counts["too_quiet"] += 1
    
    return dict(reason_counts)


def create_quality_plots(all_metrics: List[Dict], kept_metrics: List[Dict], output_dir: Path):
    """Create quality comparison plots"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics_to_plot = ['snr_db', 'dynamic_range_db', 'clipping_ratio', 
                      'low_freq_ratio', 'duration', 'spectral_centroid']
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // 3, i % 3]
        
        all_values = [m[metric] for m in all_metrics if metric in m]
        kept_values = [m[metric] for m in kept_metrics if metric in m]
        
        if all_values and kept_values:
            ax.hist(all_values, bins=30, alpha=0.7, label='Original', color='red')
            ax.hist(kept_values, bins=30, alpha=0.7, label='Cleaned', color='green')
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Count')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "quality_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def upload_to_hf(dataset_dir: Path, repo_id: str, private: bool = True, 
                exists_ok: bool = False, report: Dict = None, phoneme_analysis: dict = None):
    """Upload cleaned dataset to HuggingFace Hub"""
    try:
        from huggingface_hub import HfApi, upload_folder
    except ImportError:
        print("⚠️  HuggingFace Hub not available - install huggingface_hub")
        return
    
    api = HfApi()
    
    # Create repo if needed
    if not api.repo_exists(repo_id, repo_type="dataset"):
        api.create_repo(
            repo_id,
            private=private,
            exist_ok=exists_ok,
            repo_type="dataset",
        )
    
    # Write README with report and phoneme analysis
    write_dataset_card(dataset_dir, report, repo_id, phoneme_analysis)
    
    print(f"Uploading cleaned dataset {dataset_dir} → hf://datasets/{repo_id}")
    upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(dataset_dir),
        commit_message="Add cleaned dataset with quality improvements",
    )
    print("✓ Upload complete")


def write_dataset_card(dataset_dir: Path, report: Dict = None, repo_id: str = None, phoneme_analysis: dict = None):
    """Write README with dataset info, quality metrics, and phoneme analysis"""
    
    # Count files for metadata
    train_dir = dataset_dir / "train"
    validation_dir = dataset_dir / "validation"
    test_dir = dataset_dir / "test"
    
    train_count = len(list(train_dir.glob("*.wav"))) if train_dir.exists() else 0
    val_count = len(list(validation_dir.glob("*.wav"))) if validation_dir.exists() else 0
    test_count = len(list(test_dir.glob("*.wav"))) if test_dir.exists() else 0
    total_count = train_count + val_count + test_count
    
    # Calculate total duration if report available
    total_duration_hours = 0
    if report and 'quality_stats' in report and 'cleaned' in report['quality_stats']:
        stats = report['quality_stats']['cleaned']
        if 'duration' in stats and 'mean' in stats['duration']:
            avg_duration = stats['duration']['mean']
            total_duration_hours = (total_count * avg_duration) / 3600
    
    # YAML frontmatter with proper HuggingFace dataset metadata
    readme_content = f"""---
dataset_info:
  features:
  - name: file_name
    dtype: string
  - name: text
    dtype: string
  - name: split
    dtype: string
  configs:
  - config_name: default
    data_files:
    - split: train
      path: "train/*"
    - split: validation
      path: "validation/*"
    - split: test
      path: "test/*"
    default: true
task_categories:
- text-to-speech
language:
- en
tags:
- kokoro-tts
- voice-cloning
- speech-synthesis
- audio-dataset
- cleaned-dataset
size_categories:
- {get_size_category(total_count)}
license: cc-by-nc-nd-4.0
multilinguality:
- monolingual
pretty_name: "Cleaned Voice Dataset for Kokoro TTS"
dataset_summary: "High-quality voice dataset automatically cleaned and optimized for Kokoro TTS training"
---

# Cleaned Voice Dataset

This dataset has been automatically cleaned and optimized for Kokoro TTS voice training.

## Dataset Statistics

- **Total audio files**: {total_count:,}
  - Training: {train_count:,}
  - Validation: {val_count:,}
  - Test: {test_count:,}
- **Estimated total duration**: {total_duration_hours:.1f} hours
- **Language**: English
- **Sample rate**: 24 kHz
- **Audio format**: WAV (mono)

## Quality Improvements

"""
    
    if report:
        summary = report['summary']
        readme_content += f"""
### Processing Summary
- **Original files**: {summary['total_files']:,}
- **Kept files**: {summary['kept_files']:,} 
- **Rejected files**: {summary['rejected_files']:,}
- **Rejection rate**: {summary['rejection_rate']:.1%}

### Quality Metrics (Cleaned Dataset)
"""
        if 'cleaned' in report['quality_stats']:
            stats = report['quality_stats']['cleaned']
            for metric, values in stats.items():
                if values:
                    readme_content += f"- **{metric.replace('_', ' ').title()}**: {values['mean']:.2f} ± {values['std']:.2f}\n"
        
        if 'rejection_reasons' in report:
            readme_content += "\n### Rejection Reasons\n"
            for reason, count in report['rejection_reasons'].items():
                readme_content += f"- {reason.replace('_', ' ').title()}: {count} files\n"
    
    # Add phoneme analysis if available
    if phoneme_analysis and phoneme_analysis.get('total_phonemes', 0) > 0:
        readme_content += f"""
## Phoneme Distribution Analysis

This dataset contains **{phoneme_analysis['total_phonemes']:,} total phonemes** with **{phoneme_analysis['unique_phonemes']} unique phonemes**.

**Phoneme Coverage for Kokoro Training:**
- 🗣️ Vowels: {phoneme_analysis['coverage']['vowels']:,} ({phoneme_analysis['coverage']['vowel_percentage']:.1f}%)
- 🗨️ Consonants: {phoneme_analysis['coverage']['consonants']:,} ({phoneme_analysis['coverage']['consonant_percentage']:.1f}%)

**Top 10 Most Common Phonemes:**
"""
        
        for i, (phoneme, percentage) in enumerate(phoneme_analysis['top_10_phonemes'], 1):
            count = phoneme_analysis['phoneme_counts'][phoneme]
            readme_content += f"{i:2d}. `/{phoneme}/` - {percentage:5.2f}% ({count:,} occurrences)\n"
        
        if phoneme_analysis.get('rare_phonemes'):
            rare_count = len(phoneme_analysis['rare_phonemes'])
            readme_content += f"\n**⚠️ Training Considerations:**\n"
            readme_content += f"- **{rare_count} rare phonemes** (<0.5% frequency) detected\n"
            readme_content += f"- Consider adding more diverse content for comprehensive phoneme coverage\n"
            readme_content += f"- Kokoro voice embeddings train on phoneme distributions - ensure your target use case matches this distribution\n"
        
        readme_content += f"\n*Full phoneme analysis available in `phoneme_analysis.json`*\n"
    
    else:
        readme_content += "\n*Note: Phoneme analysis not available (requires espeak-ng and phonemizer)*\n"
    
    readme_content += f"""
## Cleaning Process

The following automated cleaning steps were applied:

1. **Quality Analysis**: SNR, dynamic range, clipping detection, spectral analysis
2. **DC Offset Removal**: Centered audio around zero
3. **High-pass Filtering**: Removed low-frequency artifacts and rumble (>80Hz)
4. **Silence Trimming**: Removed leading/trailing silence
5. **Loudness Normalization**: Consistent peak levels (-3dB)
6. **Quality Filtering**: Removed problematic samples based on:
   - Signal-to-noise ratio (>15dB)
   - Clipping detection (<1%)
   - Low-frequency artifacts (<10%)
   - Duration limits (0.5-15 seconds)
7. **Phoneme Analysis**: Distribution analysis for Kokoro TTS compatibility

## Dataset Format

This dataset follows the format:

```
dataset/
├── train/
│   ├── metadata.jsonl
│   └── *.wav
├── validation/
│   ├── metadata.jsonl  
│   └── *.wav
└── test/
    ├── metadata.jsonl
    └── *.wav
```

Each `metadata.jsonl` contains lines like:
```json
{{"file_name": "segment_0001.wav", "text": "Hello world.", "split": "train"}}
```

## License

This dataset is released under CC-BY-NC-ND-4.0 license. Please provide appropriate attribution when using this dataset.

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{cleaned_voice_dataset,
  title={{Cleaned Voice Dataset for Kokoro TTS}},
  author={{Auto-generated by clean_dataset.py}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/{repo_id or 'username/dataset-name'}}}
}}
```

*Dataset automatically cleaned and analyzed with `clean_dataset.py`*
"""
    
    # Write README
    readme_path = dataset_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📝 Dataset card written to: {readme_path}")


def get_size_category(file_count: int) -> str:
    """Get HuggingFace size category based on file count"""
    if file_count < 1000:
        return "n<1K"
    elif file_count < 10000:
        return "1K<n<10K"
    elif file_count < 100000:
        return "10K<n<100K"
    elif file_count < 1000000:
        return "100K<n<1M"
    else:
        return "n>1M"


def analyze_phoneme_distribution(texts: list) -> dict:
    """Comprehensive phoneme distribution analysis for cleaned dataset"""
    if not PHONEMIZER_AVAILABLE:
        print("⚠️  Phonemizer not available - install espeak-ng and phonemizer for phoneme analysis")
        return {}
    
    try:
        print("🔍 Analyzing phoneme distribution...")
        
        # Get phonemes for all texts
        all_phonemes = []
        for text in tqdm(texts, desc="Processing phonemes"):
            try:
                backend = EspeakBackend(
                    language='en-us', preserve_punctuation=True, with_stress=True,
                    tie='^', language_switch='remove-flags'
                )
                phones = backend.phonemize([text])[0].strip()
                all_phonemes.extend(phones.split())
            except Exception as e:
                print(f"⚠️  Failed to phonemize: '{text[:50]}...' - {e}")
                continue
        
        if not all_phonemes:
            print("⚠️  No phonemes extracted")
            return {}
        
        # Count phonemes
        phoneme_counts = Counter(all_phonemes)
        total_phonemes = sum(phoneme_counts.values())
        
        # Calculate percentages
        phoneme_percentages = {
            phoneme: (count / total_phonemes) * 100 
            for phoneme, count in phoneme_counts.items()
        }
        
        # Sort by frequency
        sorted_phonemes = sorted(phoneme_percentages.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize phonemes
        vowels = ['i', 'ɪ', 'e', 'ɛ', 'æ', 'ɑ', 'ɔ', 'o', 'ʊ', 'u', 'ʌ', 'ə', 'ɚ', 'ɛɹ', 'aɪ', 'aʊ', 'ɔɪ']
        consonants = ['p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h', 'm', 'n', 'ŋ', 'l', 'ɹ', 'w', 'j']
        
        vowel_count = sum(phoneme_counts.get(v, 0) for v in vowels)
        consonant_count = sum(phoneme_counts.get(c, 0) for c in consonants)
        other_count = total_phonemes - vowel_count - consonant_count
        
        # Create analysis report
        analysis = {
            'total_phonemes': total_phonemes,
            'unique_phonemes': len(phoneme_counts),
            'phoneme_counts': phoneme_counts,
            'phoneme_percentages': phoneme_percentages,
            'sorted_phonemes': sorted_phonemes,
            'coverage': {
                'vowels': vowel_count,
                'consonants': consonant_count,
                'other': other_count,
                'vowel_percentage': (vowel_count / total_phonemes) * 100,
                'consonant_percentage': (consonant_count / total_phonemes) * 100
            },
            'top_10_phonemes': sorted_phonemes[:10],
            'rare_phonemes': [p for p, pct in sorted_phonemes if pct < 0.5]  # Less than 0.5%
        }
        
        return analysis
        
    except Exception as e:
        print(f"⚠️  Phoneme analysis failed: {e}")
        traceback.print_exc()
        return {}


def print_phoneme_analysis(analysis: dict):
    """Print formatted phoneme distribution analysis"""
    if not analysis:
        return
        
    print("\n" + "="*60)
    print("📊 PHONEME DISTRIBUTION ANALYSIS")
    print("="*60)
    
    print(f"📈 Total phonemes: {analysis['total_phonemes']:,}")
    print(f"🔤 Unique phonemes: {analysis['unique_phonemes']}")
    print(f"🗣️  Vowels: {analysis['coverage']['vowels']:,} ({analysis['coverage']['vowel_percentage']:.1f}%)")
    print(f"🗨️  Consonants: {analysis['coverage']['consonants']:,} ({analysis['coverage']['consonant_percentage']:.1f}%)")
    
    print(f"\n🔥 Top 10 Most Common Phonemes:")
    for i, (phoneme, percentage) in enumerate(analysis['top_10_phonemes'], 1):
        count = analysis['phoneme_counts'][phoneme]
        print(f"   {i:2d}. /{phoneme}/ - {percentage:5.2f}% ({count:,} occurrences)")
    
    if analysis['rare_phonemes']:
        print(f"\n⚠️  Rare phonemes (<0.5%): {len(analysis['rare_phonemes'])} phonemes")
        rare_list = ', '.join([f"/{p}/" for p, _ in analysis['rare_phonemes'][:10]])
        if len(analysis['rare_phonemes']) > 10:
            rare_list += f" ... and {len(analysis['rare_phonemes']) - 10} more"
        print(f"   {rare_list}")
    
    print("="*60)


def save_phoneme_analysis(analysis: dict, output_dir: Path):
    """Save phoneme analysis to JSON file"""
    if not analysis:
        return
        
    # Convert Counter objects to regular dicts for JSON serialization
    json_analysis = {
        'total_phonemes': analysis['total_phonemes'],
        'unique_phonemes': analysis['unique_phonemes'],
        'phoneme_counts': dict(analysis['phoneme_counts']),
        'phoneme_percentages': analysis['phoneme_percentages'],
        'sorted_phonemes': analysis['sorted_phonemes'],
        'coverage': analysis['coverage'],
        'top_10_phonemes': analysis['top_10_phonemes'],
        'rare_phonemes': analysis['rare_phonemes']
    }
    
    phoneme_file = output_dir / 'phoneme_analysis.json'
    with open(phoneme_file, 'w', encoding='utf-8') as f:
        json.dump(json_analysis, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Phoneme analysis saved to: {phoneme_file}")


def clean_dataset(input_dir: Path, output_dir: Path, thresholds: Dict, 
                 aggressive: bool = False, dry_run: bool = False):
    """Main dataset cleaning function"""
    
    print(f"🧹 Cleaning dataset: {input_dir} → {output_dir}")
    
    # Initialize components
    analyzer = AudioQualityAnalyzer()
    cleaner = AudioCleaner()
    
    # Load metadata
    metadata = load_metadata(input_dir)
    print(f"Found {len(metadata)} files in metadata")
    
    # Process each file
    cleaned_metadata = []
    all_metrics = []
    kept_metrics = []
    rejected_files = {}
    
    train_dir = input_dir / "train"
    output_train_dir = output_dir / "train"
    
    if not dry_run:
        output_train_dir.mkdir(parents=True, exist_ok=True)
    
    for item in tqdm(metadata, desc="Processing audio files"):
        file_path = train_dir / item['file_name']
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        try:
            # Load audio
            audio, sr = torchaudio.load(file_path)
            if audio.shape[0] > 1:  # Convert to mono
                audio = torch.mean(audio, dim=0)
            else:
                audio = audio[0]
            
            # Resample if needed
            if sr != 24000:
                audio = torchaudio.functional.resample(audio, sr, 24000)
            
            # Analyze quality
            metrics = analyzer.analyze_audio(audio)
            all_metrics.append(metrics)
            
            # Check if meets quality thresholds
            is_good, issues = analyzer.is_good_quality(metrics, thresholds)
            
            if is_good:
                # Clean the audio
                cleaned_audio = cleaner.clean_audio(audio, aggressive=aggressive)
                
                # Re-analyze after cleaning
                final_metrics = analyzer.analyze_audio(cleaned_audio)
                kept_metrics.append(final_metrics)
                
                if not dry_run:
                    # Save cleaned audio
                    output_file = output_train_dir / item['file_name']
                    torchaudio.save(output_file, cleaned_audio.unsqueeze(0), 24000)
                
                cleaned_metadata.append(item)
            else:
                rejected_files[str(file_path)] = issues
                print(f"❌ Rejected {file_path.name}: {'; '.join(issues)}")
                
        except Exception as e:
            traceback.print_exc()
            print(f"❌ Error processing {file_path}: {e}")
            rejected_files[str(file_path)] = [f"Processing error: {e}"]
    
    # Save results
    if not dry_run:
        save_metadata(cleaned_metadata, output_dir)
        
        # Copy validation/test sets if they exist
        for split in ['validation', 'test']:
            split_dir = input_dir / split
            if split_dir.exists():
                shutil.copytree(split_dir, output_dir / split, dirs_exist_ok=True)
    
    # Generate report
    report = generate_quality_report(all_metrics, kept_metrics, rejected_files, 
                                   output_dir if not dry_run else Path.cwd())
    
    # Analyze phoneme distribution
    all_texts = [item['text'] for item in metadata]
    phoneme_analysis = analyze_phoneme_distribution(all_texts)
    print_phoneme_analysis(phoneme_analysis)
    save_phoneme_analysis(phoneme_analysis, output_dir if not dry_run else Path.cwd())
    
    # Print summary
    print(f"\n📊 Cleaning Summary:")
    print(f"   Original files: {len(all_metrics)}")
    print(f"   Kept files: {len(kept_metrics)}")
    print(f"   Rejected files: {len(rejected_files)}")
    print(f"   Rejection rate: {len(rejected_files)/len(all_metrics):.1%}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Clean audio dataset for Kokoro training")
    
    # I/O arguments
    parser.add_argument("--input", type=str, required=True, 
                       help="Input dataset directory")
    parser.add_argument("--output", type=str, required=True,
                       help="Output cleaned dataset directory")
    
    # Quality thresholds
    quality_group = parser.add_argument_group('Quality Thresholds')
    quality_group.add_argument("--min-snr-db", type=float, default=15.0,
                              help="Minimum SNR in dB (default: 15.0)")
    quality_group.add_argument("--max-clipping-ratio", type=float, default=0.01,
                              help="Maximum clipping ratio (default: 0.01)")
    quality_group.add_argument("--max-dc-offset", type=float, default=0.05,
                              help="Maximum DC offset (default: 0.05)")
    quality_group.add_argument("--max-low-freq-ratio", type=float, default=0.1,
                              help="Maximum low-frequency energy ratio (default: 0.1)")
    quality_group.add_argument("--min-duration", type=float, default=0.5,
                              help="Minimum duration in seconds (default: 0.5)")
    quality_group.add_argument("--max-duration", type=float, default=15.0,
                              help="Maximum duration in seconds (default: 15.0)")
    quality_group.add_argument("--min-peak", type=float, default=0.01,
                              help="Minimum peak amplitude (default: 0.01)")
    
    # Processing options
    parser.add_argument("--aggressive", action="store_true",
                       help="Use more aggressive cleaning (stricter thresholds)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Analyze quality without making changes")
    
    # HuggingFace upload options
    hf_group = parser.add_argument_group('HuggingFace Upload')
    hf_group.add_argument("--upload-hf", action="store_true",
                         help="Upload cleaned dataset to HuggingFace Hub")
    hf_group.add_argument("--hf-repo", type=str,
                         help="HuggingFace repository ID (e.g., 'username/dataset-name')")
    hf_group.add_argument("--hf-private", action="store_true", default=True,
                         help="Create private repository (default: True)")
    hf_group.add_argument("--hf-exists-ok", action="store_true",
                         help="Allow overwriting existing repository")
    
    args = parser.parse_args()
    
    # Validate arguments
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if args.upload_hf and not args.hf_repo:
        raise ValueError("--upload-hf requires --hf-repo")
    
    # Set quality thresholds
    thresholds = {
        'min_snr_db': args.min_snr_db,
        'max_clipping_ratio': args.max_clipping_ratio,
        'max_dc_offset': args.max_dc_offset,
        'max_low_freq_ratio': args.max_low_freq_ratio,
        'min_duration': args.min_duration,
        'max_duration': args.max_duration,
        'min_peak': args.min_peak
    }
    
    # Adjust thresholds for aggressive mode
    if args.aggressive:
        thresholds['min_snr_db'] += 5.0
        thresholds['max_clipping_ratio'] /= 2
        thresholds['max_low_freq_ratio'] /= 2
        print("🔥 Aggressive mode: Using stricter quality thresholds")
    
    # Clean dataset
    report = clean_dataset(input_dir, output_dir, thresholds, 
                          aggressive=args.aggressive, dry_run=args.dry_run)
    
    # Upload to HuggingFace if requested
    if args.upload_hf and not args.dry_run:
        print("\n🚀 Uploading to HuggingFace...")
        
        # Run phoneme analysis for HF upload
        output_metadata_path = output_dir / "metadata.jsonl"
        if output_metadata_path.exists():
            print("🔬 Analyzing phonemes for HuggingFace README...")
            texts = []
            with open(output_metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    texts.append(data['text'])
            
            phoneme_analysis = analyze_phoneme_distribution(texts)
            print_phoneme_analysis(phoneme_analysis)
            
            # Save phoneme analysis
            phoneme_report_path = output_dir / "phoneme_analysis.json"
            with open(phoneme_report_path, 'w', encoding='utf-8') as f:
                json.dump(phoneme_analysis, f, indent=2, default=str)
            print(f"📊 Phoneme analysis saved: {phoneme_report_path}")
        else:
            phoneme_analysis = None
            print("⚠️  No metadata found - skipping phoneme analysis")
        
        upload_to_hf(
            dataset_dir=output_dir,
            repo_id=args.hf_repo,
            private=args.hf_private,
            report=report,
            phoneme_analysis=phoneme_analysis
        )


if __name__ == "__main__":
    main()
