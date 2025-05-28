# StyleTTS2 → Kokoro Voice Embedding Pipeline

This pipeline provides a more practical approach to creating Kokoro voice embeddings by leveraging StyleTTS2's proven voice cloning capabilities and projecting them to Kokoro's embedding space.

## Why This Approach?

The original Kokoro training approach has limitations:
- **Overfitted embedding space**: Kokoro's voice embeddings are highly specialized and difficult to optimize directly
- **Poor convergence**: Direct optimization often fails to converge to a usable voice
- **Limited generalization**: The embedding space doesn't generalize well to new voices

The StyleTTS2 → Kokoro pipeline solves these issues:
- **Proven voice cloning**: StyleTTS2 has demonstrated excellent voice cloning capabilities
- **Better embedding space**: StyleTTS2's style embeddings are more generalizable
- **Projection learning**: We learn a mapping from StyleTTS2 to Kokoro space
- **Faster training**: No need to fine-tune large models, just extract and project

## Pipeline Overview

```
Target Voice Data → StyleTTS2 Style Extraction → Projection Training → Kokoro Voice Embedding
```

### Stage 1: StyleTTS2 Style Extraction
- Load target voice samples from your dataset
- Use StyleTTS2 to extract style embeddings from each sample
- Average embeddings to create a representative style vector

### Stage 2: Kokoro Projection Training
- Train a neural network to map StyleTTS2 embeddings to Kokoro format
- Use acoustic similarity loss between generated and target audio
- Optimize for mel-spectrogram matching

### Stage 3: Voice Generation
- Generate final Kokoro-compatible voice embedding
- Expand to length-dependent format (510 × 1 × 256)
- Test with Kokoro TTS generation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# The pipeline requires:
# - styletts2>=0.1.6 (for style extraction)
# - kokoro (for final voice generation)
# - torch, torchaudio (for neural networks)
# - accelerate (for training optimization)
```

## Dataset Format

Your dataset should follow this structure:

```
dataset/
├── train/
│   ├── metadata.jsonl
│   ├── segment_0001.wav
│   ├── segment_0002.wav
│   └── ...
└── validation/  (optional)
    ├── metadata.jsonl
    ├── segment_0001.wav
    └── ...
```

**metadata.jsonl format:**
```json
{"file_name": "segment_0001.wav", "text": "Hello, this is the first sentence."}
{"file_name": "segment_0002.wav", "text": "This is another sentence to train on."}
```

## Usage

### Basic Training

```bash
python training_styletts2.py \
    --data ./my_dataset \
    --name my_voice \
    --out ./output \
    --epochs-projection 100
```

### Advanced Options

```bash
python training_styletts2.py \
    --data ./my_dataset \
    --name my_voice \
    --out ./output \
    --epochs-projection 200 \
    --lr-projection 1e-3 \
    --max-style-samples 150 \
    --device cuda \
    --tensorboard \
    --wandb \
    --wandb-project "my-voice-project"
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | Required | Path to dataset directory |
| `--name` | `my_voice` | Name for the output voice |
| `--out` | `output` | Output directory |
| `--epochs-projection` | `100` | Training epochs for projection layer |
| `--lr-projection` | `1e-3` | Learning rate for projection training |
| `--max-style-samples` | `100` | Max samples for style extraction |
| `--device` | `auto` | Device (auto/cuda/mps/cpu) |
| `--tensorboard` | `False` | Enable TensorBoard logging |
| `--wandb` | `False` | Enable Weights & Biases logging |

## Output Files

After training, you'll get:

```
output/my_voice/
├── my_voice.pt                 # Final Kokoro voice embedding
├── my_voice_artifacts.pt       # Training artifacts and metadata
├── my_voice_test_1.wav        # Test generation samples
├── my_voice_test_2.wav
├── my_voice_test_3.wav
└── checkpoints/               # Training checkpoints
```

## Using Your Voice

Once trained, use your voice with Kokoro:

```python
import torch
from kokoro import KPipeline

# Load your trained voice
voice = torch.load("output/my_voice/my_voice.pt")

# Initialize Kokoro pipeline
pipeline = KPipeline(lang_code="a")

# Generate speech with your voice
text = "Hello, this is my custom voice!"
outputs = []
for _, _, audio in pipeline(text, voice=voice):
    outputs.append(audio)

# Save the result
if outputs:
    full_audio = torch.cat(outputs)
    import soundfile as sf
    sf.write("my_speech.wav", full_audio.numpy(), 24000)
```

## Technical Details

### StyleTTS2 Style Extraction

The pipeline uses StyleTTS2's inference capabilities to extract style embeddings:

1. **Audio Processing**: Target audio is processed to 24kHz mono
2. **Style Computation**: StyleTTS2 computes style vectors from audio samples
3. **Embedding Averaging**: Multiple samples are averaged for robustness

### Projection Network Architecture

The projection layer maps StyleTTS2 embeddings to Kokoro format:

```
StyleTTS2 Embedding (256D) 
    ↓
Linear(256 → 512) + LayerNorm + ReLU + Dropout
    ↓  
Linear(512 → 512) + LayerNorm + ReLU + Dropout
    ↓
Linear(512 → 256) + Tanh
    ↓
Kokoro Embedding (256D)
```

### Loss Function

The training uses acoustic similarity loss:
- **Mel-spectrogram MSE**: Compares generated vs target mel spectrograms
- **Length Normalization**: Handles variable-length audio sequences
- **Gradient Clipping**: Prevents training instability

### Length-Dependent Embeddings

Kokoro uses length-dependent voice embeddings (510 × 1 × 256). The pipeline:
1. Generates a base 256D embedding from projection
2. Creates slight variations for different phoneme lengths
3. Applies length-based scaling for better prosody

## Troubleshooting

### Common Issues

**StyleTTS2 Import Error**
```bash
pip install styletts2>=0.1.6
```

**CUDA Out of Memory**
```bash
# Use CPU or reduce max-style-samples
python training_styletts2.py --device cpu --max-style-samples 50
```

**Poor Voice Quality**
- Ensure high-quality training data (clean audio, diverse phonemes)
- Increase training epochs: `--epochs-projection 200`
- Try different learning rates: `--lr-projection 5e-4`

**No Audio Generated**
- Check that Kokoro is properly installed
- Verify voice embedding format (should be [510, 1, 256])
- Test with a known working voice first

### Performance Tips

1. **Data Quality**: Use clean, high-quality audio samples
2. **Sample Diversity**: Include diverse phonemes and speaking styles
3. **Training Time**: More epochs generally improve quality
4. **Device**: Use GPU for faster training when available

## Comparison with Direct Training

| Aspect | Direct Kokoro Training | StyleTTS2 → Kokoro Pipeline |
|--------|----------------------|---------------------------|
| **Convergence** | Often fails | Reliable |
| **Training Time** | 2-4 days | 1-2 hours |
| **Data Requirements** | Large dataset | Moderate dataset |
| **Voice Quality** | Variable | Consistent |
| **Embedding Space** | Overfitted | Generalizable |

## Future Improvements

Potential enhancements to the pipeline:

1. **Better Style Extraction**: Access StyleTTS2's internal style encoder directly
2. **Multi-Speaker Support**: Handle multiple speakers in training data
3. **Fine-Tuning**: Optional StyleTTS2 fine-tuning before extraction
4. **Advanced Projection**: More sophisticated mapping architectures
5. **Perceptual Loss**: Add perceptual similarity metrics

## Contributing

To improve this pipeline:

1. **Style Extraction**: Help implement direct access to StyleTTS2's style encoder
2. **Loss Functions**: Experiment with better acoustic similarity metrics
3. **Architecture**: Try different projection network designs
4. **Evaluation**: Develop better voice quality metrics

## License

This pipeline builds on:
- **StyleTTS2**: MIT License (inference package)
- **Kokoro**: Original license terms
- **Pipeline Code**: MIT License

See individual component licenses for details. 