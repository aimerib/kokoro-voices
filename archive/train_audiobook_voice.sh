#!/bin/bash
# Optimized training script for audiobook voice cloning

# Set environment variables for MPS optimization
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Training configuration optimized for audiobook narration
accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    training.py \
    --data ./datasets/your_voice \
    --name audiobook_voice \
    --epochs 50 \
    --lr 1e-4 \
    --batch-size 1 \
    --grad-accumulation 4 \
    --log-audio-every 5 \
    --memory-efficient \
    --style-reg 1e-5 \
    --timbre-warning 0.35 \
    --wandb \
    --wandb-project "audiobook-voices" \
    --wandb-name "$(date +%Y%m%d_%H%M%S)_audiobook" \
    2>&1 | tee training.log

# After training, test the voice
echo "Training complete! Testing voice quality..."
python inference_test.py output/audiobook_voice/audiobook_voice.best.pt \
    --text "Once upon a time, there was a little prince who lived on a planet scarcely bigger than himself." \
    --output test_audiobook.wav

echo "Done! Check test_audiobook.wav to hear your cloned voice."
