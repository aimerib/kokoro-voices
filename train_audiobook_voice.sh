#!/bin/bash
# Optimized training script for audiobook voice cloning

# Set environment variables for MPS optimization
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Training configuration optimized for audiobook narration
python training.py \
    --data ./datasets/your_voice \
    --name audiobook_voice \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --log_audio_every 5 \
    --checkpoint_every 5 \
    --patience 15 \
    --save_best \
    --memory_efficient \
    --style_regularization 1e-5 \
    --timbre_warning_threshold 0.35 \
    --use_wandb \
    --wandb_project "audiobook-voices" \
    --wandb_name "$(date +%Y%m%d_%H%M%S)_audiobook" \
    2>&1 | tee training.log

# After training, test the voice
echo "Training complete! Testing voice quality..."
python inference_test.py output/audiobook_voice/audiobook_voice.best.pt \
    --text "Once upon a time, there was a little prince who lived on a planet scarcely bigger than himself." \
    --output test_audiobook.wav

echo "Done! Check test_audiobook.wav to hear your cloned voice."
