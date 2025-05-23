#!/bin/bash
# CUDA-optimized training script for Kokoro voice cloning
# Designed to prevent OOM errors on limited GPU memory

# Set CUDA memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Optional: Set specific GPU if you have multiple
# export CUDA_VISIBLE_DEVICES=0

# Training configuration optimized for CUDA with limited memory
python training.py \
    --data ./datasets/your_voice \
    --name audiobook_voice_cuda \
    --epochs 50 \
    --lr 5e-5 \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --log_audio_every 10 \
    --checkpoint_every 5 \
    --patience 20 \
    --save_best \
    --memory_efficient \
    --style_regularization 1e-5 \
    --timbre_warning_threshold 0.35 \
    --use_wandb \
    --wandb_project "audiobook-voices" \
    --wandb_name "$(date +%Y%m%d_%H%M%S)_cuda_optimized" \
    2>&1 | tee training_cuda.log

# Monitor GPU memory usage during training (optional)
echo "To monitor GPU usage in another terminal, run:"
echo "watch -n 1 nvidia-smi"

# After training, test the voice
echo "Training complete! Testing voice quality..."
python inference_test.py output/audiobook_voice_cuda/audiobook_voice_cuda.best.pt \
    --text "The quick brown fox jumps over the lazy dog. This is a test of the voice cloning system." \
    --output test_audiobook_cuda.wav

echo "Done! Check test_audiobook_cuda.wav to hear your cloned voice."
