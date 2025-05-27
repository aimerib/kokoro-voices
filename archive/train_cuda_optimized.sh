#!/bin/bash
# CUDA-optimized training script for Kokoro voice cloning
# Designed to prevent OOM errors on limited GPU memory

# Set CUDA memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Optional: Set specific GPU if you have multiple
# export CUDA_VISIBLE_DEVICES=0

# First, configure accelerate if not already done
if [ ! -f ~/.cache/huggingface/accelerate/default_config.yaml ]; then
    echo "Configuring accelerate for first-time use..."
    accelerate config --config_file ~/.cache/huggingface/accelerate/default_config.yaml <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: NO
mixed_precision: no
num_machines: 1
num_processes: 1
gpu_ids: 0
EOF
fi

# Training configuration optimized for CUDA with limited memory
accelerate launch \
    --mixed_precision no \
    --num_processes 1 \
    --num_machines 1 \
    --dynamo_backend no \
    training.py \
    --dataset-id aimeri/test-dataset \
    --name my_voice \
    --epochs 50 \
    --lr 5e-5 \
    --batch-size 1 \
    --grad-accumulation 8 \
    --log-audio-every 10 \
    --memory-efficient \
    --style-reg 1e-5 \
    --timbre-warning 0.35 \
    --wandb \
    --wandb-project "audiobook-voices" \
    --wandb-name "$(date +%Y%m%d_%H%M%S)_cuda_optimized" \
    2>&1 | tee training_cuda.log

# Monitor GPU memory usage during training (optional)
echo "To monitor GPU usage in another terminal, run:"
echo "watch -n 1 nvidia-smi"

# After training, test the voice
echo "Training complete! Testing voice quality..."
accelerate launch inference_test.py output/my_voice/my_voice.best.pt \
    --text "The quick brown fox jumps over the lazy dog. This is a test of the voice cloning system." \
    --output test_audiobook_cuda.wav

echo "Done! Check test_audiobook_cuda.wav to hear your cloned voice."
