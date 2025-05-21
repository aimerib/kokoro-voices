
## Training

### Examples

#### Mac

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run training.py \
  --dataset-id your-repo/your-dataset\
  --epoch 5 \
  --wandb \
  --wandb-project "kokoro-voice" \
  --memory-efficient \
  --grad-accumulation 4 \
  --log-audio-every 1
```

#### NVIDIA GPU

```bash
uv run training.py \
  --dataset-id your-repo/your-dataset\
  --epoch 5 \
  --wandb \
  --wandb-project "kokoro-voice" \
  --log-audio-every 1
```

### Training a voice on runpod

```bash
git clone https://github.com/aimerib/kokoro-voices.git
```

```bash
cd kokoro-voices
pip install uv
uv venv
uv pip install -r requirements.txt
```

```bash
wandb login
```

Edit .env with your HF_TOKEN if you want to use a Huggingface dataset or upload to Huggingface.

```bash
uv run training.py \
  --dataset-id your-repo/your-dataset \
  --epoch 100 \
  --wandb \
  --wandb-project "kokoro-voice" \
  --log-audio-every 5 \
  --upload-to-hf \
  --hf-repo-id your-repo/your-voice
```