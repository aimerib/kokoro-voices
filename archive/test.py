# # Using SpeechBrain for batch processing
# import os
# from os.path import isdir
# import torchaudio
# from speechbrain.inference import SepformerSeparation as separator

# # Load model once
# model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix")

# # Process directory of audio files
# input_dir = "/Users/aimeri/Downloads/processed/mastering-memory-allocation-techniques.mp3"
# output_dir = "processed/wav"

# if isdir(input_dir):
#     for filename in os.listdir(input_dir):
#         if filename.endswith('.mp3'):
#             input_path = os.path.join(input_dir, filename)
#             est_sources = model.separate_file(path=input_path)
        
#             # Save separated sources
#             torchaudio.save(
#                 os.path.join(output_dir, f"clean_{filename}.wav"), 
#                 est_sources[:, :, 0].cpu(), 
#                 8000
#             )
# elif input_dir.endswith('.mp3'):
#     est_sources = model.separate_file(path=input_dir)

#     # Save separated sources
#     torchaudio.save(
#         os.path.join(output_dir, f"clean_{input_dir}.wav"), 
#         est_sources[:, :, 0].cpu(), 
#         8000
#     )    






import subprocess, pathlib, multiprocessing as mp
import torch
import torchaudio
from df import enhance, init_df
import tempfile

SRC_DIR  = pathlib.Path("/Users/aimeri/Downloads/processed/unt")
STEM_DIR = pathlib.Path("stems")      # speech/music split
CLEAN_DIR = pathlib.Path("clean")     # denoised speech

# Initialize DeepFilter model once (this can be moved to main() if memory is a concern)
print("Loading DeepFilter model...")
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model with device specification
model, df_state, _ = init_df(device=device)

def separate(path):
    subprocess.run(["demucs", "-d", "mps", "-j", "10", "--two-stems=vocals", "-o", STEM_DIR, str(path)], check=True)

def denoise(stem):
    """Enhanced denoise function using DeepFilter Python API"""
    try:
        # Load audio
        audio, sr = torchaudio.load(stem)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # DeepFilter expects specific sample rate (usually 48kHz)
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr, 48000)
            audio = resampler(audio)
            sr = 48000
        
        # Move audio to same device as model
        audio = audio.to(device)
        
        # Enhance audio using DeepFilter
        enhanced_audio = enhance(model, df_state, audio.squeeze(0))
        
        # Convert back to CPU for saving
        enhanced_audio = enhanced_audio.cpu().unsqueeze(0)
        
        # Create output path in CLEAN_DIR
        output_path = CLEAN_DIR / stem.name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save enhanced audio
        torchaudio.save(str(output_path), enhanced_audio, sr)
        print(f"✓ Denoised: {stem.name}")
        
    except Exception as e:
        print(f"✗ Error processing {stem}: {e}")

def main():
    # Ensure clean directory exists
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1) parallel source-separation
    # with mp.Pool() as pool:
    #     pool.map(separate, SRC_DIR.glob("*.mp3"))

    # 2) denoise the isolated speech
    print(f"Denoising vocals using DeepFilter on {device}...")
    
    # For multiprocessing with DeepFilter, we need to be careful about model sharing
    # Using sequential processing for now to avoid model loading issues
    for stem in STEM_DIR.rglob("vocals.wav"):
        denoise(stem.absolute())
    
    # Alternative: Use multiprocessing but initialize model in each worker
    # This requires moving model initialization inside denoise() function
    # with mp.Pool() as pool:
    #     pool.map(denoise, [stem.absolute() for stem in STEM_DIR.rglob("vocals.wav")])

if __name__ == "__main__":
    mp.freeze_support()
    main()
