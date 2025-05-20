import torch
import soundfile as sf
from kokoro import KPipeline
import glob
import os

# Function to expand a voice tensor to the required format
def expand_voice_tensor(voice_tensor, max_len=512):
    """
    Expands a [1, 256] voice tensor to [max_len, 256] by repeating it
    to match the Kokoro model's expectation for different phoneme lengths
    """
    if voice_tensor.dim() == 1 and voice_tensor.size(0) == 256:
        voice_tensor = voice_tensor.unsqueeze(0)  # [256] -> [1, 256]
    
    if voice_tensor.size(0) == 1:
        # Expand to [max_len, 256]
        return voice_tensor.repeat(max_len, 1)
    
    return voice_tensor

def generate_with_custom_voice(text, voice_path, output_file="output.wav"):
    # Initialize the pipeline
    pipeline = KPipeline(lang_code='a')
    
    # Load and expand the voice tensor
    try:
        voice_tensor = torch.load(voice_path, map_location='cpu')['voice']
# # Register under a name, e.g. 'my_voice'
# pipeline.voices['my_voice'] = voice_tensor.squeeze(0)  
#         voice_tensor = torch.load(voice_path)
        print(f"Loaded voice tensor with shape: {voice_tensor.shape}")
        
        # Make sure tensor has the right format for Kokoro
        # expanded_voice = expand_voice_tensor(voice_tensor)
        
        # Generate audio
        outputs = []
        for _, _, audio in pipeline(text, voice=voice_tensor):
            print(f"Generated {audio.shape[0]} samples")
            outputs.append(audio)
        
        # Combine and save
        if outputs:
            full_audio = torch.cat(outputs)
            sf.write(output_file, full_audio.numpy(), 24000)
            print(f"Saved audio to {output_file}")
            return True
    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to stock voice")
        
        # Fallback to a stock voice
        outputs = []
        for _, _, audio in pipeline(text, voice="af_heart"):
            outputs.append(audio)
        
        if outputs:
            full_audio = torch.cat(outputs)
            sf.write(output_file, full_audio.numpy(), 24000)
            print(f"Saved audio with fallback voice to {output_file}")
            return True
    
    return False

def generate_with_standard_voice(text, output_file="output.wav"):
    # Initialize the pipeline
    pipeline = KPipeline(lang_code='a')
    
    # Generate audio
    outputs = []
    for _, _, audio in pipeline(text, voice="af_heart"):
        outputs.append(audio)
    
    # Combine and save
    if outputs:
        full_audio = torch.cat(outputs)
        sf.write(output_file, full_audio.numpy(), 24000)
        print(f"Saved audio to {output_file}")
        return True
    
    return False

if __name__ == "__main__":
    # Your text to synthesize
    text = "The quick brown fox jumps over the lazy dog. This is a test of my custom voice for Kokoro."
    
    # Path to folder with voice checkpoints
    voices_path = "output/my_voice/"
    voice_name = "my_voice"

    # iterate over each .epoch*.pt file in the folder ignoring _compact.pt
    for voice_path in glob.glob(os.path.join(voices_path, f"{voice_name}.epoch*.pt")):
        if "compact" in voice_path:
            continue
        print(f"Processing {voice_path}")
        generate_with_custom_voice(text, voice_path, f"{voice_name}.{voice_path.split(".")[-2]}.wav")
    generate_with_custom_voice(text, f"{voice_name}.pt", f"{voice_name}.pt.wav")
    # Generate and save audio
    # generate_with_custom_voice(text, voice_path, "my_custom_voice.wav")
    generate_with_standard_voice(text, "standard_voice.wav")
