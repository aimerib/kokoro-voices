from utils import generate_with_custom_voice, generate_with_standard_voice

if __name__ == "__main__":
    # Your text to synthesize
    text = "The quick brown fox jumps over the lazy dog. This is a test of my custom voice for Kokoro."
    
    # Path to folder with voice checkpoints
    voices_path = "output/my_voice/"
    voice_name = "my_voice"

    # # iterate over each .epoch*.pt file in the folder ignoring _compact.pt
    # for voice_path in glob.glob(os.path.join(voices_path, f"{voice_name}.epoch*.pt")):
    #     if "compact" in voice_path:
    #         continue
    #     print(f"Processing {voice_path}")
    #     generate_with_custom_voice(text, voice_path, f"{voice_name}.{voice_path.split(".")[-2]}.wav")
    generate_with_custom_voice(text, f"{voices_path}/{voice_name}.best.pt", f"{voice_name}.best.wav")
    # Generate and save audio
    # generate_with_custom_voice(text, voice_path, "my_custom_voice.wav")
    generate_with_standard_voice(text, "standard_voice.wav")
