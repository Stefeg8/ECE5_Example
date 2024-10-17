import torch
from TTS.api import TTS
import time
# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

# Init TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
#wav = tts.tts(text="Hello world!", speaker_wav="my/cloning/audio.wav", language="en")
# Text to speech to a file
time1 = time.time()

tts.tts_to_file(text="Balls are squishy and I love having cocks in my mouth but sometimes it hurts so I can't have it in my mouth all the time. It is very unfortunate but I'll have to make do.", speaker_wav=["emily1.wav", "IMG_1306.wav", "IMG_1307.wav", "IMG_1308.wav","IMG_1309.wav","IMG_1310.wav","IMG_1313.wav","IMG_1314.wav","IMG_1315.wav"], language="en", file_path="balsshd.wav")
print(f"Time taken for TTS: {time.time() - time1}")
