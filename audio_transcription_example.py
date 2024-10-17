import whisper

AUDIO_FILE = "test.wav"
transcription_model = whisper.load_model("base").to("cuda")
result = transcription_model.transcribe(AUDIO_FILE) 
print(f"Transcription: " + result["text"])