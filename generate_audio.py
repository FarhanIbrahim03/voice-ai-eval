from gtts import gTTS

sentences = [
    ("What is the capital of France?", "sample_data/q1.mp3"),
    ("What is the boiling point of water in Celsius?", "sample_data/q2.mp3"),
    ("Who wrote Romeo and Juliet?", "sample_data/q3.mp3"),
]

for text, path in sentences:
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(path)
    print(f"Saved: {path}")