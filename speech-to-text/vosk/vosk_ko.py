import pyaudio
from vosk import Model, KaldiRecognizer

model = Model("./models/vosk-model-ko")     # 한국어 모델 사용
recognizer = KaldiRecognizer(model, 16000)  # 16kHz 설정
mic = pyaudio.PyAudio()
stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
stream.start_stream()

print("말하세요...")
while True:
    data = stream.read(4096)
    if recognizer.AcceptWaveform(data):
        print(recognizer.Result())
