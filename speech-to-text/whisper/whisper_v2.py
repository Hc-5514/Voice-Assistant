import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import threading
import queue
import time

from pywhispercpp.model import Model

# 🧠 Whisper.cpp 모델 경로
model_path = "../..//whisper.cpp/models/ggml-base.bin"  # 수정 필요
model = Model(model_path, n_threads=4)

# 🎙 오디오 설정
samplerate = 16000
channels = 1
block_duration = 2  # 초 단위 (STT 처리 간격)

# 🎧 입력 버퍼
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    """sounddevice.InputStream의 콜백"""
    if status:
        print(f"⚠️ 녹음 오류: {status}")
    audio_queue.put(indata.copy())

def stt_worker():
    """큐에 쌓인 오디오를 STT 처리"""
    while True:
        audio_data = []

        # 일정 시간 동안 오디오 모으기
        start_time = time.time()
        while time.time() - start_time < block_duration:
            try:
                data = audio_queue.get(timeout=block_duration)
                audio_data.append(data)
            except queue.Empty:
                continue

        if not audio_data:
            continue

        audio_chunk = np.concatenate(audio_data, axis=0)

        # 임시 WAV 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_chunk, samplerate)
            wav_path = tmp_file.name

        try:
            # Whisper.cpp로 텍스트 변환
            # segments = model.transcribe(wav_path)
            segments = model.transcribe(wav_path, language="auto")
            for segment in segments:
                print("📝", segment.text.strip())
        except Exception as e:
            print(f"[ERROR] STT 실패: {e}")

def main():
    print("🎙 실시간 음성 인식을 시작합니다. 마이크에 말하세요.")
    stt_thread = threading.Thread(target=stt_worker, daemon=True)
    stt_thread.start()

    with sd.InputStream(samplerate=samplerate, channels=channels, dtype='int16', callback=audio_callback):
        while True:
            time.sleep(0.1)

if __name__ == "__main__":
    main()
