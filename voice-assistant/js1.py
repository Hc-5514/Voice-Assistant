"""
로컬 환경 STT + Wake Word + GPT-4o (logging 적용)
STT: Whisper (base)
TTS: gTTS
Wake Word: API 사용 x
GPT: GPT-4o
"""

import ctypes  # ALSA 에러 숨김용
import logging
import os
import subprocess
import sys
import time
import timeit
import wave

import alsaaudio
import openai
import serial  # 시리얼
import whisper
from dotenv import load_dotenv
from gtts import gTTS

# ALSA 에러 로그 숨기기
asound = ctypes.cdll.LoadLibrary('libasound.so')
asound.snd_lib_error_set_handler(None)

# ----------- 로그 설정 -----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/v3.log"),
        logging.StreamHandler()
    ]
)

# ----------- 환경 설정 및 초기화 -----------
load_dotenv()
whisper_model = whisper.load_model("base")
os.environ["DISPLAY"] = ""  # xcb error 메세지 제거용 코드
sys.stderr = open(os.devnull, 'w')

# GPT API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("[ERROR] API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

SYSTEM_PROMPT = """
당신은 사용자의 음성 비서를 담당하는 AI입니다.
당신의 이름은 나로봇입니다.
질문에 대해 즉시 답변하세요. "잠시만 기다려주세요" 같은 문장은 사용하지 마세요.
불필요한 단어를 제거하고 간결한 문장으로 정리하세요.
너무 긴 답변은 사용자가 쉽게 이해할 수 있도록 50자 이내로 요약하여 제공하세요.
이모티콘을 사용하지 마세요.
친절하고 자연스러운 말투로 응답하세요.
"""


# ----------- TTS (gTTS + ffmpeg + mpg123) -----------

def speak_text(text, speed=1.3):
    try:
        timestamp = int(time.time())
        mp3_file = f"tts_{timestamp}.mp3"
        wav_file = f"tts_{timestamp}.wav"

        # 1. gTTS 음성 생성
        tts = gTTS(text=text, lang='ko')
        tts.save(mp3_file)

        # 2. ffmpeg로 mp3 → wav 변환 + 속도 조절
        subprocess.run([
            "ffmpeg", "-y", "-i", mp3_file,
            "-filter:a", f"atempo={speed}",
            "-ar", "44100", "-ac", "2", "-f", "wav",
            wav_file
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. ALSA (aplay) 로 재생 - WM8960 (card 3)
        subprocess.run(["aplay", "-D", "hw:3,0", wav_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 4. 임시 파일 삭제
        os.remove(mp3_file)
        os.remove(wav_file)

    except Exception as e:
        logging.error(f"[ERROR] 음성 출력 실패: {e}")


# ----------- 시리얼 초기화 -----------
try:
    ser = serial.Serial('/dev/serial0', 115200, timeout=1)
    logging.info("🔌 시리얼 포트 열림 (/dev/serial0)")
except Exception as e:
    ser = None
    logging.error(f"[ERROR] 시리얼 포트 열기 실패: {e}")


# ----------- Wake Word 정의 (시리얼 명령 포함) -----------

def send_serial_command(cmd):
    if ser and ser.is_open:
        try:
            ser.write(f"{cmd}\n".encode())
            logging.info(f"📤 시리얼로 명령 전송: {cmd}")
        except Exception as e:
            logging.error(f"[ERROR] 시리얼 전송 실패: {e}")


wake_word_actions = {
    "우울": lambda: (send_serial_command("ob"), speak_text("우울하면 나와 함께 춤을 추자~")),
    "행복": lambda: (send_serial_command("c1"), speak_text("와! 기분이 좋으시군요! 무슨 일이 있었나요?")),
    "춤": lambda: (send_serial_command("c2"), speak_text("신나는 음악을 틀어줄 수는 없지만, 기분 좋게 흔들어 보세요!")),
}


# ----------- STT -----------

def transcribe_audio_to_text(file_path):
    try:
        logging.info("🔄 Whisper로 오디오 파일 처리 중...")
        result = whisper_model.transcribe(file_path, language="ko", fp16=False)
        os.remove(file_path)
        text = result.get("text", "").strip()
        logging.info(f"📝 변환된 텍스트: {text}")
        return text
    except Exception as e:
        logging.error(f"[ERROR] STT 변환 실패: {e}")
        return None


def handle_audio_input():
    logging.info("🎙 ALSA 마이크로부터 직접 녹음 시작...")
    device = 'default'
    channels = 2
    rate = 44100
    format = alsaaudio.PCM_FORMAT_S16_LE
    periodsize = 160
    duration_sec = 5  # 녹음 시간 설정

    try:
        inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NONBLOCK, channels=channels, rate=rate, format=format,
                            periodsize=periodsize, device=device)
        temp_filename = "temp.wav"

        wf = wave.open(temp_filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16bit -> 2 bytes
        wf.setframerate(rate)

        total_loops = int((rate * duration_sec) / periodsize)
        loops = 0
        while loops < total_loops:
            l, data = inp.read()
            if l:
                wf.writeframes(data)
            time.sleep(0.01)
            loops += 1

        wf.close()
        logging.info(f"📁 녹음 완료: {temp_filename}")
        return temp_filename  # 오디오 파일 경로 반환

    except Exception as e:
        logging.error(f"[ERROR] ALSA 녹음 실패: {e}")
        return None


def process_wake_word(text):
    wake_words = set(wake_word_actions.keys())
    for wake_word in wake_words:
        if wake_word in text:
            logging.info(f"✅ Wake Word 감지됨: {wake_word}")
            wake_word_actions[wake_word]()
            return True
    return False


# ----------- GPT -----------

def generate_response(user_input):
    try:
        logging.info("GPT 응답 생성 중...")
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            max_tokens=256,
            temperature=0.5,
        )
        assistant_response = response["choices"][0]["message"]["content"].strip()
        logging.info(f"🤖 GPT 응답: {assistant_response}")
        return assistant_response
    except Exception as e:
        logging.error(f"[ERROR] GPT 응답 생성 중 오류 발생: {e}")
        return None


# ----------- Main -----------

def main():
    while True:
        try:
            audio_file = handle_audio_input()
            if not audio_file:
                continue

            start_time = timeit.default_timer()
            transcribed_text = transcribe_audio_to_text(audio_file)

            if not transcribed_text:
                logging.warning("⚠️ 텍스트 변환 실패: 다시 질문해주세요.")
                continue

            if process_wake_word(transcribed_text):
                continue

            response = generate_response(transcribed_text)
            if not response:
                logging.warning("[WARNING] GPT 응답 생성 실패")
                continue

            end_time = timeit.default_timer()
            elapsed_time = end_time - start_time
            logging.info(f"⏳ 실행 시간: {elapsed_time:.3f}초")

            speak_text(response)

        except KeyboardInterrupt:
            logging.info("\n🚪 프로그램을 종료합니다.")
            if ser and ser.is_open:
                ser.close()
                logging.info("🔒 시리얼 포트 닫힘 (/dev/serial0)")
            break
        except Exception as e:
            logging.error(f"[ERROR] 예외 발생: {e}")


if __name__ == "__main__":
    main()
