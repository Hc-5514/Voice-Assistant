"""
로컬 환경 STT + Wake Word + GPT-4o + Google Cloud TTS (logging 적용)
STT: Whisper (base)
TTS: Google Cloud Text-to-Speech
Wake Word: API 사용 x
GPT: GPT-4o
"""

import logging
import multiprocessing
import os
import subprocess
import time

import openai
import speech_recognition as sr
import whisper
from dotenv import load_dotenv
from gtts import gTTS

# ----------- Whisper 모델 로드 -----------
whisper_model = whisper.load_model("base")

# ----------- 로깅 설정 -----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ----------- 환경 변수 로드 -----------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ----------- SYSTEM PROMPT 설정 -----------
SYSTEM_PROMPT = """
당신은 사용자의 음성 비서를 담당하는 AI입니다.
당신의 이름은 나로봇입니다.
질문에 대해 즉시 답변하세요. "잠시만 기다려주세요" 같은 문장은 사용하지 마세요.
불필요한 단어를 제거하고, 간결하고 짧은 문장으로 답변하세요.
항상 50자 이내로 요약해서 답변하세요.
답변은 1~2개의 문장으로만 구성하세요.
이모티콘을 사용하지 마세요.
친절하고 자연스러운 말투를 유지하세요.
"""


# ----------- Whisper STT 워커 -----------
def whisper_stt_worker(temp_filename, result_queue):
    try:
        result = whisper_model.transcribe(temp_filename, language="ko", fp16=False, beam_size=1, best_of=1)
        text = result.get("text", "").strip()
        result_queue.put(text)
    except Exception:
        result_queue.put(None)


# ----------- STT 통합 함수 -----------
def transcribe_audio_to_text(audio_data, timeout=5):
    temp_filename = "temp.wav"
    try:
        logging.info("🔄 오디오 데이터 처리 중...")

        with open(temp_filename, "wb") as f:
            f.write(audio_data.get_wav_data(convert_rate=16000, convert_width=2))

        result_queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=whisper_stt_worker, args=(temp_filename, result_queue))
        p.start()
        p.join(timeout)

        if p.is_alive():
            logging.warning("⚠️ Whisper STT 시간이 초과되었습니다. 프로세스를 종료합니다.")
            p.terminate()
            p.join()
            return None

        if not result_queue.empty():
            text = result_queue.get()
            logging.info(f"📝 변환된 텍스트: {text}")
            return text
        else:
            logging.warning("⚠️ STT 결과가 없습니다.")
            return None

    except Exception as e:
        logging.error(f"[ERROR] STT 변환 실패: {e}")
        return None
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


# ----------- 오디오 입력 통합 함수 -----------
def handle_audio_input():
    recognizer = sr.Recognizer()
    try:
        microphone = sr.Microphone(device_index=3, sample_rate=48000, chunk_size=1024)
    except Exception as e:
        logging.error(f"[ERROR] 마이크 장치 초기화 실패: {e}")
        return None

    logging.info("=======================================================")
    logging.info("🎤 음성 비서가 준비되었습니다.")

    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = 500
    recognizer.pause_threshold = 1.0

    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=1.5)
                logging.info("🎙 질문을 듣는 중...")
                audio = recognizer.listen(source, timeout=3)

            return audio

        except sr.UnknownValueError:
            logging.warning("⚠️ 음성을 이해하지 못했습니다. 다시 말씀해주세요.")
        except Exception as e:
            logging.error(f"[ERROR] 음성 입력 오류: {e}")
            return None


# ----------- GPT 응답 생성 함수 -----------
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


def speak_text(text, speed=1.3):
    try:
        timestamp = int(time.time())
        original = f"tts_{timestamp}.mp3"
        adjusted = f"tts_{timestamp}_fast.mp3"

        # 1. gTTS 음성 생성
        tts = gTTS(text=text, lang='ko')
        tts.save(original)

        # 2. ffmpeg로 재생 속도 조절
        subprocess.run([
            "ffmpeg", "-y", "-i", original,
            "-filter:a", f"atempo={speed}",
            adjusted
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 3. mpg123로 mp3 재생
        subprocess.run(["mpg123", adjusted], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 4. 임시 파일 삭제
        os.remove(original)
        os.remove(adjusted)

        logging.info(f"🗣️ 음성 출력 (1.3x): {text}")
    except Exception as e:
        logging.error(f"[ERROR] 음성 출력 실패: {e}")


# ----------- 메인 루프 -----------
def main():
    while True:
        try:
            audio_data = handle_audio_input()

            if not audio_data:
                continue

            transcribed_text = transcribe_audio_to_text(audio_data, timeout=5)

            if not transcribed_text:
                logging.warning("⚠️ 텍스트 변환 실패: 다시 질문해주세요.")
                continue

            response = generate_response(transcribed_text)

            if not response:
                logging.warning("[WARNING] GPT 응답 생성 실패")
                continue

            logging.info(f"✅ 최종 응답: {response}")
            speak_text(response)

        except KeyboardInterrupt:
            logging.info("\n🚪 프로그램을 종료합니다.")
            break
        except Exception as e:
            logging.error(f"[ERROR] 예외 발생: {e}")


if __name__ == "__main__":
    multiprocessing.set_start_method('fork')  # Mac/Linux
    main()
