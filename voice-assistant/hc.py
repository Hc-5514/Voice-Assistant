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
import platform
import subprocess
import sys
import time
import timeit

import openai
import speech_recognition as sr
import whisper
from dotenv import load_dotenv
from google.cloud import texttospeech

# ----------- 로그 설정 -----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/hc.log"),
        logging.StreamHandler()
    ]
)

# ----------- 환경 설정 및 초기화 -----------
load_dotenv()
whisper_model = whisper.load_model("base")
sys.stderr = open(os.devnull, 'w')

# GPT API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
TTS_SPEAKING_RATE = float(os.getenv("TTS_SPEAKING_RATE", 1.0))
TTS_VOICE = os.getenv("TTS_VOICE", "ko-KR-Standard-A")

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


# ----------- Wake Word 로딩 (.env에서 읽어오기) -----------
def load_wake_word_actions():
    raw = os.getenv("WAKE_WORDS", "")
    wake_word_actions = {}
    if raw:
        pairs = raw.split(',')
        for pair in pairs:
            if ':' in pair:
                keyword, response = pair.split(':', 1)
                wake_word_actions[keyword.strip()] = response.strip()
    return wake_word_actions


wake_word_actions = load_wake_word_actions()


def process_wake_word(text):
    for wake_word, response in wake_word_actions.items():
        if wake_word in text:
            logging.info(f"✅ Wake Word 감지됨: {wake_word}")
            speak_text(response)
            return True
    return False


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


def handle_audio_input():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

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
                audio = recognizer.listen(source, timeout=None)
            return audio
        except sr.UnknownValueError:
            logging.warning("⚠️ 음성을 이해하지 못했습니다. 다시 말씀해주세요.")
        except Exception as e:
            logging.error(f"[ERROR] 음성 입력 오류: {e}")


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
            max_tokens=100,
            temperature=0.5,
        )
        assistant_response = response["choices"][0]["message"]["content"].strip()
        logging.info(f"🤖 GPT 응답: {assistant_response}")
        return assistant_response
    except Exception as e:
        logging.error(f"[ERROR] GPT 응답 생성 중 오류 발생: {e}")
        return None


# ----------- Google Cloud TTS 음성 출력 함수 -----------
def speak_text(text):
    try:
        timestamp = int(time.time())
        wav_file = f"tts_{timestamp}.wav"

        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name=TTS_VOICE,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=TTS_SPEAKING_RATE
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        with open(wav_file, "wb") as out:
            out.write(response.audio_content)

        if platform.system() == "Darwin":
            subprocess.run(["afplay", wav_file])
        else:
            subprocess.run(["aplay", wav_file])

        os.remove(wav_file)

    except Exception as e:
        logging.error(f"[ERROR] 음성 출력 실패: {e}")


# ----------- Main -----------

def main():
    while True:
        try:
            audio_data = handle_audio_input()

            if not audio_data:
                logging.warning("⚠️ 오디오 녹음 실패")
                continue

            start_time = timeit.default_timer()

            transcribed_text = transcribe_audio_to_text(audio_data)

            if not transcribed_text:
                logging.warning("⚠️ 텍스트 변환 실패: 다시 질문해주세요.")
                continue

            if process_wake_word(transcribed_text):
                continue

            response = generate_response(transcribed_text)
            if not response:
                logging.warning("[WARNING] GPT 응답 생성 실패")
                continue

            speak_text(response)
            end_time = timeit.default_timer()
            elapsed_time = end_time - start_time
            logging.info(f"⏳ 실행 시간: {elapsed_time:.3f}초")

        except KeyboardInterrupt:
            logging.info("\n🚪 프로그램을 종료합니다.")
            break
        except Exception as e:
            logging.error(f"[ERROR] 예외 발생: {e}")


if __name__ == "__main__":
    main()
