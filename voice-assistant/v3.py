"""
로컬 환경 STT-TTS + Wake Word + GPT-4o (logging 적용)
STT: Whisper (base)
TTS: Pyttsx3
Wake Word: API 사용 x
GPT: GPT-4o
"""

import logging
import os
import timeit
import sys

import openai
import pyttsx3
import speech_recognition as sr
import whisper
from dotenv import load_dotenv

# ----------- 로그 설정 -----------
logging.basicConfig(
    level=logging.INFO,  # 모든 로그 출력
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("./logs/v3.log"),  # 로그 파일 저장
        logging.StreamHandler()  # 콘솔 출력
    ]
)

# ----------- 환경 설정 및 초기화 -----------
load_dotenv()  # .env 파일에서 API 키 로드
whisper_model = whisper.load_model("base")  # STT 모델 로드
engine = pyttsx3.init()  # TTS 엔진 초기화
engine.setProperty('voice', 'com.apple.voice.compact.ko-KR.Yuna')
engine.setProperty("rate", 180)  # 음성 속도 조절
engine.setProperty("volume", 1.0)  # 볼륨 설정
sys.stderr = open(os.devnull, 'w')  # ALSA 등의 시스템 에러 메시지를 무시

# Wake Word 정의 및 실행할 액션 매핑
wake_word_actions = {
    "우울해": lambda: speak_text("괜찮아요! 힘내세요. 제가 항상 응원할게요."),
    "행복해": lambda: speak_text("와! 기분이 좋으시군요! 계속 행복하세요."),
    "춤춰줘": lambda: speak_text("신나는 음악을 틀어줄 수는 없지만, 기분 좋게 흔들어 보세요!"),
}

# GPT API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("[ERROR] API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

# GPT 시스템 프롬프트
SYSTEM_PROMPT = """
당신은 사용자의 음성 비서를 담당하는 AI입니다.
질문에 대해 즉시 답변하세요. "잠시만 기다려주세요" 같은 문장은 사용하지 마세요.
불필요한 단어를 제거하고 간결한 문장으로 정리하세요.
너무 긴 답변은 사용자가 쉽게 이해할 수 있도록 50자 이내로 요약하여 제공하세요.
이모티콘을 사용하지 마세요.
친절하고 자연스러운 말투로 응답하세요.
"""


# ----------- STT, TTS, Wake Word -----------

def speak_text(text):
    """텍스트를 음성으로 변환하여 출력 (TTS)"""
    try:
        engine.say(text)
        engine.runAndWait()
        logging.info(f"🗣️ 음성 출력: {text}")
    except Exception as e:
        logging.error(f"[ERROR] 음성 출력 실패: {e}")


def transcribe_audio_to_text(audio_data):
    """오디오 데이터를 텍스트로 변환 (STT)"""
    try:
        logging.info("🔄 오디오 데이터 처리 중...")
        temp_filename = "temp.wav"

        with open(temp_filename, "wb") as f:
            f.write(audio_data.get_wav_data())

        result = whisper_model.transcribe(temp_filename, language="ko", fp16=False)
        os.remove(temp_filename)
        text = result.get("text", "").strip()
        logging.info(f"📝 변환된 텍스트: {text}")
        return text
    except Exception as e:
        logging.error(f"[ERROR] STT 변환 실패: {e}")
        return None


def handle_audio_input():
    """마이크를 통해 음성을 입력받아 오디오 데이터를 반환"""
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    logging.info("=======================================================")
    logging.info("🎤 음성 비서가 준비되었습니다.")

    # 마이크 감도 및 잡음 조정
    recognizer.dynamic_energy_threshold = False  # 자동 감도 조절 비활성화
    recognizer.energy_threshold = 500  # 감도 높여서 작은 잡음 무시
    recognizer.pause_threshold = 1.2  # 긴 문장도 인식 가능하도록 조정

    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source, duration=1.5)  # 배경 소음 보정 강화
                logging.info("🎙 질문을 듣는 중...")
                audio = recognizer.listen(source, timeout=None)
            return audio
        except sr.UnknownValueError:
            logging.warning("⚠️ 음성을 이해하지 못했습니다. 다시 말씀해주세요.")
        except Exception as e:
            logging.error(f"[ERROR] 음성 입력 오류: {e}")


def process_wake_word(text):
    """입력된 텍스트에서 Wake Word 감지 후 실행"""
    wake_words = set(wake_word_actions.keys())  # Wake Word 집합화 (탐색 속도 개선)

    for wake_word in wake_words:
        if wake_word in text:
            logging.info(f"✅ Wake Word 감지됨: {wake_word}")
            wake_word_actions[wake_word]()  # 매칭된 함수 실행
            return True
    return False


# ----------- GPT API 호출 -----------

def generate_response(user_input):
    """GPT-4o API를 호출하여 응답 생성"""
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


# ----------- 메인 실행 함수 -----------

def main():
    """메인 실행 함수: Wake Word 감지 후 명령 실행"""
    while True:
        try:
            audio_data = handle_audio_input()
            start_time = timeit.default_timer()  # 실행 시작 시간

            transcribed_text = transcribe_audio_to_text(audio_data)

            if not transcribed_text:
                logging.warning("⚠️ 텍스트 변환 실패: 다시 질문해주세요.")
                continue

            # Wake Word 실행 후 즉시 다음 입력 대기
            if process_wake_word(transcribed_text):
                continue

            # GPT 응답 생성
            response = generate_response(transcribed_text)
            if not response:
                logging.warning("[WARNING] GPT 응답 생성 실패")
                continue

            # 실행 시간 측정 및 출력
            end_time = timeit.default_timer()
            elapsed_time = end_time - start_time
            logging.info(f"⏳ 실행 시간: {elapsed_time:.3f}초")

            speak_text(response)

        except KeyboardInterrupt:
            logging.info("\n🚪 프로그램을 종료합니다.")
            break
        except Exception as e:
            logging.error(f"[ERROR] 예외 발생: {e}")


if __name__ == "__main__":
    main()
