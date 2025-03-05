"""
로컬 환경 STT-TTS + Wake Word 인식 테스트
STT: Whisper (base)
TTS: Pyttsx3
Wake Word: API 사용 x
"""

import os

import pyttsx3  # TTS 라이브러리
import speech_recognition as sr  # 음성 인식 라이브러리
import whisper  # Whisper 음성 인식 모델

# ----------- 초기 설정 -----------
# Whisper 모델 로드 (STT)
whisper_model = whisper.load_model("base")

# pyttsx3 TTS 엔진 초기화
engine = pyttsx3.init()
engine.setProperty("rate", 180)  # 음성 속도 조절 (기본값: 200)
engine.setProperty("volume", 1.0)  # 볼륨 설정 (0.0 ~ 1.0)

# Wake Word와 실행할 함수 매핑
wake_word_actions = {
    "우울해": lambda: speak_text("괜찮아요! 힘내세요. 제가 항상 응원할게요."),
    "행복해": lambda: speak_text("와! 기분이 좋으시군요! 계속 행복하세요."),
    "춤춰줘": lambda: speak_text("신나는 음악을 틀어줄 수는 없지만, 기분 좋게 흔들어 보세요!"),
}


def transcribe_audio_to_text(audio_data):
    """
    오디오 데이터를 텍스트로 변환하는 함수 (STT)

    :param audio_data: 변환할 오디오 데이터 (speech_recognition.AudioData 객체)
    :return: 변환된 텍스트 문자열
    """
    try:
        print("🔄 오디오 데이터 처리 중...")

        # 오디오 데이터를 임시 WAV 파일로 저장
        temp_filename = "temp.wav"
        with open(temp_filename, "wb") as f:
            f.write(audio_data.get_wav_data())

        # Whisper를 사용하여 음성을 텍스트로 변환 (fp16=False는 CPU 환경 필수)
        result = whisper_model.transcribe(temp_filename, language="ko", fp16=False)

        # 변환된 텍스트 반환
        transcribed_text = result.get("text", "")

        # 임시 파일 삭제
        os.remove(temp_filename)

        return transcribed_text
    except Exception as e:
        print(f"[오류] STT 변환 실패: {e}")
        return None


def handle_audio_input():
    """
    마이크를 통해 음성을 입력받아 오디오 데이터를 반환하는 함수

    :return: 녹음된 오디오 데이터 (speech_recognition.AudioData 객체)
    """
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("===========================================")
    print("🎤 음성 비서가 준비되었습니다. Wake Word를 말씀하세요.")

    recognizer.pause_threshold = 1.2  # 사용자의 일시정지가 1.2초 이상이면 발화 종료로 인식

    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                print("🎙 질문을 듣는 중...")
                audio = recognizer.listen(source, timeout=None)

            return audio
        except sr.UnknownValueError:
            print("⚠️ 음성을 이해하지 못했습니다. 다시 말씀해주세요.")
        except Exception as e:
            print(f"[오류] 음성 입력 오류: {e}")


def speak_text(text):
    """
    변환된 텍스트를 음성으로 출력하는 함수 (TTS)

    :param text: 출력할 텍스트 문자열
    """
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[오류] 음성 출력 실패: {e}")


def process_wake_word(text):
    """
    변환된 텍스트에서 Wake Word를 감지하고 해당 동작을 실행하는 함수
    """
    for wake_word, action in wake_word_actions.items():
        if wake_word in text:
            print(f"✅ Wake Word 감지됨: {wake_word}")
            action()  # 매칭된 함수 실행
            return True
    return False  # Wake Word가 감지되지 않음


def main():
    """
    메인 실행 함수: Wake Word 감지 후 명령 실행
    """
    while True:
        try:
            # 음성 입력 받기
            audio_data = handle_audio_input()

            # 음성을 텍스트로 변환
            transcribed_text = transcribe_audio_to_text(audio_data)

            if not transcribed_text:
                print("⚠️ 텍스트 변환 실패: 다시 질문해주세요.")
                continue

            print(f"📝 변환된 텍스트: {transcribed_text}")

            # Wake Word 감지 후 실행
            if process_wake_word(transcribed_text):
                continue  # Wake Word를 실행했으면, 새로운 음성 입력 대기

            # 일반적인 응답 처리 (GPT API 연동 가능)
            speak_text("Wake Word가 감지되지 않았어요. 다시 말씀해주세요.")
        except KeyboardInterrupt:
            print("\n🚪 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"[예외 발생] {e}")


if __name__ == "__main__":
    main()  # 메인 함수 실행
