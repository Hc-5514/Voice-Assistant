"""
로컬 환경 STT-TTS 테스트
STT: Whisper (base)
TTS: Pyttsx3
"""

import os

import pyttsx3  # Text-to-Speech (TTS) 라이브러리
import speech_recognition as sr  # 음성 인식 라이브러리
import whisper  # Whisper 음성 인식 모델

# ----------- 초기 설정 -----------
# Whisper 모델 로드: 'base' 모델을 사용하여 한국어/영어 음성 인식
whisper_model = whisper.load_model("base")

# pyttsx3 TTS 엔진 초기화
engine = pyttsx3.init()
engine.setProperty("rate", 170)  # 음성 속도 조절 (기본값: 200)
engine.setProperty("volume", 1.0)  # 볼륨 설정 (0.0 ~ 1.0)


def transcribe_audio_to_text(audio_data):
    """
    오디오 데이터를 텍스트로 변환하는 함수 (STT)

    :param audio_data: 변환할 오디오 데이터 (speech_recognition.AudioData 객체)
    :return: 변환된 텍스트 문자열
    """
    try:
        print("🔄 오디오 데이터 처리 중...")

        # 오디오 데이터를 임시 WAV 파일로 저장 (Whisper는 파일 기반 처리)
        temp_filename = "temp.wav"
        with open(temp_filename, "wb") as f:
            f.write(audio_data.get_wav_data())

        # Whisper를 사용하여 음성을 텍스트로 변환 (fp16=False는 CPU 환경 필수)
        result = whisper_model.transcribe(temp_filename, language="ko", fp16=False)

        # 변환된 텍스트 반환
        transcribed_text = result.get("text", "")

        # 임시 파일 삭제 (리소스 절약)
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

    # 사용자 안내 메시지 출력
    print("=====================================")
    print("🎤 음성 비서가 준비되었습니다. 질문을 말씀하세요.")

    recognizer.pause_threshold = 1.2  # 사용자의 일시정지가 1.2초 이상이면 발화 종료로 인식

    while True:
        try:
            with microphone as source:
                # 주변 환경의 노이즈 보정 (환경에 따라 자동 조정)
                recognizer.adjust_for_ambient_noise(source)
                print("🎙 질문을 듣는 중...")
                audio = recognizer.listen(source, timeout=None)  # 사용자 발화 입력 받기

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


def main():
    """
    메인 실행 함수: 음성을 입력받고 이를 텍스트로 변환한 후, 다시 음성으로 출력
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

            # 변환된 텍스트 출력 및 음성 출력
            print(f"📝 변환된 텍스트: {transcribed_text}")
            speak_text(transcribed_text)
        except KeyboardInterrupt:
            print("\n🚪 프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"[예외 발생] {e}")


if __name__ == "__main__":
    main()  # 메인 함수 실행
