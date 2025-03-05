import os

import speech_recognition as sr  # 음성 인식 라이브러리
import whisper  # Whisper 음성 인식 모델

# Whisper 모델 로드: 'base' 모델을 사용합니다.
whisper_model = whisper.load_model("base")


def transcribe_audio_to_text(audio_data):
    """
    오디오 데이터를 텍스트로 변환하는 함수
    :param audio_data: 변환할 오디오 데이터 (speech_recognition.AudioData 객체)
    :return: 변환된 텍스트 문자열
    """
    try:
        print("오디오 데이터 처리 중...")
        # 오디오 데이터를 일시적인 WAV 파일로 저장
        with open("temp.wav", "wb") as f:
            f.write(audio_data.get_wav_data())
        # Whisper 모델을 사용하여 오디오 파일을 텍스트로 변환
        result = whisper_model.transcribe("temp.wav", fp16=False)
        # result = whisper_model.transcribe("temp.wav", language="ko", fp16=False)
        # result = whisper_model.transcribe("temp.wav", language="en", fp16=False)
        # 임시 파일 삭제
        os.remove("temp.wav")
        # 변환된 텍스트 반환
        return result.get("text", "")
    except Exception as e:
        print(f"오류 발생 (텍스트 변환): {e}")
        return None


def handle_audio_input():
    """
    마이크로부터 음성을 입력받아 오디오 데이터를 반환하는 함수
    :return: 녹음된 오디오 데이터 (speech_recognition.AudioData 객체)
    """
    recognizer = sr.Recognizer()  # 음성 인식기 초기화
    microphone = sr.Microphone()  # 마이크 장치 초기화

    recognizer.pause_threshold = 1.2  # 음성 인식 시 일시정지 임계값 설정
    print("음성 비서가 준비되었습니다. 질문을 말씀하세요.")

    while True:
        try:
            with microphone as source:
                # 주변 소음으로부터 마이크 조정
                recognizer.adjust_for_ambient_noise(source)
                print("질문을 듣는 중...")
                # 사용자의 음성 입력을 듣고 오디오 데이터를 캡처
                audio = recognizer.listen(source, timeout=None)
            return audio
        except sr.UnknownValueError:
            print("음성을 이해하지 못했습니다. 다시 말씀해주세요.")
        except Exception as e:
            print(f"오류 발생 (오디오 입력): {e}")


def main():
    """
    메인 함수: 음성 입력을 받아 텍스트로 변환하고 출력
    """
    while True:
        try:
            # 음성 입력 처리 및 오디오 데이터 획득
            audio_data = handle_audio_input()
            # 오디오 데이터를 텍스트로 변환
            transcribed_text = transcribe_audio_to_text(audio_data)
            if not transcribed_text:
                print("텍스트 변환 실패: 다시 질문해주세요.")
                continue
            # 변환된 텍스트 출력
            print(transcribed_text)
        except KeyboardInterrupt:
            print("프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"예외 발생: {e}")


if __name__ == "__main__":
    main()  # 메인 함수 실행
