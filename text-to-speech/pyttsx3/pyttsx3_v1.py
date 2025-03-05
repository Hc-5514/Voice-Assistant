import pyttsx3  # Text-to-Speech 라이브러리

# pyttsx3 엔진 초기화
engine = pyttsx3.init()

# 음성 속도 조절 (기본값: 200)
engine.setProperty("rate", 180)  # 속도 줄이기 (값을 높이면 빠름)

# 볼륨 설정 (0.0 ~ 1.0)
engine.setProperty("volume", 1.0)  # 최대 볼륨


def speak_text(text):
    """텍스트를 음성으로 변환하여 출력"""
    try:
        engine.say(text)  # 입력된 텍스트를 음성으로 변환
        engine.runAndWait()  # 음성을 실행
    except Exception as e:
        print(f"오류 발생 (음성 출력): {e}")


# 테스트 실행ㅅ
speak_text("안녕하세요! 저는 pyttsx3을 이용한 음성 비서입니다.")
