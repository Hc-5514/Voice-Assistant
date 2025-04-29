import os
import time

from dotenv import load_dotenv

# 루트 디렉토리 경로를 강제로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path)

# 2. 환경변수 가져오기 + 절대경로 변환
key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if key_path is None:
    raise ValueError("❗ GOOGLE_APPLICATION_CREDENTIALS 가 .env 파일에 설정되지 않았습니다.")

abs_key_path = os.path.abspath(key_path)
print("✅ 변환된 절대경로:", abs_key_path)
print("✅ 파일 존재 여부:", os.path.exists(abs_key_path))

if not os.path.exists(abs_key_path):
    raise FileNotFoundError(f"🔴 JSON 파일을 찾을 수 없습니다: {abs_key_path}")

# 3. 환경변수 덮어쓰기
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = abs_key_path

# 4. 이제 GCP 라이브러리 import
from google.cloud import texttospeech

# 5. 클라이언트 생성
client = texttospeech.TextToSpeechClient()


def speak_text(text):
    try:
        timestamp = int(time.time())
        wav_file = f"tts_{timestamp}.wav"

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="ko-KR",
            name="ko-KR-Wavenet-B",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        with open(wav_file, "wb") as out:
            out.write(response.audio_content)

        print(f"🗣️ 음성 파일 저장 완료: {wav_file}")

        os.system(f"afplay {wav_file}")  # Mac
        # os.system(f"aplay {wav_file}") # Linux
        os.remove(wav_file)

    except Exception as e:
        print(f"[ERROR] 음성 생성 실패: {e}")


# 테스트 실행
if __name__ == "__main__":
    speak_text("안녕하세요. 나로봇입니다. 무엇을 도와드릴까요?")
