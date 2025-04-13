import pyttsx3

engine = pyttsx3.init()
# engine.setProperty("rate", 180)
voices = engine.getProperty('voices')  # ✅ 여기서 voices 리스트를 가져옵니다

for voice in voices:
    if 'korean' in voice.name.lower() or 'ko_' in str(voice.languages).lower():
        try:
            engine.setProperty('voice', voice.id)
            engine.say("안녕하세요. 저는 한국어 음성 비서입니다. 무엇을 도와드릴까요?")
            print(f"안녕하세요. 이 목소리는 {voice.name}입니다.")
            print(f"Name: {voice.name}")
            print(f"ID: {voice.id}")
            print(f"Languages: {voice.languages}")
            print("-" * 40)
            engine.runAndWait()
        except Exception as e:
            print(f"❌ {voice.name} 보이스 재생 실패: {e}")


# engine.say("안녕하세요. 저는 한국어 음성 비서입니다.")
# engine.runAndWait()
