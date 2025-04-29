import speech_recognition as sr

print("🎤 사용 가능한 마이크 목록:")
for index, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"[{index}] {name}")
