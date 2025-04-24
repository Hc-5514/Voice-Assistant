from TTS.api import TTS
import torch
from TTS.tts.configs.xtts_config import XttsConfig

torch.serialization.add_safe_class(XttsConfig)

# 가장 자연스러운 다국어 모델 (한국어 포함)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True, gpu=False)

# 음성 합성
tts.tts_to_file(text="안녕하세요. 무엇을 도와드릴까요?", file_path="hello.wav")
