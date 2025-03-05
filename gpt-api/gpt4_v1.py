"""
GPT-4 기반 음성 비서 응답 생성기
"""

import os

import openai
from dotenv import load_dotenv

# .env 파일 로드 (API 키 불러오기)
load_dotenv()

# GPT API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("[ERROR] API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 시스템 프롬프트 (GPT 역할 설정)
SYSTEM_PROMPT = """
당신은 사용자의 음성 비서를 담당하는 AI입니다.
질문에 대해 즉시 답변하세요. "잠시만 기다려주세요" 같은 문장은 사용하지 마세요.
불필요한 단어를 제거하고 간결한 문장으로 정리하세요.
너무 긴 답변은 사용자가 쉽게 이해할 수 있도록 50자 이내로 요약하여 제공하세요.
친절하고 자연스러운 말투로 응답하세요.
"""


def generate_response(user_input):
    """
    GPT API를 호출하여 응답을 생성하는 함수
    :param user_input: 사용자의 입력 문장
    :return: GPT-4의 응답
    """
    try:
        print("[INFO] GPT 응답 생성 중...")

        # 현재 입력된 질문만 API 요청
        response = openai.ChatCompletion.create(
            model="gpt-4",  # 최신 GPT-4 모델 사용
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            max_tokens=256,  # 응답 길이 제한
            temperature=0.5,  # 응답의 창의성 조절
        )

        # GPT 응답 추출
        assistant_response = response["choices"][0]["message"]["content"].strip()

        return assistant_response

    except Exception as e:
        print(f"[ERROR] 응답 생성 중 오류 발생: {e}")
        return None


def main():
    """
    테스트 실행 함수 (예제 질문 입력 후 GPT 응답 출력)
    """
    test_inputs = ["오늘 서울 날씨 어때?", "오늘은 어떤 옷을 입으면 좋을까?"]

    for user_query in test_inputs:
        try:
            # GPT 응답 생성
            response = generate_response(user_query)
            if not response:
                print("[WARNING] GPT 응답 생성 실패")
                continue

            # 응답 출력
            print(f"GPT 응답: {response}")

        except KeyboardInterrupt:
            print("\n[INFO] 프로그램을 종료합니다.")
            exit(0)
        except Exception as e:
            print(f"[ERROR] 알 수 없는 오류 발생: {e}")


if __name__ == "__main__":
    main()
