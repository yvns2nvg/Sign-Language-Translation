import os
from dotenv import load_dotenv
from anthropic import Anthropic

# .env 파일에서 환경 변수 로드
load_dotenv()

# ANTHROPIC_API_KEY 환경 변수가 설정되어 있어야 합니다.
api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    print("WARNING: ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("같은 폴더에 '.env' 파일을 생성하고 아래와 같이 API 키를 입력하세요:")
    print("ANTHROPIC_API_KEY=your_actual_api_key_here")
else:
    # 클라이언트 초기화
    client = Anthropic(api_key=api_key)

    try:
        print("Claude API 호출 중...")
        # API 호출 예시 (Claude 3.5 Sonnet 모델 사용)
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0,
            system="당신은 친절한 인공지능 비서입니다.",
            messages=[
                {
                    "role": "user",
                    "content": "안녕하세요! 연결이 잘 되었는지 확인하기 위해 인사말 한마디 부탁드려요."
                }
            ]
        )
        print("\n=== Claude 답변 ===")
        print(message.content[0].text)
        print("====================")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
