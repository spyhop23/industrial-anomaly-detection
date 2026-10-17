# serving/agent/agent.py
"""
7주차: 단일 에이전트 — 자연어로 시스템과 상호작용.

inspect_image_tool / query_experiment_results_tool 두 개를 LangChain 에이전트에
연결해서, 터미널에서 자연어로 질의응답할 수 있게 한다.

이 파일은 serving/agent/ 폴더 안에 둔다 (app.py가 있는 serving/ 바로 아래가 아님).
기존 서빙 코드(app.py, inference.py)와 섞이지 않도록 분리된 서브폴더다.

실행 전 준비:
    1. serving/agent 폴더에서: pip install -r ../requirements.txt -r requirements-agent.txt
    2. .env.example을 복사해 .env 파일을 만들고 OPENAI_API_KEY=sk-... 설정

실행 (반드시 serving/agent 폴더 안에서):
    cd serving/agent
    python agent.py

예시 질문:
    - "bottle_bad.png 이미지 검사해줘"
    - "bottle이 TensorRT에서 AUROC 얼마나 나왔어?"
    - "capsule PyTorch 레이턴시 알려줘"
"""

import os

from dotenv import load_dotenv
# LangChain 1.0부터는 AgentExecutor + create_tool_calling_agent(구버전 문법) 대신
# create_agent 하나로 에이전트를 만든다 (내부적으로 LangGraph 기반으로 재작성됨).
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from agent_tools import inspect_image_tool, query_experiment_results_tool

load_dotenv()

SYSTEM_PROMPT = """당신은 공장 품질 검사 담당자를 돕는 어시스턴트입니다.
두 가지 도구를 사용할 수 있습니다:
1. inspect_image_tool: 이미지 파일을 이상 탐지 모델로 검사합니다.
2. query_experiment_results_tool: 과거 실험(벤치마크) 기록에서 모델 성능 수치를 찾습니다.

사용자가 이미지 검사를 요청하면 inspect_image_tool을 쓰고,
모델 성능이나 과거 실험 결과를 물어보면 query_experiment_results_tool을 쓰세요.
도구 결과를 그대로 나열하지 말고, 자연스러운 한국어 문장으로 설명해주세요.
"""


def build_agent():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 만들고 "
            "OPENAI_API_KEY=sk-... 를 추가해주세요 (.env.example 참고)."
        )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [inspect_image_tool, query_experiment_results_tool]

    # create_agent가 "Tool 목록 + LLM + 시스템 프롬프트"를 받아서
    # 알아서 판단-실행 루프(LangGraph 그래프)를 구성해준다.
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def main():
    agent = build_agent()
    print("=== 품질 검사 에이전트 (종료: exit) ===")
    while True:
        try:
            user_input = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "종료"}:
            break

        # create_agent로 만든 에이전트는 메시지 목록을 입력/출력으로 주고받는다.
        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        answer = result["messages"][-1].content
        print(f"\n답변: {answer}")


if __name__ == "__main__":
    main()
