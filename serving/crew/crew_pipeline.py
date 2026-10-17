# serving/crew/crew_pipeline.py
"""
8주차: AI Agent 레이어 ② — 멀티 에이전트 (CrewAI)

세 개의 역할로 나뉜 에이전트가 순서대로 협업한다:
    1. 검사 에이전트 (Inspector)  — 이미지를 모델로 검사
    2. 분석 에이전트 (Analyst)    — 이상 패턴/심각도 판단
    3. 보고서 작성 에이전트 (Reporter) — 불량 보고서 자동 생성

실행 전 준비:
    1. serving/crew 폴더에서: pip install -r ../requirements.txt -r requirements-crew.txt
    2. .env.example을 복사해 .env 파일을 만들고 OPENAI_API_KEY=sk-... 설정
       (이미 serving/agent/.env를 만들어뒀다면 그 내용을 그대로 복사해도 된다)

실행 (반드시 serving/crew 폴더 안에서):
    cd serving/crew
    python crew_pipeline.py ../bottle_bad.png
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv

from crew_tools import crew_inspect_image, crew_query_experiment_results

load_dotenv()

CREW_DIR = Path(__file__).resolve().parent
REPORTS_DIR = CREW_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# anomaly_score를 심각도 등급으로 나누는 기준.
# 현재 보유한 소량의 샘플 이미지 점수 분포(0.49~0.68 구간)를 바탕으로 잡은 임시 기준이며,
# 실제 생산 데이터가 쌓이면 재보정이 필요하다 (9주차 의사결정 문서화에 기록할 것).
SEVERITY_GUIDE = """
anomaly_score 기준 심각도 판단 가이드 (0.50이 정상/이상 판정 경계):
- 0.50 미만         : 정상 (불량 아님)
- 0.50 이상 0.60 미만: 경미한 이상 (육안 재확인 권장)
- 0.60 이상 0.70 미만: 심각한 이상 (라인에서 제외 필요)
- 0.70 이상         : 매우 심각한 이상 (즉시 라인 중단 검토)
"""


def build_crew(image_path: str) -> Crew:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 만들고 "
            "OPENAI_API_KEY=sk-... 를 추가해주세요 (.env.example 참고)."
        )

    llm = LLM(model="gpt-4o-mini", temperature=0)

    inspector = Agent(
        role="품질 검사 에이전트",
        goal="주어진 이미지를 이상 탐지 모델로 검사하고, anomaly_score와 판정을 정확히 보고한다.",
        backstory=(
            "공장 생산 라인에서 제품 이미지를 검사하는 역할을 맡고 있다. "
            "반드시 inspect_image 도구를 사용해서 실제 모델 추론 결과를 얻어야 하며, "
            "도구를 쓰지 않고 임의로 결과를 지어내서는 안 된다."
        ),
        tools=[crew_inspect_image],
        llm=llm,
        verbose=True,
    )

    analyst = Agent(
        role="이상 패턴 분석 에이전트",
        goal="검사 에이전트의 결과를 받아 이상 패턴의 심각도를 판단하고 원인을 추정한다.",
        backstory=(
            "품질 데이터 분석가로서 anomaly_score를 보고 심각도를 분류한다.\n"
            f"{SEVERITY_GUIDE}\n"
            "필요하면 query_experiment_results 도구로 과거 벤치마크 기록(해당 카테고리의 평소 정확도 등)을 "
            "참고해서 판단 근거를 보강해도 좋다."
        ),
        tools=[crew_query_experiment_results],
        llm=llm,
        verbose=True,
    )

    reporter = Agent(
        role="보고서 작성 에이전트",
        goal="검사 결과와 분석 내용을 종합해 공장 관리자가 바로 읽을 수 있는 불량 보고서를 작성한다.",
        backstory=(
            "품질관리팀에 제출할 보고서를 작성하는 역할이다. 마크다운 형식으로, "
            "제목/검사 파일명/anomaly_score/판정/심각도/권장 조치 섹션을 포함한 "
            "간결하고 명확한 보고서를 쓴다. 불필요한 수식어 없이 사실 위주로 작성한다."
        ),
        llm=llm,
        verbose=True,
    )

    inspect_task = Task(
        description=f"이미지 파일 '{image_path}'를 inspect_image 도구로 검사하라.",
        expected_output="filename, anomaly_score, is_anomaly가 포함된 검사 결과 문자열",
        agent=inspector,
    )

    analyze_task = Task(
        description=(
            "검사 에이전트의 결과를 바탕으로 심각도(정상/경미한 이상/심각한 이상/매우 심각한 이상)를 "
            "판단하고, 판단 근거를 1~2문장으로 설명하라."
        ),
        expected_output="심각도 등급과 그 판단 근거",
        agent=analyst,
        context=[inspect_task],
    )

    report_task = Task(
        description=(
            "검사 결과와 분석 내용을 종합해 마크다운 형식의 불량 보고서를 작성하라. "
            "제목, 검사 파일명, anomaly_score, 판정, 심각도, 권장 조치 섹션을 반드시 포함해야 한다."
        ),
        expected_output="마크다운 형식의 완성된 불량 보고서 전문",
        agent=reporter,
        context=[inspect_task, analyze_task],
    )

    return Crew(
        agents=[inspector, analyst, reporter],
        tasks=[inspect_task, analyze_task, report_task],
        process=Process.sequential,
        verbose=True,
    )


def run_pipeline(image_path: str) -> Path:
    crew = build_crew(image_path)
    result = crew.kickoff()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(image_path).stem
    report_path = REPORTS_DIR / f"{timestamp}_{image_name}_report.md"
    report_path.write_text(str(result), encoding="utf-8")
    return report_path


def main():
    if len(sys.argv) < 2:
        print("사용법: python crew_pipeline.py <이미지_경로>")
        print("예시:   python crew_pipeline.py ../bottle_bad.png")
        sys.exit(1)

    image_path = sys.argv[1]
    report_path = run_pipeline(image_path)
    print(f"\n{'=' * 60}")
    print(f"보고서가 저장되었습니다: {report_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
