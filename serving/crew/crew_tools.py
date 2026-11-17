# serving/crew/crew_tools.py
"""
8주차: CrewAI용 Tool.

7주차에서 만든 agent_tools.py의 핵심 로직(추론 API 호출, 벤치마크 문서 파싱)을
재사용하되, CrewAI의 Tool 형식(@tool 데코레이터)으로 다시 감싼다.
LangChain Tool과 CrewAI Tool은 서로 다른 타입이라 그대로 가져다 쓸 수 없어서,
내부 함수(inspect_via_api, _load_benchmark_tables)만 가져와 재사용한다.
이미지 검사는 FastAPI 서버(/inspect)를 거치므로, 실행 전 서버를 먼저 띄워야 한다.
"""

import re
import sys
from pathlib import Path

from crewai.tools import tool

# --- 경로 설정: serving/agent/agent_tools.py 의 내부 함수를 재사용 ---------
CREW_DIR = Path(__file__).resolve().parent            # .../serving/crew
SERVING_DIR = CREW_DIR.parent                           # .../serving
AGENT_DIR = SERVING_DIR / "agent"                        # .../serving/agent

sys.path.insert(0, str(AGENT_DIR))
from agent_tools import InspectionError, _load_benchmark_tables, inspect_via_api  # noqa: E402


@tool("inspect_image")
def crew_inspect_image(image_path: str, category: str = "bottle") -> str:
    """주어진 경로의 이미지 파일을 이상 탐지 모델로 검사한다.
    category는 제품 종류다 (예: bottle, capsule). 기본값은 bottle.
    filename, category, anomaly_score, threshold, is_anomaly를 포함한 결과 문자열을 반환한다.
    """
    try:
        result = inspect_via_api(image_path, category)
    except InspectionError as e:
        return f"오류: {e}"

    return (
        f"filename={result['filename']}, category={result['category']}, "
        f"anomaly_score={result['anomaly_score']:.4f}, threshold={result['threshold']:.3f}, "
        f"is_anomaly={result['is_anomaly']}"
    )


@tool("query_experiment_results")
def crew_query_experiment_results(query: str) -> str:
    """실험 결과 기록(.md 문서)에서 카테고리나 백엔드 이름으로 관련 결과를 찾는다.
    예: "bottle TensorRT AUROC".
    """
    records = _load_benchmark_tables()
    if not records:
        return "실험 기록 문서(scripts/docs/*.md)를 찾을 수 없습니다."

    keywords = [w.lower() for w in re.findall(r"[a-zA-Z0-9_가-힣]+", query)]
    matches = [
        rec for rec in records
        if any(kw in " ".join(rec["row"].values()).lower() for kw in keywords)
    ]

    if not matches:
        return f"'{query}'에 해당하는 기록을 찾지 못했습니다."

    return "\n".join(
        f"[{rec['source_file']}] " + ", ".join(f"{k}={v}" for k, v in rec["row"].items())
        for rec in matches
    )
