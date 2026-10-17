# serving/crew/crew_tools.py
"""
8주차: CrewAI용 Tool.

7주차에서 만든 agent_tools.py의 핵심 로직(모델 로드, 벤치마크 문서 파싱)을
재사용하되, CrewAI의 Tool 형식(@tool 데코레이터)으로 다시 감싼다.
LangChain Tool과 CrewAI Tool은 서로 다른 타입이라 그대로 가져다 쓸 수 없어서,
내부 함수(_get_detector, _load_benchmark_tables)만 가져와 재사용한다.
"""

import re
import sys
from pathlib import Path

import numpy as np
from crewai.tools import tool
from PIL import Image, UnidentifiedImageError

# --- 경로 설정: serving/agent/agent_tools.py 의 내부 함수를 재사용 ---------
CREW_DIR = Path(__file__).resolve().parent            # .../serving/crew
SERVING_DIR = CREW_DIR.parent                           # .../serving
AGENT_DIR = SERVING_DIR / "agent"                        # .../serving/agent

sys.path.insert(0, str(AGENT_DIR))
from agent_tools import _get_detector, _load_benchmark_tables  # noqa: E402


@tool("inspect_image")
def crew_inspect_image(image_path: str) -> str:
    """주어진 경로의 이미지 파일을 이상 탐지 모델로 검사한다.
    filename, anomaly_score, is_anomaly를 포함한 결과 문자열을 반환한다.
    """
    path = Path(image_path)
    if not path.exists():
        fallback = SERVING_DIR / image_path
        if fallback.exists():
            path = fallback
        else:
            return f"오류: '{image_path}' 파일을 찾을 수 없습니다."

    try:
        image = Image.open(path)
        image.verify()
        image = Image.open(path)
    except UnidentifiedImageError:
        return f"오류: '{image_path}'는 올바른 이미지 파일이 아닙니다."
    except Exception as e:
        return f"오류: 이미지를 여는 중 문제가 발생했습니다 ({e})."

    try:
        detector = _get_detector()
        result = detector.predict(image)
    except Exception as e:
        return f"오류: 모델 추론 중 문제가 발생했습니다 ({e})."

    score = float(np.array(result["pred_score"]).flatten()[0])
    is_anomaly = bool(np.array(result["pred_label"]).flatten()[0])

    return f"filename={path.name}, anomaly_score={score:.4f}, is_anomaly={is_anomaly}"


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
