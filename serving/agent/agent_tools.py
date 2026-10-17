# serving/agent/agent_tools.py
"""
7주차: 기존 추론 함수(AnomalyDetector)와 실험 기록(.md 문서)을
LangChain Tool로 wrapping한다.

이 파일은 serving/agent/ 폴더 안에 둔다 (serving/ 바로 아래가 아님).
경로는 모두 이 파일의 실제 위치(__file__) 기준으로 계산하므로,
python을 어느 폴더에서 실행하든(serving/agent에서 실행하든, 다른 곳에서 실행하든)
항상 올바른 inference.py / models / scripts/docs를 찾는다.

폴더 구조 가정:
    industrial-anomaly-detection/
    ├── scripts/docs/*.md        <- 벤치마크 결과가 적힌 문서들
    └── serving/
        ├── inference.py         <- 기존 AnomalyDetector
        ├── models/bottle_model.onnx
        └── agent/               <- 이 파일들이 여기 위치
            ├── agent.py
            ├── agent_tools.py   <- (현재 파일)
            ├── requirements-agent.txt
            └── .env / .env.example

- inspect_image_tool: inference.py의 AnomalyDetector.predict()를 그대로 감싼 Tool.
  "이 이미지 검사해줘" 같은 자연어 요청에서 이미지 경로를 받아 실행한다.
- query_experiment_results_tool: scripts/docs/*.md 에 쌓아온 실험 기록(MLflow 대신
  마크다운으로 남긴 벤치마크 표)을 읽어서, 카테고리/백엔드 키워드로 관련 내용을 찾아준다.
  "bottle이 TensorRT에서 AUROC 얼마야?" 같은 질문에 답하기 위한 Tool.
"""

import re
import sys
from pathlib import Path

from langchain_core.tools import tool
from PIL import Image, UnidentifiedImageError

# --- 경로 설정 (이 파일의 실제 위치 기준으로 계산 — cwd에 의존하지 않음) -------
AGENT_DIR = Path(__file__).resolve().parent          # .../serving/agent
SERVING_DIR = AGENT_DIR.parent                        # .../serving
PROJECT_ROOT = SERVING_DIR.parent                     # .../industrial-anomaly-detection

# inference.py가 serving/ 바로 아래에 있으므로, import 할 수 있도록 경로를 추가해준다.
sys.path.insert(0, str(SERVING_DIR))
from inference import AnomalyDetector  # noqa: E402  (sys.path 설정 후에 import해야 함)

# app.py와 동일한 모델을 사용. 에이전트 프로세스가 시작될 때 한 번만 로드한다.
MODEL_PATH = SERVING_DIR / "models" / "bottle_model.onnx"
# 실험 기록이 쌓여있는 마크다운 문서들 (주차별 트러블슈팅 기록에 벤치마크 표가 들어있음)
DOCS_DIR = PROJECT_ROOT / "scripts" / "docs"

_detector: AnomalyDetector | None = None


def _get_detector() -> AnomalyDetector:
    """모델은 최초 호출 시 한 번만 로드하고 재사용한다 (매 요청마다 로드하면 느림)."""
    global _detector
    if _detector is None:
        _detector = AnomalyDetector(str(MODEL_PATH))
    return _detector


# --- Tool 1: 모델 추론 ----------------------------------------------------

@tool
def inspect_image_tool(image_path: str) -> str:
    """주어진 경로의 이미지 파일을 이상 탐지 모델로 검사한다.
    사용자가 "이 이미지 검사해줘", "불량인지 봐줘" 같은 요청을 할 때 사용한다.
    image_path는 로컬 디스크에 있는 이미지 파일의 경로여야 한다 (예: bottle_bad.png).
    이상 점수(anomaly_score)와 최종 판정(is_anomaly)을 반환한다.
    """
    path = Path(image_path)
    if not path.exists():
        # 상대경로로 주어졌는데 못 찾으면, serving/ 폴더 기준으로도 한 번 더 찾아본다.
        # (bottle_good.png 같은 테스트 이미지들이 serving/ 바로 아래 있기 때문)
        fallback = SERVING_DIR / image_path
        if fallback.exists():
            path = fallback
        else:
            return f"오류: '{image_path}' 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요."

    try:
        image = Image.open(path)
        image.verify()
        image = Image.open(path)  # verify() 이후에는 다시 열어야 함
    except UnidentifiedImageError:
        return f"오류: '{image_path}'는 올바른 이미지 파일이 아닙니다."
    except Exception as e:
        return f"오류: 이미지를 여는 중 문제가 발생했습니다 ({e})."

    try:
        detector = _get_detector()
        result = detector.predict(image)
    except Exception as e:
        return f"오류: 모델 추론 중 문제가 발생했습니다 ({e})."

    import numpy as np

    score = float(np.array(result["pred_score"]).flatten()[0])
    is_anomaly = bool(np.array(result["pred_label"]).flatten()[0])
    verdict = "불량(이상)" if is_anomaly else "정상"

    return (
        f"검사 결과 — 파일: {path.name}\n"
        f"- anomaly_score: {score:.4f}\n"
        f"- 판정: {verdict}"
    )


# --- Tool 2: 실험 기록(MLflow 대체) 조회 ------------------------------------

# "bottle이 PyTorch에서", "capsule TensorRT 레이턴시", "screw PatchCore AUROC" 처럼
# 카테고리/백엔드 키워드가 섞인 자연어 질의를 처리하기 위해, 문서 안의 마크다운 표를
# 통째로 파싱해서 카테고리별 레코드로 만들어둔다.
_TABLE_ROW_RE = re.compile(r"^\|\s*(\w[\w.]*)\s*\|(.+)\|\s*$", re.MULTILINE)


def _load_benchmark_tables() -> list[dict]:
    """scripts/docs/*.md 안의 마크다운 표를 찾아 {category, raw_row, source_file} 리스트로 반환."""
    records = []
    if not DOCS_DIR.exists():
        return records

    for md_file in sorted(DOCS_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        lines = text.splitlines()
        header_cols = None
        for line in lines:
            if not line.strip().startswith("|"):
                continue
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if header_cols is None:
                header_cols = cols
                continue
            if set(cols) <= {"---"} or all(set(c) <= {"-", ":"} for c in cols):
                continue  # 구분선(|---|---|) 스킵
            if len(cols) != len(header_cols):
                header_cols = None  # 다음 표로 넘어감
                continue
            row = dict(zip(header_cols, cols))
            records.append({"source_file": md_file.name, "row": row})
    return records


@tool
def query_experiment_results_tool(query: str) -> str:
    """실험 결과 기록(MLflow 대신 주차별 .md 문서에 정리된 벤치마크 표)에서
    카테고리나 백엔드 이름으로 관련 결과를 찾아준다.
    예: "bottle이 TensorRT에서 AUROC 얼마야?", "capsule 레이턴시 알려줘".
    query에는 찾고 싶은 카테고리명, 모델/백엔드 이름 등 키워드를 자유롭게 넣는다.
    """
    records = _load_benchmark_tables()
    if not records:
        return "실험 기록 문서(scripts/docs/*.md)를 찾을 수 없습니다."

    keywords = [w.lower() for w in re.findall(r"[a-zA-Z0-9_가-힣]+", query)]
    matches = []
    for rec in records:
        row_text = " ".join(rec["row"].values()).lower()
        if any(kw in row_text for kw in keywords):
            matches.append(rec)

    if not matches:
        available = sorted({rec["row"].get("category", "") for rec in records if rec["row"].get("category")})
        return (
            f"'{query}'에 해당하는 기록을 찾지 못했습니다. "
            f"기록에 있는 카테고리: {', '.join(available) if available else '(표 형식을 인식하지 못함)'}"
        )

    lines = []
    for rec in matches:
        row_str = ", ".join(f"{k}={v}" for k, v in rec["row"].items())
        lines.append(f"[{rec['source_file']}] {row_str}")
    return "\n".join(lines)
