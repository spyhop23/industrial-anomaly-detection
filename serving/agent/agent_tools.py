# serving/agent/agent_tools.py
"""
7주차: 기존 추론 함수(AnomalyDetector)와 실험 기록(.md 문서)을
LangChain Tool로 wrapping한다.

이 파일은 serving/agent/ 폴더 안에 둔다 (serving/ 바로 아래가 아님).
경로는 모두 이 파일의 실제 위치(__file__) 기준으로 계산하므로,
python을 어느 폴더에서 실행하든 항상 올바른 scripts/docs를 찾는다.

폴더 구조 가정:
    industrial-anomaly-detection/
    ├── scripts/docs/*.md        <- 벤치마크 결과가 적힌 문서들
    └── serving/
        ├── app.py               <- FastAPI 추론 서버 (/inspect)
        ├── models/<category>_model.onnx
        └── agent/               <- 이 파일들이 여기 위치
            ├── agent.py
            ├── agent_tools.py   <- (현재 파일)
            ├── requirements-agent.txt
            └── .env / .env.example

- inspect_image_tool: FastAPI 서버의 /inspect 엔드포인트를 호출하는 Tool.
  "이 이미지 검사해줘" 같은 자연어 요청에서 이미지 경로를 받아 실행한다.
  모델을 직접 로드하지 않고 API를 거치기 때문에, 에이전트 요청도 서버의 입력 검증·
  구조화 로깅·드리프트 감지·캘리브레이션 임계값을 똑같이 적용받는다.
  (실행 전 serving/ 폴더에서 `uvicorn app:app --port 8000`으로 서버를 먼저 띄워야 한다.)
- query_experiment_results_tool: scripts/docs/*.md 에 쌓아온 실험 기록(MLflow 대신
  마크다운으로 남긴 벤치마크 표)을 읽어서, 카테고리/백엔드 키워드로 관련 내용을 찾아준다.
  "bottle이 TensorRT에서 AUROC 얼마야?" 같은 질문에 답하기 위한 Tool.
"""

import os
import re
from pathlib import Path

import requests
from langchain_core.tools import tool

# --- 경로 설정 (이 파일의 실제 위치 기준으로 계산 — cwd에 의존하지 않음) -------
AGENT_DIR = Path(__file__).resolve().parent          # .../serving/agent
SERVING_DIR = AGENT_DIR.parent                        # .../serving
PROJECT_ROOT = SERVING_DIR.parent                     # .../industrial-anomaly-detection

# 추론 서버 주소는 환경변수 ANOMALY_API_URL로 바꿀 수 있다 (.env에 넣어도 됨).
# load_dotenv()가 이 모듈 import 이후에 호출되므로, 값은 호출 시점에 읽는다.
DEFAULT_API_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_SEC = 30
# 실험 기록이 쌓여있는 마크다운 문서들 (주차별 트러블슈팅 기록에 벤치마크 표가 들어있음)
DOCS_DIR = PROJECT_ROOT / "scripts" / "docs"

_MIME_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


class InspectionError(Exception):
    """사용자에게 그대로 보여줄 수 있는 검사 실패 사유."""


def _resolve_image_path(image_path: str) -> Path:
    path = Path(image_path)
    if path.exists():
        return path
    # 상대경로로 주어졌는데 못 찾으면, serving/ 폴더 기준으로도 한 번 더 찾아본다.
    fallback = SERVING_DIR / image_path
    if fallback.exists():
        return fallback
    raise InspectionError(f"'{image_path}' 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")


def inspect_via_api(image_path: str, category: str = "bottle") -> dict:
    """이미지를 FastAPI /inspect 로 보내고 JSON 결과를 반환한다 (CrewAI Tool에서도 재사용)."""
    api_url = os.getenv("ANOMALY_API_URL", DEFAULT_API_URL).rstrip("/")
    path = _resolve_image_path(image_path)
    content_type = _MIME_TYPES.get(path.suffix.lower())
    if content_type is None:
        raise InspectionError(f"'{path.name}'는 지원하지 않는 형식입니다. PNG/JPEG만 검사할 수 있습니다.")

    try:
        with path.open("rb") as f:
            resp = requests.post(
                f"{api_url}/inspect",
                params={"category": category},
                files={"file": (path.name, f, content_type)},
                timeout=REQUEST_TIMEOUT_SEC,
            )
    except requests.ConnectionError:
        raise InspectionError(
            f"추론 서버({api_url})에 연결할 수 없습니다. "
            "serving/ 폴더에서 `uvicorn app:app --port 8000`으로 서버를 먼저 실행해주세요."
        )
    except requests.Timeout:
        raise InspectionError(f"추론 서버 응답이 {REQUEST_TIMEOUT_SEC}초 안에 오지 않았습니다.")

    if resp.status_code != 200:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise InspectionError(f"서버가 요청을 거부했습니다 ({resp.status_code}): {detail}")
    return resp.json()


# --- Tool 1: 모델 추론 ----------------------------------------------------

@tool
def inspect_image_tool(image_path: str, category: str = "bottle") -> str:
    """주어진 경로의 이미지 파일을 이상 탐지 모델로 검사한다.
    사용자가 "이 이미지 검사해줘", "불량인지 봐줘" 같은 요청을 할 때 사용한다.
    image_path는 로컬 디스크에 있는 이미지 파일의 경로여야 한다 (예: bottle_bad.png).
    category는 제품 종류다 (예: bottle, capsule). 사용자가 말하지 않으면 bottle을 쓴다.
    이상 점수(anomaly_score)와 최종 판정(is_anomaly)을 반환한다.
    """
    try:
        result = inspect_via_api(image_path, category)
    except InspectionError as e:
        return f"오류: {e}"

    verdict = "불량(이상)" if result["is_anomaly"] else "정상"
    return (
        f"검사 결과 — 파일: {result['filename']} (카테고리: {result['category']})\n"
        f"- anomaly_score: {result['anomaly_score']:.4f} (판정 임계값 {result['threshold']:.3f})\n"
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
