# serving/app.py
import io
import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from inference import load_detectors

# --- 로깅 설정 ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("anomaly_api")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(LOG_DIR / "requests.log", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())

app = FastAPI(title="Industrial Anomaly Detection API")

# models/<category>_model.onnx 를 모두 로드한다 (예: bottle_model.onnx, capsule_model.onnx).
# 모델 파일은 용량 때문에 저장소에 포함하지 않으므로, README의 "모델 준비" 절차대로 먼저 만들어야 한다.
MODELS_DIR = Path(__file__).resolve().parent / "models"
detectors = load_detectors(MODELS_DIR)
if not detectors:
    raise RuntimeError(
        f"{MODELS_DIR} 에서 <category>_model.onnx 파일을 찾지 못했습니다. "
        "README의 '모델 준비' 절차를 참고해 ONNX 모델을 먼저 준비하세요."
    )
DEFAULT_CATEGORY = "bottle" if "bottle" in detectors else next(iter(detectors))

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MAX_FILE_SIZE_MB = 10

# --- 드리프트 감지용 상태 (메모리 기반, 서버 재시작하면 초기화됨) ---
# 제품 라인(카테고리)마다 정상 이상 비율이 다르므로 윈도우를 카테고리별로 따로 둔다.
WINDOW_SIZE = 20
DRIFT_THRESHOLD = 0.5  # 최근 WINDOW_SIZE개 중 50% 넘게 이상 판정되면 경고
recent_results: dict[str, deque] = defaultdict(lambda: deque(maxlen=WINDOW_SIZE))


def log_event(level: str, **fields):
    entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "level": level, **fields}
    getattr(logger, level, logger.info)(json.dumps(entry, ensure_ascii=False))


def get_detector(category: str):
    if category not in detectors:
        raise HTTPException(
            status_code=400,
            detail=f"로드된 모델이 없는 카테고리입니다: {category}. 사용 가능: {sorted(detectors)}",
        )
    return detectors[category]


def check_drift(category: str):
    window = recent_results[category]
    if len(window) < WINDOW_SIZE:
        return None
    anomaly_rate = sum(window) / len(window)
    if anomaly_rate > DRIFT_THRESHOLD:
        log_event(
            "warning",
            event="possible_drift",
            category=category,
            anomaly_rate=round(anomaly_rate, 2),
            window_size=WINDOW_SIZE,
            message="최근 요청 중 이상 판정 비율이 비정상적으로 높습니다. 카메라/조명/제품 라인 변화를 확인하세요.",
        )
    return anomaly_rate


@app.get("/health")
def health():
    return {"status": "ok", "categories": sorted(detectors)}


@app.get("/drift-status")
def drift_status(category: str = Query(None, description="검사 카테고리 (기본: bottle)")):
    category = category or DEFAULT_CATEGORY
    get_detector(category)
    window = recent_results[category]
    if len(window) < WINDOW_SIZE:
        return {"category": category, "status": "insufficient_data", "collected": len(window), "needed": WINDOW_SIZE}
    anomaly_rate = sum(window) / len(window)
    return {
        "category": category,
        "status": "drift_detected" if anomaly_rate > DRIFT_THRESHOLD else "normal",
        "recent_anomaly_rate": round(anomaly_rate, 2),
        "window_size": WINDOW_SIZE,
        "threshold": DRIFT_THRESHOLD,
    }


@app.post("/inspect")
async def inspect(
    file: UploadFile = File(...),
    category: str = Query(None, description="검사 카테고리 (기본: bottle). 예: /inspect?category=capsule"),
):
    start = time.time()
    category = category or DEFAULT_CATEGORY
    detector = get_detector(category)

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        log_event("warning", filename=file.filename, error="unsupported_content_type", content_type=file.content_type)
        raise HTTPException(status_code=400, detail=f"지원하지 않는 파일 형식입니다: {file.content_type}. PNG/JPEG만 허용됩니다.")

    contents = await file.read()

    if not contents:
        log_event("warning", filename=file.filename, error="empty_file")
        raise HTTPException(status_code=400, detail="빈 파일입니다.")

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        log_event("warning", filename=file.filename, error="file_too_large", size_mb=round(len(contents) / 1024 / 1024, 2))
        raise HTTPException(status_code=400, detail=f"파일이 너무 큽니다. {MAX_FILE_SIZE_MB}MB 이하만 허용됩니다.")

    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        log_event("warning", filename=file.filename, error="invalid_image")
        raise HTTPException(status_code=400, detail="이미지 파일을 읽을 수 없습니다. 손상되었거나 지원하지 않는 형식입니다.")

    try:
        result = detector.predict(image)
    except Exception:
        log_event("error", filename=file.filename, error="inference_failed")
        raise HTTPException(status_code=500, detail="모델 추론 중 오류가 발생했습니다.")

    # 최종 판정은 ONNX 내장 pred_label이 아니라 카테고리별 캘리브레이션 임계값 기준 (inference.py 참고)
    score_val = result["anomaly_score"]
    label_val = result["is_anomaly"]
    latency_ms = (time.time() - start) * 1000

    recent_results[category].append(label_val)
    check_drift(category)

    log_event(
        "info",
        filename=file.filename,
        category=category,
        anomaly_score=round(score_val, 4),
        is_anomaly=label_val,
        threshold=result["threshold_used"],
        latency_ms=round(latency_ms, 2),
    )

    return {
        "filename": file.filename,
        "category": category,
        "anomaly_score": round(score_val, 4),
        "is_anomaly": label_val,
        "threshold": result["threshold_used"],
    }