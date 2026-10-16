# serving/app.py
import io
import json
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from inference import AnomalyDetector

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
detector = AnomalyDetector("models/bottle_model.onnx")

ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg"}
MAX_FILE_SIZE_MB = 10

# --- 드리프트 감지용 상태 (메모리 기반, 서버 재시작하면 초기화됨) ---
WINDOW_SIZE = 20
DRIFT_THRESHOLD = 0.5  # 최근 WINDOW_SIZE개 중 50% 넘게 이상 판정되면 경고
recent_results = deque(maxlen=WINDOW_SIZE)


def log_event(level: str, **fields):
    entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "level": level, **fields}
    getattr(logger, level, logger.info)(json.dumps(entry, ensure_ascii=False))


def check_drift():
    if len(recent_results) < WINDOW_SIZE:
        return None
    anomaly_rate = sum(recent_results) / len(recent_results)
    if anomaly_rate > DRIFT_THRESHOLD:
        log_event(
            "warning",
            event="possible_drift",
            anomaly_rate=round(anomaly_rate, 2),
            window_size=WINDOW_SIZE,
            message="최근 요청 중 이상 판정 비율이 비정상적으로 높습니다. 카메라/조명/제품 라인 변화를 확인하세요.",
        )
    return anomaly_rate


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/drift-status")
def drift_status():
    if len(recent_results) < WINDOW_SIZE:
        return {"status": "insufficient_data", "collected": len(recent_results), "needed": WINDOW_SIZE}
    anomaly_rate = sum(recent_results) / len(recent_results)
    return {
        "status": "drift_detected" if anomaly_rate > DRIFT_THRESHOLD else "normal",
        "recent_anomaly_rate": round(anomaly_rate, 2),
        "window_size": WINDOW_SIZE,
        "threshold": DRIFT_THRESHOLD,
    }


@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    start = time.time()

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

    score_val = float(np.array(result["pred_score"]).flatten()[0])
    label_val = bool(np.array(result["pred_label"]).flatten()[0])
    latency_ms = (time.time() - start) * 1000

    recent_results.append(label_val)
    check_drift()

    log_event(
        "info",
        filename=file.filename,
        anomaly_score=round(score_val, 4),
        is_anomaly=label_val,
        latency_ms=round(latency_ms, 2),
    )

    return {
        "filename": file.filename,
        "anomaly_score": round(score_val, 4),
        "is_anomaly": label_val,
    }