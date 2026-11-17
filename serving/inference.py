# serving/inference.py
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_SIZE = (256, 256)

# 카테고리별 캘리브레이션된 임계값 (evaluation/robustness_eval.py 분석 결과 기반)
# ONNX 그래프에 내장된 pred_label은 학습 시점의 기본 임계값으로 계산되므로,
# 최종 판정(is_anomaly)은 항상 아래 임계값과 pred_score를 비교해서 만든다.
CALIBRATED_THRESHOLDS = {
    "bottle": 0.500,    # 변화 없음 — 분리력이 이미 충분
    "capsule": 0.504,   # 15000-step 체크포인트 기준, Precision 0.832→0.872
}
DEFAULT_THRESHOLD = 0.5

# models/ 폴더 안의 파일 이름 규칙: <category>_model.onnx (예: bottle_model.onnx)
MODEL_FILENAME_SUFFIX = "_model.onnx"


class AnomalyDetector:
    def __init__(self, onnx_path: str, category: str = "bottle"):
        self.category = category
        self.threshold = CALIBRATED_THRESHOLDS.get(category, DEFAULT_THRESHOLD)
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        arr = np.array(image, dtype=np.float32) / 255.0  # 0~1 스케일링만, 추가 정규화 없음
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return arr[np.newaxis, ...].astype(np.float32)  # (1, 3, H, W)

    def predict(self, image: Image.Image) -> dict:
        """원시 ONNX 출력과 함께, 캘리브레이션 임계값으로 판정한 결과를 반환한다.

        반환 키:
            anomaly_score  : float, 이미지 단위 이상 점수 (pred_score)
            is_anomaly     : bool, anomaly_score >= threshold_used
            threshold_used : float, 이 카테고리에 적용한 임계값
            raw            : dict, ONNX 원시 출력 (pred_score, pred_label, anomaly_map 등)
        """
        input_tensor = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        raw = dict(zip(self.output_names, outputs))

        score = float(np.asarray(raw["pred_score"]).ravel()[0])
        return {
            "anomaly_score": score,
            "is_anomaly": score >= self.threshold,
            "threshold_used": self.threshold,
            "raw": raw,
        }


def load_detectors(models_dir: str | Path) -> dict[str, AnomalyDetector]:
    """models_dir 안의 <category>_model.onnx 파일을 모두 찾아 카테고리별 detector를 만든다."""
    models_dir = Path(models_dir)
    detectors = {}
    for onnx_path in sorted(models_dir.glob(f"*{MODEL_FILENAME_SUFFIX}")):
        category = onnx_path.name[: -len(MODEL_FILENAME_SUFFIX)]
        detectors[category] = AnomalyDetector(onnx_path, category=category)
    return detectors
