# serving/inference.py
import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_SIZE = (256, 256)

# 카테고리별 캘리브레이션된 임계값 (robustness_eval.py 분석 결과 기반)
CALIBRATED_THRESHOLDS = {
    "bottle": 0.500,    # 변화 없음 — 분리력이 이미 충분
    "capsule": 0.504,   # 15000-step 체크포인트 기준, Precision 0.832→0.872
}
DEFAULT_THRESHOLD = 0.5

class AnomalyDetector:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        print("입력 이름:", self.input_name)
        print("출력 이름들:", self.output_names)

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        arr = np.array(image, dtype=np.float32) / 255.0  # 0~1 스케일링만, 추가 정규화 없음
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return arr[np.newaxis, ...].astype(np.float32)  # (1, 3, H, W)

    def predict(self, image: Image.Image, category: str = "bottle") -> dict:
        input_tensor = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        result = dict(zip(self.output_names, outputs))
        threshold = CALIBRATED_THRESHOLDS.get(category, DEFAULT_THRESHOLD)
        result["is_anomaly"] = bool(result["pred_score"] >= threshold)
        result["threshold_used"] = threshold
        return result