# serving/inference.py
import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_SIZE = (256, 256)

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

    def predict(self, image: Image.Image) -> dict:
        input_tensor = self.preprocess(image)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        return dict(zip(self.output_names, outputs))