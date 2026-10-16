# serving/test_inference.py
import numpy as np
from PIL import Image
from inference import AnomalyDetector

detector = AnomalyDetector("models/bottle_model.onnx")

for filename in ["bottle_test_a.png", "bottle_test_b.png"]: 
    image = Image.open(filename)
    print(f"\n=== {filename} ===")
    print("원본 이미지 크기:", image.size, image.mode)

    input_tensor = detector.preprocess(image)
    print("전처리 후 shape:", input_tensor.shape)
    print("전처리 후 값 범위: min=%.4f max=%.4f mean=%.4f" % (
        input_tensor.min(), input_tensor.max(), input_tensor.mean()
    ))

    result = detector.predict(image)
    for name, value in result.items():
        arr = np.array(value)
        if arr.size > 1:
            print(f"{name}: shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")
        else:
            print(f"{name}: {arr}")