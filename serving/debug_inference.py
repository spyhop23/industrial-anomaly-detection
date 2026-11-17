# serving/debug_inference.py
"""
전처리/추론 결과를 눈으로 확인하는 디버그 스크립트 (5주차 전처리 버그 추적에 사용).
자동화된 테스트가 아니라, 이미지 몇 장을 넣고 텐서 값 범위와 ONNX 원시 출력을 출력해 보는 용도다.

사용법 (serving/ 폴더에서):
    python debug_inference.py /path/to/mvtec_ad/bottle/test/good/000.png /path/to/mvtec_ad/bottle/test/broken_large/000.png
    python debug_inference.py --category capsule /path/to/image.png
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from inference import AnomalyDetector

MODELS_DIR = Path(__file__).resolve().parent / "models"


def main():
    parser = argparse.ArgumentParser(description="ONNX 추론 결과 디버그 출력")
    parser.add_argument("images", nargs="+", help="검사할 이미지 경로")
    parser.add_argument("--category", default="bottle", help="models/<category>_model.onnx 를 사용")
    args = parser.parse_args()

    detector = AnomalyDetector(MODELS_DIR / f"{args.category}_model.onnx", category=args.category)

    for filename in args.images:
        image = Image.open(filename)
        print(f"\n=== {filename} ===")
        print("원본 이미지 크기:", image.size, image.mode)

        input_tensor = detector.preprocess(image)
        print("전처리 후 shape:", input_tensor.shape)
        print("전처리 후 값 범위: min=%.4f max=%.4f mean=%.4f" % (
            input_tensor.min(), input_tensor.max(), input_tensor.mean()
        ))

        result = detector.predict(image)
        print(f"anomaly_score={result['anomaly_score']:.4f}  threshold={result['threshold_used']:.3f}  "
              f"is_anomaly={result['is_anomaly']}")
        for name, value in result["raw"].items():
            arr = np.array(value)
            if arr.size > 1:
                print(f"  raw.{name}: shape={arr.shape} min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}")
            else:
                print(f"  raw.{name}: {arr}")


if __name__ == "__main__":
    main()
