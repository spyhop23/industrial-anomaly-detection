# serving/app.py
import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from inference import AnomalyDetector

app = FastAPI(title="Industrial Anomaly Detection API")
detector = AnomalyDetector("models/bottle_model.onnx")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/inspect")
async def inspect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = detector.predict(image)
    score_val = float(np.array(result["pred_score"]).flatten()[0])
    label_val = bool(np.array(result["pred_label"]).flatten()[0])
    return {
        "filename": file.filename,
        "anomaly_score": round(score_val, 4),
        "is_anomaly": label_val,
    }