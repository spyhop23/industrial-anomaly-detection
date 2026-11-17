# 빌드 전 serving/models/ 에 <category>_model.onnx 가 있어야 한다.
# (*.onnx 는 용량 때문에 git에 포함하지 않음 — README "모델 준비" 참고)
#   docker build -t anomaly-api serving/
#   docker run -p 8000:8000 anomaly-api
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries=10 -r requirements.txt

COPY app.py inference.py ./
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]