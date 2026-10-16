# 5주차 로컬 배포 — FastAPI + ONNX Runtime 서빙 트러블슈팅 기록

4주차까지는 Kaggle GPU 노트북에서 실험을 진행했지만, 5주차부터는 "지속적으로 떠 있는 API 서버"와 "Docker 이미지 빌드"가 필요해서 로컬(개인 노트북)로 작업 환경을 옮겼다. TensorRT 엔진은 GPU/드라이버 버전에 종속적이라 일반 노트북에 그대로 가져갈 수 없으므로, 배포 단계에서는 **portable한 ONNX Runtime(CPU)**으로 bottle 카테고리의 FP32 ONNX 모델을 서빙하는 방향으로 전환했다. (TensorRT는 "GPU 서빙 최적화", ONNX Runtime은 "이식성 있는 배포"라는 트레이드오프로 정리.)

로컬 환경을 처음부터 세팅하는 과정에서 겪은 문제들과, 그중 가장 치명적이었던 "정상/이상 구분이 전혀 안 되던" 버그의 원인을 정리한다.

## 1. 로컬 Python 환경 세팅 과정에서의 시행착오

- **Windows Python 설치 경로 변경**: 최근 python.org 설치 프로그램은 개별 버전 설치가 아니라 "Python Install Manager"를 통해 여러 버전을 관리하는 방식으로 바뀌었다. `py install 3.12`, `py list`로 버전을 확인/설치해야 했다.
- **PowerShell 실행 정책**: venv 활성화 스크립트(`venv\Scripts\Activate.ps1`)가 기본 보안 정책에 막혀 실행이 안 됐다. `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`로 해결.
- **venv 활성화 명령어 실수**: PowerShell에서는 `activate`(확장자 없이)가 아니라 `Activate.ps1`로 명시해야 한다. 확장자 없이 치면 `activate.bat`으로 잘못 해석되어 cmd 창이 떴다 사라지는 증상이 났다.
- **Python 버전과 onnxruntime 호환성**: 처음 설치된 Python 3.15.0은 너무 최신이라 onnxruntime wheel이 아예 없었다(`from versions: none`). Python 3.12로 venv를 다시 만들어서 해결. — 이번 프로젝트 전체에서 반복된 "최신 버전이 오히려 호환성을 깬다"는 패턴이 로컬 환경에서도 똑같이 나타났다.
- **OneDrive 동기화 폴더 안에 venv를 두는 것의 위험성**: 프로젝트 폴더가 `OneDrive\문서\...` 안에 있다 보니, venv(수많은 작은 파일을 읽고 쓰는 폴더)를 그 안에 두면 동기화 충돌/잠금 문제가 생길 수 있다. 실제로 onnxruntime import 시 `0xC0000005`(Access Violation) 크래시가 발생했는데, 이 원인을 좁히는 과정에서 venv를 OneDrive 밖(`C:\venvs\...`)으로 옮겼다. (최종적으로 이 크래시의 결정적 원인은 아니었던 것으로 보이지만, venv는 애초에 동기화 폴더 밖에 두는 게 일반적으로도 권장된다.)
- **onnxruntime 버전 자체의 문제**: 최신 onnxruntime(1.31.0)에서 import 시점에 바로 access violation이 발생했다. 더 널리 쓰이는 1.18.1로 내리니, 적어도 "조용히 죽는" 크래시 대신 제대로 된 파이썬 예외가 뜨기 시작했다 — 디버깅 가능한 상태가 된 것만으로도 큰 진전이었다.

## 2. ONNX external data 파일명 불일치

4주차 때도 겪었던 문제의 변형이다. `.onnx` 파일 내부에는 가중치가 담긴 external data 파일의 이름이 export 당시의 원래 이름(`model.onnx.data`)으로 고정되어 박혀 있는데, Kaggle에서 다운로드하면서 실제 폴더의 파일명이 `bottle_model.onnx.data`로 바뀌어 있었다. 그 결과:

```
GetFileLength for models\model.onnx.data failed: open file model.onnx.data fail, errcode = 2
```

**해결**: `.data` 파일명을 ONNX가 내부적으로 참조하는 이름(`model.onnx.data`)에 맞게 다시 변경. (메인 `.onnx` 파일명은 자유롭게 바꿔도 되지만, `.data` 파일명은 export 당시 이름과 반드시 일치해야 한다.)

## 3. 가장 치명적이었던 버그: 전처리 파이프라인 불일치 (정규화 중복/누락)

로컬 ONNX Runtime 추론이 정상적으로 돌아가기 시작한 뒤, `good`(정상) 이미지와 `bad`(결함) 이미지를 각각 테스트했는데 **둘 다 이상(anomaly) 점수가 0.98~1.0으로 거의 동일하게 높게 나왔다.** 4주차 PyTorch/TensorRT 벤치마크에서는 bottle 카테고리 image AUROC가 1.0000(완벽한 구분)이었으므로, 모델 자체의 문제는 아니라고 보고 전처리 코드를 의심했다.

처음엔 "ONNX 그래프 안에 정규화가 이미 포함되어 있어서 수동 정규화와 중복되는 것 아닌가" 가설을 세웠지만, 4주차 `export_tensorrt.py`를 다시 보니 TensorRT 엔진 벤치마크 때도 anomalib 데이터모듈이 만든 정규화된 텐서를 그대로 입력했었다는 걸 확인했다 — 즉 "정규화를 두 번 하는 것"이 원래 맞는 방식인 줄 알았다.

**진짜 원인은 반대였다.** Kaggle에서 체크포인트를 직접 불러와 모델의 `pre_processor.transform`을 출력해서 확인해보니:

```python
model = EfficientAd.load_from_checkpoint(ckpt_path)
print(model.pre_processor.transform)
# Compose(    Resize(size=[256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True))
```

**`Normalize`가 아예 없었다.** 이 모델은 ImageNet mean/std로 정규화된 입력이 아니라, 그냥 **0~1로 스케일링만 된 이미지**를 입력으로 받는다. 그런데 로컬 `inference.py`는 여기에 ImageNet 정규화(mean 빼고 std로 나누기)까지 추가로 적용하고 있었다 — 모델이 학습 때 한 번도 본 적 없는 범위의 값을 입력받은 셈이라, 모든 이미지가 "비정상적인 입력"으로 인식되어 이상 점수가 전체적으로 치솟았던 것이다.

**해결**: 전처리에서 ImageNet 정규화 단계를 완전히 제거하고, 0~1 스케일링(`/255.0`)만 남겼다.

```python
# 수정 전 (잘못된 버전)
arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

# 수정 후
arr = np.array(image, dtype=np.float32) / 255.0  # 스케일링만, 추가 정규화 없음
```

수정 후 재테스트 결과:

| 파일 | anomaly_score | is_anomaly | 실제 라벨 |
|---|---|---|---|
| bottle_good.png | 0.4980 | False | 정상 |
| bottle_bad.png | 0.6751 | True | 결함 |
| bottle_test_a.png | 0.4964 | False | 정상 |
| bottle_test_b.png | 0.5134 | True | 결함 |

4장 모두 정상/결함이 정확히 갈렸다.

**교훈**: export된 모델이 어떤 전처리를 기대하는지는 "4주차 코드에서 쓰던 방식을 그대로 재사용"하는 게 아니라, **모델 자체(`pre_processor.transform`)에서 직접 확인**해야 한다. 같은 프로젝트 안에서도 평가 파이프라인(데이터로더 기반)과 배포 파이프라인(수동 전처리)은 겉보기엔 비슷해 보여도 실제로 통과하는 변환 단계가 다를 수 있다.

## 4. 참고: 점수 스케일에 대한 오해

전처리를 고친 뒤에도 `anomaly_score`가 0.5 근처(0.49~0.68)에 몰려 있어서 "분류 성능이 떨어진 것 아닌가" 걱정했었는데, 이건 정상이다. 4주차 평가에서 본 "0 또는 1에 가까운" 점수는 anomalib이 검증 데이터셋 통계로 min-max 정규화를 거친 값이고, 로컬 ONNX Runtime에서 직접 받는 `pred_score`는 그 정규화 이전의 원시 점수다. 중요한 건 점수의 절대적 크기가 아니라 **`pred_label`(모델에 내장된 threshold로 이미 비교된 최종 판정)이 정확한가**이며, 이 부분은 4장 전부 정확했다.

## 5. 드리프트(drift) 감지 기능 테스트

서빙 API에 최근 N건(윈도우 크기 20)의 `is_anomaly` 판정 비율을 추적해서, 비율이 임계값(50%)을 넘으면 `possible_drift` 경고를 로그에 남기고 `/drift-status`에서 `drift_detected`를 반환하도록 구현해뒀는데, 이 로직이 실제로 의도대로 동작하는지 로컬에서 직접 검증했다.

**테스트 절차**

1. 서버를 `uvicorn app:app --reload --port 8000`으로 띄운 상태에서, 먼저 `GET /drift-status`를 호출해 베이스라인을 확인.
2. `bottle_good.png`(정상 이미지)를 `/inspect`에 20회 연속 업로드.
3. 다시 `GET /drift-status` 호출.
4. `bottle_bad.png`(결함 이미지)를 15회 연속 업로드해 윈도우 내 이상 비율을 끌어올림.
5. `GET /drift-status`와 `logs/requests.log`를 확인.

반복 요청은 PowerShell에서 아래 방식으로 보냈다:

```powershell
1..20 | ForEach-Object {
    curl.exe -X POST "http://127.0.0.1:8000/inspect" -F "file=@bottle_good.png;type=image/png"
}
```

**결과**

| 단계 | `/drift-status` 응답 |
|---|---|
| 베이스라인 (요청 0건) | `{"status": "insufficient_data", "collected": 0, "needed": 20}` |
| 정상 이미지 20건 | `{"status": "normal", "recent_anomaly_rate": 0.0, "window_size": 20, "threshold": 0.5}` |
| 결함 이미지 15건 추가 | `{"status": "drift_detected", "recent_anomaly_rate": 0.75, "window_size": 20, "threshold": 0.5}` |

슬라이딩 윈도우(최근 20건 기준)가 요청마다 실시간으로 갱신되는 것도 로그에서 직접 확인됐다. 결함 이미지가 쌓이면서 `recent_anomaly_rate`가 11번째 요청(윈도우 내 이상 비율이 55%로 임계값 50%를 처음 넘긴 시점)부터 `possible_drift` 경고를 매 요청마다 남겼다:

```json
{"timestamp": "2026-10-16T14:24:10", "level": "warning", "event": "possible_drift", "anomaly_rate": 0.55, "window_size": 20, "message": "최근 요청 중 이상 판정 비율이 비정상적으로 높습니다. 카메라/조명/제품 라인 변화를 확인하세요."}
{"timestamp": "2026-10-16T14:24:10", "level": "warning", "event": "possible_drift", "anomaly_rate": 0.6,  "window_size": 20, "message": "..."}
{"timestamp": "2026-10-16T14:24:11", "level": "warning", "event": "possible_drift", "anomaly_rate": 0.65, "window_size": 20, "message": "..."}
{"timestamp": "2026-10-16T14:24:12", "level": "warning", "event": "possible_drift", "anomaly_rate": 0.7,  "window_size": 20, "message": "..."}
{"timestamp": "2026-10-16T14:24:12", "level": "warning", "event": "possible_drift", "anomaly_rate": 0.75, "window_size": 20, "message": "..."}
```

**확인된 점**

- 윈도우 집계(`collected`/`window_size`)가 요청 건수를 정확히 추적함.
- 정상 이미지만 들어올 때는 `recent_anomaly_rate`가 0으로 유지되어 불필요한 경고가 발생하지 않음.
- 이상 비율이 임계값(0.5)을 넘는 순간부터 매 요청마다 `possible_drift` 경고가 로그에 기록되고, `/drift-status`의 상태도 `drift_detected`로 즉시 전환됨.
- 실제 운영 환경에서 카메라/조명/제품 라인 변화로 이상 탐지 비율이 비정상적으로 치솟는 상황을 조기에 감지할 수 있는 최소 기능 수준의 모니터링이 로컬에서 의도대로 동작함을 확인.

## 최종 결과

- FastAPI 서버(`/health`, `/inspect`)가 로컬에서 정상 기동
- ONNX Runtime(CPU) 기반 추론이 정상/결함 이미지를 정확히 구분
- 레이턴시는 아직 정밀 측정 전이지만, GPU 없이 CPU 추론만으로 실시간 API 서빙이 가능한 수준임을 확인
- 드리프트 감지 기능(슬라이딩 윈도우 기반 이상 비율 모니터링 + 로그 경고)이 설계한 대로 정상 동작함을 로컬 테스트로 검증 완료