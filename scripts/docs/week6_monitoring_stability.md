# 6주차 모니터링 & 안정성 — "돌아가는 모델"을 "믿을 수 있는 시스템"으로

5주차에서 로컬 ONNX Runtime 서빙 서버(FastAPI)를 구축하는 과정에서, 사실상 6주차 목표였던 로깅/입력 검증/드리프트 감지 로직을 함께 구현하고 테스트했다. 이 문서는 그 세 가지 기능을 6주차 관점에서 정리한 기록이다.

## 1. 추론 요청/결과 로깅

`/inspect` 엔드포인트가 호출될 때마다 구조화된 JSON 로그를 `logs/requests.log`에 남기도록 구현했다.

**정상 추론 로그 예시**

```json
{"timestamp": "2026-10-16T14:24:12", "level": "info", "filename": "bottle_bad.png", "anomaly_score": 0.6751, "is_anomaly": true, "latency_ms": 592.2}
```

요청마다 파일명, anomaly score, 최종 판정(`is_anomaly`), 추론 레이턴시(`latency_ms`)까지 함께 남겨서, 나중에 이 로그만으로도 "언제 어떤 이미지가 어떻게 판정됐는지", "레이턴시가 튀는 구간이 있는지"를 추적할 수 있게 했다.

## 2. 입력 검증 및 기본 에러 핸들링

처음엔 이미지가 아닌 파일이 올라왔을 때 서버가 어떻게 반응하는지가 불확실했다. 테스트를 위해 실제로 `.txt` 파일(`mm.txt`)을 `/inspect`에 업로드해봤다.

**테스트 결과**

터미널 로그:

```
{"timestamp": "2026-10-16T14:40:14", "level": "warning", "filename": "mm.txt", "error": "unsupported_content_type", "content_type": "text/plain"}
INFO:     127.0.0.1:55623 - "POST /inspect HTTP/1.1" 400 Bad Request
```

- 서버가 크래시(500 에러)하지 않고, content-type을 검증해서 **400 Bad Request**로 정상 응답했다.
- 거부된 요청도 `level: warning`으로 로그에 남아서, 이후에 "잘못된 파일이 얼마나 자주 올라오는지" 같은 운영 지표로도 활용 가능하다.

**확인된 점**: 이미지가 아닌 파일이 들어와도 서버 전체가 죽지 않고, 명확한 에러 코드와 사유를 반환하며, 그 사실이 로그에도 남는다. 최소한의 입력 검증 요건은 충족.

## 3. 데이터 드리프트 감지 (선택 항목)

최근 N건(윈도우 크기 20)의 `is_anomaly` 판정 비율을 추적해서, 비율이 임계값(50%)을 넘으면 `possible_drift` 경고를 로그에 남기고 `/drift-status`에서 `drift_detected`를 반환하도록 구현했다. (자세한 테스트 절차와 결과는 `week5_local_deployment_debugging.md`의 "5. 드리프트 감지 기능 테스트" 참고.)

요약:

| 단계 | `/drift-status` 응답 |
|---|---|
| 베이스라인 | `{"status": "insufficient_data", "collected": 0, "needed": 20}` |
| 정상 이미지 20건 | `{"status": "normal", "recent_anomaly_rate": 0.0}` |
| 결함 이미지 15건 추가 | `{"status": "drift_detected", "recent_anomaly_rate": 0.75}` |

이상 비율이 임계값을 넘는 순간부터 매 요청마다 `possible_drift` 경고가 로그에 기록되는 것까지 확인했다.

## 최종 결과

6주차 목표 세 가지 모두 구현 및 로컬 테스트 완료:

- **로깅**: 모든 `/inspect` 요청의 입력/결과/레이턴시가 구조화된 JSON으로 `logs/requests.log`에 기록됨.
- **입력 검증/에러 핸들링**: 지원하지 않는 파일 타입(`.txt` 등) 업로드 시 서버가 안전하게 400 에러로 응답하고, 거부 사유가 로그에 남음.
- **드리프트 감지**: 슬라이딩 윈도우 기반 이상 비율 모니터링이 의도대로 동작하며, 임계값 초과 시 경고 로그와 상태 API가 즉시 반응함.

다음 단계(7주차)는 이 추론 함수(`inspect_image()`)와 드리프트/로그 조회 기능을 LangChain Tool로 래핑해서, 자연어로 시스템과 상호작용하는 단일 에이전트 레이어를 구축하는 것이다.