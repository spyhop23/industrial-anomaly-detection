# 4주차 경량화 — TensorRT FP16 변환 트러블슈팅 기록

EfficientAD 모델(bottle, capsule)을 ONNX → TensorRT FP16으로 변환하는 과정에서 만난 문제들과 해결 과정을 정리한다. 단순히 "버전이 안 맞았다"는 수준이 아니라, TensorRT 10+ 세대의 API 자체가 이전 세대와 설계 철학이 크게 바뀌면서 생긴 문제들이 많아서, 비슷한 작업을 하게 될 경우 참고할 만한 내용을 남겨둔다.

## 1. ONNX export: `onnxscript` 의존성 누락

anomalib의 `engine.export(export_type=ExportType.ONNX, ...)`는 내부적으로 PyTorch의 새로운 dynamo 기반 exporter(`torch.onnx.dynamo_export`)를 사용하는데, 이 경로는 `onnxscript` 패키지를 필요로 한다. anomalib 자체 설치에는 포함되지 않아서 별도 설치가 필요했다.

```bash
pip install onnxscript -q
```

## 2. ONNX opset 버전과 external data

Export된 모델은 opset 18, IR version 10으로 생성됐다. export 과정에서 onnxscript가 opset을 낮추려는 시도(14로 다운그레이드)를 했지만 `Resize` 연산자에서 실패해 fallback으로 원래 opset(18)을 그대로 유지했다 — 이건 경고였을 뿐 export 자체를 막지는 않았다.

더 중요한 문제는 따로 있었다. 모델 가중치가 `.onnx` 파일 안에 전부 담기지 않고 `model.onnx.data`라는 별도 파일에 "external data"로 분리 저장됐는데, TensorRT의 `OnnxParser.parse(bytes)`로 바이트만 읽어서 넘기면 TensorRT가 원본 파일 위치를 몰라서 같은 폴더의 `.data` 파일을 못 찾는다. 결과적으로 `model.teacher.conv1.bias` 같은 initializer를 못 읽어서 파싱이 통째로 실패했다.

**해결**: `parser.parse(f.read())` 대신 `parser.parse_from_file(str(onnx_path))`를 사용 — 파일 경로를 직접 넘기면 TensorRT가 같은 디렉토리에서 external data 파일을 자동으로 찾는다.

## 3. TensorRT 10+/11 API의 대대적인 변경

이번에 사용한 TensorRT 버전(11.4.0.106)은 이전 세대(TensorRT 7~9) 기준으로 작성된 튜토리얼/예제 코드와 호환되지 않는 부분이 많았다.

- `trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH` — 제거됨. Implicit batch 모드 자체가 완전히 사라지고 explicit batch가 기본값이 되면서, `builder.create_network()`를 플래그 없이 호출하면 된다.
- `builder.platform_has_fast_fp16` — 제거됨. 대부분의 최신 GPU가 FP16을 기본 지원하게 되면서 이 체크 자체가 의미 없어졌다.
- `trt.BuilderFlag.FP16` — **아예 존재하지 않는다.** TensorRT 10+는 "Strongly Typed Network"라는 새로운 패러다임을 도입했는데, 이 모드에서는 builder 플래그로 정밀도를 지정하는 게 아니라 **입력 ONNX 모델 자체의 텐서 타입을 그대로 따라간다.** 즉 FP16 추론을 원하면 TensorRT 설정이 아니라 ONNX 모델 자체를 FP16으로 변환해야 한다.

```python
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
network = builder.create_network(network_flags)
```

## 4. ONNX FP16 변환 시 비교 연산(Greater) 타입 불일치

`onnxconverter_common.float16.convert_float_to_float16(model, keep_io_types=True)`로 ONNX 모델을 FP16으로 변환했는데, `keep_io_types=True`는 그래프의 "입출력"만 FP32로 지켜줄 뿐, 중간에 있는 `Greater`(임계값 비교) 노드까지는 챙기지 못했다. 그 결과 `pred_label`/`pred_mask`를 만드는 비교 연산에서 한쪽 입력(pred_score/anomaly_map)은 FP16, 다른 쪽(임계값 상수)은 FP32로 남아 타입 불일치 에러가 발생했다.

```
ElementWiseOperation GREATER must have same input types. But they are of types Float and Half.
```

**해결**: 비교 연산은 애초에 FP16으로 바꿔봤자 속도 이득이 없는 부분이므로, 변환 대상에서 제외했다.

```python
model_fp16 = float16.convert_float_to_float16(
    model, keep_io_types=True, op_block_list=["Greater"],
)
```

## 5. TensorRT 추론 실행 API도 변경됨

구버전 예제에 흔한 `context.execute_v2(bindings=[...])` + `pycuda` 조합 대신, TensorRT 10+는 `execute_async_v3` + `set_tensor_address` 기반 API를 쓴다. PyTorch CUDA 텐서의 `.data_ptr()`을 그대로 바인딩에 넘길 수 있어서 `pycuda` 없이 처리할 수 있었다.

```python
context.set_input_shape(input_name, tuple(input_tensor.shape))
context.set_tensor_address(input_name, input_tensor.data_ptr())
context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
```

## 6. 가장 치명적이었던 버그: Optimization Profile을 잘못 고정

엔진을 처음 빌드했을 때, ONNX 입력 shape이 `(-1, 3, -1, -1)`(배치/높이/너비가 전부 동적)이었는데, optimization profile을 만들 때 아래처럼 "-1이면 전부 1로" 치환하는 코드를 썼다.

```python
static_shape = tuple(dim if dim != -1 else 1 for dim in input_shape)
```

문제는 이게 배치 차원뿐 아니라 **높이/너비까지 1×1로 고정**해버렸다는 것. 엔진 빌드도 에러 없이 성공했고, 추론 실행도 에러 없이 끝났다(TensorRT가 shape 범위를 벗어난 입력에 대해 Python 레벨 예외를 던지지 않았다). 그래서 겉으로는 다 정상으로 보였는데, 실제로는 완전히 잘못된(사실상 의미 없는) 출력이 나오고 있었다.

이게 드러난 건 **정확도 벤치마크 단계**였다. bottle 카테고리의 TensorRT image AUROC가 정확히 0.5가 나왔는데, 이건 "모델이 정상/이상을 전혀 구분 못 한다"는 신호다. 지연시간이나 엔진 빌드 과정에서는 전혀 티가 안 나고, 실제 정확도를 측정해봐야만 드러나는 종류의 버그였다.

**해결**: 더미값(1) 대신 실제 사용할 입력 크기(데이터로더에서 실제 샘플을 하나 가져와서 그 shape)로 min=opt=max를 고정했다.

```python
sample_shape = tuple(next(iter(dataloader))["image"].shape)
profile.set_shape(input_tensor.name, sample_shape, sample_shape, sample_shape)
```

**교훈**: TensorRT는 shape이 잘못돼도 조용히 실패(silent failure)할 수 있다. 엔진 빌드 성공 = 올바른 엔진이 아니다. 반드시 실제 정확도 지표로 검증해야 한다.

## 7. (디버깅 과정의) 두 번째 함정: 메모리에 남은 구버전 엔진 객체

위 버그를 고치고 엔진을 다시 빌드했는데도 벤치마크 결과가 수정 전과 소수점까지 완전히 똑같이 나온 적이 있었다. 원인은 `.engine` 파일은 새로 저장됐지만, 그 파일을 읽어서 메모리에 올려놓은 `TRTEngineWrapper` 객체(`trt_engines` 딕셔너리)는 커널을 재시작하지 않는 한 예전 바이트를 그대로 들고 있었기 때문. 엔진을 재빌드한 뒤에는 반드시 wrapper 객체도 다시 생성해야 한다.

## 최종 결과

| category | PyTorch 레이턴시 | TensorRT FP16 레이턴시 | speedup | PyTorch image AUROC | TRT image AUROC | PyTorch pixel AUROC | TRT pixel AUROC | 체크포인트 크기 | TRT 엔진 크기 |
|---|---|---|---|---|---|---|---|---|---|
| bottle | 40.56ms | 4.12ms | 9.86x | 1.0000 | 1.0000 | 0.9764 | 0.9763 | 71.7MB | 26.8MB |
| capsule | 40.75ms | 4.19ms | 9.73x | 0.9039 | 0.9039 | 0.9508 | 0.9509 | 71.7MB | 28.6MB |

정확도 손실은 반올림 오차 수준(소수점 넷째 자리)에 불과한 반면, 추론 속도는 약 10배 빨라졌고 엔진 크기는 체크포인트 대비 60% 이상 줄었다. bottle(쉬운 케이스)뿐 아니라 capsule(3주차에서 under-training 문제를 발견하고 고쳤던 어려운 케이스)에서도 동일한 수준의 결과가 나왔다는 점이, 이 최적화가 특정 카테고리에만 우연히 잘 맞은 게 아니라는 근거가 된다.