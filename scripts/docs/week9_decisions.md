# 9주차 — 주요 의사결정 과정 문서화

"결과"가 아니라 그 결과에 이르기까지의 **판단 근거**를 남기는 문서. 각 결정이 왜 필요했고, 어떤 증거로 검증했는지를 정리한다.

## 1. EfficientAD under-training 발견 및 수정

### 발견 (3주차)

5개 카테고리(bottle, cable, capsule, hazelnut, screw)에 대해 PatchCore/PaDiM/EfficientAD 세 모델을 모두 학습시키고 image AUROC를 비교했다 (PatchCore/PaDiM은 `max_epochs=1`, EfficientAD는 `max_steps=5000`, `batch_size=1`로 학습).

| category | PatchCore | PaDiM | EfficientAD (5000 step) |
|---|---|---|---|
| bottle | 1.000 | 0.997 | 1.000 |
| cable | 0.988 | 0.859 | 0.950 |
| capsule | 0.991 | 0.878 | **0.706** |
| hazelnut | 1.000 | 0.650 | **0.843** |
| screw | 0.966 | 0.776 | **0.766** |

bottle/cable은 EfficientAD도 준수했지만, **capsule/hazelnut/screw에서 PatchCore 대비 뚜렷하게 낮은 성능**이 나왔다. 특히 capsule은 0.706으로, "거의 랜덤에 가까운" 수준은 아니지만 실사용하기엔 불안한 수치였다.

### 가설과 검증

PatchCore는 메모리 뱅크 기반이라 "학습"이라는 개념이 약하고(feature 추출 후 저장), EfficientAD는 teacher-student knowledge distillation 구조라 **step 수가 부족하면 student 네트워크가 teacher를 충분히 따라가지 못해(under-fitting) 이상 탐지 경계가 불안정해질 수 있다**는 가설을 세웠다. 이를 검증하기 위해 가장 성능이 낮았던 capsule과 hazelnut에 대해서만 `max_steps`를 5000 → 15000으로 3배 늘려 재학습하는 ablation을 별도 노트북으로 진행했다.

### 결과

| category | 5000 step | 15000 step | 변화 |
|---|---|---|---|
| capsule | 0.706 | **0.878** | +17.3%p |
| hazelnut | 0.843 | **0.906** | +6.3%p |

두 카테고리 모두 뚜렷하게 개선됐고, 특히 capsule은 17%p 이상 올랐다. **이로써 "성능이 낮았던 원인이 모델/데이터 자체의 한계가 아니라 under-training이었다"는 가설이 검증됐다.** 이후 4주차 TensorRT 변환 단계에서는 capsule의 최종 image AUROC가 0.9039로 기록되어 있는데(ablation 때의 0.878보다 조금 더 높음), 이는 이후 추가적인 step 조정이 있었음을 시사한다.

**교훈**: 모델 성능이 기대보다 낮게 나왔을 때, 곧바로 "이 모델/카테고리는 안 맞는다"고 결론 내리기 전에 학습이 충분했는지부터 의심해봐야 한다. 특히 teacher-student distillation처럼 학습 과정이 명시적인 구조에서는 더더욱 그렇다. 반대로, 아무 근거 없이 step을 늘리는 게 아니라 "가장 의심되는 카테고리만 골라 통제된 ablation으로 검증"한 것이 비용 대비 효율적이었다.

## 2. 모델 선택 기준: PatchCore가 아니라 EfficientAD를 배포 모델로 택한 이유

3주차 비교 실험만 놓고 보면, bottle/capsule 모두에서 **PatchCore가 EfficientAD보다 정확도가 높거나 비슷했다** (bottle: 1.000/0.985 vs 1.000/0.974, capsule: 0.991 vs 0.878~0.906). 순수하게 정확도만 보면 PatchCore를 선택하는 게 자연스러워 보인다.

그럼에도 **EfficientAD를 최종 배포 모델로 선택한 이유는 두 가지**다.

1. **ONNX/TensorRT 배포 적합성**: PatchCore는 메모리 뱅크(feature를 저장해두고 k-NN으로 거리를 계산하는 방식) 기반이라, 학습 후 "고정된 연산 그래프"로 export하기가 까다롭다. k-NN 검색 자체가 정적 그래프 변환이나 TensorRT 최적화와 궁합이 잘 안 맞는다. 반면 EfficientAD는 순수 feed-forward 네트워크(teacher-student)라 ONNX export → TensorRT FP16 변환까지 자연스럽게 이어진다 (실제로 4주차에서 이 경로를 그대로 탔다).
2. **프로젝트의 목적 자체가 "경량화·배포까지 해봤다"는 것을 보여주는 것**이었다. 이 프로젝트는 단순히 "가장 정확한 모델을 찾는 것"이 아니라, 이상 탐지 모델을 실제로 경량화하고 엣지/서버에 배포 가능한 형태로 만드는 전체 파이프라인 경험을 쌓고 이를 포트폴리오로 보여주는 데 목적이 있었다. 그 목적에는 "정적 그래프로 export 가능하고, 양자화/가속이 쉬운 모델"이 더 적합했고, EfficientAD가 정확히 그 조건에 맞았다.

**정리하면**: 정확도만 보면 PatchCore가 우위였지만, 이 프로젝트가 증명하려는 역량(경량화 + 배포)에 맞춰 EfficientAD를 선택했다. 이는 "실무에서는 최고 정확도 모델이 항상 정답이 아니며, 배포 제약 조건과 프로젝트 목표에 맞는 모델을 선택해야 한다"는 실질적인 트레이드오프 판단이었다.

## 3. 그 외 트러블슈팅에서 나온 의사결정들

각 항목의 전체 맥락은 해당 주차 문서를 참고.

- **ONNX external data 파일명 불일치** ([4주차](week4_tensorrt_debugging.md)): export 당시 파일명과 실제 폴더의 파일명이 달라 로딩이 실패. `.data` 파일명은 export 시점 이름과 반드시 일치해야 한다는 제약을 확인하고 대응.
- **TensorRT Optimization Profile silent failure** ([4주차](week4_tensorrt_debugging.md)): shape을 잘못 고정해도 엔진 빌드/추론이 에러 없이 "성공"해버리는 silent failure를 겪음. 이후 "엔진 빌드 성공 = 올바른 엔진"이 아니라는 원칙을 세우고, 반드시 정확도 지표로 검증하는 절차를 추가.
- **전처리 파이프라인 불일치(정규화 중복)** ([5주차](week5_local_deployment_debugging.md)): 로컬 서빙 코드에 불필요한 ImageNet 정규화가 추가되어 정상/결함 이미지가 전혀 구분되지 않는 치명적 버그 발생. "export된 모델이 기대하는 전처리는 이전 코드를 재사용하지 말고 모델 자체(`pre_processor.transform`)에서 직접 확인해야 한다"는 원칙 확립.
- **심각도(severity) 판단 기준** ([8주차](week8_crewai_pipeline.md)): CrewAI 분석 에이전트가 anomaly_score를 등급으로 나누는 기준(0.5/0.6/0.7)은 통계적으로 도출한 게 아니라 보유 샘플의 점수 분포(0.49~0.68)를 바탕으로 잡은 임시값이다. 실제 운영 데이터가 쌓이면 재보정이 필요하다는 한계를 명시적으로 남겨둠.
- **CrewAI Tool 이름에 한글 사용 불가** ([8주차](week8_crewai_pipeline.md)): OpenAI function-calling 스펙이 함수 이름에 `[a-zA-Z0-9_-]`만 허용한다는 걸 에러를 통해 확인, Tool "이름"과 "설명"의 역할을 구분해서 수정.
- **LangChain 1.0 API 마이그레이션** ([7주차](week7_langchain_agent.md)): 예제/튜토리얼에서 흔히 보이는 `AgentExecutor` 구문이 최신 버전에서는 레거시로 밀려났음을 발견, 실제 설치된 버전을 확인하고 `create_agent` 기반으로 재작성.

## 4. 종합 — 이 프로젝트가 보여주는 것

1주차부터 8주차까지 일관되게 나타난 패턴은, **"잘 되는 것처럼 보이는 상태"와 "실제로 올바른 상태"를 구분하기 위해 항상 정량적 지표로 재검증했다는 점**이다. TensorRT 엔진 빌드 성공, 전처리 코드가 에러 없이 돌아가는 것, EfficientAD 학습이 끝나는 것 — 이런 것들은 전부 "실패 없이 끝났다"는 신호일 뿐, "올바르게 동작한다"는 보장이 아니었다. 매번 AUROC, anomaly_score 분포, 실제 정상/결함 이미지 테스트 같은 구체적인 증거로 검증하고 나서야 다음 단계로 넘어갔다.
