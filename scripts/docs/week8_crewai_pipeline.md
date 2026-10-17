# 8주차 AI Agent 레이어 ② — 멀티 에이전트 (CrewAI)

목표: 에이전트 간 협업 파이프라인 구성 — 검사/분석/보고서 작성 역할을 나눈 에이전트들이 end-to-end로 불량 보고서를 만들어내게 한다.

## 1. 구조

기존 서빙 코드와 섞이지 않도록 `serving/crew/` 서브폴더에 분리했다. (7주차 `serving/agent/`와 동일한 방식.)

```
serving/
├── agent/   (7주차 — 단일 에이전트)
└── crew/    (8주차 — 신규)
    ├── crew_tools.py       — CrewAI용 Tool (7주차 로직 재사용)
    ├── crew_pipeline.py    — 검사→분석→보고서 3단계 파이프라인
    ├── requirements-crew.txt
    ├── .env.example
    └── reports/             — 실행 때마다 생성되는 보고서 .md 파일들
```

**역할 분리 (Process.sequential)**

1. **검사 에이전트 (Inspector)** — `inspect_image` 도구로 실제 모델 추론 실행. 도구를 쓰지 않고 결과를 지어내지 못하도록 backstory에 명시.
2. **분석 에이전트 (Analyst)** — anomaly_score를 심각도 등급(정상/경미/심각/매우 심각)으로 분류. 필요하면 `query_experiment_results` 도구로 과거 벤치마크 기록을 참고할 수 있게 했다.
3. **보고서 작성 에이전트 (Reporter)** — 앞 두 에이전트의 결과를 받아 마크다운 불량 보고서를 작성. Tool 없이 LLM만으로 글쓰기만 담당.

7주차에서 만든 Tool 로직(`_get_detector`, `_load_benchmark_tables`)을 그대로 재사용해서, 모델 로딩이나 문서 파싱 코드를 중복 작성하지 않았다.

## 2. 심각도 판단 기준 (설계상의 가정)

anomaly_score를 등급으로 나누는 기준은 다음과 같이 잡았다:

| anomaly_score | 등급 |
|---|---|
| 0.50 미만 | 정상 (불량 아님) |
| 0.50 ~ 0.60 | 경미한 이상 |
| 0.60 ~ 0.70 | 심각한 이상 |
| 0.70 이상 | 매우 심각한 이상 |

**이 기준은 통계적으로 도출한 게 아니라, 현재 보유한 소량의 샘플 이미지 점수 분포(0.49~0.68 구간)를 보고 임시로 잡은 경험적 기준이다.** 실제 생산 데이터가 쌓이면 정상/이상 각각의 점수 분포를 분석해서 재보정이 필요하다 — 이건 9주차 "주요 의사결정 과정 문서화"에 반드시 남겨야 할 항목이다 (수치를 왜 이렇게 잡았는지, 어떤 한계가 있는지).

## 3. 트러블슈팅: CrewAI Tool 이름에 한글을 쓰면 안 됨

처음에 Tool 이름을 `"이미지_검사"`, `"실험_기록_조회"`처럼 한글로 지었는데, 실행하자마자 아래 에러가 반복적으로 발생했다.

```
ERROR:root:OpenAI: Error converting tool: OpenAI function name cannot be empty
ValueError: OpenAI function name cannot be empty
```

**원인**: CrewAI가 내부적으로 Tool을 OpenAI function-calling 스펙으로 변환할 때, 함수 이름을 `[a-zA-Z0-9_-]`만 허용하는 정규식으로 정제(sanitize)한다. Tool 이름 전체가 한글이었기 때문에 이 정제 과정에서 글자가 전부 걸러지고 빈 문자열만 남아, "함수 이름이 비어있다"는 에러로 이어졌다.

**해결**: Tool 이름을 영문 식별자(`inspect_image`, `query_experiment_results`)로 바꾸고, 사람이 읽을 설명(docstring)만 한글로 남겼다. Tool의 "이름"과 "설명"의 역할이 다르다는 걸 명확히 구분해서 접근해야 하는 문제였다.

```python
# 수정 전 (에러 발생)
@tool("이미지_검사")
def crew_inspect_image(image_path: str) -> str: ...

# 수정 후
@tool("inspect_image")
def crew_inspect_image(image_path: str) -> str: ...
```

**교훈**: LLM에 함수를 노출시키는 프레임워크(LangChain, CrewAI 등)는 내부적으로 OpenAI function-calling 스펙을 거치는 경우가 많고, 이 스펙은 함수 "이름"에 엄격한 문자 제약이 있다. 사람이 보는 설명(description)과 기계가 참조하는 이름(name)을 섞어 쓰면 안 된다는 걸 직접 겪으며 확인했다.

## 4. 테스트 결과

동일한 명령어로 정상/결함 이미지 각각을 end-to-end로 돌려봤다.

```powershell
python crew_pipeline.py ../bottle_bad.png
python crew_pipeline.py ../bottle_good.png
```

**결함 이미지 (`bottle_bad.png`)**

```markdown
# 불량 보고서
검사 파일명: bottle_bad.png
anomaly_score: 0.6751
판정: 이상
심각도: 심각한 이상
권장 조치: 해당 제품은 라인에서 제외하고, 추가 검사를 실시해야 합니다.
```

**정상 이미지 (`bottle_good.png`)**

```markdown
# 불량 보고서
검사 파일명: bottle_good.png
anomaly_score: 0.4980
판정: 정상
심각도: 정상 (불량 아님)
권장 조치: 추가 조치 필요 없음
판단 근거: anomaly_score가 0.4980으로 0.50 미만이므로 정상 범주에 해당하며, 불량이 아님을 나타냅니다.
```

두 케이스 모두 설계한 심각도 기준에 맞게 정확히 분류됐고, 5주차/7주차에서 확인한 동일 이미지의 anomaly_score(0.6751 / 0.4980)와도 일치한다.

## 5. 실행 흐름 관찰 — 에이전트가 Tool을 "선택적으로" 사용함

`verbose=True`로 실행했을 때 찍힌 실제 이벤트 순서는 다음과 같다.

```
Crew Execution Started
→ Task Started (검사)      → Agent Started → Tool Execution Started → Tool Execution Completed → Agent Final Answer → Task Completion
→ Task Started (분석)      → Agent Started → Agent Final Answer → Task Completion
→ Task Started (보고서 작성) → Agent Started → Agent Final Answer → Task Completion
→ Crew Completion
```

흥미로운 점은, 분석 에이전트에게도 `query_experiment_results` 도구를 줬지만 **이번 실행에서는 Tool Execution 단계 없이 바로 답을 냈다** — anomaly_score만으로 심각도를 판단하기에 충분하다고 LLM이 스스로 판단하고 Tool 호출을 생략한 것이다. 이는 Agent가 "Tool을 쓸지 말지도 스스로 판단한다"는 걸 보여주는 사례로, 9주차 아키텍처 다이어그램에 "검사 단계만 Tool 호출 필수, 분석 단계는 선택적 Tool 호출"로 표시해둘 가치가 있다.

## 최종 결과

- 검사/분석/보고서 작성 역할을 나눈 CrewAI 멀티 에이전트 파이프라인을 `Process.sequential`로 구성 완료.
- Tool 이름에 한글을 쓸 수 없다는 제약(OpenAI function-calling 스펙)을 확인하고 해결.
- 정상/결함 이미지 각각에 대해 end-to-end 파이프라인을 테스트, 두 경우 모두 올바른 심각도로 분류되고 보고서가 `reports/`에 자동 저장되는 것을 확인.
- 에이전트가 Tool 호출 여부를 스스로 판단하는 것을 실제 실행 로그에서 관찰 — 9주차 아키텍처 다이어그램에 반영 예정.

다음 단계(9주차)는 README 작성, 아키텍처 다이어그램(이번 주 관찰한 실행 흐름 포함), 그리고 주요 의사결정 과정(심각도 기준 설정 근거 포함)을 문서화하는 것이다.
