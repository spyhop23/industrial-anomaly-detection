# 7주차 AI Agent 레이어 ① — 단일 에이전트

목표: 자연어로 시스템과 상호작용 — "이 이미지 검사해줘", "모델 성능 어때?" 같은 질문에 LLM이 적절한 함수를 스스로 골라 실행하고 답하게 만든다.

## 1. LangChain 기본 개념

- **Tool**: LLM이 호출할 수 있는 함수. `@tool` 데코레이터와 docstring(이 도구가 언제 쓰이는지 설명)만 붙이면 된다. 이 docstring이 LLM에게 "언제 이 도구를 써야 하는지" 알려주는 매뉴얼 역할을 한다.
- **Agent**: 사용자 질문을 보고 어떤 Tool을 쓸지, 몇 번 쓸지, 언제 끝낼지를 LLM이 스스로 판단하며 실행하는 루프. 개발자가 미리 if문으로 분기하는 게 아니라 LLM이 그때그때 판단한다.
- **Chain**: Agent와 달리 판단 없이 정해진 순서로만 흘러가는 고정 파이프라인. 이번 프로젝트는 "이미지 검사"와 "실험 결과 조회" 중 사용자가 뭘 원하는지 미리 알 수 없으므로 Chain이 아닌 Agent 방식을 택했다.

## 2. 구현 — 기존 코드를 Tool로 wrapping

새 코드는 기존 서빙 코드(`app.py`, `inference.py`)와 섞이지 않도록 `serving/agent/` 서브폴더에 분리했다.

```
serving/
├── app.py, inference.py   (5~6주차, 안 건드림)
├── models/
└── agent/                  (7주차 신규)
    ├── agent.py
    ├── agent_tools.py
    ├── requirements-agent.txt
    └── .env / .env.example
```

**Tool 1 — `inspect_image_tool`**: `inference.py`의 `AnomalyDetector.predict()`를 그대로 감싸서, 이미지 경로를 받아 anomaly score와 판정을 반환한다. 모델은 프로세스 시작 시 한 번만 로드해서 재사용한다 (매 요청마다 로드하면 느림).

**Tool 2 — `query_experiment_results_tool`**: 원래 계획은 MLflow 실험 서버를 조회하는 것이었지만, 실제로는 실험 결과를 MLflow 서버가 아니라 `scripts/docs/*.md`에 마크다운 표로 기록해왔다. 그래서 이 Tool은 MLflow API 대신 해당 문서들의 마크다운 표를 파싱해서, 카테고리/백엔드 키워드로 관련 행을 찾아 반환하도록 만들었다. "계획에 쓰인 기술(MLflow)"이 아니라 "실제로 쓰고 있는 기록 방식(.md)"에 맞춰 구현을 조정한 것.

경로는 전부 `__file__` 기준으로 계산해서, 어느 폴더에서 `python agent.py`를 실행하든 `inference.py`/`models`/`scripts/docs`를 항상 올바르게 찾도록 했다.

## 3. 트러블슈팅: LangChain 1.0의 Agent API 변경

처음 작성한 코드는 `AgentExecutor` + `create_tool_calling_agent` (구버전 문법)를 사용했는데, 실행하자마자 아래 에러가 발생했다.

```
ImportError: cannot import name 'AgentExecutor' from 'langchain.agents'
```

설치된 버전을 확인해보니 `langchain 1.4.6`으로, 2025년 공개된 LangChain 1.0 계열이었다. 이 버전에서 에이전트 생성 방식이 크게 바뀌어, `AgentExecutor`/`create_tool_calling_agent` 조합(구버전 문법)이 레거시로 밀려나고, 대신 `create_agent()` 함수 하나로 LLM + Tool 목록 + 시스템 프롬프트를 받아 에이전트(내부적으로 LangGraph 기반)를 구성하는 방식으로 단순화됐다.

**해결**: `ChatPromptTemplate` + `AgentExecutor` 조합을 걷어내고, `langchain.agents.create_agent(model=llm, tools=tools, system_prompt=...)`로 교체했다. 입출력도 `{"input": ...}` / `result["output"]` 형태에서, 메시지 리스트를 주고받는 형태(`{"messages": [...]}` / `result["messages"][-1].content`)로 바뀌었다.

**교훈**: 빠르게 발전하는 라이브러리(LangChain 같은)는 예제/튜토리얼이 쓰인 시점과 실제 설치되는 버전 사이에 메이저 API 변경이 있을 수 있다. 에러 메시지만 보고 패키지를 재설치하기보다, 설치된 실제 버전을 먼저 확인하고 그 버전의 API에 맞춰 코드를 고치는 게 맞는 접근이었다.

## 4. 테스트 결과

세 가지 자연어 질의 모두 올바른 Tool을 선택하고, 정확한 수치로 답했다. 모든 수치가 이전 주차(4주차 TensorRT 벤치마크, 5주차 로컬 배포) 기록과 정확히 일치한다.

| 질문 | 선택된 Tool | 답변 요약 | 기존 기록과 일치 여부 |
|---|---|---|---|
| "bottle_bad.png 이미지 검사해줘" | `inspect_image_tool` | anomaly_score 0.6751, 판정 "불량(이상)" | 5주차 테스트 결과와 일치 |
| "capsule pytorch 레이턴시 알려줘" | `query_experiment_results_tool` | PyTorch 40.75ms, TensorRT FP16 4.19ms, 9.73배 속도 향상 | 4주차 벤치마크 표와 일치 |
| "bottle이 TensorRT에서 AUROC 얼마나 나왔어?" | `query_experiment_results_tool` | AUROC 1.0000 | 4주차 벤치마크 표와 일치 |

도구 결과를 그대로 나열하지 않고, 자연스러운 한국어 문장으로 설명하는 것까지 시스템 프롬프트에 지정한 대로 동작했다.

## 최종 결과

- 기존 추론 함수와 실험 기록 조회를 각각 LangChain Tool로 wrapping 완료.
- `create_agent`로 단일 에이전트를 구성해, 자연어 질의에 맞는 Tool을 스스로 선택하고 정확한 수치로 응답하는 것을 확인.
- LangChain 1.0 API 변경사항을 반영해 최신 버전과 호환되는 코드로 정리.
- MLflow 서버 대신 실제로 쓰고 있는 기록 방식(.md 문서)에 맞춰 실험 결과 조회 Tool을 구현 — "계획"보다 "실제 환경"에 맞춘 설계.

다음 단계(8주차)는 이 단일 에이전트를 역할별로 쪼개서(검사/분석/보고서 작성), 멀티 에이전트 파이프라인으로 확장하는 것이다.
