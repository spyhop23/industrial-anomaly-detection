# serving/dashboard/dashboard.py
"""
7주차(단일 에이전트) + 8주차(멀티 에이전트 CrewAI 파이프라인)를
브라우저에서 쓸 수 있는 대시보드로 합친 것.

새 폴더를 또 만든 이유: 7주차(agent/), 8주차(crew/)와 마찬가지로
기존 코드를 건드리지 않고, 이 둘을 "불러다 쓰는" 얇은 UI 레이어만
따로 분리해두기 위해서다. (서버 코드 자체를 수정하지 않는다.)

실행 전 준비:
    1. serving/dashboard 폴더에서:
       pip install -r ../requirements.txt -r ../agent/requirements-agent.txt -r ../crew/requirements-crew.txt -r requirements-dashboard.txt
    2. .env.example을 복사해 .env 파일을 만들고 OPENAI_API_KEY=sk-... 설정
       (agent/.env 또는 crew/.env를 이미 만들어뒀다면 같은 키를 그대로 복사해도 된다)

실행 (반드시 serving/dashboard 폴더 안에서):
    cd serving/dashboard
    streamlit run dashboard.py
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# --- 경로 설정: 7주차(agent)와 8주차(crew) 코드를 그대로 불러다 쓴다 ---------
DASHBOARD_DIR = Path(__file__).resolve().parent         # .../serving/dashboard
SERVING_DIR = DASHBOARD_DIR.parent                        # .../serving
AGENT_DIR = SERVING_DIR / "agent"                          # 7주차
CREW_DIR = SERVING_DIR / "crew"                             # 8주차

load_dotenv(DASHBOARD_DIR / ".env")  # 이 폴더의 .env를 우선 사용

sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(CREW_DIR))

from agent import build_agent          # 7주차: 단일 에이전트  # noqa: E402
from crew_pipeline import run_pipeline  # 8주차: 멀티 에이전트 파이프라인  # noqa: E402

st.set_page_config(page_title="품질 검사 AI 대시보드", page_icon="🔍", layout="centered")


# --- 7주차 단일 에이전트는 매번 새로 만들지 않고 캐싱해서 재사용 ------------
@st.cache_resource(show_spinner="에이전트를 준비하는 중...")
def get_chat_agent():
    return build_agent()


st.title("🔍 품질 검사 AI 대시보드")
st.caption("7주차(단일 에이전트) + 8주차(멀티 에이전트 CrewAI 파이프라인)를 한 화면에서 사용합니다.")

tab_chat, tab_inspect = st.tabs(["💬 질문하기 (7주차)", "📋 이미지 검사 → 불량 보고서 (8주차)"])

# =========================================================================
# 탭 1 — 7주차: 자연어로 질문 (단일 에이전트)
# =========================================================================
with tab_chat:
    st.markdown(
        "예시 질문: *\"bottle_bad.png 이미지 검사해줘\"*, "
        "*\"bottle이 TensorRT에서 AUROC 얼마나 나왔어?\"*"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("질문을 입력하세요")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            agent = get_chat_agent()
        except RuntimeError as e:
            st.error(str(e))
        else:
            with st.chat_message("assistant"):
                with st.spinner("생각하는 중..."):
                    # create_agent로 만든 에이전트는 메시지 목록을 주고받는다.
                    # 매번 전체 대화 기록을 같이 넘겨서 멀티턴 맥락을 유지한다.
                    result = agent.invoke({"messages": st.session_state.chat_history})
                    answer = result["messages"][-1].content
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

# =========================================================================
# 탭 2 — 8주차: 이미지 업로드 → 검사/분석/보고서 작성 멀티 에이전트 파이프라인
# =========================================================================
with tab_inspect:
    st.markdown("이미지를 올리면 **검사 → 분석 → 보고서 작성** 3개 에이전트가 순서대로 실행됩니다.")

    uploaded_file = st.file_uploader("검사할 이미지", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption=uploaded_file.name, width=300)

        if st.button("검사 시작", type="primary"):
            # 업로드된 이미지를 임시 파일로 저장한 뒤, 그 경로를 파이프라인에 넘긴다.
            suffix = Path(uploaded_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                with st.spinner("3개 에이전트가 순서대로 작업 중입니다... (검사 → 분석 → 보고서 작성)"):
                    report_path = run_pipeline(tmp_path)
                report_text = report_path.read_text(encoding="utf-8")
            except RuntimeError as e:
                st.error(str(e))
            else:
                st.success(f"완료! 보고서가 저장됐습니다: {report_path.name}")
                st.markdown(report_text)
                st.download_button(
                    "보고서 다운로드 (.md)",
                    data=report_text,
                    file_name=report_path.name,
                    mime="text/markdown",
                )
