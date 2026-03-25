from pathlib import Path
# step8-1) API Key 입력
import os

import streamlit as st

# step1-1) .env 파일의 내용을 환경 변수로 로드
from dotenv import load_dotenv

# # step1-2) 모델 객체 생성 및 Invoke()
# from langchain.chat_models import init_chat_model

# step2-3) Vector Store 구축
from vector_store import build_vector_store

# step3-3) DB import 및 tool 작성
from langchain.tools import tool
from vector_store import get_retriever
from langchain.agents import create_agent

# step1-1) .env 파일의 내용을 환경 변수로 로드
# step8-2) API Key 입력
# load_dotenv()

# # step1-2) 모델 객체 생성 및 Invoke()
# llm = init_chat_model("gpt-5.4-nano")

# # step3-3) DB import 및 tool 작성
@tool
def rag_tool(query: str):
    """
    2025년 국민연금기금의 운용수익률 및 자산 포트폴리오 성과 데이터를 검색하는 도구입니다.
    사용자가 다음과 같은 정보를 질문할 때 이 도구를 호출하세요:

    1. 기금 전체 수익률 및 규모: 2025년 국민연금 총 수익률, 기금 평가액 등
    2. 자산군별 세부 성과: 국내주식, 해외주식, 국내채권, 해외채권, 대체투자 등 각 자산군의 수익률, 비중, 평가액 현황
    3. 성과 배경 및 시장 요인: 인공지능(AI) 및 반도체 강세, 정부 정책, 국내외 기준금리 인하 등 수익률 변동에 영향을 미친 원인
    4. 과거 평균 수익률: 3년, 5년, 설립 이후 장기 평균 수익률 비교 분석

    '국민연금', '운용수익률', '기금 평가액', '자산 배분', '국내외 주식/채권 수익률' 등의 키워드가 포함된 질문에 필수적으로 사용해야 합니다.
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)
    print(docs)

    return "\n\n".join([doc.page_content for doc in docs])

# step3-3) Agentic RAG
# agent = create_agent(model="gpt-5.4-mini", tools=tools)

# step2-4) Vector Store 구축
def save_uploaded_file(uploaded_file):
    upload_dir = Path("./uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return str(file_path)

def render_sidebar():
    with st.sidebar:
        # step8-3) API Key 입력
        api_key_input = st.text_input("OpenAI API Key", type="password")
        if api_key_input:
            st.session_state.openai_api_key = api_key_input
            os.environ["OPENAI_API_KEY"] = api_key_input

        # step4-1) 한글/엑셀 파일 업로드 확장
        uploaded_files = st.file_uploader(
            "파일 업로드",
            type=["pdf", "hwp", "hwpx", "xlsx"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.session_state.uploaded_files_meta = [
                {"name": file.name, "size": file.size} for file in uploaded_files
            ]
        else:
            st.session_state.uploaded_files_meta = []

        st.subheader("업로드된 파일")
        if st.session_state.uploaded_files_meta:
            for item in st.session_state.uploaded_files_meta:
                size_kb = item["size"] / 1024
                st.write(f"- {item['name']} ({size_kb:.1f} KB)")
        else:
            st.caption("아직 업로드된 파일이 없습니다.")

        # step2-3) Vector Store 구축
        if uploaded_files and st.button("벡터스토어 생성"):
            # step4-2) 업로드한 여러 파일을 한번에 벡터스토어로 만들기
            file_paths = []
            for uploaded_file in uploaded_files:
                file_paths.append(save_uploaded_file(uploaded_file))

            result = build_vector_store(file_paths)
            st.session_state.vector_store_ready = True
            st.success(result)

        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# step7-2)
@st.cache_resource(show_spinner=False)
def get_agent():
    return create_agent(
        model="gpt-5.4-mini", 
        tools = [rag_tool],
        system_prompt = """
            당신은 국민연금 기금 관련 질문에 답변하는 전문 어시스턴트입니다.

            규칙:
            1. 국민연금의 수익률, 평가액, 자산배분, 자산군별 성과 질문이면 rag_tool을 사용하세요.
            2. rag_tool이 반환한 문맥을 근거로만 답변하세요.
            3. 문맥에 없는 내용은 추측하지 말고 모른다고 답하세요.
            4. 수치가 있으면 가능한 한 구체적으로 설명하세요.
        """
    )

# step9-2) Stream Answer
def stream_answer(history):
    agent = get_agent()

    for token, metadata in agent.stream(
        {"messages": history},
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") != "model":
            continue

        text = getattr(token, "text", None)
        if not text:
            content = getattr(token, "content", "")
            text = content if isinstance(content, str) else ""

        if text:
            yield text

def render_chat():
    st.title("NPS X RAG")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("질문을 입력해 주세요.")
    if not query:
        return

    # step7-1)
    with st.chat_message("user"):
        st.write(query)

    st.session_state.messages.append({"role": "user", "content": query})

    # # step1-2) 모델 객체 생성 및 Invoke()
    # response = llm.invoke(query)
    # answer = response.content

    # # step3-4) agent 답변 생성
    # response = agent.invoke(
    #     {"messages" : [{"role": "user", "content": query}]}
    # )
    # answer = response['messages'][-1].content

    # step5-1) 과거 대화 맥락 주입
    # st.session_state.messages.append({"role": "assistant", "content": answer})
    history = st.session_state.messages.copy()
    # history.append({"role": "user", "content": query})

    # # step7-3)
    # agent = get_agent()


    # response = agent.invoke({"messages": history})
    # answer = response['messages'][-1].content

    # step9-1) Stream Answer
    with st.chat_message("assistant"):
        answer = st.write_stream(stream_answer(history), cursor="▌")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


st.set_page_config(page_title="기초 챗봇 UI", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files_meta" not in st.session_state:
    st.session_state.uploaded_files_meta = []
if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if st.session_state.openai_api_key:
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

render_sidebar()
render_chat()
