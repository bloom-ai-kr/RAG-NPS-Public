from pathlib import Path
# step6-2) API Key 입력
import os

import streamlit as st

# step1-1) .env 파일의 내용을 환경 변수로 로드
from dotenv import load_dotenv

# # step1-2) 모델 객체 생성 및 Invoke()
# from langchain.chat_models import init_chat_model

# step2-3) Vector Store 구축
from vector_store import build_vector_store

# step3-3) Agentic RAG
from langchain.tools import tool
from vector_store import get_retriever
from langchain.agents import create_agent

# step1-1) .env 파일의 내용을 환경 변수로 로드
load_dotenv()

@tool
def rag_tool(query: str):
    """
    국민연금 2025년 9월 운용 관련 데이터 조회 시 사용
    """
    retriever = get_retriever()
    docs = retriever.invoke(query)
    print(docs)

    return "\n\n".join([doc.page_content for doc in docs])

tools = [rag_tool]

# step2-4) Vector Store 구축
def save_uploaded_file(uploaded_file):
    upload_dir = Path("./uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / uploaded_file.name
    file_path.write_bytes(uploaded_file.getbuffer())
    return str(file_path)

def render_sidebar():
    with st.sidebar:
        # step6-2) API Key 입력
        api_key_input = st.text_input("OpenAI API Key", type="password")
        if api_key_input:
            st.session_state.openai_api_key = api_key_input
            os.environ["OPENAI_API_KEY"] = api_key_input

        # step5-1) 한글/엑셀 파일 업로드 확장
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
            # step5-2) 업로드한 여러 파일을 한번에 벡터스토어로 만들기
            file_paths = []
            for uploaded_file in uploaded_files:
                file_paths.append(save_uploaded_file(uploaded_file))

            result = build_vector_store(file_paths)
            st.session_state.vector_store_ready = True
            st.success(result)

        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_chat():
    st.title("NPS X RAG")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("질문을 입력해 주세요.")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})

    # # step1-2) 모델 객체 생성 및 Invoke()
    # llm = init_chat_model("gpt-5-nano")

    # response = llm.invoke(query)
    # answer = response.content

    # st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # step3-3) Agentic RAG
    agent = create_agent(model="gpt-5-mini", tools=tools)

    # step6-1) 과거 대화 맥락 주입
    history = st.session_state.messages.copy()
    history.append({"role": "user", "content": query})

    response = agent.invoke({"messages": history})
    answer = response['messages'][-1].content

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
