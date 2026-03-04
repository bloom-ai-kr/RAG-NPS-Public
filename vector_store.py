from pathlib import Path

# step2-1) Vector Store 구축
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_hwp_hwpx import HwpHwpxLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore

# step2-2) Vector Store 구축
CHROMA_PATH = "./chroma_db"
CACHE_PATH = "./cache"
COLLECTION_NAME = "rag_collection"

# step5-4) 한글/엑셀 파일 로더 분기
def load_documents(file_path: str):
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        loader = PDFPlumberLoader(file_path)
    elif extension in [".hwp", ".hwpx"]:
        loader = HwpHwpxLoader(file_path, mode="elements")
    elif extension in [".xlsx", ".xls"]:
        loader = UnstructuredExcelLoader(file_path, mode="elements")
    else:
        raise ValueError("지원하지 않는 파일 형식입니다. pdf, hwp, hwpx, xlsx만 지원합니다.")

    return loader.load()

# step5-3) 여러 파일 문서를 하나로 합치기
def build_vector_store(file_paths: list[str]) -> str:
    docs = []
    for file_path in file_paths:
        new_docs = load_documents(file_path)
        docs.extend(new_docs)

    # split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    recursive_docs = text_splitter.split_documents(docs)

    # Embed
    underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = LocalFileStore(CACHE_PATH)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        store,
        namespace=underlying_embeddings.model
    )

    # Store
    Chroma.from_documents(
        documents=recursive_docs,
        embedding=cached_embedder,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print('벡터스토어 생성 완료')

    return "벡터스토어 생성 완료"


# step3-2) Agentic RAG
def load_vector_store():
    underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = LocalFileStore(CACHE_PATH)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings,
        store,
        namespace=underlying_embeddings.model
    )

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=cached_embedder,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return db

# step3-1) Agentic RAG
def get_retriever(k: int = 3):
    db = load_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever
