import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


docs = []
for file in DATA_DIR.glob("*.md"):
    content = file.read_text(encoding="utf-8")
    
    doc= Document(page_content= content, metadata={"source":file.name})
    docs.append(doc)





splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""],
)
chunks = splitter.split_documents(docs)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#初始化embedding模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},      # 没 GPU 就用 CPU
    encode_kwargs={"normalize_embeddings": True},  # 归一化，检索更准
)

#存入Charm(自动向量化)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(CHROMA_DIR),   # 持久化位置   Path(...) 是 pathlib 的 Path 对象（方便拼接路径、跨平台）
# Chroma 的 API 要字符串，所以用 str() 转一下
)

# 3. 验证：搜一下
results = vectorstore.similarity_search("什么是 Agent?", k=3)
for r in results:
    print(r.metadata["source"], r.page_content[:100])