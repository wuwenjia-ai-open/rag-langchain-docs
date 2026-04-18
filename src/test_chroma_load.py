"""快速测试：直接加载已有的 Chroma（不重新向量化），证明持久化有效"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

print("[...] 加载 embedding 模型（已缓存，秒级）...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

print("[...] 打开已有的 Chroma 向量库（不重新建）...")
vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
)
print(f"[OK] 向量库中有 {vectorstore._collection.count()} 个向量\n")

# 多搜几个问题试试
queries = [
    "什么是 LangChain Agent",
    "如何实现 RAG 检索",
    "tool calling 怎么用",
]

for q in queries:
    print(f"\n问题: {q}")
    print("-" * 60)
    results = vectorstore.similarity_search(q, k=3)
    for i, r in enumerate(results, 1):
        preview = r.page_content.replace("\n", " ")[:80]
        print(f"  [{i}] {r.metadata['source']}")
        print(f"      {preview}...")
