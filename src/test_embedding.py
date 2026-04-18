"""单独测试：bge embedding 模型能不能下载和调用"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_huggingface import HuggingFaceEmbeddings

print("[...] 初始化 embedding（第一次会下载约 100MB，耐心等）...")
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("[OK] 模型加载完成")

print("[...] 测试向量化...")
v = embeddings.embed_query("什么是 LangChain")
print(f"[OK] 向量维度: {len(v)}, 前5维: {v[:5]}")
