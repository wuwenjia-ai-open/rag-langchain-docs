"""增量更新向量库：只处理新增/修改/删除的文档，避免全量重建"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 加载环境
load_dotenv(Path(__file__).parent.parent.parent / ".env")

DATA_DIR = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
INDEX_FILE = DATA_DIR / "index.json"


# ========== 工具函数 ==========

def file_hash(path: Path) -> str:
    """计算文件 MD5 哈希（内容指纹）"""
    content = path.read_bytes()
    return hashlib.md5(content).hexdigest()


def load_index() -> dict:
    """读取 index.json,返回 {文件名: hash} 字典"""
    if not INDEX_FILE.exists():
        return {}
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_index(index: dict):
    """保存 {文件名: hash} 到 index.json"""
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def split_document(file: Path) -> list[Document]:
    """读取单个文件并切分成 chunks"""
    content = file.read_text(encoding="utf-8")
    doc = Document(page_content=content, metadata={"source": file.name})

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""],
    )
    return splitter.split_documents([doc])


# ========== 主逻辑 ==========

def main():
    print("=" * 60)
    print("增量更新向量库")
    print("=" * 60)

    # 1. 初始化 embedding + Chroma
    print("\n[1/6] 加载 embedding 模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("[2/6] 打开向量库...")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    print(f"      当前向量数: {vectorstore._collection.count()}")

    # 2. 读取旧索引
    print("\n[3/6] 对比文件变化...")
    old_index = load_index()
    current_files = {f.name: file_hash(f) for f in DATA_DIR.glob("*.md")}

    # 3. 找出变化
    new_files = set(current_files.keys()) - set(old_index.keys())
    deleted_files = set(old_index.keys()) - set(current_files.keys())
    changed_files = {
        name for name in current_files
        if name in old_index and current_files[name] != old_index[name]
    }

    print(f"      新增: {len(new_files)} 个")
    print(f"      修改: {len(changed_files)} 个")
    print(f"      删除: {len(deleted_files)} 个")
    print(f"      不变: {len(current_files) - len(new_files) - len(changed_files)} 个")

    if not (new_files or changed_files or deleted_files):
        print("\n[OK] 没有变化，无需更新")
        return

    # 4. 处理删除
    if deleted_files:
        print(f"\n[4/6] 删除 {len(deleted_files)} 个文件的向量...")
        for name in deleted_files:
            vectorstore._collection.delete(where={"source": name})
            print(f"      - {name}")

    # 5. 处理新增和修改
    to_add = new_files | changed_files
    if to_add:
        print(f"\n[5/6] 处理 {len(to_add)} 个文件（新增+修改）...")

        # 先删除修改过的旧版本
        for name in changed_files:
            vectorstore._collection.delete(where={"source": name})

        # 切分并添加
        all_chunks = []
        for name in to_add:
            chunks = split_document(DATA_DIR / name)
            all_chunks.extend(chunks)
            tag = "新增" if name in new_files else "更新"
            print(f"      [{tag}] {name} → {len(chunks)} 片")

        print(f"      正在向量化 {len(all_chunks)} 个 chunks...")
        vectorstore.add_documents(all_chunks)

    # 6. 保存新索引
    print("\n[6/6] 保存索引...")
    save_index(current_files)

    print(f"\n[OK] 更新完成！当前向量数: {vectorstore._collection.count()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
    
