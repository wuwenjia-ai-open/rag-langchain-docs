"""第2步:把 data/ 目录下所有 .md 文件切分成小块"""
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent.parent / "data"


# ========== 任务 1: 把 data/ 下的所有 .md 读成 Document 列表 ==========

docs = []
for file in DATA_DIR.glob("*.md"):
    content = file.read_text(encoding="utf-8")
    # TODO 1: 构造一个 Document 对象，append 到 docs 里
    # 提示: Document(page_content=???, metadata={"source": ???})
    # page_content 放什么？metadata 里 source 放什么（用 file.name）？
    doc= Document(page_content= content, metadata={"source":file.name})
    docs.append(doc)


print(f"读取了 {len(docs)} 个文档")


# ========== 任务 2: 切分（这块你之前写对了，保留） ==========

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""],
)
chunks = splitter.split_documents(docs)


# ========== 任务 3: 打印验证 ==========
# TODO 2: 打印总共切成多少片
# 提示: len(chunks)
print(f"切成了 {len(chunks)} 片")  # 把 ??? 改掉

# TODO 3: 算每片的字符长度，打印最长/最短/平均
# 提示:
lengths = [len(c.page_content) for c in chunks]
# max(lengths), min(lengths), sum(lengths) // len(lengths)
print(f"每片最长字符为 : {max(lengths)}")
print(f"每片最短字符串为: {min(lengths)}")
print(f"每片平均字符为: {sum(lengths) // len(lengths)}")
# ...你来写...



# TODO 4: 打印第 0、50、100 片的内容，看看切得合不合理
# 提示:
for i in [0, 50, 100]:
    print(f"--- 第 {i} 片 (来自 {chunks[i].metadata['source']}) ---")
    print(chunks[i].page_content)
# ...你来写...
