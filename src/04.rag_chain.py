import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 必须在所有 import 之前

from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#1.加载api环境
load_dotenv(Path(__file__).parent.parent.parent / ".env")

#2.初始化模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    temperature=0.3,
)

#3.把向量库变成检索器
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

#4.构建Prompt
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
你是一个 LangChain 文档助手，根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，直接说"文档中没有找到相关内容"，不要编造。

参考资料：
{context}

用户问题：{question}
""")

#5把context拼接成字符串

# retriever 返回的是 Document 列表，要拼成字符串喂给 prompt
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

#6.用 LCEL 把所有组件串起来

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)#RunnablePassthrough() =“不加工、不转换、不修改，输入什么就传什么。保留原问题个检索到的内容一起传入Prompt”

#调用
# answer = chain.invoke("什么是 LangChain Agent?")
# print(answer)


#7.问答循环
while True:
    question = input("\n请输入您的问题: ").strip()
    if question.lower() in ("quit", "exit", "q", "退出"):#忽略大小写比较
        break
    answer = chain.invoke(question)
    print(f"\n小智: {answer}")
