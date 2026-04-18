"""Streamlit 前端：LangChain 文档问答助手"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 加载环境变量
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# 页面配置
st.set_page_config(
    page_title="LangChain 文档助手",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 LangChain 文档问答助手")
st.markdown("基于 RAG 技术，回答关于 LangChain 官方文档的问题")


# 缓存模型加载（避免每次刷新都重新加载）
@st.cache_resource
def load_rag_chain():
    """初始化 RAG chain（只加载一次）"""
    # 1. LLM
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0.3,
    )

    # 2. 向量库
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

    # 3. Prompt
    prompt = ChatPromptTemplate.from_template("""
你是一个 LangChain 文档助手，根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，直接说"文档中没有找到相关内容"，不要编造。

参考资料：
{context}

用户问题：{question}
""")

    # 4. Chain
    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# 加载 chain（显示加载状态）
with st.spinner("正在加载模型..."):
    chain, retriever = load_rag_chain()

st.success("✅ 模型加载完成，可以开始提问了！")

# 侧边栏：示例问题
st.sidebar.header("💡 示例问题")
example_questions = [
    "什么是 LangChain Agent？",
    "如何实现 RAG 检索？",
    "tool calling 怎么配置？",
    "LangChain 支持哪些模型？",
    "如何使用 memory 实现多轮对话？",
]
for q in example_questions:
    if st.sidebar.button(q, key=q):
        st.session_state.question = q

# 主界面：问答区
question = st.text_input(
    "请输入你的问题：",
    value=st.session_state.get("question", ""),
    placeholder="例如：什么是 LangChain Agent？",
)

if st.button("🔍 提问", type="primary") or question:
    if question.strip():
        with st.spinner("正在思考..."):
            try:
                # 获取答案
                answer = chain.invoke(question)

                # 显示答案
                st.markdown("### 📝 回答")
                st.markdown(answer)

                # 显示引用来源
                with st.expander("📚 查看引用来源"):
                    docs = retriever.invoke(question)
                    for i, doc in enumerate(docs, 1):
                        st.markdown(f"**来源 {i}：{doc.metadata['source']}**")
                        st.text(doc.page_content[:300] + "...")
                        st.divider()

            except Exception as e:
                st.error(f"❌ 出错了：{e}")
    else:
        st.warning("请输入问题")

# 底部说明
st.markdown("---")
st.caption("💡 技术栈：LangChain + DeepSeek + Chroma + bge-small-zh-v1.5 + Streamlit")
