# LangChain 文档问答助手

基于 RAG（检索增强生成）技术的智能文档问答系统，爬取 LangChain 官方文档并实现中文问答。

![技术栈](https://img.shields.io/badge/LangChain-0.3-blue) ![Python](https://img.shields.io/badge/Python-3.11+-green) ![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-orange)

## 项目背景

LangChain 官方文档全英文且内容庞大，开发者查找信息效率低。本项目通过 RAG 技术，让 AI 能基于官方文档准确回答问题，避免 LLM 幻觉，提升学习效率。

## 技术栈

| 技术 | 用途 | 选型理由 |
|------|------|---------|
| **LangChain** | RAG 流程编排 | 成熟的 LLM 应用框架，组件丰富 |
| **DeepSeek** | 大语言模型 | 中文能力强，API 便宜（¥1/百万 tokens） |
| **Chroma** | 向量数据库 | 轻量级，本地部署，零配置 |
| **bge-small-zh-v1.5** | 文本向量化 | 中文 embedding 效果好，模型小（~100MB） |
| **Streamlit** | 前端界面 | 纯 Python 实现 Web UI，开发快 |

## 核心功能

- ✅ 爬取 LangChain 官方文档（66 篇核心文档）
- ✅ 文本切分与向量化（4149 个 chunks）
- ✅ 基于语义相似度的文档检索
- ✅ 引用来源展示（可溯源）
- ✅ **增量更新机制**（工程亮点）

## 项目亮点

### 1. 增量更新机制

生产环境中，文档会频繁更新。全量重建向量库耗时且浪费资源。

**解决方案**：基于 MD5 哈希的增量更新
- 用 `index.json` 记录每个文档的内容指纹
- 每次更新时对比新旧指纹，只处理变化的文档
- 支持新增、修改、删除三种操作

**效果**：66 个文档中只有 2 个变化时，更新时间从 3 分钟降到 10 秒。

```python
# 核心逻辑
new_files = set(current_files.keys()) - set(old_index.keys())
changed_files = {name for name in current_files 
                 if name in old_index and current_files[name] != old_index[name]}
deleted_files = set(old_index.keys()) - set(current_files.keys())
```

### 2. 文本切分策略

使用 `RecursiveCharacterTextSplitter`，优先按 Markdown 标题切分，保证语义完整性。

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,       # 每片最大字符数
    chunk_overlap=100,    # 相邻片重叠，防止切断句子
    separators=["\n## ", "\n### ", "\n\n", "\n", "。", " ", ""],
)
```

**参数调优**：
- `chunk_size=800`：平衡检索精度和上下文完整性
- `chunk_overlap=100`：避免关键信息被切断

### 3. Prompt 工程

设计了防幻觉的 Prompt 模板：

```
你是一个 LangChain 文档助手，根据以下参考资料回答用户问题。
如果参考资料中没有相关信息，直接说"文档中没有找到相关内容"，不要编造。

参考资料：{context}
用户问题：{question}
```

明确要求"不知道就说不知道"，避免 LLM 编造答案。

## 项目结构

```
rag-langchain-docs/
├── src/
│   ├── load_docs.py           # 爬取 LangChain 官方文档
│   ├── split_docs.py           # 文本切分验证
│   ├── build_vectorstore.py    # 全量建库（首次使用）
│   ├── update_vectorstore.py   # 增量更新（核心）
│   ├── rag_chain.py            # 命令行问答
│   └── app.py                  # Streamlit 前端
├── data/                       # 文档存储（66 个 .md 文件）
├── chroma_db/                  # 向量库持久化（4149 个向量）
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/你的用户名/rag-langchain-docs.git
cd rag-langchain-docs

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 初始化向量库

```bash
# 爬取文档
python src/load_docs.py

# 全量建库（首次运行，约 3-5 分钟）
python src/build_vectorstore.py
```

### 4. 启动应用

```bash
# 启动 Web 界面
streamlit run src/app.py

# 或使用命令行版本
python src/rag_chain.py
```

访问 `http://localhost:8501` 即可使用。

## 增量更新

文档有变化时，只需运行：

```bash
python src/update_vectorstore.py
```

脚本会自动检测新增、修改、删除的文档，只处理变化部分。

## 技术细节

### RAG 流程

```
用户提问
    ↓ bge-small-zh 向量化
    ↓ Chroma 检索 top-5 相关文档
    ↓ 拼接 Prompt（参考资料 + 问题）
    ↓ DeepSeek 生成回答
用户看到答案 + 引用来源
```

### 性能指标

| 指标 | 数值 |
|------|------|
| 文档数量 | 66 篇 |
| 向量数量 | 4149 个 |
| 向量库大小 | 52 MB |
| 平均响应时间 | 2-3 秒 |
| 检索准确率 | ~85%（主观评估） |

## 遇到的挑战与解决

### 1. HuggingFace 模型下载超时

**问题**：国内访问 `huggingface.co` 经常超时

**解决**：使用国内镜像站 `hf-mirror.com`

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### 2. 向量库重复数据

**问题**：多次运行 `Chroma.from_documents()` 导致数据重复

**解决**：实现增量更新机制，避免全量重建

### 3. DeepSeek API 兼容性

**问题**：OpenAI SDK v2.x 参数名变更导致连接失败

**解决**：使用 `openai_api_key` / `openai_api_base` 而非新版参数名

## 后续优化方向

- [ ] 支持多轮对话（添加 Memory）
- [ ] 接入更多数据源（GitHub Issues、社区问答）
- [ ] 优化检索策略（混合检索：向量 + 关键词）
- [ ] 添加评估指标（RAGAS 框架）
- [ ] 部署到云端（Docker + Vercel）

## 开发环境

- Python 3.11+
- Windows 11 / macOS / Linux
- 8GB+ RAM（向量化时占用约 2GB）

## 许可证

MIT License

## 联系方式

- GitHub: [@wuwenjia-ai-open](https://github.com/wuwenjia-ai-open)
- Email: wuwenjia@outlook.com

---

⭐ 如果这个项目对你有帮助，欢迎 Star！
