"""环境验证脚本：确认 .env 读取 + DeepSeek API 能通"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("DEEPSEEK_API_KEY")
assert api_key, f"未读到 DEEPSEEK_API_KEY，检查 {env_path} 是否存在"
print(f"[OK] .env 读取成功，key 前缀: {api_key[:10]}...")

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=api_key,
    openai_api_base="https://api.deepseek.com/v1",
    temperature=0.3,
)

print("[...] 正在调用 DeepSeek...")
resp = llm.invoke("用一句话介绍 LangChain 是什么")
print(f"[OK] DeepSeek 响应: {resp.content}")
