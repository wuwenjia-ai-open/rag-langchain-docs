"""
第1步：从 LangChain 官方文档下载核心 Markdown 文件

LangChain 文档用 Mintlify 托管，支持 llms.txt 标准:
  - /llms.txt 是给 LLM 读的目录索引 (markdown 格式)
  - 每个文档页面都有对应的 .md 纯 Markdown 版本
这比爬 HTML 再清洗干净太多。

流程：
  /llms.txt 拿目录 → 过滤目标 URL → 逐个下载 .md → 保存到 data/
"""
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

LLMS_TXT_URL = "https://python.langchain.com/llms.txt"
INCLUDE_PATH_PATTERNS = [
    r"docs\.langchain\.com/oss/python/langchain/",
    r"docs\.langchain\.com/oss/python/concepts/",
]

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def get_target_urls() -> list[str]:
    """下载 llms.txt，解析出目标 markdown URL 列表"""
    resp = requests.get(LLMS_TXT_URL, timeout=30)
    resp.raise_for_status()

    all_urls = re.findall(r"\((https?://[^\s\)]+\.md)\)", resp.text)
    target = [u for u in all_urls if any(re.search(p, u) for p in INCLUDE_PATH_PATTERNS)]
    return sorted(set(target))


def url_to_filename(url: str) -> str:
    """URL -> 安全文件名，比如
    https://docs.langchain.com/oss/python/langchain/agents.md -> langchain_agents.md
    """
    path = urlparse(url).path.strip("/")
    path = path.removeprefix("oss/python/")
    return path.replace("/", "_")


def download_md(url: str) -> str:
    """下载单个 .md 文件内容"""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def main():
    print("[1/2] 从 llms.txt 收集目标 URL...")
    urls = get_target_urls()
    print(f"      共 {len(urls)} 个 markdown 文件")
    if not urls:
        print("[ERROR] 未找到目标 URL")
        return

    print(f"\n[2/2] 下载并保存到 {DATA_DIR}...")
    success, failed = 0, []
    for i, url in enumerate(urls, 1):
        filename = url_to_filename(url)
        try:
            content = download_md(url)
            header = f"---\nsource: {url}\n---\n\n"
            (DATA_DIR / filename).write_text(header + content, encoding="utf-8")
            success += 1
            print(f"  [{i:3d}/{len(urls)}] OK  {filename}  ({len(content)} chars)")
            time.sleep(0.3)
        except Exception as e:
            failed.append((url, str(e)))
            print(f"  [{i:3d}/{len(urls)}] FAIL {filename}: {e}")

    print(f"\n完成: 成功 {success}, 失败 {len(failed)}")
    if failed:
        print("失败列表:")
        for url, err in failed:
            print(f"  - {url}: {err}")


if __name__ == "__main__":
    main()
