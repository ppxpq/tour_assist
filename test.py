import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "api_key.env")

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

MIMO_API_KEY = os.getenv("MIMO_API_KEY")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")

print(f"Base URL: {MIMO_BASE_URL}")
print(f"API Key: {MIMO_API_KEY[:10]}...")
print()

for model in ["mimo-v2.5-pro", "mimo-v2-omni"]:
    print(f"--- 测试 {model} ---")
    try:
        llm = ChatOpenAI(model=model, api_key=MIMO_API_KEY, base_url=MIMO_BASE_URL, temperature=0.1)
        resp = llm.invoke([HumanMessage(content="你好，用一句话介绍你自己")])
        print(f"  回复: {resp.content[:200]}")
        print(f"  状态: OK")
    except Exception as e:
        print(f"  错误: {e}")
    print()