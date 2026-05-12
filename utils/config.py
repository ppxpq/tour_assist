
import os
from pathlib import Path

from dotenv import load_dotenv

# --- 基础路径 ---
BASE_DIR = Path(__file__).resolve().parents[1]
ENV_PATH = BASE_DIR / "api_key.env"
PERSIST_PATH = str(BASE_DIR / "data" / "chroma_db")
UPLOAD_DIR = str(BASE_DIR / "data" / "uploads")

# 加载本地密钥文件；系统环境变量优先，便于部署和 CI 注入密钥。
load_dotenv(dotenv_path=ENV_PATH, override=False)

# --- 高德地图 API 配置 ---
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

# --- 智谱 AI 配置 ---
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")

# --- 代理/Gemini 配置 ---
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
PROXY_ENDPOINT = os.getenv("PROXY_ENDPOINT", "http://127.0.0.1:7897")

# --- 阿里云 API 配置 ---
ALI_API_KEY = os.getenv("ALI_API_KEY")
ALI_BASE_URL = os.getenv("ALI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# --- MiMo API 配置 ---
MIMO_API_KEY = os.getenv("MIMO_API_KEY")
MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")

# --- Claude 代理配置 ---
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "sk-f040701f1a944bb8b239ad5ac096ef78")
CLAUDE_BASE_URL = os.getenv("CLAUDE_BASE_URL", "http://127.0.0.1:8045/v1")

# --- 模型列表 ---
MODEL_LIST = [
    "glm-4.5-air",
    "glm-4-flash",
    "glm-4.7-flash",
    "gemini-3-flash",
    "gemini-3.1-pro",
    "claude-opus-4-6-thinking",
    "qwen-turbo",
    "qwen-plus",
    "qwen-max",
    "qwen-vl-max",
    "qwen-audio-turbo",
    "mimo-v2.5-pro",
    "mimo-v2-omni",
]

# --- 初始化环境 (移除代理) ---
def init_env():
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)
    os.environ.pop('ALL_PROXY', None)
