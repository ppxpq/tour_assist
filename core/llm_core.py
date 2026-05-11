
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from utils import config

# --- 全局 Embedding 实例 ---
@lru_cache(maxsize=1)
def get_embeddings():
    """缓存 Embeddings 实例，避免重复初始化开销。"""
    return OpenAIEmbeddings(
        model="embedding-3",
        openai_api_key=config.ZHIPU_API_KEY,
        openai_api_base=config.ZHIPU_BASE_URL,
        chunk_size=60
    )

# --- LLM 工厂函数（已支持：智谱 + Gemini + 阿里云通义千问）---
@lru_cache(maxsize=16)
def get_llm(model_name):
    model_name = (model_name or "").strip()
    if not model_name:
        raise ValueError("model_name must not be empty")
    model_lower = model_name.lower()
    
    # 1. 智谱 GLM 系列
    if "glm" in model_lower:
        return ChatOpenAI(
            model=model_name,
            openai_api_key=config.ZHIPU_API_KEY,
            openai_api_base=config.ZHIPU_BASE_URL,
            temperature=0.1
        )
    
    # 2. MiMo 系列
    elif "mimo" in model_lower:
        return ChatOpenAI(
            model=model_name,
            api_key=config.MIMO_API_KEY,
            base_url=config.MIMO_BASE_URL,
            temperature=0.1,
        )

    # 3. 阿里云通义千问 qwen 系列（新增）
    elif "qwen" in model_lower:
        return ChatOpenAI(
            model=model_name,
            api_key=config.ALI_API_KEY,
            base_url=config.ALI_BASE_URL,
            temperature=0.1
        )
    
    # 4. Gemini / Claude 等代理模型
    else:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=config.PROXY_API_KEY,
            transport="rest",
            client_options={"api_endpoint": config.PROXY_ENDPOINT},
            temperature=0.1
        )

# --- RAG Chain 构建器 ---
def create_rag_chain(vector_db, llm):
    # 1. 定义检索器 (MMR 算法保障检索结果的多样性)
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 30}
    )

    # 2. 历史感知 Prompt (Pro 级优化)
    history_system_prompt = (
        "你是一个专门负责重构查询请求的专家系统。请分析下方提供的对话历史记录与用户的最新提问。\n"
        "【任务目标】\n"
        "如果最新提问中包含指代词（如“它”、“这个”）或依赖前文语境，请将其重写为一个完全独立、语义清晰且无歧义的完整问题。\n"
        "如果最新提问本身已经独立完整，请直接返回原提问。\n"
        "【绝对禁令】\n"
        "严格限制：你的任务仅仅是提取和重写问题，绝不能尝试回答该问题，也不要输出任何前缀或解释性语言。"
    )
    history_prompt = ChatPromptTemplate.from_messages([
        ("system", history_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_retriever = create_history_aware_retriever(llm, retriever, history_prompt)

    # 3. QA Prompt (Pro 级优化)
    qa_system_prompt = (
        "你是一个严谨且专业的 AI 知识助手。你的核心任务是仅基于提供的上下文信息来准确解答用户的问题。\n\n"
        "【遵循原则】\n"
        "1. 信息溯源：回答必须能直接在提供的上下文中找到依据，绝对禁止凭空捏造或引入外部知识。\n"
        "2. 拒绝猜测：如果上下文信息不足以回答问题，请直接回复：“根据提供的上下文，我无法回答这个问题。”\n"
        "3. 结构清晰：对复杂信息进行总结，合理使用Markdown排版（如列表、粗体）以提升可读性。\n\n"
        "【上下文信息】\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_retriever, qa_chain)
