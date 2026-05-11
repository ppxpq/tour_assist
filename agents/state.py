from typing import Annotated, Any, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages


def human_texts(messages: list[BaseMessage]) -> list[str]:
    """提取对话中所有非空 HumanMessage 的文本内容。"""
    return [
        msg.content.strip()
        for msg in messages
        if isinstance(msg, HumanMessage) and (msg.content or "").strip()
    ]

class TravelState(TypedDict, total=False):
    # 通过 add_messages 在节点间自动追加消息
    messages: Annotated[Sequence[BaseMessage], add_messages]
    intent: str
    city: str
    days: int
    start_date: str
    preference: str
    raw_materials: str
    missing_fields: list[str]
    router_reason: str
    router_model: str
    planner_model: str
    user_query: str
    vector_db: Any
    