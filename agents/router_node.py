from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agents.state import TravelState, human_texts
from core.llm_core import get_llm


VALID_INTENTS = frozenset(
    {
        "need_plan",
        "need_more_info",
        "need_answer",
        "need_ticket",
        "general_chat",
        "other",
    }
)

TICKET_QUERY_RE = re.compile(
    r"车票|火车票|高铁票|动车票|列车票|余票|抢票|查票|候补|12306|"
    r"(?:查询|查|看看|看一下|有没有|预订|订|买|购买|改签|退)[^，。！？\n]*"
    r"(?:高铁|动车|火车|列车|铁路|车次)[^，。！？\n]*(?:票|余票)?"
)
RAIL_TICKET_HINT_RE = re.compile(
    r"火车票|高铁票|动车票|列车票|余票|抢票|查票|候补|12306|"
    r"高铁|动车|火车|列车|铁路|车次"
)
NON_TRAIN_TICKET_RE = re.compile(
    r"机票|船票|汽车票|门票|景区|演出票|演唱会|电影票|电影|迪士尼"
)

MISSING_FIELD_LABELS: dict[str, str] = {
    "city": "目的地城市",
    "days": "出行天数",
    "start_date": "出行日期（从哪一天开始，如 5月20日）",
    "preference": "旅行偏好（如美食 / 自然 / 人文 / 亲子 / 摄影 / 休闲）",
}

CLASSIFY_SYSTEM_TEMPLATE = """你是旅游助手的路由模块。
今天是 {today}。请根据对话历史返回一个 JSON 对象，字段如下：

{{
  "intent": "need_plan | need_more_info | need_answer | need_ticket | general_chat | other",
  "city": "目的地城市名，没有则为空字符串",
  "days": 0,  // 旅行天数，注意区分"玩x天"(days=x)和"y天后出发"(days=0，这是出发时间不是行程天数)
  "start_date": "出发日期，格式为 YYYY-MM-DD，没有则为空字符串。如用户说"5月20号出发"则填"{year}-05-20"，说"下周一"则根据今天日期推算具体日期",
  "preference": "旅行偏好，多个用 + 连接，没有则为空字符串",
  "reason": "一句话说明判断依据"
}}

意图定义：
- need_plan: 用户明确要你生成旅游计划/攻略/行程，并且信息基本齐全
- need_more_info: 用户想规划行程，但目的地 / 天数 / 出发日期 / 偏好不完整
- need_ticket: 用户需要查询火车票/高铁票/动车票/列车票/12306 余票、订票、改签、退票等车票信息。只要用户明确要查车票，就优先归为 need_ticket，不要归为 need_answer
- need_answer: 用户在问具体问题，例如天气、景点、美食、交通、当前位置、图片识别、语音内容、知识库内容、个人偏好
- general_chat: 问候、感谢、闲聊
- other: 无法归类

如果用户是在问"我的偏好是什么""知识库里写了什么""根据我上传的资料回答"等，这属于 need_answer，不属于 need_plan。

只输出 JSON，不要输出额外说明。"""


def _get_classify_system() -> str:
    now = datetime.now()
    return CLASSIFY_SYSTEM_TEMPLATE.format(
        today=now.strftime("%Y年%m月%d日"),
        year=now.strftime("%Y"),
    )


def _missing_fields(city: str, days: int, start_date: str, preference: str) -> list[str]:
    missing: list[str] = []
    if not city:
        missing.append("city")
    if days <= 0:
        missing.append("days")
    if not start_date:
        missing.append("start_date")
    if not preference:
        missing.append("preference")
    return missing


def _build_missing_prompt(missing: list[str]) -> str:
    readable = "、".join(
        MISSING_FIELD_LABELS[field] for field in missing if field in MISSING_FIELD_LABELS
    )
    return (
        f"为了给你生成准确的行程，还需要确认：**{readable}**。\n"
        "把这些信息告诉我后，我就可以继续规划。"
    )


def _normalize_intent(raw: str) -> str:
    intent = (raw or "").strip().lower()
    return intent if intent in VALID_INTENTS else "other"


def _sanitize_days(raw: object) -> int:
    try:
        days = int(raw)  # type: ignore[arg-type]
        return days if 1 <= days <= 30 else 0
    except (TypeError, ValueError):
        return 0


def _safe_parse_json(text: str) -> Optional[dict]:
    cleaned = re.sub(r"```(?:json)?\s*|```", "", text).strip()
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def _looks_like_ticket_query(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text or "")
    if not normalized:
        return False
    if NON_TRAIN_TICKET_RE.search(normalized) and not RAIL_TICKET_HINT_RE.search(normalized):
        return False
    return bool(TICKET_QUERY_RE.search(normalized))


def router_agent(state: TravelState) -> dict:
    messages: list[BaseMessage] = list(state.get("messages", []))
    user_texts = human_texts(messages)
    user_input = user_texts[-1] if user_texts else ""

    if not user_input:
        return {
            "intent": "general_chat",
            "city": "",
            "days": 0,
            "start_date": "",
            "preference": "",
            "missing_fields": [],
            "router_reason": "empty_input",
            "user_query": "",
            "messages": [
                AIMessage(
                    content="你好，告诉我你想去哪里、玩几天、从哪天出发，以及更偏好的旅行风格，我就可以开始帮你规划。"
                )
            ],
        }

    if _looks_like_ticket_query(user_input):
        return {
            "intent": "need_ticket",
            "city": (state.get("city") or "").strip(),
            "days": _sanitize_days(state.get("days", 0)),
            "start_date": (state.get("start_date") or "").strip(),
            "preference": (state.get("preference") or "").strip(),
            "missing_fields": [],
            "router_reason": "ticket_keyword",
            "user_query": user_input,
        }

    recent_msgs = messages[-6:]
    context_lines = []
    for msg in recent_msgs[:-1]:
        if isinstance(msg, HumanMessage):
            context_lines.append(f"[用户] {msg.content}")
        elif isinstance(msg, AIMessage):
            context_lines.append(f"[AI] {msg.content}")
    context_for_llm = "\n".join(context_lines) + f"\n[当前] {user_input}"

    target_model = (state.get("router_model") or "glm-4-flash").strip()
    try:
        llm = get_llm(target_model)
    except Exception:
        return {
            "intent": "other",
            "city": "",
            "days": 0,
            "start_date": "",
            "preference": "",
            "missing_fields": [],
            "router_reason": "llm_init_failed",
            "user_query": user_input,
        }

    try:
        response = llm.invoke(
            [
                SystemMessage(content=_get_classify_system()),
                HumanMessage(content=context_for_llm),
            ]
        )
        raw_text = response.content if hasattr(response, "content") else str(response)
        parsed = _safe_parse_json(raw_text)
    except Exception:
        parsed = None

    if parsed is None:
        return {
            "intent": "other",
            "city": "",
            "days": 0,
            "start_date": "",
            "preference": "",
            "missing_fields": [],
            "router_reason": "llm_parse_failed",
            "user_query": user_input,
        }

    state_city = (state.get("city") or "").strip()
    state_days = _sanitize_days(state.get("days", 0))
    state_start_date = (state.get("start_date") or "").strip()
    state_preference = (state.get("preference") or "").strip()

    intent = _normalize_intent(str(parsed.get("intent", "")))
    city_new = str(parsed.get("city", "") or "").strip()
    days_new = _sanitize_days(parsed.get("days", 0))
    start_date_new = str(parsed.get("start_date", "") or "").strip()
    preference_new = str(parsed.get("preference", "") or "").strip()

    city = city_new or state_city
    days = days_new or state_days
    start_date = start_date_new or state_start_date
    preference = preference_new or state_preference

    missing: list[str] = []
    if intent in {"need_plan", "need_more_info"}:
        missing = _missing_fields(city, days, start_date, preference)
        intent = "need_more_info" if missing else "need_plan"

    reply: Optional[AIMessage] = None
    if intent == "need_more_info":
        reply = AIMessage(content=_build_missing_prompt(missing))
    elif intent in {"general_chat", "other"}:
        try:
            chat_llm = get_llm("glm-4-flash")
            chat_response = chat_llm.invoke(
                [
                    SystemMessage(
                        content="你是一个友好的旅游规划助手。请用自然、简洁的方式回复用户的闲聊或无法归类的消息。"
                        "可以适当引导用户了解你的能力（旅游规划、景点/天气/路线查询、知识库问答等），但不要每次都重复自我介绍。"
                    ),
                    HumanMessage(content=user_input),
                ]
            )
            reply = AIMessage(
                content=chat_response.content if hasattr(chat_response, "content") else str(chat_response)
            )
        except Exception:
            reply = AIMessage(
                content="你好，我是旅游规划小助手。我可以帮你做旅游规划，也可以回答景点、天气、路线、知识库内容等具体问题。"
            )

    output: dict = {
        "intent": intent,
        "city": city,
        "days": days,
        "start_date": start_date,
        "preference": preference,
        "missing_fields": missing,
        "router_reason": "llm",
        "user_query": user_input,
    }
    if reply is not None:
        output["messages"] = [reply]

    return output
