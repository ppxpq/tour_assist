from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
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
NON_TRAVEL_PURCHASE_RE = re.compile(
    r"(?:我要|我想|想|准备|打算)?(?:买|购买|入手|下单|预订|订购|找|看)"
    r"[^，。！？\n]*(?:ps5|ps5pro|playstation|ns2|switch\s*2|switch2|xbox|"
    r"游戏机|主机|显卡|手机|电脑|平板|耳机|相机|镜头|家电)",
    re.IGNORECASE,
)
AFFECTION_CHAT_RE = re.compile(r"我爱你|喜欢你|爱你|比心|贴贴")
TRAVEL_SIGNAL_RE = re.compile(
    r"旅游|旅行|出行|行程|攻略|路线|景点|酒店|住宿|民宿|餐厅|美食|天气|"
    r"交通|打车|公交|地铁|高铁|动车|火车|车票|机票|门票|自驾|租车|"
    r"亲子|老人|情侣|朋友|周末|几天|一日游|两日游|三日游|自由行|citywalk",
    re.IGNORECASE,
)
TAG_QUESTION_RE = re.compile(r"什么意思|是什么|是啥|区别|怎么选|如何选|解释|含义")
PLAN_SIGNAL_RE = re.compile(r"规划|安排|路线|行程|攻略|怎么玩|玩|旅行|旅游|出行|去.+?(?:玩|旅行|旅游)")
DAY_RE = re.compile(r"(\d{1,2})\s*(?:天|日游)")
CITY_RE = re.compile(r"(?:去|到|在|游|玩|目的地[:：]?)\s*([\u4e00-\u9fa5]{2,8})(?:玩|旅游|旅行|出行|[，。！？\s]|$)")
FLEXIBLE_START_DATE = "日期灵活"
FLEXIBLE_DATE_RE = re.compile(
    r"(?:随便|任意|都行|都可以|均可|不限|无所谓|不固定|没定|未定|灵活)"
    r"[^，。！？\n]*(?:日期|时间|哪天|哪一天|出发|启程|周末|工作日)|"
    r"(?:日期|时间|出发|启程)[^，。！？\n]*(?:随便|任意|都行|都可以|均可|不限|无所谓|不固定|没定|未定|灵活)"
)
GENERIC_FLEXIBLE_REPLY_RE = re.compile(
    r"^(?:都可以|都行|随便|无所谓|不限|任意|没定|未定|不固定|灵活|"
    r"看你安排|看你推荐|你安排|你定|你决定|按你推荐|听你的)[吧啊呀啦嘛。！!，,]*$"
)
DATE_CLARIFICATION_RE = re.compile(r"哪一天|哪天|出发日期|出行日期|出发时间|日期|时间|这个周末|下周六")
KNOWN_CITIES = [
    "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "无锡", "常州", "扬州", "成都", "重庆",
    "西安", "武汉", "长沙", "厦门", "青岛", "大连", "天津", "宁波", "福州", "泉州", "洛阳", "开封",
]
PREFERENCE_WORDS = [
    "休闲", "美食", "少走路", "无障碍", "自然", "人文", "摄影", "夜游", "小众", "省钱", "亲子",
    "老人", "品质", "舒适", "经济", "公共交通", "打车", "自驾", "骑行",
]
TAG_EXPLANATIONS: dict[str, str] = {
    "经济": "经济预算会优先控制总花费，更多使用公共交通、平价餐饮、免费或低价景点，适合学生党或想压低预算的旅行。",
    "舒适": "舒适预算会在花费和体验之间平衡，减少过度折腾，适当选择更省力的交通、更稳妥的餐饮和节奏更舒服的安排。",
    "品质": "品质预算更重视体验质量，会减少排队和转场，优先考虑更好的餐饮、住宿、交通或特色体验。",
    "少走路": "少走路表示行程会控制步行距离，景点尽量集中，必要时用打车或短途接驳替代长距离步行。",
    "步行友好": "步行友好表示你可以接受较多步行，适合老街、湖区、街区漫游等慢游路线。",
    "无障碍优先": "无障碍优先会优先考虑老人、轮椅、婴儿车等需求，减少台阶，增加休息点，并尽量选择交通和动线更顺的景点。",
    "骑行": "骑行适合城市慢游、绿道、湖区或短距离点位串联，规划时会避免安排不适合骑车的长距离或复杂路段。",
    "打车": "打车偏好会减少换乘和步行，适合亲子、老人同行、赶时间或不想折腾的行程。",
    "公共交通": "公共交通偏好会优先考虑地铁、公交等方式，通常更省钱，但可能需要更多换乘和步行。",
    "自驾/租车": "自驾/租车适合郊区、多点位、跨城或公共交通不方便的路线，但需要考虑停车、限行和驾驶疲劳。",
    "老人": "老人同行会让规划更关注少走路、休息频率、无障碍、就近餐饮和服务点，避免过密转场。",
    "亲子": "亲子同行会更关注安全、节奏、餐饮便利度、洗手间和适合儿童的体验，避免过度赶路。",
}

MISSING_FIELD_LABELS: dict[str, str] = {
    "city": "目的地城市",
    "days": "出行天数",
    "start_date": "出行日期（从哪一天开始，如 5月20日）",
    "preference": "旅行偏好（如美食 / 自然 / 人文 / 亲子 / 摄影 / 休闲）",
}

MISSING_FIELD_QUESTIONS: dict[str, str] = {
    "city": "目的地是哪里？",
    "days": "计划玩几天？",
    "start_date": "哪一天出发？也可以说“这个周末”或“下周六”。",
    "preference": "偏好什么类型的体验？比如美食、自然、人文、亲子、少走路或休闲。",
}

CLASSIFY_SYSTEM_TEMPLATE = """你是旅游助手的路由模块。
今天是 {today}。请根据对话历史返回一个 JSON 对象，字段如下：

{{
  "intent": "need_plan | need_more_info | need_answer | need_ticket | general_chat | other",
  "city": "目的地城市名，没有则为空字符串",
  "days": 0,  // 旅行天数，注意区分"玩x天"(days=x)和"y天后出发"(days=0，这是出发时间不是行程天数)
  "start_date": "出发日期，格式为 YYYY-MM-DD，没有则为空字符串。如用户说"5月20号出发"则填"{year}-05-20"，说"下周一"则根据今天日期推算具体日期；如果用户明确表示哪天都可以/日期不限/随便哪天出发，则填"日期灵活"",
  "preference": "旅行偏好，多个用 + 连接，没有则为空字符串",
  "reason": "一句话说明判断依据"
}}

意图定义：
- need_plan: 用户明确要你生成旅游计划/攻略/行程，并且信息基本齐全
- need_more_info: 用户想规划行程，但目的地 / 天数 / 出发日期 / 偏好不完整。若用户明确表示日期灵活，不要继续追问出发日期
- need_ticket: 用户需要查询火车票/高铁票/动车票/列车票/12306 余票、订票、改签、退票等车票信息。只要用户明确要查车票，就优先归为 need_ticket，不要归为 need_answer
- need_answer: 用户在问具体问题，例如天气、景点、美食、交通、当前位置、图片识别、语音内容、知识库内容、个人偏好
- general_chat: 问候、感谢、闲聊
- other: 无法归类，或用户提出与旅行无关的购买、游戏、数码、泛娱乐、泛生活咨询

如果用户是在问"我的偏好是什么""知识库里写了什么""根据我上传的资料回答"等，这属于 need_answer，不属于 need_plan。
如果用户是在问 PS5、NS2、Switch、手机、电脑、显卡等非旅行商品购买，不要归为 need_plan 或 need_answer，应归为 other。

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
    questions = [MISSING_FIELD_QUESTIONS[field] for field in missing if field in MISSING_FIELD_QUESTIONS]
    if not questions:
        readable = "、".join(
            MISSING_FIELD_LABELS[field] for field in missing if field in MISSING_FIELD_LABELS
        )
        questions = [f"请补充：{readable}。"]
    return (
        "可以，我先帮你把出行需求收拢一下。还需要确认：\n"
        + "\n".join(f"- {question}" for question in questions)
        + "\n\n补齐后我就能继续生成路线。"
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


def _recent_ai_asked_start_date(messages: list[BaseMessage] | None) -> bool:
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            continue
        if isinstance(msg, AIMessage):
            content = str(getattr(msg, "content", "") or "")
            return bool(DATE_CLARIFICATION_RE.search(content))
    return False


def _is_flexible_start_date(text: str, messages: list[BaseMessage] | None = None) -> bool:
    normalized = re.sub(r"\s+", "", text or "")
    if not normalized:
        return False
    if FLEXIBLE_DATE_RE.search(normalized):
        return True
    return bool(GENERIC_FLEXIBLE_REPLY_RE.fullmatch(normalized) and _recent_ai_asked_start_date(messages))


def _normalize_start_date_value(value: str, messages: list[BaseMessage] | None = None) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if _is_flexible_start_date(value, messages):
        return FLEXIBLE_START_DATE
    return value


def _next_weekday(start: datetime, weekday: int) -> datetime:
    delta = (weekday - start.weekday()) % 7
    return start + timedelta(days=delta or 7)


def _extract_start_date(text: str, messages: list[BaseMessage] | None = None) -> str:
    normalized = re.sub(r"\s+", "", text or "")
    now = datetime.now()

    if _is_flexible_start_date(normalized, messages):
        return FLEXIBLE_START_DATE

    explicit = re.search(r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})(?:日|号)?", normalized)
    if explicit:
        year = int(explicit.group(1) or now.year)
        month = int(explicit.group(2))
        day = int(explicit.group(3))
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return ""

    if "今天" in normalized:
        return now.strftime("%Y-%m-%d")
    if "明天" in normalized:
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")
    if "后天" in normalized:
        return (now + timedelta(days=2)).strftime("%Y-%m-%d")
    if "周末" in normalized or "这个周末" in normalized:
        saturday = _next_weekday(now, 5)
        if now.weekday() == 5:
            saturday = now
        if now.weekday() == 6:
            saturday = now - timedelta(days=1)
        return saturday.strftime("%Y-%m-%d")

    weekday_map = {
        "一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
    }
    weekday_match = re.search(r"(?:下周|这周|本周|周|星期)([一二三四五六日天])", normalized)
    if weekday_match:
        target = weekday_map[weekday_match.group(1)]
        date = _next_weekday(now, target)
        if weekday_match.group(0).startswith(("这周", "本周")):
            delta = target - now.weekday()
            if delta >= 0:
                date = now + timedelta(days=delta)
        return date.strftime("%Y-%m-%d")

    return ""


def _extract_city(text: str) -> str:
    normalized = re.sub(r"\s+", "", text or "")
    for city in KNOWN_CITIES:
        if city in normalized:
            return city
    match = CITY_RE.search(text or "")
    if match:
        candidate = match.group(1).strip()
        candidate = re.sub(r"(周末|明天|后天|今天|这个|下周|本周|一次)$", "", candidate)
        return candidate[:8]
    return ""


def _extract_days(text: str) -> int:
    normalized = re.sub(r"\s+", "", text or "")
    match = DAY_RE.search(normalized)
    if match:
        return _sanitize_days(match.group(1))
    if "周末" in normalized:
        return 2
    if "一日游" in normalized:
        return 1
    if "两日游" in normalized:
        return 2
    if "三日游" in normalized:
        return 3
    return 0


def _extract_preference(text: str) -> str:
    normalized = re.sub(r"\s+", "", text or "")
    matched = [word for word in PREFERENCE_WORDS if word in normalized]
    return "+".join(dict.fromkeys(matched))


def _fallback_parse_travel_request(
    text: str,
    state: TravelState,
    messages: list[BaseMessage] | None = None,
) -> dict:
    recent_human_text = "\n".join(human_texts(messages or [])[-4:])
    context_text = "\n".join(part for part in [recent_human_text, text] if part).strip()
    parse_text = context_text or text

    city = _extract_city(parse_text) or (state.get("city") or "").strip()
    days = _extract_days(parse_text) or _sanitize_days(state.get("days", 0))
    start_date = (
        _extract_start_date(text, messages)
        or _extract_start_date(parse_text, messages)
        or _normalize_start_date_value((state.get("start_date") or "").strip(), messages)
    )
    preference = _extract_preference(parse_text) or (state.get("preference") or "").strip()
    normalized = re.sub(r"\s+", "", text or "")
    has_plan_signal = bool(PLAN_SIGNAL_RE.search(normalized) or city or days or start_date or preference)
    intent = "need_more_info"
    if has_plan_signal:
        missing = _missing_fields(city, days, start_date, preference)
        intent = "need_more_info" if missing else "need_plan"
    else:
        missing = []
        intent = "other"
    return {
        "intent": intent,
        "city": city,
        "days": days,
        "start_date": start_date,
        "preference": preference,
        "missing_fields": missing,
    }


def _looks_like_ticket_query(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text or "")
    if not normalized:
        return False
    if NON_TRAIN_TICKET_RE.search(normalized) and not RAIL_TICKET_HINT_RE.search(normalized):
        return False
    return bool(TICKET_QUERY_RE.search(normalized))


def _tag_explanation_answer(text: str) -> str:
    normalized = re.sub(r"\s+", "", text or "")
    if not normalized or not TAG_QUESTION_RE.search(normalized):
        return ""

    matched = [name for name in TAG_EXPLANATIONS if name in normalized]
    if "预算" in normalized and not matched:
        matched = ["经济", "舒适", "品质"]
    if "交通" in normalized and not matched:
        matched = ["公共交通", "打车", "自驾/租车", "骑行", "步行友好", "少走路", "无障碍优先"]

    if not matched:
        return ""

    lines = ["这些标签是为了帮我更准确地控制行程节奏和推荐方式："]
    for name in matched:
        lines.append(f"- **{name}**：{TAG_EXPLANATIONS[name]}")
    lines.append("你也可以不选标签，直接用自然语言告诉我你的真实情况。")
    return "\n".join(lines)


def _is_non_travel_purchase(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text or "")
    return bool(NON_TRAVEL_PURCHASE_RE.search(normalized)) and not bool(TRAVEL_SIGNAL_RE.search(normalized))


def _domain_boundary_answer(text: str, *, general_chat: bool = False) -> str:
    normalized = re.sub(r"\s+", "", text or "")
    if any(word in normalized for word in ("我爱你", "喜欢你", "爱你")):
        return "谢谢你呀。我会把这份喜欢认真用在帮你规划旅行上：目的地、天数、同行人和偏好告诉我，我就能开始整理路线。"

    if _is_non_travel_purchase(text):
        return (
            "这个需求看起来不是旅行相关，我就不展开做购买建议啦。\n\n"
            "我主要能帮你做：旅行路线规划、景点/美食/天气/交通查询、资料库攻略整理。"
            "如果你是想把购物安排进某次旅行里，可以告诉我目的地、日期和行程天数，我会帮你把购物点和游玩路线排顺。"
        )

    if general_chat:
        return "你好，我是旅行规划助手。你可以告诉我目的地、出发日期、天数、同行人和偏好，我来帮你整理路线。"

    return (
        "这个问题有点超出旅行助手的范围，我就不往非旅行方向展开了。\n\n"
        "我可以帮你规划行程、查询景点/美食/天气/交通，或根据资料库里的攻略做路线建议。"
    )


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

    if AFFECTION_CHAT_RE.search(re.sub(r"\s+", "", user_input)):
        return {
            "intent": "general_chat",
            "city": (state.get("city") or "").strip(),
            "days": _sanitize_days(state.get("days", 0)),
            "start_date": (state.get("start_date") or "").strip(),
            "preference": (state.get("preference") or "").strip(),
            "missing_fields": [],
            "router_reason": "affection_chat",
            "user_query": user_input,
            "messages": [AIMessage(content=_domain_boundary_answer(user_input, general_chat=True))],
        }

    if _is_non_travel_purchase(user_input):
        return {
            "intent": "other",
            "city": (state.get("city") or "").strip(),
            "days": _sanitize_days(state.get("days", 0)),
            "start_date": (state.get("start_date") or "").strip(),
            "preference": (state.get("preference") or "").strip(),
            "missing_fields": [],
            "router_reason": "non_travel_purchase",
            "user_query": user_input,
            "messages": [AIMessage(content=_domain_boundary_answer(user_input))],
        }

    tag_answer = _tag_explanation_answer(user_input)
    if tag_answer:
        return {
            "intent": "general_chat",
            "city": (state.get("city") or "").strip(),
            "days": _sanitize_days(state.get("days", 0)),
            "start_date": (state.get("start_date") or "").strip(),
            "preference": (state.get("preference") or "").strip(),
            "missing_fields": [],
            "router_reason": "tag_explanation",
            "user_query": user_input,
            "messages": [AIMessage(content=tag_answer)],
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
        fallback = _fallback_parse_travel_request(user_input, state, messages)
        reply = None
        if fallback["intent"] == "need_more_info":
            reply = AIMessage(content=_build_missing_prompt(fallback["missing_fields"]))
        elif fallback["intent"] in {"general_chat", "other"}:
            reply = AIMessage(content=_domain_boundary_answer(user_input))
        output = {
            **fallback,
            "router_reason": "local_fallback_llm_init_failed",
            "user_query": user_input,
        }
        if reply is not None:
            output["messages"] = [reply]
        return output

    try:
        response = llm.invoke(
            [
                SystemMessage(content=_get_classify_system()),
                HumanMessage(content=context_for_llm),
            ]
        )
        raw_text = response.content if hasattr(response, "content") else str(response)
        parsed = _safe_parse_json(raw_text)
        llm_error = ""
    except Exception as exc:
        parsed = None
        llm_error = str(exc)

    if parsed is None:
        fallback = _fallback_parse_travel_request(user_input, state, messages)
        reply = None
        if fallback["intent"] == "need_more_info":
            reply = AIMessage(content=_build_missing_prompt(fallback["missing_fields"]))
        elif fallback["intent"] in {"general_chat", "other"}:
            reply = AIMessage(content=_domain_boundary_answer(user_input))
        output = {
            **fallback,
            "router_reason": "local_fallback_llm_parse_failed",
            "user_query": user_input,
        }
        if reply is not None:
            output["messages"] = [reply]
        return output

    state_city = (state.get("city") or "").strip()
    state_days = _sanitize_days(state.get("days", 0))
    state_start_date = _normalize_start_date_value((state.get("start_date") or "").strip(), messages)
    state_preference = (state.get("preference") or "").strip()

    intent = _normalize_intent(str(parsed.get("intent", "")))
    city_new = str(parsed.get("city", "") or "").strip()
    days_new = _sanitize_days(parsed.get("days", 0))
    start_date_new = _normalize_start_date_value(str(parsed.get("start_date", "") or "").strip(), messages)
    preference_new = str(parsed.get("preference", "") or "").strip()

    fallback = _fallback_parse_travel_request(user_input, state, messages)
    city = city_new or str(fallback.get("city") or "") or state_city
    days = days_new or int(fallback.get("days") or 0) or state_days
    start_date = start_date_new or str(fallback.get("start_date") or "") or state_start_date
    preference = preference_new or str(fallback.get("preference") or "") or state_preference

    if intent == "other" and fallback.get("intent") in {"need_plan", "need_more_info"}:
        intent = str(fallback["intent"])

    missing: list[str] = []
    if intent in {"need_plan", "need_more_info"}:
        if _is_non_travel_purchase(user_input):
            intent = "other"
        else:
            missing = _missing_fields(city, days, start_date, preference)
            intent = "need_more_info" if missing else "need_plan"

    reply: Optional[AIMessage] = None
    if intent == "need_more_info":
        reply = AIMessage(content=_build_missing_prompt(missing))
    elif intent in {"general_chat", "other"}:
        reply = AIMessage(content=_domain_boundary_answer(user_input, general_chat=intent == "general_chat"))

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
