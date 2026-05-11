import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agents.state import TravelState, human_texts
from core.llm_core import get_llm
from core.tools import (
    get_current_location,
    get_route_distance,
    get_travel_tools,
    get_weather,
    get_weather_forecast,
    recognize_scenic_spot,
    search_restaurant,
    search_scenic_spot,
    speech_to_text,
)


CURRENT_LOCATION_HINTS = (
    "我现在在哪里",
    "我在哪",
    "我在哪里",
    "当前位置",
    "当前在哪",
    "我现在在什么地方",
)
CURRENT_PLACE_REFS = ("这里", "这儿", "我这", "我这里", "当前位置", "当前所在地", "当前所在位置")
HISTORY_PLACE_REFS = ("那边", "那里", "那儿", "那个地方", "那个城市")
WEATHER_HINTS = ("天气", "温度", "气温", "下雨", "晴", "阴", "湿度")
SCENIC_HINTS = ("景点", "去哪玩", "哪里玩", "推荐", "打卡", "值得去")
FOOD_HINTS = ("美食", "餐厅", "吃什么", "吃啥", "好吃的", "推荐吃")
ROUTE_HINTS = ("距离", "多远", "路线", "怎么走", "路程", "交通")
PROFILE_HINTS = ("偏好", "喜好", "喜欢", "知识库", "资料", "文档", "rag")

ROUTE_PATTERN = re.compile(
    r"(?:从(?P<origin>[\u4e00-\u9fa5]{1,10}|这里|这儿|我这|我这里|当前位置)?\s*)?"
    r"(?:到|去)\s*(?P<dest>[\u4e00-\u9fa5]{2,10})"
)
CITY_PATTERNS = [
    re.compile(r"(?:去|到|在|前往|想去)\s*([\u4e00-\u9fa5]{2,10})(?:旅游|旅行|玩|逛|天气|景点|美食|路线|距离|[，。！？\s]|$)"),
    re.compile(r"([\u4e00-\u9fa5]{2,10})(?:天气|景点|美食|餐厅|攻略|行程|路线|距离|路程)"),
]


def _safe_tool_invoke(tool_obj, payload: dict) -> str:
    try:
        output = tool_obj.invoke(payload)
        return (output or "").strip()
    except Exception as exc:
        return f"工具 {tool_obj.name} 调用失败：{exc}"


def _pick_scenic_keyword(preference: str) -> str:
    mapping = {
        "自然": "自然风光",
        "人文": "历史人文",
        "美食": "热门景点",
        "亲子": "亲子景点",
        "摄影": "拍照景点",
        "休闲": "休闲景点",
    }
    return mapping.get((preference or "").strip(), "热门景点")


def _pick_food_keyword(preference: str) -> str:
    return "本地必吃" if (preference or "").strip() == "美食" else "特色餐厅"


def _extract_media_path(text: str, kind: str) -> Optional[str]:
    if kind == "image":
        pattern = r"路径[:：]\s*([^\n]+?\.(?:jpg|jpeg|png|webp|bmp|gif))"
    else:
        pattern = r"路径[:：]\s*([^\n]+?\.(?:mp3|wav|m4a|ogg|webm|flac|aac))"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _normalize_city_name(raw: str) -> str:
    city = (raw or "").strip(" ，。！？,.!?：:")
    city = re.sub(r"(天气|景点|美食|餐厅|攻略|行程|路线|路程|距离)$", "", city)
    city = re.sub(r"(市|地区|自治州|盟|特别行政区)$", "", city)
    if city in CURRENT_PLACE_REFS or city in HISTORY_PLACE_REFS:
        return city
    return city if 2 <= len(city) <= 8 else ""


def _extract_city_mentions(text: str) -> list[str]:
    results: list[str] = []
    for pattern in CITY_PATTERNS:
        for match in pattern.finditer(text or ""):
            city = _normalize_city_name(match.group(1))
            if city and city not in results:
                results.append(city)
    return results


def _extract_city_from_location_text(location_text: str) -> str:
    match = re.search(r"当前.*?城市[:：]\s*([^\n]+)", location_text)
    if not match:
        match = re.search(r"城市[:：]\s*([^\n]+)", location_text)
    if not match:
        return ""
    return _normalize_city_name(match.group(1))


def _extract_route_targets(query: str) -> tuple[str, str, bool]:
    match = ROUTE_PATTERN.search(query)
    if match:
        origin = _normalize_city_name(match.group("origin") or "")
        destination = _normalize_city_name(match.group("dest") or "")
        use_current_origin = origin in CURRENT_PLACE_REFS or not origin
        if origin in CURRENT_PLACE_REFS:
            origin = ""
        return origin, destination, use_current_origin

    cities = _extract_city_mentions(query)
    destination = cities[-1] if cities else ""
    use_current_origin = any(ref in query for ref in CURRENT_PLACE_REFS) or "从我" in query
    return "", destination, use_current_origin


def _last_explicit_city(user_texts: list[str]) -> str:
    for text in reversed(user_texts):
        cities = [
            city
            for city in _extract_city_mentions(text)
            if city not in CURRENT_PLACE_REFS and city not in HISTORY_PLACE_REFS
        ]
        if cities:
            return cities[-1]
    return ""


def _needs_current_location(query: str) -> bool:
    return any(keyword in query for keyword in CURRENT_LOCATION_HINTS) or any(
        ref in query for ref in CURRENT_PLACE_REFS
    )


def _needs_weather(query: str) -> bool:
    return any(keyword in query for keyword in WEATHER_HINTS)


def _needs_scenic(query: str) -> bool:
    return any(keyword in query for keyword in SCENIC_HINTS)


def _needs_food(query: str) -> bool:
    return any(keyword in query for keyword in FOOD_HINTS)


def _needs_route(query: str) -> bool:
    return any(keyword in query for keyword in ROUTE_HINTS)


def _needs_profile_answer(query: str) -> bool:
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in PROFILE_HINTS)


def _should_append_kb_for_answer(query: str, sections: list[str]) -> bool:
    if _needs_profile_answer(query):
        return True
    if not sections:
        return True
    return False


def _format_answer_message(query: str, answer_text: str) -> str:
    content = (answer_text or "").strip()
    if not content:
        return "暂时没有查到可直接回复的结果。"

    if content.startswith(("目前缺少", "资料不足", "无法", "未能", "查询失败")):
        return content

    content = re.sub(r"【([^】]+)】\s*\n?", r"### \1\n", content)

    if _needs_weather(query) and "### 天气" in content:
        prefix = "已为你查到当前天气："
    elif _needs_current_location(query) and "### 当前位置" in content:
        prefix = "这是当前定位信息："
    elif _needs_route(query) and "### 距离与路线" in content:
        prefix = "路线信息如下："
    elif _needs_scenic(query) or _needs_food(query):
        prefix = "整理到的信息如下："
    else:
        prefix = "查到的信息如下："

    return f"{prefix}\n\n{content}".strip()


def _compact_doc_text(text: str, limit: int = 420) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    return cleaned if len(cleaned) <= limit else cleaned[:limit].rstrip() + "..."


def _doc_source(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {})
    if isinstance(metadata, dict):
        source = metadata.get("source") or metadata.get("file_path") or metadata.get("path")
        if source:
            return str(source)
    return "未知来源"


def _build_rag_queries(query: str, city: str, days: int, preference: str) -> list[str]:
    queries: list[str] = []
    base_query = (query or "").strip()
    if base_query:
        queries.append(base_query)

    parts: list[str] = []
    if city:
        parts.append(city)
    if days > 0:
        parts.append(f"{days}天")
    if preference and preference != "综合":
        parts.append(preference)
    if parts:
        queries.append(" ".join(parts + ["旅游攻略", "景点", "美食", "交通"]))

    if _needs_profile_answer(base_query):
        queries.append("旅游偏好 喜好 喜欢 个人偏好")

    deduped: list[str] = []
    for item in queries:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _search_knowledge_base(
    vector_db: Any,
    *,
    query: str,
    city: str,
    days: int,
    preference: str,
    k: int = 4,
) -> str:
    if vector_db is None:
        return "【知识库检索】当前未加载知识库，已跳过 RAG 检索。"

    search_queries = _build_rag_queries(query, city, days, preference)
    if not search_queries:
        return "【知识库检索】缺少检索关键词，已跳过 RAG 检索。"

    docs = []
    seen_keys = set()
    tried_queries: list[str] = []

    for search_query in search_queries:
        tried_queries.append(search_query)
        try:
            current_docs = vector_db.similarity_search(search_query, k=k)
        except Exception as exc:
            return f"【知识库检索】检索失败：{exc}"

        for doc in current_docs:
            key = (
                _doc_source(doc),
                hashlib.md5(getattr(doc, "page_content", "").encode("utf-8")).hexdigest()
                if getattr(doc, "page_content", "")
                else id(doc),
            )
            if key not in seen_keys:
                seen_keys.add(key)
                docs.append(doc)

    if not docs:
        return "【知识库检索】未命中相关文档。"

    snippets: list[str] = []
    for index, doc in enumerate(docs[:k], start=1):
        content = _compact_doc_text(getattr(doc, "page_content", ""))
        source = _doc_source(doc)
        snippets.append(f"片段{index}（来源：{source}）\n{content}")

    query_line = "；".join(tried_queries)
    return f"【知识库检索】查询词：{query_line}\n\n" + "\n\n".join(snippets)


def _build_transport_hint(spot_text: str, days: int) -> str:
    if not spot_text.strip():
        return "景点分布待补充，建议优先地铁加步行，跨区再打车。"
    if days <= 1:
        return "天数较短，建议把行程集中在同一片区，减少折返。"
    if days >= 4:
        return "天数较充足，建议按片区拆分游玩，同片区步行，跨区地铁或打车。"
    return "建议按片区拆分每天行程，同片区步行，跨片区优先地铁。"


def _research_for_answer(
    query: str,
    messages: list[BaseMessage],
    router_city: str,
    days: int,
    preference: str,
    vector_db: Any,
) -> str:
    user_texts = human_texts(messages)
    previous_texts = user_texts[:-1]
    history_city = _last_explicit_city(previous_texts)
    explicit_cities = [
        city
        for city in _extract_city_mentions(query)
        if city not in CURRENT_PLACE_REFS and city not in HISTORY_PLACE_REFS
    ]

    location_text: Optional[str] = None
    current_city = ""

    def ensure_location() -> tuple[str, str]:
        nonlocal location_text, current_city
        if location_text is None:
            location_text = _safe_tool_invoke(get_current_location, {})
            current_city = _extract_city_from_location_text(location_text)
        return location_text, current_city

    def resolve_context_city(*, prefer_current: bool = False) -> str:
        if prefer_current and any(ref in query for ref in CURRENT_PLACE_REFS):
            _, city = ensure_location()
            return city
        if any(ref in query for ref in HISTORY_PLACE_REFS) and history_city:
            return history_city
        if any(ref in query for ref in CURRENT_PLACE_REFS):
            _, city = ensure_location()
            return city
        if explicit_cities:
            return explicit_cities[-1]
        if router_city:
            return router_city
        if history_city:
            return history_city
        return ""

    image_path = _extract_media_path(query, "image")
    if image_path:
        result = _safe_tool_invoke(
            recognize_scenic_spot,
            {"image_input": image_path, "city_hint": resolve_context_city() or None},
        )
        kb_text = _search_knowledge_base(
            vector_db,
            query=query,
            city=router_city,
            days=days,
            preference=preference,
            k=3,
        )
        return f"【图片识别】\n{result}\n\n{kb_text}".strip()

    audio_path = _extract_media_path(query, "audio")
    if audio_path:
        transcript = _safe_tool_invoke(speech_to_text, {"audio_input": audio_path})
        kb_text = _search_knowledge_base(
            vector_db,
            query=query,
            city=router_city,
            days=days,
            preference=preference,
            k=3,
        )
        return f"【语音识别】\n{transcript}\n\n{kb_text}".strip()

    sections: list[str] = []

    if _needs_current_location(query):
        location, _ = ensure_location()
        sections.append(f"【当前位置】\n{location}")

    if _needs_weather(query):
        target_city = resolve_context_city(prefer_current=True)
        if not target_city:
            _, target_city = ensure_location()
        if target_city:
            weather_text = _safe_tool_invoke(get_weather, {"city": target_city})
            sections.append(f"【天气】\n{weather_text}")

    if _needs_route(query):
        origin, destination, use_current_origin = _extract_route_targets(query)
        if not destination and any(ref in query for ref in HISTORY_PLACE_REFS):
            destination = history_city
        if use_current_origin and not origin:
            _, current = ensure_location()
            origin = current or origin

        if destination:
            payload = {"destination": destination}
            if origin:
                payload["origin"] = origin
            route_text = _safe_tool_invoke(get_route_distance, payload)
            sections.append(f"【距离与路线】\n{route_text}")
        else:
            sections.append("【距离与路线】\n缺少明确的目的地城市，暂时无法计算距离。")

    if _needs_scenic(query):
        target_city = resolve_context_city(prefer_current=True)
        if not target_city:
            _, target_city = ensure_location()
        if target_city:
            scenic_text = _safe_tool_invoke(
                search_scenic_spot,
                {"city": target_city, "keyword": _pick_scenic_keyword(preference)},
            )
            sections.append(f"【景点】\n{scenic_text}")

    if _needs_food(query):
        target_city = resolve_context_city(prefer_current=True)
        if not target_city:
            _, target_city = ensure_location()
        if target_city:
            food_text = _safe_tool_invoke(
                search_restaurant,
                {"city": target_city, "keyword": _pick_food_keyword(preference)},
            )
            sections.append(f"【餐饮】\n{food_text}")

    if _should_append_kb_for_answer(query, sections):
        kb_city = resolve_context_city() or router_city
        kb_text = _search_knowledge_base(
            vector_db,
            query=query,
            city=kb_city,
            days=days,
            preference=preference,
            k=4,
        )
        if kb_text:
            sections.append(kb_text)

    if not sections:
        return "目前缺少可执行检索的关键信息，请补充后我再继续。"

    return "\n\n".join(sections).strip()


def _parse_structured_materials(raw: str) -> dict:
    """尝试从 raw_materials 中解析 JSON 结构。"""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


# ─── 研究节点：LLM 工具调用（need_plan）───

_RESEARCHER_SYSTEM = """你是一个旅游信息搜集助手。你的任务是：
1. 根据用户需求，自主决定调用哪些工具来搜集信息
2. 尽量并行搜集天气、景点、餐饮、路线等信息
3. 搜集完毕后，将所有信息整合为结构化 JSON 输出

【输出格式】
最终回复必须是如下 JSON 结构（不要包裹在 markdown 代码块中）：
{
  "weather": "天气信息摘要",
  "scenic_spots": "景点推荐摘要",
  "restaurants": "餐饮推荐摘要",
  "route_info": "路线与交通摘要",
  "knowledge_base": "知识库参考摘要（如有）"
}

每个字段的值应是结构化的文字摘要，而非原始 API 返回。如果没有相关信息，填空字符串。"""

_RESEARCHER_ANSWER_SYSTEM = """You are a travel research assistant for direct Q&A.
Use the bound tools to research the user's current question instead of relying on keyword rules.
When a tool can answer or verify the question, call it first. You may call multiple tools before answering.
If the question asks about uploaded documents, user profile, preferences, notes, or the local knowledge base,
use the search_knowledge_base tool.
After tools finish, answer the user directly in the user's language. Do not output JSON in direct-answer mode.
"""


def _make_knowledge_base_tool(vector_db: Any, city: str, days: int, preference: str):
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search uploaded/local knowledge base for relevant travel notes, preferences, and documents."""
        return _search_knowledge_base(
            vector_db,
            query=query,
            city=city,
            days=days,
            preference=preference,
            k=4,
        )

    return search_knowledge_base


def _format_recent_history(messages: Optional[list[BaseMessage]]) -> str:
    if not messages:
        return ""

    lines: list[str] = []
    for msg in messages[-6:]:
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            role = "Message"
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


def _build_research_user_message(
    *,
    intent: str,
    city: str,
    days: int,
    start_date: str,
    preference: str,
    query: str,
    messages: Optional[list[BaseMessage]],
) -> str:
    history = _format_recent_history(messages)
    common_context = (
        f"Current user request: {query}\n"
        f"Router intent: {intent or 'unknown'}\n"
        f"Router city: {city or 'unknown'}\n"
        f"Trip days: {days if days > 0 else 'unknown'}\n"
        f"Start date: {start_date or 'unknown'}\n"
        f"Preference: {preference or 'unknown'}\n"
        f"Recent conversation:\n{history or '(none)'}\n\n"
    )

    if intent != "need_plan":
        return (
            common_context
            + "Research this as a direct travel Q&A task. Let the LLM decide which bound tools to call. "
            + "Use search_knowledge_base for uploaded documents, profile, preferences, notes, or RAG questions. "
            + "Use weather, route, location, scenic spot, restaurant, image, or audio tools when they match the request. "
            + "Then answer directly."
        )

    return (
        common_context
        + "Research this as trip-planning material. Let the LLM decide which bound tools to call, including "
        + "search_knowledge_base when local documents or preferences are useful. Final output must be strict JSON "
        + "with keys: weather, scenic_spots, restaurants, route_info, knowledge_base."
    )


def _research_with_llm(
    city: str,
    days: int,
    start_date: str,
    preference: str,
    query: str,
    vector_db: Any,
    intent: str = "need_plan",
    messages: Optional[list[BaseMessage]] = None,
) -> dict:
    """使用 LLM 工具调用搜集旅行信息，返回结构化结果。"""
    tools = [*get_travel_tools(), _make_knowledge_base_tool(vector_db, city, days, preference)]
    tool_map = {t.name: t for t in tools}
    llm = get_llm("mimo-v2.5-pro").bind_tools(tools)
    normalized_intent = (intent or "need_plan").strip().lower()
    direct_answer_mode = normalized_intent != "need_plan"
    system_prompt = _RESEARCHER_ANSWER_SYSTEM if direct_answer_mode else _RESEARCHER_SYSTEM

    user_msg = _build_research_user_message(
        intent=normalized_intent,
        city=city,
        days=days,
        start_date=start_date,
        preference=preference,
        query=query,
        messages=messages,
    )
    msgs: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ]
    called_tool_names: set[str] = set()

    for _ in range(10):
        resp = llm.invoke(msgs)
        msgs.append(resp)

        tool_calls = resp.tool_calls or []
        if not tool_calls:
            break

        # 并行执行所有工具调用
        results_map: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=5) as pool:
            future_to_id = {}
            for tc in tool_calls:
                tc_name = tc.get("name", "")
                tc_id = tc.get("id", "")
                called_tool_names.add(tc_name)
                if tc_name not in tool_map:
                    results_map[tc_id] = f"未知工具：{tc_name}"
                    continue
                future_to_id[pool.submit(
                    _safe_tool_invoke, tool_map[tc_name], tc.get("args", {})
                )] = tc_id

            for future in as_completed(future_to_id):
                tc_id = future_to_id[future]
                try:
                    results_map[tc_id] = future.result()
                except Exception as exc:
                    results_map[tc_id] = f"工具执行异常：{exc}"

        for tc in tool_calls:
            tc_id = tc.get("id", "")
            result_text = results_map.get(tc_id, "工具执行失败")
            msgs.append(ToolMessage(content=result_text, tool_call_id=tc_id))

    kb_text = ""
    if not direct_answer_mode and "search_knowledge_base" not in called_tool_names:
        kb_text = _search_knowledge_base(
            vector_db,
            query=query,
            city=city,
            days=days,
            preference=preference,
            k=4,
        )

    # 解析 LLM 输出的 JSON
    final_content = getattr(msgs[-1], "content", "") if msgs else ""
    if direct_answer_mode:
        answer_text = final_content.strip() or "No usable research result was returned; please try again."
        return {
            "raw_materials": answer_text,
            "messages": [AIMessage(content=answer_text)],
        }

    parsed = _parse_structured_materials(final_content)
    if parsed:
        if kb_text and not (parsed.get("knowledge_base") or "").strip():
            parsed["knowledge_base"] = kb_text
        raw_materials = json.dumps(parsed, ensure_ascii=False, indent=2)
    else:
        raw_materials = final_content
        if kb_text:
            raw_materials = f"{raw_materials}\n\n{kb_text}".strip()

    return {
        "raw_materials": raw_materials,
        "messages": [AIMessage(content=f"已完成资料搜集（{len(raw_materials)} 字符）。")],
    }


# ─── 研究节点主入口 ───

def researcher_agent(state: TravelState) -> dict:
    intent = (state.get("intent") or "").strip().lower()
    query = (state.get("user_query") or "").strip()
    city = (state.get("city") or "").strip()
    days = int(state.get("days") or 0)
    start_date = (state.get("start_date") or "").strip()
    preference = (state.get("preference") or "综合").strip()
    vector_db = state.get("vector_db")
    messages: list[BaseMessage] = list(state.get("messages", []))

    # Researcher always uses LLM tool-calling on its main path; old deterministic
    # answer research is kept only as a failure fallback.
    if intent == "need_plan" and not city:
        return {"raw_materials": "【资料采集失败】缺少目的地城市，无法执行天气、景点和美食检索。"}

    try:
        return _research_with_llm(
            city=city,
            days=days,
            start_date=start_date,
            preference=preference,
            query=query,
            vector_db=vector_db,
            intent=intent,
            messages=messages,
        )
    except Exception as exc:
        if intent != "need_plan":
            answer_text = _research_for_answer(
                query=query,
                messages=messages,
                router_city=city,
                days=days,
                preference=preference,
                vector_db=vector_db,
            )
            return {
                "raw_materials": f"LLM tool-calling failed and fell back: {exc}\n\n{answer_text}",
                "messages": [AIMessage(content=_format_answer_message(query, answer_text))],
            }

        # LLM 工具调用失败时，回退到原有并行搜集逻辑
        return _research_fallback(city, days, preference, query, vector_db, str(exc))


def _research_fallback(
    city: str,
    days: int,
    preference: str,
    query: str,
    vector_db: Any,
    error_hint: str = "",
) -> dict:
    """LLM 工具调用失败时的回退逻辑：并行搜集 + 原始文本拼接。"""
    scenic_keyword = _pick_scenic_keyword(preference)
    food_keyword = _pick_food_keyword(preference)

    tasks = {
        "location": (get_current_location, {}),
        "weather": (get_weather, {"city": city}),
        "weather_forecast": (get_weather_forecast, {"city": city}),
        "scenic": (search_scenic_spot, {"city": city, "keyword": scenic_keyword}),
        "food": (search_restaurant, {"city": city, "keyword": food_keyword}),
        "route": (get_route_distance, {"destination": city}),
    }

    results: dict[str, str] = {key: "" for key in tasks}
    with ThreadPoolExecutor(max_workers=5) as pool:
        future_map = {
            pool.submit(_safe_tool_invoke, tool_obj, payload): key
            for key, (tool_obj, payload) in tasks.items()
        }
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                results[key] = f"并发任务 {key} 执行失败：{exc}"

    kb_text = _search_knowledge_base(
        vector_db, query=query, city=city, days=days, preference=preference, k=4,
    )
    transport_hint = _build_transport_hint(results["scenic"], days)

    fallback_note = f"（LLM 工具调用失败已回退：{error_hint}）" if error_hint else ""
    gathered_info = (
        f"【资料采集摘要】{fallback_note}\n"
        f"目的地：{city}\n"
        f"天数：{days if days > 0 else '未指定'}\n"
        f"偏好：{preference}\n\n"
        f"【当前位置参考】\n{results['location']}\n\n"
        f"【实时天气】\n{results['weather']}\n\n"
        f"【天气预报】\n{results['weather_forecast']}\n\n"
        f"【景点检索】\n{results['scenic']}\n\n"
        f"【餐饮检索】\n{results['food']}\n\n"
        f"【到达距离参考】\n{results['route']}\n\n"
        f"【市内交通建议】\n{transport_hint}\n\n"
        f"{kb_text}"
    )
    return {"raw_materials": gathered_info.strip()}
