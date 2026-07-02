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


TOOL_FAILURE_POLICY: dict[str, str] = {
    "get_weather": "strict",
    "get_weather_forecast": "strict",
    "get_current_location": "strict",
    "get_route_distance": "strict",
    "speech_to_text": "strict",
    "recognize_scenic_spot": "strict",
    "search_scenic_spot": "soft",
    "search_restaurant": "soft",
}


# ─── 工具调用安全包装 ───

def _safe_tool_invoke(tool_obj, payload: dict, failures: list[dict] | None = None) -> str:
    tool_name = tool_obj.name
    policy = TOOL_FAILURE_POLICY.get(tool_name, "strict")
    try:
        output = tool_obj.invoke(payload)
        result = (output or "").strip()
        if result and any(kw in result for kw in ("失败", "无法", "未能", "请稍后重试", "请确认")):
            if failures is not None:
                failures.append({"tool": tool_name, "type": policy, "error": result})
        return result
    except Exception as exc:
        if failures is not None:
            failures.append({"tool": tool_name, "type": policy, "error": str(exc)})
        return f"工具 {tool_name} 调用失败：{exc}"


# ─── 知识库检索 ───

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
    if query:
        queries.append(query.strip())
    parts = [p for p in [city, f"{days}天" if days > 0 else "", preference if preference != "综合" else ""] if p]
    if parts:
        queries.append(" ".join(parts + ["旅游攻略", "景点", "美食", "交通"]))
    return list(dict.fromkeys(filter(None, queries)))


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

    docs, seen_keys, tried_queries = [], set(), []
    for search_query in search_queries:
        tried_queries.append(search_query)
        try:
            for doc in vector_db.similarity_search(search_query, k=k):
                key = (
                    _doc_source(doc),
                    hashlib.md5(getattr(doc, "page_content", "").encode()).hexdigest(),
                )
                if key not in seen_keys:
                    seen_keys.add(key)
                    docs.append(doc)
        except Exception as exc:
            return f"【知识库检索】检索失败：{exc}"

    if not docs:
        return "【知识库检索】未命中相关文档。"

    snippets = [
        f"片段{i}（来源：{_doc_source(doc)}）\n{_compact_doc_text(getattr(doc, 'page_content', ''))}"
        for i, doc in enumerate(docs[:k], 1)
    ]
    return f"【知识库检索】查询词：{'；'.join(tried_queries)}\n\n" + "\n\n".join(snippets)


# ─── LLM 工具调用核心 ───

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

_RESEARCHER_ANSWER_SYSTEM = """你是一个旅游直接问答助手。
根据用户的问题，自主决定调用哪些工具来搜集、核实信息，可以调用多个工具。
如果问题涉及上传的文档、用户偏好、旅游笔记或本地知识库，使用 search_knowledge_base 工具。
如果用户询问天气但没有明确城市，必须先调用 get_current_location 获取当前城市，再调用 get_weather 或 get_weather_forecast；只有定位失败时才请用户补充城市。
工具调用结束后，用用户的语言直接回答问题，不要输出 JSON。"""

WEATHER_QUERY_RE = re.compile(
    r"天气|气温|温度|冷不冷|热不热|下雨|降雨|雨|雪|风力|风大|空气质量|预报"
)
FORECAST_QUERY_RE = re.compile(r"预报|未来|明天|后天|这几天|最近几天|本周|周末|星期|旅行期间")


def _make_knowledge_base_tool(vector_db: Any, city: str, days: int, preference: str):
    @tool
    def search_knowledge_base(query: str) -> str:
        """Search uploaded/local knowledge base for relevant travel notes, preferences, and documents."""
        return _search_knowledge_base(vector_db, query=query, city=city, days=days, preference=preference, k=4)
    return search_knowledge_base


def _format_recent_history(messages: Optional[list[BaseMessage]]) -> str:
    lines = []
    for msg in (messages or [])[-6:]:
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        role = "User" if isinstance(msg, HumanMessage) else "Assistant" if isinstance(msg, AIMessage) else "Message"
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
    context = (
        f"Current user request: {query}\n"
        f"Router intent: {intent or 'unknown'}\n"
        f"Router city: {city or 'unknown'}\n"
        f"Trip days: {days if days > 0 else 'unknown'}\n"
        f"Start date: {start_date or 'unknown'}\n"
        f"Preference: {preference or 'unknown'}\n"
        f"Recent conversation:\n{history or '(none)'}\n\n"
    )
    if intent != "need_plan":
        return context + (
            "Research this as a direct travel Q&A task. "
            "Decide which tools to call. Use search_knowledge_base for documents, profile, or preferences. "
            "Then answer directly."
        )
    return context + (
        "Research this as trip-planning material. "
        "Decide which tools to call, including search_knowledge_base when local documents or preferences are useful. "
        "Final output must be strict JSON with keys: weather, scenic_spots, restaurants, route_info, knowledge_base."
    )


def _parse_structured_materials(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}


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
    tools = [*get_travel_tools(), _make_knowledge_base_tool(vector_db, city, days, preference)]
    tool_map = {t.name: t for t in tools}
    # 工具调用循环必须用非思维链模型。
    # thinking mode 模型（如 mimo-v2.5-pro）每轮返回 reasoning_content，
    # LangChain 标准序列化不会把它带回下一次请求，导致 API 400 错误。
    llm = get_llm("glm-4.5-air").bind_tools(tools)

    normalized_intent = (intent or "need_plan").strip().lower()
    direct_answer_mode = normalized_intent != "need_plan"
    system_prompt = _RESEARCHER_ANSWER_SYSTEM if direct_answer_mode else _RESEARCHER_SYSTEM

    msgs: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=_build_research_user_message(
            intent=normalized_intent, city=city, days=days, start_date=start_date,
            preference=preference, query=query, messages=messages,
        )),
    ]
    called_tool_names: set[str] = set()
    tool_failures: list[dict] = []

    for _ in range(10):
        resp = llm.invoke(msgs)
        msgs.append(resp)
        tool_calls = resp.tool_calls or []
        if not tool_calls:
            break

        results_map: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=5) as pool:
            future_to_id = {}
            for tc in tool_calls:
                tc_name, tc_id = tc.get("name", ""), tc.get("id", "")
                called_tool_names.add(tc_name)
                if tc_name not in tool_map:
                    results_map[tc_id] = f"未知工具：{tc_name}"
                    continue
                future_to_id[pool.submit(
                    _safe_tool_invoke, tool_map[tc_name], tc.get("args", {}), tool_failures
                )] = tc_id

            for future in as_completed(future_to_id):
                tc_id = future_to_id[future]
                try:
                    results_map[tc_id] = future.result()
                except Exception as exc:
                    tc_name = next((t.get("name", "") for t in tool_calls if t.get("id") == tc_id), "")
                    tool_failures.append({"tool": tc_name, "type": TOOL_FAILURE_POLICY.get(tc_name, "strict"), "error": str(exc)})
                    results_map[tc_id] = f"工具执行异常：{exc}"

        for tc in tool_calls:
            msgs.append(ToolMessage(content=results_map.get(tc.get("id", ""), "工具执行失败"), tool_call_id=tc.get("id", "")))

    final_content = getattr(msgs[-1], "content", "") if msgs else ""

    if direct_answer_mode:
        answer = final_content.strip() or "未能获取有效结果，请重试。"
        return {"raw_materials": answer, "tool_failures": tool_failures, "messages": [AIMessage(content=answer)]}

    # need_plan：补充 KB（如果 LLM 未主动调用）
    kb_text = ""
    if "search_knowledge_base" not in called_tool_names:
        kb_text = _search_knowledge_base(vector_db, query=query, city=city, days=days, preference=preference, k=4)

    parsed = _parse_structured_materials(final_content)
    if parsed:
        if kb_text and not (parsed.get("knowledge_base") or "").strip():
            parsed["knowledge_base"] = kb_text
        raw_materials = json.dumps(parsed, ensure_ascii=False, indent=2)
    else:
        raw_materials = f"{final_content}\n\n{kb_text}".strip() if kb_text else final_content

    return {
        "raw_materials": raw_materials,
        "tool_failures": tool_failures,
        "messages": [AIMessage(content=f"已完成资料搜集（{len(raw_materials)} 字符）。")],
    }


# ─── 兜底回退（LLM 完全不可用时）───

def _pick_scenic_keyword(preference: str) -> str:
    return {"自然": "自然风光", "人文": "历史人文", "亲子": "亲子景点", "摄影": "拍照景点", "休闲": "休闲景点"}.get(
        (preference or "").strip(), "热门景点"
    )


def _pick_food_keyword(preference: str) -> str:
    return "本地必吃" if (preference or "").strip() == "美食" else "特色餐厅"


def _city_from_location(location_text: str) -> str:
    """从 get_current_location 返回文本中提取城市名。"""
    match = re.search(r"城市[：:]\s*(\S+?)(?:[市区\s]|$)", location_text)
    return match.group(1).strip() if match else ""


def _is_weather_query(query: str) -> bool:
    return bool(WEATHER_QUERY_RE.search(query or ""))


def _needs_weather_forecast(query: str) -> bool:
    return bool(FORECAST_QUERY_RE.search(query or ""))


def _research_weather_answer(city: str, query: str) -> dict:
    """Answer weather questions deterministically so missing city can use location first."""
    tool_failures: list[dict] = []
    location_result = ""
    effective_city = city

    if not effective_city:
        location_result = _safe_tool_invoke(get_current_location, {}, tool_failures)
        effective_city = _city_from_location(location_result)

    if not effective_city:
        answer = (
            f"{location_result}\n\n"
            "我没能从定位工具拿到当前城市，请告诉我你所在的城市，我再帮你查今天的天气。"
        ).strip()
        return {"raw_materials": answer, "tool_failures": tool_failures, "messages": [AIMessage(content=answer)]}

    weather = _safe_tool_invoke(get_weather, {"city": effective_city}, tool_failures)
    parts = []
    if location_result:
        parts.append(f"我先用定位工具识别到你当前所在城市是 **{effective_city}**。")
    parts.append(weather)

    if _needs_weather_forecast(query):
        forecast = _safe_tool_invoke(get_weather_forecast, {"city": effective_city}, tool_failures)
        if forecast:
            parts.append(forecast)

    answer = "\n\n".join(part for part in parts if part).strip()
    return {
        "raw_materials": answer,
        "tool_failures": tool_failures,
        "messages": [AIMessage(content=answer)],
        "city": effective_city,
    }


def _research_fallback(city: str, days: int, preference: str, query: str, vector_db: Any, error_hint: str = "") -> dict:
    """LLM 工具调用完全失败时的兜底：先获取位置确定城市，再并发调其他工具。"""
    tool_failures: list[dict] = []

    # 先同步获取位置，城市为空时用定位结果填充
    location_result = _safe_tool_invoke(get_current_location, {}, tool_failures)
    effective_city = city or _city_from_location(location_result)

    tasks = {
        "weather": (get_weather, {"city": effective_city}),
        "weather_forecast": (get_weather_forecast, {"city": effective_city}),
        "scenic": (search_scenic_spot, {"city": effective_city, "keyword": _pick_scenic_keyword(preference)}),
        "food": (search_restaurant, {"city": effective_city, "keyword": _pick_food_keyword(preference)}),
        "route": (get_route_distance, {"destination": effective_city}),
    }
    results: dict[str, str] = {k: "" for k in tasks}

    with ThreadPoolExecutor(max_workers=5) as pool:
        future_map = {
            pool.submit(_safe_tool_invoke, tool_obj, payload, tool_failures): key
            for key, (tool_obj, payload) in tasks.items()
        }
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                tool_failures.append({"tool": tasks[key][0].name, "type": "strict", "error": str(exc)})
                results[key] = f"任务 {key} 失败：{exc}"

    kb_text = _search_knowledge_base(vector_db, query=query, city=effective_city, days=days, preference=preference, k=4)
    fallback_note = f"（LLM 工具调用失败已回退：{error_hint}）\n" if error_hint else ""

    raw_materials = (
        f"【资料采集摘要】{fallback_note}"
        f"目的地：{effective_city or '未知'}｜天数：{days or '未指定'}｜偏好：{preference}\n\n"
        f"【当前位置】\n{location_result}\n\n"
        f"【实时天气】\n{results['weather']}\n\n"
        f"【天气预报】\n{results['weather_forecast']}\n\n"
        f"【景点】\n{results['scenic']}\n\n"
        f"【餐饮】\n{results['food']}\n\n"
        f"【路线距离】\n{results['route']}\n\n"
        f"{kb_text}"
    ).strip()

    return {"raw_materials": raw_materials, "tool_failures": tool_failures}


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

    if intent == "need_plan" and not city:
        return {"raw_materials": "【资料采集失败】缺少目的地城市，无法执行天气、景点和美食检索。"}

    if intent != "need_plan" and _is_weather_query(query):
        return _research_weather_answer(city, query)

    try:
        return _research_with_llm(
            city=city, days=days, start_date=start_date, preference=preference,
            query=query, vector_db=vector_db, intent=intent, messages=messages,
        )
    except Exception as exc:
        result = _research_fallback(city, days, preference, query, vector_db, str(exc))
        # need_answer 路由到 END，没有 planner 兜底，必须在这里写入 messages，
        # 否则 state 里没有新 AIMessage，UI 会显示"没有可展示文本"。
        if intent != "need_plan":
            answer = result.get("raw_materials") or "抱歉，查询遇到问题，请稍后重试。"
            result["messages"] = [AIMessage(content=answer)]
        return result
