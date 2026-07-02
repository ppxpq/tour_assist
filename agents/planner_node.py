import json
import re
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage

from agents.state import TravelState
from core.llm_core import get_llm

try:
    from langgraph.config import get_stream_writer
except Exception:  # pragma: no cover - keeps older LangGraph installs importable
    get_stream_writer = None


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return "" if content is None else str(content)


def _get_writer():
    if get_stream_writer is None:
        return None
    try:
        return get_stream_writer()
    except Exception:
        return None


def _stream_llm_text(llm: Any, prompt: str, *, node: str = "planner") -> str:
    writer = _get_writer()
    chunks: list[str] = []

    try:
        for chunk in llm.stream(prompt):
            text = _chunk_text(chunk)
            if not text:
                continue
            chunks.append(text)
            if writer is not None:
                writer({"type": "message_delta", "node": node, "delta": text})
    except Exception:
        # Some OpenAI-compatible providers do not support streaming. If the
        # stream failed before any text was emitted, fall back to the original
        # blocking invoke path so the user still gets an answer.
        if chunks:
            raise
        response = llm.invoke(prompt)
        content = getattr(response, "content", None) or str(response)
        if writer is not None and content:
            writer({"type": "message_delta", "node": node, "delta": content})
        return content

    return "".join(chunks)


def _build_failure_notice(tool_failures: list[dict]) -> str:
    """根据 tool_failures 构建数据可用性提示，区分 strict/soft 失败。"""
    if not tool_failures:
        return ""

    strict_failures = [f for f in tool_failures if f.get("type") == "strict"]
    soft_failures = [f for f in tool_failures if f.get("type") == "soft"]

    lines: list[str] = []

    if strict_failures:
        lines.append("以下工具调用失败，相关数据不可用，请勿编造或猜测这些数据：")
        for f in strict_failures:
            tool = f.get("tool", "未知工具")
            error = f.get("error", "未知错误")
            lines.append(f"  - {tool}：{error}")
        lines.append('处理方式：跳过依赖这些数据的排期逻辑，或用通用建议替代（如天气部分注明"天气信息暂不可用，建议出发前确认"）。')

    if soft_failures:
        lines.append("以下工具未返回实时数据，你可以用自身知识补充推荐：")
        for f in soft_failures:
            tool = f.get("tool", "未知工具")
            lines.append(f"  - {tool}")
        lines.append('处理方式：基于你的知识给出推荐，但在推荐前加上"根据常见推荐"等提示词，让用户知道非实时数据。')

    return "\n".join(lines)


def _parse_raw_materials(raw: str) -> str:
    """Parse JSON from raw_materials and convert to readable sections."""
    text = raw.strip()
    if not text:
        return ""

    # Try to parse JSON
    candidate = text
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return text  # Not JSON, return as-is

    if not isinstance(data, dict):
        return text

    sections = []
    field_map = [
        ("weather", "天气与预报"),
        ("scenic_spots", "景点推荐"),
        ("restaurants", "餐饮推荐"),
        ("route_info", "路线与交通"),
        ("knowledge_base", "知识库参考"),
    ]
    for key, label in field_map:
        value = (data.get(key) or "").strip()
        if value:
            sections.append(f"【{label}】\n{value}")

    return "\n\n".join(sections) if sections else text


def _extract_budget_level(text: str) -> str:
    """Extract budget level from the natural-language prompt assembled by the frontend."""
    match = re.search(r"预算[：:\s]*(经济|舒适|品质)", text or "")
    return match.group(1) if match else ""


def planner_agent(state: TravelState) -> dict:
    """根据 researcher 汇总素材，生成最终行程文本。"""
    intent = (state.get("intent") or "").strip().lower()
    user_query = (state.get("user_query") or "").strip()
    city = (state.get("city") or "").strip()
    days = int(state.get("days") or 0)
    start_date = (state.get("start_date") or "").strip()
    preference = (state.get("preference") or "综合").strip()
    travel_mode = (state.get("travel_mode") or "").strip()
    raw_materials = (state.get("raw_materials") or "").strip()
    tool_failures: list[dict] = state.get("tool_failures") or []
    budget_level = _extract_budget_level(user_query)

    if not city:
        return {
            "messages": [
                AIMessage(content="还缺少目的地城市，暂时无法生成完整行程。")
            ]
        }

    if not raw_materials:
        return {
            "messages": [
                AIMessage(content="暂未采集到有效资料，请稍后重试或补充更具体需求。")
            ]
        }

    # Parse structured JSON into readable sections for the planner prompt
    formatted_materials = _parse_raw_materials(raw_materials)

    planner_model = (state.get("planner_model") or "glm-4.5-air").strip()
    llm = get_llm(planner_model)

    failure_notice = _build_failure_notice(tool_failures)
    availability_section = f"\n【数据可用性】\n{failure_notice}\n" if failure_notice else ""

    today_str = datetime.now().strftime("%Y年%m月%d日")

    # 构建出行方式说明
    travel_mode_section = ""
    if travel_mode:
        travel_mode_section = f"- 出行方式：{travel_mode}"

    budget_section = f"- 预算档位：{budget_level}" if budget_level else "- 预算档位：未指定"
    budget_requirement = (
        f'最后给出"注意事项"与"预算安排"，预算安排只围绕用户已选择的「{budget_level}」档位展开，'
        "不要再输出低/中/高三档对比。"
        if budget_level
        else '最后给出"注意事项"与"预算建议（低/中/高三档）"。'
    )

    prompt = f"""
你是一位资深旅游规划师。今天是 {today_str}。请基于提供的资料，生成可执行的 {city} 行程方案。

【用户约束】
- 原始需求：{user_query or '未提供'}
- 目的地：{city}
- 天数：{days if days > 0 else '未指定'}
- 出发日期：{start_date if start_date else '未指定'}
- 偏好：{preference}
{travel_mode_section}
{budget_section}

【已采集资料】
{formatted_materials}
{availability_section}
【输出要求】
1. 使用 Markdown，但必须严格按下面的标题结构输出，便于前端卡片化展示：
   - 一级标题：# {city} {days if days > 0 else ''}日行程
   - 二级标题：## 行程概览
   - 二级标题：## Day 1 · 真实当日主题概要
   - 三级标题：### 上午 / ### 下午 / ### 晚上 / ### 餐饮建议
   - 二级标题：## 注意事项
   - 二级标题：## 预算安排 或 ## 预算建议
2. 「行程概览」必须包含：主题、强度、适合人群、交通策略、预算档位（如有）。
3. 按天拆分；每一天包含上午、下午、晚上。每个 Day 标题必须根据当天核心路线生成 8-16 字主题概要，例如「老城文化与夜游美食」「太湖风光与园林慢游」，严禁输出「当天主题」「主题待定」「综合游览」等占位词或泛词。
4. 每个 Day 标题下第一行必须标注「日期 + 天气预报信息」 ，格式固定为：「📅 日期：XXXX 年 XX 月 XX 日 | 🌤 天气：晴转多云，15-23℃」（日期格式统一为 “XXXX 年 XX 月 XX 日”，天气预报需包含天气状况、温度范围）。{f"出发日期为 {start_date}，请从该日期开始依次推算每一天的具体日期。" if start_date else ""}
   第二行必须给出「本日概要：一句话说明主要动线、体验重点和行程节奏」，例如：「本日概要：上午集中游览老城文化点位，下午转向运河街区，晚上以本地餐饮和夜景收尾。」
5. 每个上午/下午/晚上时间段必须用列表给出以下字段：地点/活动、推荐理由、建议停留、交通建议。
6. **天气适配规则（重要）**：
   - 如果某天预报有雨、雪、雷暴、冰雹、大雾、霾、沙尘等恶劣天气，该天的上午/下午时段**必须安排室内活动**（如博物馆、室内景点、商场、美食探店、文化体验等），避免安排户外徒步、公园游览、户外拍照等。
   - 恶劣天气的晚上可以安排室内餐饮或演出。
   - 仅在天气良好时才推荐户外景点和活动。
7. 每天补充 1-2 个餐饮建议。
8. {budget_requirement}
9. 不要编造资料中完全不存在的硬性事实；不确定信息用"建议/可考虑"表述。
10. **交通建议要求**：
   - 如果出行方式是自驾，请在交通建议中说明驾车路线和预计行驶时间。
"""

    try:
        content = _stream_llm_text(llm, prompt).strip()
        if not content:
            response = llm.invoke(prompt)
            content = getattr(response, "content", None) or str(response)
        return {"messages": [AIMessage(content=content)]}
    except Exception as exc:
        return {
            "messages": [
                AIMessage(content=f"行程生成失败，请稍后重试。错误信息：{exc}")
            ]
        }
