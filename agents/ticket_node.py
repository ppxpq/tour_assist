from __future__ import annotations

import asyncio
import json
import re
from datetime import date as Date, datetime, timedelta
from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agents.state import TravelState, human_texts
from core.llm_core import get_llm
from core.mcp_client import get_tickets as mcp_get_tickets


_TICKET_SYSTEM = """你是一个车票查询助手。根据用户的消息，提取车票查询所需的参数。

请返回一个 JSON 对象，包含以下字段（无法提取的字段返回空字符串或 false）：
- departure: 出发城市或站点名
- destination: 到达城市或站点名
- date: 出发日期，格式 yyyy-MM-dd，无明确日期则为空字符串
- train_filter: 车型筛选，G=高铁 D=动车 Z=直达 T=特快 K=快速，多个可组合如"GD"，不限则为空字符串
- need_transfer: 是否需要中转，布尔值

示例输入：帮我查后天北京到上海的高铁
示例输出：
{{"departure": "北京", "destination": "上海", "date": "2024-05-15", "train_filter": "G", "need_transfer": false}}

注意：
- 只提取用户明确提到的信息，未提及的字段留空
- 今天是 {today}，请将"明天""后天"等推算为 yyyy-MM-dd
- 只输出 JSON，不要任何额外说明"""


def _safe_parse_json(text: str) -> Optional[dict]:
    cleaned = re.sub(r"```(?:json)?\s*|```", "", text).strip()
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


_CN_NUMBER_MAP = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_WEEKDAY_MAP = {
    "一": 0,
    "二": 1,
    "三": 2,
    "四": 3,
    "五": 4,
    "六": 5,
    "日": 6,
    "天": 6,
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
    "5": 4,
    "6": 5,
    "7": 6,
}


def _parse_cn_int(raw: str) -> Optional[int]:
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    if raw in _CN_NUMBER_MAP:
        return _CN_NUMBER_MAP[raw]
    if "十" in raw:
        tens_raw, _, ones_raw = raw.partition("十")
        tens = 1 if not tens_raw else _CN_NUMBER_MAP.get(tens_raw)
        ones = 0 if not ones_raw else _CN_NUMBER_MAP.get(ones_raw)
        if tens is not None and ones is not None:
            return tens * 10 + ones
    return None


def _format_date(value: Date) -> str:
    return value.strftime("%Y-%m-%d")


def _safe_build_date(year: int, month: int, day: int) -> Optional[Date]:
    try:
        return Date(year, month, day)
    except ValueError:
        return None


def _next_month_day(month: int, day: int, today: Date) -> Optional[Date]:
    candidate = _safe_build_date(today.year, month, day)
    if candidate is None:
        return None
    if candidate < today:
        candidate = _safe_build_date(today.year + 1, month, day)
    return candidate


def _parse_local_ticket_date(text: str, today: Date | None = None) -> str:
    """Parse common Chinese date expressions deterministically for ticket queries."""
    normalized = re.sub(r"\s+", "", text or "")
    if not normalized:
        return ""

    today = today or datetime.now().date()

    for word, offset in (
        ("大后天", 3),
        ("大後天", 3),
        ("后天", 2),
        ("後天", 2),
        ("明天", 1),
        ("今天", 0),
    ):
        if word in normalized:
            return _format_date(today + timedelta(days=offset))

    days_later_match = re.search(r"(?P<days>\d+|[一二两三四五六七八九十]{1,3})天后", normalized)
    if days_later_match:
        days = _parse_cn_int(days_later_match.group("days"))
        if days is not None and 0 <= days <= 60:
            return _format_date(today + timedelta(days=days))

    full_date_match = re.search(
        r"(?P<year>\d{4})[年/-](?P<month>\d{1,2})[月/-](?P<day>\d{1,2})(?:日|号)?",
        normalized,
    )
    if full_date_match:
        parsed_date = _safe_build_date(
            int(full_date_match.group("year")),
            int(full_date_match.group("month")),
            int(full_date_match.group("day")),
        )
        return _format_date(parsed_date) if parsed_date else ""

    month_day_match = re.search(
        r"(?<!\d)(?P<month>\d{1,2})(?:月|[/-])(?P<day>\d{1,2})(?:日|号)?(?!\d)",
        normalized,
    )
    if month_day_match:
        parsed_date = _next_month_day(
            int(month_day_match.group("month")),
            int(month_day_match.group("day")),
            today,
        )
        return _format_date(parsed_date) if parsed_date else ""

    next_week_match = re.search(r"(?:下周|下星期|下礼拜)(?P<weekday>[一二三四五六日天1234567])", normalized)
    if next_week_match:
        weekday = _WEEKDAY_MAP.get(next_week_match.group("weekday"))
        if weekday is not None:
            days_to_next_monday = 7 - today.weekday()
            return _format_date(today + timedelta(days=days_to_next_monday + weekday))

    week_match = re.search(r"(?:这周|本周|周|星期|礼拜)(?P<weekday>[一二三四五六日天1234567])", normalized)
    if week_match:
        weekday = _WEEKDAY_MAP.get(week_match.group("weekday"))
        if weekday is not None:
            days_ahead = (weekday - today.weekday()) % 7
            return _format_date(today + timedelta(days=days_ahead))

    return ""


_TRAIN_RE = re.compile(
    r"(?P<train>[A-Z]{1,3}\d{1,5}[A-Z]?)\s+"
    r"(?P<from>.+?)\s*→\s*(?P<to>.+?)\s+"
    r"(?P<start>\d{2}:\d{2})\s*→\s*(?P<end>\d{2}:\d{2})\s*"
    r"历时[:：]\s*(?P<duration>\d{2}:\d{2})",
    re.S,
)
_SEAT_RE = re.compile(
    r"(?P<name>[\u4e00-\u9fa5A-Za-z0-9]+):\s*"
    r"(?P<status>剩余\d+张票|有票|无票|候补|--)\s*"
    r"(?P<price>\d+(?:\.\d+)?元)"
)
_CITY_PAIR_PATTERNS = (
    re.compile(
        r"(?:从|由)(?P<departure>[\u4e00-\u9fa5A-Za-z]{2,10})(?:出发)?"
        r"(?:到|去|至|->|→)"
        r"(?P<destination>[\u4e00-\u9fa5A-Za-z]{2,10})"
    ),
    re.compile(
        r"(?P<departure>[\u4e00-\u9fa5A-Za-z]{2,10})"
        r"(?:到|至|->|→)"
        r"(?P<destination>[\u4e00-\u9fa5A-Za-z]{2,10})"
    ),
)
_COMMON_STATION_NAMES = [
    "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "无锡", "常州", "扬州", "成都", "重庆",
    "西安", "武汉", "长沙", "厦门", "青岛", "大连", "天津", "宁波", "福州", "泉州", "洛阳", "开封",
    "南京南", "无锡东", "无锡新区", "上海虹桥", "杭州东", "苏州北",
]


def _strip_ticket_noise(value: str) -> str:
    cleaned = re.sub(
        r"(?:的)?(?:高铁票|动车票|火车票|列车票|车票|票|余票|查询|查一下|查|看看|预订|订|买|购买).*$",
        "",
        value or "",
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"(?:明天|后天|大后天|今天|周[一二三四五六日天]|星期[一二三四五六日天])", "", cleaned)
    cleaned = re.sub(r"^(?:从|由)", "", cleaned)
    return re.sub(r"[，。！？、,\s]", "", cleaned).strip()


def _normalize_station_name(value: str) -> str:
    cleaned = _strip_ticket_noise(value)
    cleaned = re.sub(r"(?:出发地|目的地|到达地|出发站|到达站)[：:]", "", cleaned)
    cleaned = re.sub(r"(?:车票信息|票务信息|查询结果|高铁|动车|火车|列车|车票|余票)$", "", cleaned)
    cleaned = cleaned.strip()
    for station in sorted(_COMMON_STATION_NAMES, key=len, reverse=True):
        if station in cleaned:
            return station
    return cleaned[:10]


def _parse_local_ticket_route(text: str) -> tuple[str, str]:
    normalized = re.sub(r"\s+", "", text or "")
    normalized = re.sub(r"^(?:帮我)?(?:查询|查一下|查|看看|看一下|买|购买|预订|订)", "", normalized)
    for pattern in _CITY_PAIR_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        departure = _strip_ticket_noise(match.group("departure"))
        destination = _strip_ticket_noise(match.group("destination"))
        if departure and destination and departure != destination:
            return departure[:10], destination[:10]
    return "", ""


def _parse_local_train_filter(text: str) -> str:
    normalized = re.sub(r"\s+", "", text or "")
    flags = []
    if "高铁" in normalized or "城际" in normalized:
        flags.append("G")
    if "动车" in normalized:
        flags.append("D")
    if "直达" in normalized:
        flags.append("Z")
    if "特快" in normalized:
        flags.append("T")
    if "快速" in normalized or "普快" in normalized:
        flags.append("K")
    return "".join(dict.fromkeys(flags))


def _clean_ticket_text(text: str) -> str:
    """Remove noisy MCP details while keeping the useful ticket content."""
    cleaned = re.sub(r"\(telecode:[^)]+\)", "", text or "")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"车次\|出发站\s*→\s*到达站\|出发时间\s*→\s*到达时间\|历时", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def _clean_station(station: str) -> str:
    return re.sub(r"\s+", "", station or "").strip()


def _format_seat(name: str, status: str, price: str) -> str:
    status = status.strip()
    status = re.sub(r"剩余(\d+)张票", r"余\1张", status)
    return f"{name}：{status} / {price}"


def _parse_ticket_records(raw: str) -> list[dict[str, str]]:
    text = _clean_ticket_text(raw)
    matches = list(_TRAIN_RE.finditer(text))
    records: list[dict[str, str]] = []

    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        seat_text = text[match.end():next_start]
        seats = [
            _format_seat(
                seat_match.group("name"),
                seat_match.group("status"),
                seat_match.group("price"),
            )
            for seat_match in _SEAT_RE.finditer(seat_text)
        ]
        records.append(
            {
                "train": match.group("train").strip(),
                "route": f"{_clean_station(match.group('from'))} → {_clean_station(match.group('to'))}",
                "time": f"{match.group('start')} → {match.group('end')}",
                "duration": match.group("duration"),
                "seats": "<br>".join(seats) if seats else "暂无余票详情",
            }
        )

    return records


def _format_ticket_result(raw: str) -> str:
    records = _parse_ticket_records(raw)
    if not records:
        cleaned = _clean_ticket_text(raw)
        return f"```text\n{cleaned}\n```" if cleaned else ""

    rows = [
        f"共找到 **{len(records)}** 趟车，按 12306 返回顺序展示：",
        "",
        "| 车次 | 区间 | 时间 | 历时 | 余票 / 价格 |",
        "|---|---|---|---:|---|",
    ]
    rows.extend(
        f"| {record['train']} | {record['route']} | {record['time']} | {record['duration']} | {record['seats']} |"
        for record in records
    )
    return "\n".join(rows)


def _run_async(coro):
    """Run an async coroutine synchronously."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=60)
    return asyncio.run(coro)


def _query_tickets(
    departure: str,
    destination: str,
    date: str,
    train_filter: str = "",
    limited_num: int = 10,
) -> str:
    """Query train tickets via MCP client."""
    result = _run_async(
        mcp_get_tickets(
            date=date,
            from_station=departure,
            to_station=destination,
            train_filter=train_filter,
            limited_num=limited_num,
            format="text",
        )
    )
    return result


def ticket_agent(state: TravelState) -> dict:
    """Ticket query agent - queries 12306 train tickets via MCP."""
    messages: list[BaseMessage] = list(state.get("messages", []))
    user_texts = human_texts(messages)
    user_input = user_texts[-1] if user_texts else ""
    travel_mode = (state.get("travel_mode") or "").strip()

    if not user_input:
        return {
            "messages": [
                AIMessage(content="请告诉我出发地、目的地和出行日期，我来帮你查询车票。")
            ]
        }

    now = datetime.now()
    local_date = _parse_local_ticket_date(user_input, today=now.date())
    local_departure, local_destination = _parse_local_ticket_route(user_input)
    local_train_filter = _parse_local_train_filter(user_input)
    system_prompt = _TICKET_SYSTEM.format(today=now.strftime("%Y年%m月%d日"))

    llm = None
    try:
        llm = get_llm("glm-4-flash")
    except Exception:
        llm = None

    try:
        if llm:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ])
            raw_text = response.content if hasattr(response, "content") else str(response)
            parsed = _safe_parse_json(raw_text)
        else:
            parsed = None
    except Exception:
        parsed = None

    # Fallback: use state fields
    departure = ""
    destination = ""
    date = ""
    train_filter = ""
    need_transfer = False

    if parsed:
        departure = str(parsed.get("departure", "") or "").strip()
        destination = str(parsed.get("destination", "") or "").strip()
        date = str(parsed.get("date", "") or "").strip()
        train_filter = str(parsed.get("train_filter", "") or "").strip().upper()
        need_transfer = bool(parsed.get("need_transfer", False))

    if local_departure:
        departure = local_departure
    if local_destination:
        destination = local_destination
    if local_train_filter:
        train_filter = local_train_filter

    # Supplement from state
    if not departure:
        departure = (state.get("departure") or "").strip()
    if not destination:
        destination = (state.get("city") or "").strip()
    if local_date:
        date = local_date
    if not date:
        date = (state.get("start_date") or "").strip()

    departure = _normalize_station_name(departure)
    destination = _normalize_station_name(destination)

    # 根据出行方式设置车型筛选
    if not train_filter and travel_mode:
        if travel_mode == "高铁":
            train_filter = "G"
        elif travel_mode == "火车":
            train_filter = "KTZ"  # K=快速, T=特快, Z=直达

    # Validate required fields
    missing = []
    if not departure:
        missing.append("出发地")
    if not destination:
        missing.append("目的地")

    if missing:
        readable = "和".join(missing)
        return {
            "messages": [
                AIMessage(
                    content=f"要查询车票，请告诉我**{readable}**。例如：帮我查5月20号北京到上海的高铁票。"
                )
            ]
        }

    if not date:
        date = now.strftime("%Y-%m-%d")

    # Query tickets via MCP
    try:
        result = _query_tickets(
            departure=departure,
            destination=destination,
            date=date,
            train_filter=train_filter,
            limited_num=10,
        )
    except Exception as e:
        return {
            "messages": [
                AIMessage(content=f"查询车票时出错：{e}。请确认出发地和目的地名称是否正确，或稍后重试。")
            ]
        }

    if not result or not result.strip():
        return {
            "messages": [
                AIMessage(
                    content=f"未找到 **{departure}** 到 **{destination}** 在 **{date}** 的车票信息。"
                    "请检查城市名称是否正确，或尝试查询其他日期。"
                )
            ]
        }

    # Format the result nicely
    filter_desc = ""
    if train_filter:
        filter_map = {"G": "高铁", "D": "动车", "Z": "直达", "T": "特快", "K": "快速"}
        filter_desc = "（" + "+".join(filter_map.get(c, c) for c in train_filter) + "）"

    formatted_result = _format_ticket_result(result)
    reply = (
        f"### 🚄 {departure} → {destination} {date} 车票信息{filter_desc}\n\n"
        f"{formatted_result}\n\n"
        f"*数据来源：12306*"
    )

    return {
        "messages": [AIMessage(content=reply)],
        "departure": departure,
        "city": destination,
        "start_date": date,
    }
