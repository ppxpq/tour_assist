

# tools/travel_tools.py
import base64
import time
import tempfile
from pathlib import Path
from functools import lru_cache
# 加上这一行
from typing import List, Dict, Optional
from datetime import datetime

import requests
from langchain_core.tools import tool
from openai import OpenAI

from utils import config


# ══════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════

def _amap_get(endpoint: str, params: dict) -> dict | None:
    """统一调用高德 REST API，失败返回 None。"""
    params["key"] = config.AMAP_API_KEY
    try:
        resp = requests.get(
            f"https://restapi.amap.com/v3/{endpoint}", params=params, timeout=5
        )
        resp.raise_for_status()
        data = resp.json()
        return data if data.get("status") == "1" else None
    except Exception:
        return None


def _geocode(city: str) -> str | None:
    """城市名 → 高德坐标字符串 'lng,lat'"""
    data = _amap_get("geocode/geo", {"address": city})
    if data and data.get("geocodes"):
        return data["geocodes"][0]["location"]
    return None



@lru_cache(maxsize=1)
def _ip_locate() -> dict | None:
    """通过 IP 获取用户当前位置（城市级精度），带缓存。"""
    data = _amap_get("ip", {})
    if data and data.get("status") == "1":
        return {
            "city":      data.get("city"),
            "location":  data.get("location"),
            "province":  data.get("province"),
            "cached_at": time.time(),
        }
    return None


@lru_cache(maxsize=1)
def _get_llm():
    """LLM 单例（用于 generate_travel_plan）。"""
    from core.llm_core import get_llm
    return get_llm("glm-4.5-air")


@lru_cache(maxsize=1)
def _get_ali_client() -> OpenAI:
    """阿里云 OpenAI 兼容客户端单例（qwen-vl-max / qwen-audio-turbo）。"""
    return OpenAI(
        api_key=config.ALI_API_KEY,
        base_url=config.ALI_BASE_URL,
    )


@lru_cache(maxsize=1)
def _get_mimo_client() -> OpenAI:
    """MiMo OpenAI 兼容客户端单例（MiMo-V2-Omni 多模态）。"""
    return OpenAI(
        api_key=config.MIMO_API_KEY,
        base_url=config.MIMO_BASE_URL,
    )


def _to_base64(file_input: str) -> tuple[str, str]:
    """
    统一处理文件输入：本地路径 → base64 编码；已是 base64 → 直接透传。
    返回 (base64_str, 后缀小写)
    """
    path = Path(file_input)
    if path.exists():
        return base64.b64encode(path.read_bytes()).decode(), path.suffix.lower()
    return file_input, ".bin"


def _parse_spot(text: str) -> tuple[str, str]:
    """从 Vision 返回文本中提取景点名和城市名。"""
    spot, city = "", ""
    for line in text.splitlines():
        if line.startswith("景点名称："):
            spot = line.removeprefix("景点名称：").strip()
        elif line.startswith("所在城市："):
            city = line.removeprefix("所在城市：").strip()
    return spot, city


def _fetch_amap_poi(spot: str, city: str) -> str:
    """根据景点名 + 城市查询高德，返回格式化补充信息。"""
    if not spot or not city:
        return ""
    data = _amap_get(
        "place/text",
        {"keywords": spot, "city": city, "types": "110000", "offset": 1, "page": 1},
    )
    if not data or not data.get("pois"):
        return ""
    poi = data["pois"][0]
    return (
        f"\n\n【高德补充信息】\n"
        f"评分：{poi.get('rating') or '暂无'} 分\n"
        f"地址：{poi.get('address', '暂无')}\n"
        f"电话：{poi.get('tel') or '暂无'}"
    )


# ══════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════

@tool
def get_current_location() -> str:
    """
    获取用户当前所在城市和大致位置。
    当用户没有明确说明出发地，需要查询距离、路线或附近信息时优先调用。
    """
    loc = _ip_locate()
    if not loc or not loc.get("city"):
        return "无法获取您的当前位置，请手动输入出发城市。"
    return (
        f"您当前所在城市：{loc['city']}\n"
        f"大致坐标：{loc['location']}"
    )


@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的实时天气。
    当需要了解目的地天气或判断是否适合出行时调用。

    Args:
        city: 城市名称，如「成都」「北京」。
    """
    # Step 1：城市名 → adcode（高德天气API用adcode更准确）
    geo_data = _amap_get("geocode/geo", {"address": city})
    if not geo_data or not geo_data.get("geocodes"):
        return f"无法解析城市「{city}」，请确认城市名称是否正确。"
    
    adcode = geo_data["geocodes"][0].get("adcode", "")
    city_name = geo_data["geocodes"][0].get("city") or geo_data["geocodes"][0].get("province") or city

    if not adcode:
        return f"无法获取「{city}」的行政区划代码。"

    # Step 2：用 adcode 查天气
    data = _amap_get("weather/weatherInfo", {"city": adcode, "extensions": "base"})
    if not data or not data.get("lives"):
        return f"未能查询到「{city}」的天气，请稍后重试。"

    live = data["lives"][0]
    return (
        f"【{live['city']}】实时天气\n"
        f"天气：{live['weather']}\n"
        f"温度：{live['temperature']}℃\n"
        f"湿度：{live['humidity']}%\n"
        f"风向：{live['winddirection']}风 {live['windpower']} 级\n"
        f"发布时间：{live['reporttime']}"
    )


@tool
def get_weather_forecast(city: str) -> str:
    """
    查询指定城市未来几天的天气预报。
    当需要规划多日行程、了解旅行期间天气时调用。

    Args:
        city: 城市名称，如「成都」「北京」。
    """
    geo_data = _amap_get("geocode/geo", {"address": city})
    if not geo_data or not geo_data.get("geocodes"):
        return f"无法解析城市「{city}」，请确认城市名称是否正确。"

    adcode = geo_data["geocodes"][0].get("adcode", "")
    city_name = geo_data["geocodes"][0].get("city") or geo_data["geocodes"][0].get("province") or city

    if not adcode:
        return f"无法获取「{city}」的行政区划代码。"

    data = _amap_get("weather/weatherInfo", {"city": adcode, "extensions": "all"})
    if not data or not data.get("forecasts"):
        return f"未能查询到「{city}」的天气预报，请稍后重试。"

    forecast = data["forecasts"][0]
    city_display = forecast.get("city", city_name)
    report_time = forecast.get("reporttime", "")
    casts = forecast.get("casts", [])

    if not casts:
        return f"「{city_display}」暂无天气预报数据。"

    lines = [f"【{city_display}】天气预报（发布于 {report_time}）"]
    for day in casts:
        date = day.get("date", "")
        week = day.get("week", "")
        day_weather = day.get("dayweather", "")
        night_weather = day.get("nightweather", "")
        day_temp = day.get("daytemp", "")
        night_temp = day.get("nighttemp", "")
        day_wind = day.get("daywind", "")
        night_wind = day.get("nightwind", "")
        day_power = day.get("daypower", "")
        night_power = day.get("nightpower", "")

        lines.append(
            f"\n📅 {date}（周{week}）"
            f"\n  白天：{day_weather} | {day_temp}℃ | {day_wind}风 {day_power}级"
            f"\n  夜间：{night_weather} | {night_temp}℃ | {night_wind}风 {night_power}级"
        )
        # 标注不适合户外的天气
        bad_weather_keywords = ("雨", "雪", "雷", "冰雹", "雾", "霾", "沙尘", "暴")
        if any(kw in day_weather for kw in bad_weather_keywords):
            lines.append(f"  ⚠️ 白天有{day_weather}，建议安排室内活动")

    return "\n".join(lines)


def _search_poi(city: str, keyword: str, poi_type: str, label: str) -> str:
    """高德 POI 通用搜索，返回格式化结果。"""
    data = _amap_get(
        "place/text",
        {"keywords": keyword, "city": city, "types": poi_type, "offset": 6, "page": 1},
    )
    if not data or not data.get("pois"):
        return f"未在「{city}」找到相关{label}，换个关键词试试。"
    lines = [
        f"• {p['name']}（{p.get('rating') or '暂无'}分）\n  地址：{p.get('address', '暂无')}"
        for p in data["pois"]
    ]
    return f"「{city}」{label}推荐（关键词：{keyword}）：\n" + "\n".join(lines)


@tool
def search_scenic_spot(city: str, keyword: Optional[str] = "景点") -> str:
    """
    搜索城市景点、名胜、公园等旅游地点。
    当用户询问某城市有哪些地方值得游览时调用。

    Args:
        city:    目标城市，如「杭州」。
        keyword: 可选，景点关键词，如「故宫」「西湖」，默认搜全部景点。
    """
    return _search_poi(city, keyword, "110000", "景点")


@tool
def search_restaurant(city: str, keyword: Optional[str] = "特色餐厅") -> str:
    """
    搜索城市特色餐厅和美食推荐。
    当用户询问当地吃什么、有哪些好餐厅时调用。

    Args:
        city:    目标城市，如「成都」。
        keyword: 可选，菜系或关键词，如「火锅」「早茶」，默认搜特色餐厅。
    """
    return _search_poi(city, keyword, "050000", "餐饮")


@tool
def get_route_distance(
    destination: str,
    origin: Optional[str] = None,
) -> str:
    """
    查询两座城市之间的驾车距离与预计耗时。
    当评估城市间交通成本或规划多城联游时调用。
    如果未提供出发地，将自动使用用户当前定位。

    Args:
        destination: 目的城市，如「苏州」。
        origin:      可选，出发城市，如「上海」，默认使用当前定位。
    """
    if not origin:
        loc = _ip_locate()
        if not loc or not loc.get("city"):
            return "无法获取您的当前位置，请手动输入出发城市。"
        origin = loc["city"]
        origin_loc = loc["location"]
    else:
        origin_loc = _geocode(origin)

    dest_loc = _geocode(destination)
    if not origin_loc or not dest_loc:
        return "无法解析城市坐标，请确认城市名称。"

    data = _amap_get(
        "direction/driving",
        {"origin": origin_loc, "destination": dest_loc, "strategy": 0},
    )
    if not data or not data.get("route", {}).get("paths"):
        return "路线查询失败，请稍后重试。"

    path = data["route"]["paths"][0]
    distance_km = round(int(path["distance"]) / 1000, 1)
    hours, mins = divmod(round(int(path["duration"]) / 60), 60)
    duration_str = f"{hours}小时{mins}分钟" if hours else f"{mins}分钟"
    return (
        f"【{origin} → {destination}】\n"
        f"驾车距离：约 {distance_km} 公里\n"
        f"预计用时：约 {duration_str}"
    )

@tool
def speech_to_text(audio_input: str) -> str:
    """
    将用户上传的语音/音频转换为文字。
    当用户以语音方式表达旅游需求时，Agent 应首先调用此工具，
    将识别结果作为文本再传递给其他工具。

    Args:
        audio_input: 音频文件的本地路径 或 base64 字符串。
                     支持格式：mp3 / m4a / wav / ogg / webm。
    """
    path = Path(audio_input)
    tmp_path = None

    if not path.exists():
        try:
            raw = base64.b64decode(audio_input)
        except Exception:
            return "音频输入无效：既不是有效路径，也不是合法 base64 字符串。"
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.write(raw)
        tmp.flush()
        path = tmp_path = Path(tmp.name)

    try:
        client = _get_mimo_client()
        with open(path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="mimo-v2-omni",
                file=f,
                response_format="text",
            )
        text = transcript if isinstance(transcript, str) else transcript.text
        return f"[语音识别结果]\n{text.strip()}"
    except Exception as e:
        return f"语音识别失败：{e}"
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@tool
def recognize_scenic_spot(
    image_input: str,
    city_hint: Optional[str] = None,
) -> str:
    """
    识别图片中的景点或地标，并附上高德地图补充信息。
    当用户上传照片询问「这是哪里」「这个景点怎么样」时调用。

    Args:
        image_input: 图片的本地路径（jpg/png/webp）或 base64 字符串。
        city_hint:   可选，用户提供的城市线索，如「好像是杭州的」。
    """
    b64_data, suffix = _to_base64(image_input)
    mime_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")

    city_context = f"图片可能拍摄于{city_hint}，请优先考虑该地区的景点。" if city_hint else ""
    prompt = f"""你是一位旅游地理专家。{city_context}
请识别图片中的景点或地标，严格按以下格式输出，无法识别时如实说明：

景点名称：<名称>
所在城市：<城市>
简介：<2-3句话>
最佳游览时间：<建议>
门票参考：<价格或免费>"""

    try:
        client = _get_mimo_client()
        resp = client.chat.completions.create(
            model="mimo-v2-omni",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_data}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        vision_text = resp.choices[0].message.content.strip()
    except Exception as e:
        return f"图片识别失败：{e}"

    spot_name, city_name = _parse_spot(vision_text)
    amap_info = _fetch_amap_poi(spot_name, city_name)
    return f"【景点识别结果】\n{vision_text}{amap_info}"


# ══════════════════════════════════════════════
# 统一导出
# ══════════════════════════════════════════════

def get_travel_tools():
    """返回旅游助手的全部工具列表。"""
    return [
        get_current_location,
        get_weather,
        get_weather_forecast,
        search_scenic_spot,
        search_restaurant,
        get_route_distance,
        # generate_travel_plan,
        speech_to_text,
        recognize_scenic_spot,
    ]