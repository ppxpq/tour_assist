from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agents.graph import build_travel_graph
from core.db_manager import clear_database, ingest_documents, load_db
from core.session_store import SessionStore
from core.xhs_importer import XhsImportError, fetch_xhs_note_as_text
from utils import config


DEFAULT_MODEL = "glm-4.5-air"
ROUTER_MODEL = "glm-4-flash"
XHS_IMPORT_COOLDOWN_SECONDS = 60

NODE_ORDER = ["router", "researcher", "planner", "ticket_agent"]
NODE_TITLE = {
    "router": "Router 意图解析",
    "researcher": "Researcher 资料搜集",
    "planner": "Planner 行程生成",
    "ticket_agent": "Ticket 车票查询",
}
STATUS_LABEL = {
    "pending": "待执行",
    "running": "运行中",
    "completed": "已完成",
    "skipped": "已跳过",
    "failed": "失败",
}

MEDIA_PATH_RE = re.compile(
    r"路径[:：]\s*([^\n，,]+?\.(?:jpg|jpeg|png|webp|bmp|gif|mp3|wav|m4a|ogg|webm|flac|aac))",
    re.IGNORECASE,
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".webm"}
LOW_QUALITY_TITLE_RE = re.compile(r"^(新会话|我想|帮我|请帮|查询|根据|用户上传)")


@dataclass(slots=True)
class UploadedFileData:
    name: str
    content: bytes
    content_type: str = ""

    def getbuffer(self) -> memoryview:
        return memoryview(self.content)


class TravelServiceError(RuntimeError):
    pass


def to_langchain_history(messages: Iterable[dict[str, str]]) -> list[BaseMessage]:
    history: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
    return history


def _now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _sanitize_messages(messages: Any) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    if not isinstance(messages, list):
        return cleaned

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "").strip()
        content = str(msg.get("content") or "")
        if role in {"user", "assistant"}:
            cleaned.append({"role": role, "content": content})
    return cleaned


def _init_node_runtime() -> dict[str, dict[str, Any]]:
    return {
        node: {
            "title": NODE_TITLE[node],
            "status": "pending",
            "status_label": STATUS_LABEL["pending"],
            "start": None,
            "end": None,
            "duration": 0.0,
            "note": "-",
        }
        for node in NODE_ORDER
    }


def _runtime_snapshot(runtime: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for node in NODE_ORDER:
        item = runtime[node]
        status = item["status"]
        snapshot[node] = {
            "title": item["title"],
            "status": status,
            "status_label": STATUS_LABEL.get(status, status),
            "duration": round(float(item.get("duration") or 0.0), 3),
            "note": item.get("note") or "-",
        }
    return snapshot


def _mark_running(runtime: dict[str, dict[str, Any]], node: str) -> None:
    item = runtime[node]
    if item["status"] in {"completed", "skipped", "failed"}:
        return
    if item["start"] is None:
        item["start"] = time.perf_counter()
    item["status"] = "running"
    item["status_label"] = STATUS_LABEL["running"]


def _mark_completed(runtime: dict[str, dict[str, Any]], node: str, note: str) -> None:
    item = runtime[node]
    if item["start"] is None:
        item["start"] = time.perf_counter()
    item["end"] = time.perf_counter()
    item["duration"] = max(0.0, item["end"] - item["start"])
    item["status"] = "completed"
    item["status_label"] = STATUS_LABEL["completed"]
    item["note"] = note or "-"


def _mark_skipped(runtime: dict[str, dict[str, Any]], node: str, note: str) -> None:
    item = runtime[node]
    if item["status"] in {"completed", "failed"}:
        return
    if item["status"] == "running" and item["start"] is not None:
        item["end"] = time.perf_counter()
        item["duration"] = max(0.0, item["end"] - item["start"])
    item["status"] = "skipped"
    item["status_label"] = STATUS_LABEL["skipped"]
    item["note"] = note


def _mark_first_running_failed(runtime: dict[str, dict[str, Any]], note: str) -> None:
    for node in NODE_ORDER:
        item = runtime[node]
        if item["status"] == "running":
            if item["start"] is None:
                item["start"] = time.perf_counter()
            item["end"] = time.perf_counter()
            item["duration"] = max(0.0, item["end"] - item["start"])
            item["status"] = "failed"
            item["status_label"] = STATUS_LABEL["failed"]
            item["note"] = note
            return


def _extract_ai_text(messages: Any) -> str:
    if not messages:
        return ""

    iterable = messages if isinstance(messages, (list, tuple)) else [messages]
    for msg in reversed(iterable):
        if isinstance(msg, AIMessage):
            content = getattr(msg, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()

        if isinstance(msg, dict):
            role = str(msg.get("type") or msg.get("role") or "").lower()
            content = msg.get("content", "")
            if role in {"ai", "assistant"} and isinstance(content, str) and content.strip():
                return content.strip()

        msg_type = str(getattr(msg, "type", "")).lower()
        content = getattr(msg, "content", "")
        if msg_type == "ai" and isinstance(content, str) and content.strip():
            return content.strip()

    return ""


def _split_graph_stream_event(event: Any) -> tuple[str, Any] | None:
    if isinstance(event, dict) and "type" in event and "data" in event:
        mode = str(event.get("type") or "")
        if mode:
            return mode, event.get("data")

    if isinstance(event, tuple) and len(event) == 2 and isinstance(event[0], str):
        return event[0], event[1]

    if isinstance(event, dict):
        return "updates", event

    return None


def _build_node_note(node: str, update: dict[str, Any]) -> str:
    if not isinstance(update, dict):
        return "-"

    if node == "router":
        intent = update.get("intent") or "-"
        missing = update.get("missing_fields") or []
        reason = update.get("router_reason") or "-"
        note = f"intent={intent}"
        if missing:
            note += f", missing={','.join(missing)}"
        note += f", source={reason}"
        return note

    if node == "researcher":
        output = _extract_ai_text(update.get("messages"))
        if output:
            return f"已直接答复，输出字符数={len(output)}"
        materials = str(update.get("raw_materials") or "")
        return f"素材字符数={len(materials)}"

    if node == "planner":
        output = _extract_ai_text(update.get("messages"))
        return f"输出字符数={len(output)}" if output else "已生成回复"

    if node == "ticket_agent":
        departure = update.get("departure") or "-"
        city = update.get("city") or "-"
        date = update.get("start_date") or "-"
        return f"{departure} -> {city} {date}"

    return "-"


def _public_state_update(node: str, update: dict[str, Any]) -> dict[str, Any]:
    """Expose only JSON-safe planning fields to the frontend."""
    if node == "router":
        return {
            "intent": str(update.get("intent") or ""),
            "city": str(update.get("city") or ""),
            "days": int(update.get("days") or 0),
            "start_date": str(update.get("start_date") or ""),
            "preference": str(update.get("preference") or ""),
            "missing_fields": list(update.get("missing_fields") or []),
            "user_query": str(update.get("user_query") or ""),
        }
    if node == "ticket_agent":
        return {
            "departure": str(update.get("departure") or ""),
            "city": str(update.get("city") or ""),
            "start_date": str(update.get("start_date") or ""),
        }
    return {}


def _extension(filename: str) -> str:
    return Path(filename or "").suffix.lower()


def _is_image_upload(upload: UploadedFileData) -> bool:
    return upload.content_type.lower().startswith("image/") or _extension(upload.name) in IMAGE_EXTENSIONS


def _is_audio_upload(upload: UploadedFileData) -> bool:
    return upload.content_type.lower().startswith("audio/") or _extension(upload.name) in AUDIO_EXTENSIONS


def _short_title(prompt: str) -> str:
    compact = re.sub(r"\s+", " ", prompt).strip()
    compact = re.sub(r"路径[:：]\s*\S+", "", compact).strip()
    if not compact:
        return "新会话"
    return compact[:18]


def _clean_title_part(text: str, limit: int = 8) -> str:
    cleaned = re.sub(r"\s+", "", text or "")
    cleaned = re.sub(r"[。！？；;，,].*$", "", cleaned)
    return cleaned[:limit]


CN_NUMBER_MAP = {
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
    "十": 10,
}


def _parse_title_number(raw: str) -> int:
    raw = (raw or "").strip()
    if raw.isdigit():
        return int(raw)
    if raw in CN_NUMBER_MAP:
        return CN_NUMBER_MAP[raw]
    if "十" in raw:
        left, _, right = raw.partition("十")
        tens = CN_NUMBER_MAP.get(left, 1 if not left else 0)
        ones = CN_NUMBER_MAP.get(right, 0) if right else 0
        return tens * 10 + ones
    return 0


def _extract_current_title_parts(title: str) -> tuple[str, str, str]:
    """Return city, days, theme from an existing generated title when possible."""
    match = re.match(r"(?P<city>[^ ·]+)\s+(?P<days>\d+)日(?:\s*·\s*(?P<theme>.+))?", title or "")
    if not match:
        return "", "", ""
    return match.group("city") or "", match.group("days") or "", match.group("theme") or ""


def _extract_trip_title(prompt: str, current_title: str = "") -> str:
    text = re.sub(r"\s+", " ", prompt or "").strip()
    if not text:
        return ""

    old_city, old_days, old_theme = _extract_current_title_parts(current_title)

    city = ""
    for pattern in (
        r"目的地[：:\s]*([^\n。；;，,]+)",
        r"我想去([^\n。；;，,]+?)(?:[，,\s]*(?:玩|旅游|旅行|出发)|$)",
        r"想去([^\n。；;，,]+?)(?:[，,\s]*(?:玩|旅游|旅行|出发)|$)",
        r"去([^\n。；;，,]+?)(?:[，,\s]*(?:玩|旅游|旅行|出发)|$)",
        r"去([^\n。；;，,]+?)(?:\d{1,2}|[一二两三四五六七八九十]+)\s*(?:天|日|周)",
        r"换成([^\n。；;，,]+)",
    ):
        match = re.search(pattern, text)
        if match:
            city = _clean_title_part(match.group(1), 6)
            break
    city = city or old_city

    days = ""
    for pattern in (
        r"玩\s*(\d{1,2})\s*天",
        r"(\d{1,2})\s*[天日]",
        r"改成\s*(\d{1,2})\s*天",
        r"([一二两三四五六七八九十]+)\s*[天日]",
    ):
        match = re.search(pattern, text)
        if match:
            days_number = _parse_title_number(match.group(1))
            days = str(days_number) if days_number else match.group(1)
            break
    if not days:
        week_match = re.search(r"(\d{1,2}|[一二两三四五六七八九十]+)\s*周", text)
        if week_match:
            week_number = _parse_title_number(week_match.group(1))
            if week_number:
                days = str(min(30, week_number * 7))
    days = days or old_days

    theme = ""
    preference_match = re.search(r"偏好[：:\s]*([^\n。]+)", text)
    if preference_match:
        raw_items = re.split(r"[、,+，,\s]+", preference_match.group(1))
        theme = "".join(_clean_title_part(item, 3) for item in raw_items if item and item != "未指定")[:8]

    if not theme:
        travelers_match = re.search(r"同行人[：:\s]*([^\n。]+)", text)
        if travelers_match:
            raw_items = re.split(r"[、,+，,\s]+", travelers_match.group(1))
            theme = "".join(_clean_title_part(item, 3) for item in raw_items if item and item != "未指定")[:8]

    if not theme:
        preference_words = ("美食", "人文", "自然", "摄影", "休闲", "小众", "省钱", "购物", "夜游", "亲子", "家庭", "老人")
        matched_words = [word for word in preference_words if word in text]
        theme = "".join(matched_words[:3])[:8]

    theme = theme or old_theme

    if city and days:
        return f"{city} {days}日" + (f" · {theme}" if theme else "")

    ticket_match = None
    if re.search(r"车票|高铁票|火车票|动车票|余票|查票|抢票|候补|12306", text):
        ticket_text = re.sub(r"\d{1,2}\s*月\s*\d{1,2}\s*(?:日|号)?", "", text)
        ticket_text = re.sub(r"(?:帮我)?(?:查询|查一下|查|看看|看一下)", "", ticket_text)
        for pattern in (
            r"从(?P<departure>[\u4e00-\u9fa5A-Za-z]{1,8})到(?P<destination>[\u4e00-\u9fa5A-Za-z]{1,8}?)(?:的)?(?:高铁票|火车票|动车票|车票|票)",
            r"(?:^|[，。,.\s])(?P<departure>[\u4e00-\u9fa5A-Za-z]{1,8})到(?P<destination>[\u4e00-\u9fa5A-Za-z]{1,8}?)(?:的)?(?:高铁票|火车票|动车票|车票|票)",
        ):
            ticket_match = re.search(pattern, ticket_text)
            if ticket_match:
                break
    if ticket_match:
        departure = _clean_title_part(ticket_match.group("departure"), 5)
        destination = _clean_title_part(ticket_match.group("destination"), 5)
        if departure and destination:
            return f"{departure}到{destination}车票"

    return ""


def _is_low_quality_title(title: str) -> bool:
    compact = re.sub(r"\s+", "", title or "")
    return not compact or bool(LOW_QUALITY_TITLE_RE.match(compact)) or len(compact) > 18


def _compact_title_theme(preference: str) -> str:
    parts = [part for part in re.split(r"[、,+，,\s]+", preference or "") if part and part not in {"综合", "未指定"}]
    return "".join(_clean_title_part(part, 3) for part in parts)[:8]


def _build_title_from_node_update(node: str, update: dict[str, Any]) -> str:
    if node == "router":
        intent = str(update.get("intent") or "").strip().lower()
        city = _clean_title_part(str(update.get("city") or ""), 8)
        days = int(update.get("days") or 0)
        preference = _compact_title_theme(str(update.get("preference") or ""))

        if intent in {"need_plan", "need_more_info"} and city and days > 0:
            return f"{city} {days}日" + (f" · {preference}" if preference else "")
        if intent == "need_answer" and city:
            return f"{city}旅行问答"

    if node == "ticket_agent":
        departure = _clean_title_part(str(update.get("departure") or ""), 6)
        destination = _clean_title_part(str(update.get("city") or ""), 6)
        if departure and destination:
            return f"{departure}到{destination}车票"

    return ""


def _title_score(title: str) -> int:
    if not title or title == "新会话":
        return 0
    score = 1
    if re.search(r"\d+日", title):
        score += 2
    if " · " in title:
        score += 1
    if "车票" in title or "问答" in title:
        score += 2
    if _is_low_quality_title(title):
        score -= 1
    return score


class TravelService:
    def __init__(self) -> None:
        config.init_env()
        self._lock = RLock()
        self._vector_lock = RLock()
        self.travel_graph = build_travel_graph()
        self._vector_dbs: dict[str, Any] = {}
        self._session_store = SessionStore()
        self._last_xhs_import_at = 0.0
        self._ensure_default_session()

    def _ensure_default_session(self, user_id: str = "local") -> None:
        with self._lock:
            self._session_store.ensure_default_session(user_id)

    def create_session(self, title: str | None = None, user_id: str = "local") -> dict[str, Any]:
        with self._lock:
            return self._session_store.create_session(title, user_id=user_id)

    def list_sessions(self, user_id: str = "local") -> list[dict[str, Any]]:
        with self._lock:
            return self._session_store.list_sessions(user_id)

    def get_session(self, session_id: str, user_id: str = "local") -> dict[str, Any]:
        with self._lock:
            return self._session_store.get_session(session_id, user_id)

    def delete_session(self, session_id: str, user_id: str = "local") -> dict[str, Any]:
        with self._lock:
            messages = self._session_store.get_session_messages(session_id, user_id)
            self._cleanup_session_media(messages)
            result = self._session_store.delete_session(session_id, user_id)

            if not result.get("current_session"):
                return self._session_store.create_session("新会话", user_id=user_id)
            return result

    def clear_session(self, session_id: str, user_id: str = "local") -> dict[str, Any]:
        with self._lock:
            messages = self._session_store.get_session_messages(session_id, user_id)
            self._cleanup_session_media(messages)
            return self._session_store.clear_session_messages(session_id, user_id)

    def _get_or_create_session_for_chat(
        self,
        session_id: str | None,
        user_id: str = "local",
    ) -> tuple[str, list[dict[str, str]]]:
        with self._lock:
            if session_id and self._session_store.get_session_summary(session_id, user_id):
                resolved_session_id = session_id
                self._session_store.set_current_session(session_id, user_id)
            else:
                session = self._session_store.create_session(user_id=user_id)
                resolved_session_id = str(session["id"])

            return resolved_session_id, self._session_store.get_session_messages(resolved_session_id, user_id)

    def _append_message(self, session_id: str, role: str, content: str, user_id: str = "local") -> None:
        with self._lock:
            summary = self._session_store.get_session_summary(session_id, user_id)
            if summary is None:
                raise KeyError(session_id)
            self._session_store.add_message(session_id, role, content, user_id=user_id)

    def _apply_session_title_candidate(self, session_id: str, candidate: str, user_id: str = "local") -> None:
        candidate = (candidate or "").strip()
        if not candidate:
            return
        with self._lock:
            summary = self._session_store.get_session_summary(session_id, user_id)
            if summary is None:
                return
            current_title = str(summary.get("title") or "")
            if candidate == current_title:
                return
            if _title_score(candidate) >= _title_score(current_title):
                self._session_store.update_session(session_id, title=candidate, user_id=user_id)

    def _maybe_update_session_title(
        self,
        session_id: str,
        prompt: str,
        current_title: str,
        previous_message_count: int,
        user_id: str = "local",
    ) -> None:
        candidate = _extract_trip_title(prompt, current_title)
        if not candidate and _is_low_quality_title(current_title):
            try:
                messages = self._session_store.get_session_messages(session_id, user_id)
                recent_user_messages = [
                    msg.get("content", "")
                    for msg in messages[-8:]
                    if msg.get("role") == "user"
                ]
                for recent_user_text in reversed(recent_user_messages):
                    candidate = _extract_trip_title(recent_user_text, current_title)
                    if candidate:
                        break
                if not candidate:
                    candidate = _extract_trip_title("\n".join(recent_user_messages), current_title)
            except Exception:
                candidate = ""

        if candidate and candidate != current_title:
            self._session_store.update_session(session_id, title=candidate, user_id=user_id)
            return

        if previous_message_count == 0 and _is_low_quality_title(current_title):
            fallback = _short_title(prompt)
            if fallback and fallback != current_title:
                self._session_store.update_session(session_id, title=fallback, user_id=user_id)

    def _cleanup_session_media(self, messages: list[dict[str, str]]) -> None:
        seen: set[str] = set()
        for msg in messages:
            content = msg.get("content", "")
            for match in MEDIA_PATH_RE.finditer(content):
                media_path = match.group(1).strip().strip("'\"")
                if media_path in seen:
                    continue
                seen.add(media_path)
                path = Path(media_path)
                try:
                    path.resolve().relative_to(Path(config.UPLOAD_DIR).resolve())
                    path.unlink(missing_ok=True)
                except Exception:
                    pass

    def save_upload(self, upload: UploadedFileData) -> str:
        suffix = _extension(upload.name) or ".bin"
        upload_dir = Path(config.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{int(time.time())}_{uuid.uuid4().hex}{suffix}"
        saved_path = upload_dir / filename
        saved_path.write_bytes(upload.content)
        return str(saved_path)

    def build_prompt(self, message: str, uploads: list[UploadedFileData] | None = None) -> str:
        prompt = (message or "").strip()
        media_hints: list[str] = []

        for upload in uploads or []:
            if _is_image_upload(upload):
                media_path = self.save_upload(upload)
                media_hints.append(f"用户上传了一张图片，路径：{media_path}，请识别图中景点并介绍。")
            elif _is_audio_upload(upload):
                media_path = self.save_upload(upload)
                media_hints.append(f"用户上传了语音文件，路径：{media_path}，请先转文字再处理。")

        if media_hints:
            prompt = "\n".join(media_hints) + (f"\n\n{prompt}" if prompt else "")

        if not prompt:
            raise TravelServiceError("请输入文字或上传图片/语音。")
        return prompt

    @staticmethod
    def _knowledge_user_key(user_id: str = "local") -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id or "local").strip("._-")
        return normalized[:80] or "local"

    def _knowledge_persist_path(self, user_id: str = "local") -> str:
        return str(Path(config.PERSIST_PATH) / "users" / self._knowledge_user_key(user_id))

    def _get_vector_db_locked(self, user_id: str = "local"):
        user_key = self._knowledge_user_key(user_id)
        if user_key not in self._vector_dbs:
            self._vector_dbs[user_key] = load_db(self._knowledge_persist_path(user_id))
        return self._vector_dbs[user_key]

    def _graph_input(
        self,
        prompt: str,
        chat_history: list[BaseMessage],
        model: str,
        user_id: str = "local",
    ) -> dict[str, Any]:
        with self._vector_lock:
            vector_db = self._get_vector_db_locked(user_id)
        return {
            "messages": chat_history + [HumanMessage(content=prompt)],
            "router_model": ROUTER_MODEL,
            "planner_model": model or DEFAULT_MODEL,
            "vector_db": vector_db,
        }

    def iter_graph_events(
        self,
        prompt: str,
        chat_history: list[BaseMessage],
        model: str = DEFAULT_MODEL,
        user_id: str = "local",
    ):
        runtime = _init_node_runtime()
        started_at = time.perf_counter()
        final_text = ""
        latest_intent = ""
        streaming_node = ""
        streaming_answer = ""

        _mark_running(runtime, "router")
        yield {
            "type": "runtime",
            "runtime": _runtime_snapshot(runtime),
            "elapsed": 0.0,
        }

        try:
            graph_input = self._graph_input(prompt, chat_history, model, user_id)
            for event in self.travel_graph.stream(graph_input, stream_mode=["updates", "custom"]):
                split_event = _split_graph_stream_event(event)
                if split_event is None:
                    continue

                mode, data = split_event

                if mode == "custom":
                    payload = data if isinstance(data, dict) else {"type": "custom", "data": data}
                    payload_type = str(payload.get("type") or "custom")
                    elapsed = time.perf_counter() - started_at

                    if payload_type == "message_delta":
                        delta_node = str(payload.get("node") or "planner")
                        reset_stream = delta_node != streaming_node
                        if delta_node != streaming_node:
                            streaming_node = delta_node
                            streaming_answer = ""
                            if delta_node == "planner":
                                final_text = ""

                        delta = str(payload.get("delta") or "")
                        if delta:
                            streaming_answer += delta
                            if delta_node == "planner":
                                final_text = streaming_answer
                        yield {
                            "type": "message_delta",
                            "node": delta_node,
                            "delta": delta,
                            "reset": reset_stream,
                            "runtime": _runtime_snapshot(runtime),
                            "elapsed": round(elapsed, 3),
                        }
                    else:
                        yield {
                            "type": payload_type,
                            "data": payload.get("data", data),
                            "runtime": _runtime_snapshot(runtime),
                            "elapsed": round(elapsed, 3),
                        }
                    continue

                if mode != "updates" or not isinstance(data, dict) or not data:
                    continue

                for node_name, update in data.items():
                    if node_name not in runtime or not isinstance(update, dict):
                        continue

                    _mark_completed(runtime, node_name, _build_node_note(node_name, update))

                    if node_name == "router":
                        latest_intent = str(update.get("intent") or "").strip().lower()
                        if latest_intent == "need_ticket":
                            _mark_running(runtime, "ticket_agent")
                            _mark_skipped(runtime, "researcher", "由路由策略跳过")
                            _mark_skipped(runtime, "planner", "由路由策略跳过")
                        elif latest_intent in {"need_plan", "need_answer"}:
                            _mark_running(runtime, "researcher")
                            _mark_skipped(runtime, "ticket_agent", "由路由策略跳过")
                        else:
                            _mark_skipped(runtime, "researcher", "由路由策略跳过")
                            _mark_skipped(runtime, "planner", "由路由策略跳过")
                            _mark_skipped(runtime, "ticket_agent", "由路由策略跳过")
                    elif node_name == "researcher":
                        if latest_intent == "need_plan":
                            _mark_running(runtime, "planner")
                        else:
                            _mark_skipped(runtime, "planner", "由问答策略跳过")

                    maybe_text = _extract_ai_text(update.get("messages"))
                    if maybe_text:
                        final_text = maybe_text

                    elapsed = time.perf_counter() - started_at
                    title_candidate = _build_title_from_node_update(node_name, update)
                    yield {
                        "type": "node_update",
                        "node": node_name,
                        "answer": maybe_text or None,
                        "title_candidate": title_candidate or None,
                        "state_update": _public_state_update(node_name, update) or None,
                        "runtime": _runtime_snapshot(runtime),
                        "elapsed": round(elapsed, 3),
                    }

        except Exception as exc:
            _mark_first_running_failed(runtime, f"运行异常：{exc}")
            elapsed = time.perf_counter() - started_at
            yield {
                "type": "error",
                "detail": str(exc),
                "runtime": _runtime_snapshot(runtime),
                "elapsed": round(elapsed, 3),
            }
            raise

        if not final_text:
            final_text = "节点执行已完成，但没有返回可展示文本。请重试一次。"

        total_elapsed = time.perf_counter() - started_at
        yield {
            "type": "final",
            "answer": final_text,
            "runtime": _runtime_snapshot(runtime),
            "elapsed": round(total_elapsed, 3),
        }

    def chat(
        self,
        *,
        message: str,
        session_id: str | None = None,
        model: str = DEFAULT_MODEL,
        history: list[dict[str, str]] | None = None,
        uploads: list[UploadedFileData] | None = None,
        persist: bool = True,
        user_id: str = "local",
        display_message: str | None = None,
    ) -> dict[str, Any]:
        prompt = self.build_prompt(message, uploads)
        stored_user_message = (display_message or "").strip() or prompt
        resolved_session_id: str | None = None

        if history is None:
            if persist:
                resolved_session_id, previous_messages = self._get_or_create_session_for_chat(session_id, user_id)
            else:
                previous_messages = []
        else:
            previous_messages = _sanitize_messages(history)
            if persist:
                resolved_session_id, _ = self._get_or_create_session_for_chat(session_id, user_id)

        chat_history = to_langchain_history(previous_messages)
        if persist and resolved_session_id:
            self._append_message(resolved_session_id, "user", stored_user_message, user_id)

        node_events: list[dict[str, Any]] = []
        final_payload: dict[str, Any] | None = None
        for event in self.iter_graph_events(prompt, chat_history, model or DEFAULT_MODEL, user_id):
            if event["type"] == "node_update":
                node_events.append(event)
                if persist and resolved_session_id:
                    self._apply_session_title_candidate(
                        resolved_session_id,
                        str(event.get("title_candidate") or ""),
                        user_id,
                    )
            elif event["type"] == "final":
                final_payload = event

        if final_payload is None:
            raise TravelServiceError("工作流未返回最终结果。")

        answer = str(final_payload.get("answer") or "")
        if persist and resolved_session_id:
            self._append_message(resolved_session_id, "assistant", answer, user_id)

        return {
            "session_id": resolved_session_id,
            "message": answer,
            "runtime": final_payload.get("runtime"),
            "elapsed": final_payload.get("elapsed"),
            "events": node_events,
        }

    def stream_chat(
        self,
        *,
        message: str,
        session_id: str | None = None,
        model: str = DEFAULT_MODEL,
        history: list[dict[str, str]] | None = None,
        persist: bool = True,
        user_id: str = "local",
        display_message: str | None = None,
    ):
        prompt = self.build_prompt(message)
        stored_user_message = (display_message or "").strip() or prompt
        resolved_session_id: str | None = None

        if history is None:
            if persist:
                resolved_session_id, previous_messages = self._get_or_create_session_for_chat(session_id, user_id)
            else:
                previous_messages = []
        else:
            previous_messages = _sanitize_messages(history)
            if persist:
                resolved_session_id, _ = self._get_or_create_session_for_chat(session_id, user_id)

        chat_history = to_langchain_history(previous_messages)
        if persist and resolved_session_id:
            self._append_message(resolved_session_id, "user", stored_user_message, user_id)

        final_answer = ""
        try:
            for event in self.iter_graph_events(prompt, chat_history, model or DEFAULT_MODEL, user_id):
                event["session_id"] = resolved_session_id
                if event["type"] == "node_update" and persist and resolved_session_id:
                    self._apply_session_title_candidate(
                        resolved_session_id,
                        str(event.get("title_candidate") or ""),
                        user_id,
                    )
                if event["type"] == "final":
                    final_answer = str(event.get("answer") or "")
                yield event
        finally:
            if persist and resolved_session_id and final_answer:
                self._append_message(resolved_session_id, "assistant", final_answer, user_id)

    def build_regenerate_prompt(self, supplement: str = "") -> tuple[str, str]:
        supplement = (supplement or "").strip()
        display_message = "按补充要求重新生成一版行程。" if supplement else "重新生成一版不同的行程方案。"
        prompt_lines = [
            "这是一次行程重新生成请求。请基于本会话历史中已经确认的出行需求、资料搜集结果和上一版行程，重新生成一版不同的旅行方案。",
            "不要把这条内部指令当成用户新需求展示；不要重新要求用户补充已经在历史中确认过的信息。",
            "如果用户此前明确表示日期不限、随便哪天出发、都可以或由系统安排，请将出发日期视为“日期灵活”，不要反复追问具体日期。",
            "新版方案必须和上一版有可比较的差异：调整景点组合、动线顺序、餐饮选择、交通策略或节奏安排，而不是简单改写措辞。",
        ]
        if supplement:
            prompt_lines.append(f"本次补充要求：{supplement}")
        else:
            prompt_lines.append("用户没有给出新的偏好，请主动换一个合理方向，例如更轻松、更美食导向、更人文或更少转场，但仍遵守既有约束。")
        return "\n".join(prompt_lines), display_message

    def knowledge_status(self, user_id: str = "local") -> dict[str, Any]:
        with self._vector_lock:
            vector_db = self._get_vector_db_locked(user_id)
            if vector_db is None:
                return {"loaded": False, "chunk_count": 0}
            try:
                chunk_count = vector_db._collection.count()
            except Exception:
                chunk_count = 0
            return {"loaded": True, "chunk_count": chunk_count}

    def ingest_knowledge(
        self,
        files: list[UploadedFileData],
        selected_model: str = DEFAULT_MODEL,
        user_id: str = "local",
    ) -> dict[str, Any]:
        if not files:
            raise TravelServiceError("没有检测到上传文件。")
        with self._vector_lock:
            user_key = self._knowledge_user_key(user_id)
            vector_db = self._get_vector_db_locked(user_id)
            self._vector_dbs[user_key], result = ingest_documents(
                files,
                vector_db,
                selected_model,
                self._knowledge_persist_path(user_id),
            )
            status = self.knowledge_status(user_id)
        return {**result, **status}

    def ingest_xhs_url(
        self,
        url: str,
        selected_model: str = DEFAULT_MODEL,
        user_id: str = "local",
    ) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            wait_seconds = int(XHS_IMPORT_COOLDOWN_SECONDS - (now - self._last_xhs_import_at))
            if wait_seconds > 0:
                raise TravelServiceError(f"小红书导入已进入保护间隔，请 {wait_seconds} 秒后再试。")
            self._last_xhs_import_at = now

        try:
            note = fetch_xhs_note_as_text(url)
        except XhsImportError as exc:
            raise TravelServiceError(str(exc)) from exc

        note_id = note.get("note_id") or "note"
        title = note.get("title") or "小红书笔记"
        upload = UploadedFileData(
            name=f"xhs_{note_id}.txt",
            content=str(note.get("text") or "").encode("utf-8"),
            content_type="text/plain",
        )
        result = self.ingest_knowledge([upload], selected_model=selected_model, user_id=user_id)
        return {
            **result,
            "source": "xhs",
            "note_id": note_id,
            "title": title,
            "url": note.get("source_url") or url,
            "message": f"已导入小红书笔记：{title}。{result.get('message', '')}",
        }

    def clear_knowledge(self, user_id: str = "local") -> dict[str, Any]:
        with self._vector_lock:
            user_key = self._knowledge_user_key(user_id)
            vector_db = self._get_vector_db_locked(user_id)
            success, message = clear_database(vector_db, self._knowledge_persist_path(user_id))
            self._vector_dbs[user_key] = None
        return {"success": success, "message": message, **self.knowledge_status(user_id)}
