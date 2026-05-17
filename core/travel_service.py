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
from utils import config


DEFAULT_MODEL = "glm-4.5-air"
ROUTER_MODEL = "glm-4-flash"

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


class TravelService:
    def __init__(self) -> None:
        config.init_env()
        self._lock = RLock()
        self._vector_lock = RLock()
        self.travel_graph = build_travel_graph()
        self.vector_db = load_db()
        self._session_store = SessionStore()
        self._ensure_default_session()

    def _ensure_default_session(self) -> None:
        with self._lock:
            self._session_store.ensure_default_session()

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        with self._lock:
            return self._session_store.create_session(title)

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._session_store.list_sessions()

    def get_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            return self._session_store.get_session(session_id)

    def delete_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            messages = self._session_store.get_session_messages(session_id)
            self._cleanup_session_media(messages)
            result = self._session_store.delete_session(session_id)

            if not result.get("current_session"):
                return self._session_store.create_session("新会话")
            return result

    def clear_session(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            messages = self._session_store.get_session_messages(session_id)
            self._cleanup_session_media(messages)
            return self._session_store.clear_session_messages(session_id)

    def _get_or_create_session_for_chat(self, session_id: str | None) -> tuple[str, list[dict[str, str]]]:
        with self._lock:
            if session_id and self._session_store.get_session_summary(session_id):
                resolved_session_id = session_id
                self._session_store.set_current_session(session_id)
            else:
                session = self._session_store.create_session()
                resolved_session_id = str(session["id"])

            return resolved_session_id, self._session_store.get_session_messages(resolved_session_id)

    def _append_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            summary = self._session_store.get_session_summary(session_id)
            if summary is None:
                raise KeyError(session_id)
            should_update_title = role == "user" and int(summary.get("message_count") or 0) == 0
            self._session_store.add_message(session_id, role, content)
            if should_update_title:
                self._session_store.update_session(session_id, title=_short_title(content))

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

    def _graph_input(self, prompt: str, chat_history: list[BaseMessage], model: str) -> dict[str, Any]:
        with self._vector_lock:
            vector_db = self.vector_db
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
            graph_input = self._graph_input(prompt, chat_history, model)
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
                    yield {
                        "type": "node_update",
                        "node": node_name,
                        "answer": maybe_text or None,
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
    ) -> dict[str, Any]:
        prompt = self.build_prompt(message, uploads)
        resolved_session_id: str | None = None

        if history is None:
            if persist:
                resolved_session_id, previous_messages = self._get_or_create_session_for_chat(session_id)
            else:
                previous_messages = []
        else:
            previous_messages = _sanitize_messages(history)
            if persist:
                resolved_session_id, _ = self._get_or_create_session_for_chat(session_id)

        chat_history = to_langchain_history(previous_messages)
        if persist and resolved_session_id:
            self._append_message(resolved_session_id, "user", prompt)

        node_events: list[dict[str, Any]] = []
        final_payload: dict[str, Any] | None = None
        for event in self.iter_graph_events(prompt, chat_history, model or DEFAULT_MODEL):
            if event["type"] == "node_update":
                node_events.append(event)
            elif event["type"] == "final":
                final_payload = event

        if final_payload is None:
            raise TravelServiceError("工作流未返回最终结果。")

        answer = str(final_payload.get("answer") or "")
        if persist and resolved_session_id:
            self._append_message(resolved_session_id, "assistant", answer)

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
    ):
        prompt = self.build_prompt(message)
        resolved_session_id: str | None = None

        if history is None:
            if persist:
                resolved_session_id, previous_messages = self._get_or_create_session_for_chat(session_id)
            else:
                previous_messages = []
        else:
            previous_messages = _sanitize_messages(history)
            if persist:
                resolved_session_id, _ = self._get_or_create_session_for_chat(session_id)

        chat_history = to_langchain_history(previous_messages)
        if persist and resolved_session_id:
            self._append_message(resolved_session_id, "user", prompt)

        final_answer = ""
        try:
            for event in self.iter_graph_events(prompt, chat_history, model or DEFAULT_MODEL):
                event["session_id"] = resolved_session_id
                if event["type"] == "final":
                    final_answer = str(event.get("answer") or "")
                yield event
        finally:
            if persist and resolved_session_id and final_answer:
                self._append_message(resolved_session_id, "assistant", final_answer)

    def knowledge_status(self) -> dict[str, Any]:
        with self._vector_lock:
            vector_db = self.vector_db
            if vector_db is None:
                return {"loaded": False, "chunk_count": 0}
            try:
                chunk_count = vector_db._collection.count()
            except Exception:
                chunk_count = 0
            return {"loaded": True, "chunk_count": chunk_count}

    def ingest_knowledge(self, files: list[UploadedFileData], selected_model: str = DEFAULT_MODEL) -> dict[str, Any]:
        if not files:
            raise TravelServiceError("没有检测到上传文件。")
        with self._vector_lock:
            self.vector_db, result = ingest_documents(files, self.vector_db, selected_model)
            status = self.knowledge_status()
        return {**result, **status}

    def clear_knowledge(self) -> dict[str, Any]:
        with self._vector_lock:
            success, message = clear_database(self.vector_db)
            self.vector_db = None
        return {"success": success, "message": message, **self.knowledge_status()}
