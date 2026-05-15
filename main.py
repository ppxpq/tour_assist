
import time
import re
import uuid
import json
from pathlib import Path
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

import utils.config as config
from agents.graph import build_travel_graph
from core.db_manager import clear_database, ingest_documents, load_db
from core.llm_core import get_llm
from UI import (
    render_page_title,
    render_sidebar,
    build_unified_chat_input,
    render_chat_history,
)


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────
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
STATUS_COLOR = {
    "pending": "#E2E8F0",
    "running": "#FDE68A",
    "completed": "#86EFAC",
    "skipped": "#CBD5E1",
    "failed": "#FCA5A5",
}
MEDIA_PATH_RE = re.compile(
    r"路径[:：]\s*([^\n，,]+?\.(?:jpg|jpeg|png|webp|bmp|gif|mp3|wav|m4a|ogg|webm|flac|aac))",
    re.IGNORECASE,
)


def save_to_upload_store(uploaded_file) -> str:
    """把 Streamlit UploadedFile 存到受管上传目录，返回路径字符串。"""
    suffix = Path(uploaded_file.name).suffix or ".bin"
    upload_dir = Path(config.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{int(time.time())}_{uuid.uuid4().hex}{suffix.lower()}"
    saved_path = upload_dir / filename
    saved_path.write_bytes(uploaded_file.getvalue())
    return str(saved_path)


def is_managed_upload_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(Path(config.UPLOAD_DIR).resolve())
        return True
    except Exception:
        return False


def cleanup_unmanaged_temp_files(paths: list[str]) -> None:
    """清理非受管临时文件；受管上传文件保留到会话清理时删除。"""
    for path in paths:
        try:
            file_path = Path(path)
            if not is_managed_upload_path(file_path):
                file_path.unlink(missing_ok=True)
        except Exception:
            pass


def cleanup_session_media(messages: list[dict[str, str]]) -> None:
    """删除当前会话历史中引用的受管上传文件。"""
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
                if is_managed_upload_path(path):
                    path.unlink(missing_ok=True)
            except Exception:
                pass


def rename_current_session(prompt: str):
    current_history = st.session_state.sessions[st.session_state.current_session]
    if len(current_history) != 2 or not current_history[-1]["content"].strip():
        return
    rename_prompt = (
        "根据用户的消息，为这段对话生成一个标题。\n"
        "要求：5-8个汉字，名词短语，提炼话题，无标点，只输出标题\n"
        f"用户消息：{prompt}"
    )
    llm = get_llm(st.session_state.current_model)
    try:
        new_title = llm.invoke([HumanMessage(content=rename_prompt)]).content.strip(' "\'\n，。！？、*')
    except Exception:
        new_title = prompt[:8]
    if not new_title:
        new_title = prompt[:8]
    old = st.session_state.current_session
    unique = new_title
    suffix = 1
    while unique in st.session_state.sessions:
        unique = f"{new_title}_{suffix}"
        suffix += 1
    renamed = {(unique if k == old else k): v for k, v in st.session_state.sessions.items()}
    st.session_state.sessions = renamed
    st.session_state.current_session = unique
    save_sessions()
    st.rerun()


def set_kb_notice(level: str, msg: str):
    st.session_state.kb_notice = {"level": level, "message": msg, "at": time.strftime("%H:%M:%S")}


def get_file_extension(f):
    name = getattr(f, "name", "").lower()
    return name.rsplit(".", 1)[-1] if "." in name else ""


def is_image_file(f):
    mime = getattr(f, "type", "").lower()
    ext = get_file_extension(f)
    return mime.startswith("image/") or ext in {"jpg", "jpeg", "png", "webp", "bmp", "gif"}


def is_audio_file(f):
    mime = getattr(f, "type", "").lower()
    ext = get_file_extension(f)
    return mime.startswith("audio/") or ext in {"mp3", "wav", "m4a", "ogg", "flac", "aac", "webm"}


def payload_value(payload, key, default=None):
    if payload is None:
        return default
    if hasattr(payload, key):
        return getattr(payload, key) or default
    if isinstance(payload, dict):
        return payload.get(key, default)
    try:
        return payload[key]
    except Exception:
        return default


def parse_chat_submission(submission):
    if isinstance(submission, str):
        return submission, [], None
    text  = payload_value(submission, "text",  "")
    files = payload_value(submission, "files", [])
    audio = payload_value(submission, "audio", None)
    files = [f for f in ([files] if not isinstance(files, list) else files) if f]
    return text, files, audio


def to_langchain_history(messages: list[dict[str, str]]) -> list[BaseMessage]:
    history: list[BaseMessage] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            history.append(HumanMessage(content=content))
        elif role == "assistant":
            history.append(AIMessage(content=content))
    return history


def _init_node_runtime() -> dict[str, dict[str, Any]]:
    return {
        node: {
            "status": "pending",
            "start": None,
            "end": None,
            "duration": 0.0,
            "note": "-",
        }
        for node in NODE_ORDER
    }


def _mark_running(runtime: dict[str, dict[str, Any]], node: str) -> None:
    item = runtime[node]
    if item["status"] in {"completed", "skipped", "failed"}:
        return
    if item["start"] is None:
        item["start"] = time.perf_counter()
    item["status"] = "running"


def _mark_completed(runtime: dict[str, dict[str, Any]], node: str, note: str) -> None:
    item = runtime[node]
    if item["start"] is None:
        item["start"] = time.perf_counter()
    item["end"] = time.perf_counter()
    item["duration"] = max(0.0, item["end"] - item["start"])
    item["status"] = "completed"
    item["note"] = note or "-"


def _mark_skipped(runtime: dict[str, dict[str, Any]], node: str, note: str) -> None:
    item = runtime[node]
    if item["status"] in {"completed", "failed"}:
        return
    if item["status"] == "running" and item["start"] is not None:
        item["end"] = time.perf_counter()
        item["duration"] = max(0.0, item["end"] - item["start"])
    item["status"] = "skipped"
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


def _build_node_note(node: str, update: dict[str, Any]) -> str:
    if not isinstance(update, dict):
        return "-"

    if node == "router":
        intent = update.get("intent") or "-"
        missing = update.get("missing_fields") or []
        reason = update.get("router_reason") or "-"
        note = f"intent={intent}"
        if missing:
            note += f"，缺失={','.join(missing)}"
        note += f"，来源={reason}"
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
        return f"{departure}→{city} {date}"

    return "-"


def _build_runtime_dot(runtime: dict[str, dict[str, Any]]) -> str:
    lines = [
        "digraph TravelGraph {",
        "rankdir=LR;",
        "graph [fontname=\"Microsoft YaHei\", bgcolor=transparent];",
        "node [shape=box, style=\"rounded,filled\", fontname=\"Microsoft YaHei\", color=\"#334155\"];",
        "edge [color=\"#64748B\"];",
    ]
    for node in NODE_ORDER:
        status = runtime[node]["status"]
        title = NODE_TITLE[node]
        label = STATUS_LABEL.get(status, status)
        fill = STATUS_COLOR.get(status, "#E2E8F0")
        lines.append(f"{node} [label=\"{title}\\n{label}\", fillcolor=\"{fill}\"];")
    lines.append("end [label=\"END\", shape=oval, style=\"filled\", fillcolor=\"#F8FAFC\", color=\"#94A3B8\"];")
    lines.append("router -> ticket_agent [label=\"need_ticket\"];")
    lines.append("router -> researcher [label=\"need_plan / need_answer\"];")
    lines.append("router -> end [label=\"other\"];")
    lines.append("researcher -> planner [label=\"need_plan\"];")
    lines.append("researcher -> end [label=\"need_answer\"];")
    lines.append("planner -> end;")
    lines.append("ticket_agent -> end;")
    lines.append("}")
    return "\n".join(lines)


def _render_runtime_table(
    placeholder,
    runtime: dict[str, dict[str, Any]],
    total_elapsed: float,
) -> None:
    rows = [
        "| 节点 | 状态 | 耗时(秒) | 说明 |",
        "|---|---|---:|---|",
    ]
    for node in NODE_ORDER:
        item = runtime[node]
        status = STATUS_LABEL.get(item["status"], item["status"])
        duration = f"{item['duration']:.2f}" if item["duration"] else "-"
        rows.append(f"| {NODE_TITLE[node]} | {status} | {duration} | {item['note']} |")
    rows.append(f"\n总耗时：{total_elapsed:.2f} 秒")
    placeholder.markdown("\n".join(rows))


def run_travel_graph(
    prompt: str,
    chat_history: list[BaseMessage],
    answer_placeholder,
    graph_placeholder,
    table_placeholder,
) -> tuple[str, dict[str, dict[str, Any]], float]:
    runtime = _init_node_runtime()
    started_at = time.perf_counter()
    final_text = ""
    latest_intent = ""

    _mark_running(runtime, "router")
    graph_placeholder.graphviz_chart(_build_runtime_dot(runtime), use_container_width=True)
    _render_runtime_table(table_placeholder, runtime, 0.0)

    graph_input = {
        "messages": chat_history + [HumanMessage(content=prompt)],
        "router_model": "glm-4-flash",
        "planner_model": st.session_state.current_model,
        "vector_db": st.session_state.vector_db,
    }

    try:
        for event in st.session_state.travel_graph.stream(graph_input, stream_mode="updates"):
            if not isinstance(event, dict) or not event:
                continue
            node_name, update = next(iter(event.items()))
            if node_name not in runtime:
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
                answer_placeholder.markdown(final_text + " ▌")

            elapsed = time.perf_counter() - started_at
            graph_placeholder.graphviz_chart(_build_runtime_dot(runtime), use_container_width=True)
            _render_runtime_table(table_placeholder, runtime, elapsed)

    except Exception as exc:
        _mark_first_running_failed(runtime, f"运行异常：{exc}")
        elapsed = time.perf_counter() - started_at
        graph_placeholder.graphviz_chart(_build_runtime_dot(runtime), use_container_width=True)
        _render_runtime_table(table_placeholder, runtime, elapsed)
        raise

    if not final_text:
        final_text = "节点执行已完成，但没有返回可展示文本。请重试一次。"

    answer_placeholder.markdown(final_text)
    total_elapsed = time.perf_counter() - started_at
    _render_runtime_table(table_placeholder, runtime, total_elapsed)
    return final_text, runtime, total_elapsed


# ──────────────────────────────────────────────
# 初始化
# ──────────────────────────────────────────────
SESSION_STORE_FILE = Path(config.BASE_DIR) / "data" / "sessions.json"

def load_sessions():
    if SESSION_STORE_FILE.exists():
        try:
            with open(SESSION_STORE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("sessions", {}), data.get("current_session", ""), data.get("session_counter", 1)
        except Exception:
            pass
    return None, None, None

def save_sessions():
    try:
        SESSION_STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "sessions": st.session_state.sessions,
            "current_session": st.session_state.current_session,
            "session_counter": st.session_state.session_counter
        }
        with open(SESSION_STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save sessions: {e}")

config.init_env()
render_page_title()

if "sessions" not in st.session_state:
    saved_sessions, saved_current, saved_counter = load_sessions()
    if saved_sessions:
        st.session_state.sessions = saved_sessions
        st.session_state.current_session = saved_current or list(saved_sessions.keys())[0]
        st.session_state.session_counter = saved_counter or len(saved_sessions)
    else:
        st.session_state.session_counter = 1
        name = f"新会话{st.session_state.session_counter}"
        st.session_state.sessions = {name: []}
        st.session_state.current_session = name
if "vector_db" not in st.session_state:
    st.session_state.vector_db = load_db()
if "current_model" not in st.session_state:
    st.session_state.current_model = "glm-4.5-air"
if "travel_graph" not in st.session_state:
    st.session_state.travel_graph = build_travel_graph()
if "kb_notice" not in st.session_state:
    st.session_state.kb_notice = None
if "node_run_history" not in st.session_state:
    st.session_state.node_run_history = []

if st.session_state.get("needs_save", False):
    save_sessions()
    st.session_state.needs_save = False

current_messages = st.session_state.sessions[st.session_state.current_session]

# ──────────────────────────────────────────────
# 侧边栏交互
# ──────────────────────────────────────────────
(
    selected_model, selected_session,
    clear_clicked, delete_clicked,
    uploaded_files, ingest_clicked, clear_db_clicked,
) = render_sidebar(config)

if selected_model != st.session_state.current_model:
    st.session_state.current_model = selected_model
    st.toast(f"已切换模型：{selected_model}")

if selected_session != st.session_state.current_session:
    st.session_state.current_session = selected_session
    st.rerun()

if clear_clicked:
    cleanup_session_media(st.session_state.sessions[st.session_state.current_session])
    st.session_state.sessions[st.session_state.current_session] = []
    save_sessions()
    st.rerun()

if delete_clicked:
    cleanup_session_media(st.session_state.sessions[st.session_state.current_session])
    if len(st.session_state.sessions) > 1:
        del st.session_state.sessions[st.session_state.current_session]
        st.session_state.current_session = list(st.session_state.sessions.keys())[0]
    else:
        st.session_state.sessions[st.session_state.current_session] = []
    save_sessions()
    st.rerun()

if uploaded_files and ingest_clicked:
    with st.spinner(f"处理 {len(uploaded_files)} 个文件..."):
        st.session_state.vector_db, result = ingest_documents(
            uploaded_files, st.session_state.vector_db, selected_model
        )
        if result.get("success"):
            set_kb_notice("success", result.get("message"))
            st.toast("入库完成")
        else:
            set_kb_notice("error", result.get("message"))
        st.rerun()

if clear_db_clicked:
    with st.spinner("清理知识库..."):
        clear_database(st.session_state.vector_db)
        st.session_state.vector_db = None
        set_kb_notice("success", "知识库已清空")
        st.rerun()

# ──────────────────────────────────────────────
# 聊天界面
# ──────────────────────────────────────────────
render_chat_history(current_messages)
submission, supports_upload = build_unified_chat_input()
if not supports_upload:
    st.caption("升级 Streamlit 可开启图片/语音输入")

# ──────────────────────────────────────────────
# 消息处理 & 流式响应
# ──────────────────────────────────────────────
if submission:
    prompt = ""
    media_paths: list[str] = []
    try:
        prompt, files, recorded_audio = parse_chat_submission(submission)
        imgs   = [f for f in files if is_image_file(f)]
        audios = [f for f in files if is_audio_file(f)]
        if recorded_audio:
            audios.insert(0, recorded_audio)

        # 去重音频
        seen = set()
        audios = [a for a in audios if (a.name, a.size) not in seen and not seen.add((a.name, a.size))]

        img = imgs[0] if imgs else None

        # ── 把文件写入受管上传路径，交给 Agent 工具处理 ──────
        media_hints = []

        if img:
            img_path = save_to_upload_store(img)
            media_paths.append(img_path)
            media_hints.append(f"用户上传了一张图片，路径：{img_path}，请识别图中景点并介绍。")
            st.image(img, width=300)             # UI 展示

        if audios:
            for a in audios:
                audio_path = save_to_upload_store(a)
                media_paths.append(audio_path)
                media_hints.append(f"用户上传了语音文件，路径：{audio_path}，请先转文字再处理。")
                st.audio(a)                      # UI 展示

        # ── 拼最终 prompt ──────────────────────────────
        if media_hints:
            prompt = "\n".join(media_hints) + ("\n\n" + prompt if prompt else "")
        if not prompt:
            st.warning("请输入文字或上传图片/语音")
            st.stop()

        # ── 保存用户消息 ───────────────────────────────
        chat_history = to_langchain_history(current_messages)
        st.session_state.sessions[st.session_state.current_session].append(
            {"role": "user", "content": prompt}
        )
        save_sessions()
        st.chat_message("user").write(prompt)

        # ── AI 回复 ────────────────────────────────────
        st.session_state.sessions[st.session_state.current_session].append(
            {"role": "assistant", "content": ""}
        )
        ai_idx = len(st.session_state.sessions[st.session_state.current_session]) - 1
        save_sessions()

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            with st.expander("查看节点运行可视化", expanded=True):
                graph_placeholder = st.empty()
                table_placeholder = st.empty()

            try:
                full_text, runtime, total_elapsed = run_travel_graph(
                    prompt=prompt,
                    chat_history=chat_history,
                    answer_placeholder=answer_placeholder,
                    graph_placeholder=graph_placeholder,
                    table_placeholder=table_placeholder,
                )
                st.session_state.sessions[st.session_state.current_session][ai_idx]["content"] = full_text
                st.session_state.node_run_history.append(
                    {
                        "session": st.session_state.current_session,
                        "prompt": prompt,
                        "runtime": runtime,
                        "elapsed": total_elapsed,
                        "at": time.strftime("%H:%M:%S"),
                    }
                )
                st.session_state.node_run_history = st.session_state.node_run_history[-30:]                
                save_sessions()
            except Exception as exc:
                answer_placeholder.error(f"出错：{exc}")
                st.session_state.sessions[st.session_state.current_session][ai_idx]["content"] += "\n\n(出错)"
                save_sessions()
    finally:
        cleanup_unmanaged_temp_files(media_paths)

    if prompt:
        rename_current_session(prompt)
