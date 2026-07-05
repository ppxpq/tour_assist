from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

from utils import config


class XhsImportError(RuntimeError):
    pass


NOTE_ID_RE = re.compile(r"^[0-9a-fA-F]{24}$")
NOTE_PATH_RE = re.compile(r"/(?:explore|discovery/item)/([0-9a-fA-F]{24})")


def _resolve_url(url: str) -> str:
    candidate = (url or "").strip()
    if not candidate:
        raise XhsImportError("请粘贴小红书笔记链接。")

    parsed = urlparse(candidate)
    if not parsed.scheme:
        candidate = f"https://{candidate}"
        parsed = urlparse(candidate)

    if "xhslink.com" not in parsed.netloc.lower():
        return candidate

    try:
        response = requests.get(candidate, allow_redirects=True, timeout=8)
        return response.url or candidate
    except Exception:
        return candidate


def _parse_xhs_url(url: str) -> tuple[str, str, str]:
    resolved_url = _resolve_url(url)
    parsed = urlparse(resolved_url)
    query = parse_qs(parsed.query)
    xsec_token = (query.get("xsec_token") or [""])[0]

    match = NOTE_PATH_RE.search(parsed.path or "")
    if match:
        return match.group(1), xsec_token, resolved_url

    text = f"{parsed.path} {parsed.query}"
    for token in re.findall(r"[0-9a-fA-F]{24}", text):
        if NOTE_ID_RE.match(token):
            return token, xsec_token, resolved_url

    if NOTE_ID_RE.match((url or "").strip()):
        return url.strip(), xsec_token, resolved_url

    raise XhsImportError("无法从链接中解析小红书 note_id，请确认是笔记详情页链接。")


def _run_fetch_script(note_id: str, xsec_token: str, source_url: str) -> dict[str, Any]:
    project_path = Path(config.XHS_PROJECT_PATH).expanduser()
    if not project_path.exists():
        raise XhsImportError(f"未找到小红书工具项目：{project_path}")

    script_path = Path(config.BASE_DIR) / "integrations" / "xhs_fetch_note.mjs"
    if not script_path.exists():
        raise XhsImportError("缺少小红书导入脚本 integrations/xhs_fetch_note.mjs。")

    command = [
        "node",
        str(script_path),
        "--project",
        str(project_path),
        "--note-id",
        note_id,
        "--xsec-token",
        xsec_token,
        "--url",
        source_url,
    ]

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=25,
            env={**os.environ, "NO_COLOR": "1"},
        )
    except FileNotFoundError as exc:
        raise XhsImportError("未找到 node 命令，请先安装 Node.js。") from exc
    except subprocess.TimeoutExpired as exc:
        raise XhsImportError("小红书笔记抓取超时，请稍后重试。") from exc

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    try:
        payload = json.loads(stdout.splitlines()[-1]) if stdout else {}
    except json.JSONDecodeError as exc:
        raise XhsImportError(f"小红书脚本返回异常：{stderr or stdout or exc}") from exc

    if completed.returncode != 0 or not payload.get("success"):
        message = payload.get("error") or stderr or "小红书笔记抓取失败。"
        raise XhsImportError(message)

    return payload


def _resolve_project_path(value: str) -> Path:
    path = Path(value or "").expanduser()
    if not path.is_absolute():
        path = Path(config.BASE_DIR) / path
    return path


def _run_xhs_downloader(source_url: str) -> dict[str, Any]:
    project_path = _resolve_project_path(getattr(config, "XHS_DOWNLOADER_PATH", ""))
    if not project_path.exists():
        raise XhsImportError(f"未找到内嵌 XHS-Downloader：{project_path}")

    script_path = Path(config.BASE_DIR) / "integrations" / "xhs_downloader_fetch.py"
    if not script_path.exists():
        raise XhsImportError("缺少小红书导入脚本 integrations/xhs_downloader_fetch.py。")

    command = [
        str(Path(config.BASE_DIR) / ".venv" / "bin" / "python"),
        str(script_path),
        "--project",
        str(project_path),
        "--url",
        source_url,
    ]
    cookie = os.getenv("XHS_COOKIE", "").strip()
    proxy = os.getenv("XHS_PROXY", "").strip()
    if cookie:
        command.extend(["--cookie", cookie])
    if proxy:
        command.extend(["--proxy", proxy])

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=35,
            env={**os.environ, "NO_COLOR": "1"},
        )
    except FileNotFoundError as exc:
        raise XhsImportError("未找到项目虚拟环境 Python，请先创建 .venv。") from exc
    except subprocess.TimeoutExpired as exc:
        raise XhsImportError("XHS-Downloader 抓取超时，请稍后重试。") from exc

    stdout = (completed.stdout or "").strip()
    stderr = (completed.stderr or "").strip()
    try:
        payload = json.loads(stdout.splitlines()[-1]) if stdout else {}
    except json.JSONDecodeError as exc:
        raise XhsImportError(f"XHS-Downloader 返回异常：{stderr or stdout or exc}") from exc

    if completed.returncode != 0 or not payload.get("success"):
        message = payload.get("error") or stderr or "XHS-Downloader 抓取失败。"
        raise XhsImportError(message)

    return payload


def _format_count(value: Any) -> str:
    text = str(value or "").strip()
    return text or "0"


def _format_timestamp(value: Any) -> str:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return ""
    if number > 10_000_000_000:
        number = number // 1000
    try:
        return datetime.fromtimestamp(number).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def _normalize_note(payload: dict[str, Any]) -> dict[str, Any]:
    card = payload.get("note_card") or {}
    interact = card.get("interact_info") or {}
    user = card.get("user") or {}
    tags = [
        str(tag.get("name") or "").strip()
        for tag in card.get("tag_list") or []
        if isinstance(tag, dict) and str(tag.get("name") or "").strip()
    ]
    images = card.get("image_list") or []

    title = str(card.get("title") or "").strip() or "小红书笔记"
    desc = str(card.get("desc") or "").strip()
    note_id = str(card.get("note_id") or payload.get("note_id") or "").strip()
    source_url = str(payload.get("url") or "").strip()

    text = "\n".join(
        part
        for part in [
            "【来源】小红书笔记",
            f"【链接】{source_url}" if source_url else "",
            f"【笔记ID】{note_id}" if note_id else "",
            f"【标题】{title}",
            f"【作者】{user.get('nickname') or '未知'}",
            f"【发布时间】{_format_timestamp(card.get('time'))}" if card.get("time") else "",
            f"【IP属地】{card.get('ip_location')}" if card.get("ip_location") else "",
            f"【标签】{'、'.join(tags)}" if tags else "",
            (
                "【互动】"
                f"点赞 {_format_count(interact.get('liked_count'))}，"
                f"收藏 {_format_count(interact.get('collected_count'))}，"
                f"评论 {_format_count(interact.get('comment_count'))}"
            ),
            f"【图片数量】{len(images)}" if images else "",
            "",
            "【正文】",
            desc or "（无正文）",
            "",
            "【适合规划参考】",
            "- 这是用户主动提供的小红书攻略资料。",
            "- 可用于景点推荐、避坑提醒、餐饮选择、拍照点和行程节奏判断。",
        ]
        if part
    )

    return {
        "note_id": note_id,
        "title": title,
        "text": text,
        "source_url": source_url,
    }


def fetch_xhs_note_as_text(url: str) -> dict[str, Any]:
    backend = str(getattr(config, "XHS_IMPORT_BACKEND", "xhs_downloader") or "xhs_downloader").strip()
    if backend == "xhs_downloader":
        source_url = _resolve_url(url)
        try:
            payload = _run_xhs_downloader(source_url)
            return _normalize_note(payload)
        except XhsImportError:
            if str(os.getenv("XHS_ALLOW_LEGACY_FALLBACK", "")).lower() not in {"1", "true", "yes", "on"}:
                raise

    note_id, xsec_token, resolved_url = _parse_xhs_url(url)
    payload = _run_fetch_script(note_id, xsec_token, resolved_url)
    return _normalize_note(payload)
