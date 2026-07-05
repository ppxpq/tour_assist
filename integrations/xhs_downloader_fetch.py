from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any


def print_json(payload: dict[str, Any], exit_code: int = 0) -> None:
    print(json.dumps(payload, ensure_ascii=False))
    raise SystemExit(exit_code)


def normalize_item(item: dict[str, Any], source_url: str) -> dict[str, Any]:
    tags = [
        tag.strip()
        for tag in str(item.get("作品标签") or "").replace("#", " ").split()
        if tag.strip()
    ]
    return {
        "success": True,
        "source": "xhs_downloader",
        "url": item.get("作品链接") or source_url,
        "note_id": item.get("作品ID") or "",
        "note_card": {
            "note_id": item.get("作品ID") or "",
            "title": item.get("作品标题") or "",
            "desc": item.get("作品描述") or "",
            "time": item.get("时间戳") or "",
            "tag_list": [{"name": tag} for tag in tags],
            "interact_info": {
                "liked_count": item.get("点赞数量") or "",
                "collected_count": item.get("收藏数量") or "",
                "comment_count": item.get("评论数量") or "",
                "share_count": item.get("分享数量") or "",
            },
            "user": {
                "nickname": item.get("作者昵称") or "",
                "user_id": item.get("作者ID") or "",
            },
            "type": item.get("作品类型") or "",
            "image_list": item.get("下载地址") or [],
        },
        "raw": item,
    }


async def fetch(project_path: Path, url: str, cookie: str, proxy: str, timeout: int) -> dict[str, Any]:
    sys.path.insert(0, str(project_path))
    try:
        from source import XHS
    except ModuleNotFoundError as exc:
        return {
            "success": False,
            "error": f"XHS-Downloader 依赖缺失：{exc.name}。请执行 .venv/bin/pip install -r integrations/XHS-Downloader/requirements.txt",
        }

    work_path = Path(os.getenv("XHS_DOWNLOADER_WORK_PATH", str(project_path / "runtime"))).expanduser()
    work_path.mkdir(parents=True, exist_ok=True)
    async with XHS(
        work_path=str(work_path),
        folder_name="Download",
        cookie=cookie,
        proxy=proxy or None,
        timeout=timeout,
        max_retry=2,
        record_data=False,
        image_download=False,
        video_download=False,
        live_download=False,
        download_record=False,
        language="zh_CN",
    ) as xhs:
        data = await xhs.extract(url, download=False, data=True)

    if not data:
        return {"success": False, "error": "XHS-Downloader 未获取到笔记数据。"}
    item = next((entry for entry in data if isinstance(entry, dict) and entry.get("作品ID")), None)
    if not item:
        return {"success": False, "error": "XHS-Downloader 返回数据为空或格式异常。", "raw": data}
    return normalize_item(item, url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch one XHS note through embedded XHS-Downloader.")
    parser.add_argument("--project", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--cookie", default=os.getenv("XHS_COOKIE", ""))
    parser.add_argument("--proxy", default=os.getenv("XHS_PROXY", ""))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("XHS_TIMEOUT", "12")))
    args = parser.parse_args()

    project_path = Path(args.project).expanduser().resolve()
    if not project_path.exists():
        print_json({"success": False, "error": f"未找到 XHS-Downloader 项目：{project_path}"}, 2)

    try:
        payload = asyncio.run(fetch(project_path, args.url, args.cookie, args.proxy, args.timeout))
    except Exception as exc:
        payload = {
            "success": False,
            "error": str(exc) or exc.__class__.__name__,
            "name": exc.__class__.__name__,
        }
    print_json(payload, 0 if payload.get("success") else 2)


if __name__ == "__main__":
    main()
