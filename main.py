from __future__ import annotations

import json
import os
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.travel_service import DEFAULT_MODEL, TravelService, TravelServiceError, UploadedFileData
from utils import config


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(default="", description="用户本轮输入")
    session_id: str | None = Field(default=None, description="后端会话 ID，不传则自动创建")
    model: str = Field(default=DEFAULT_MODEL, description="Planner 使用的模型")
    history: list[ChatMessage] | None = Field(default=None, description="前端自管历史，可不使用后端会话")
    save_to_session: bool = Field(default=True, description="是否写入后端会话")


class SessionCreateRequest(BaseModel):
    title: str | None = None


def _cors_origins() -> list[str]:
    raw = os.getenv("BACKEND_CORS_ORIGINS", "*")
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["*"]


app = FastAPI(
    title="Travel Assistant API",
    description="FastAPI backend for the multi-agent travel planning system.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = TravelService()


def _as_history(history: list[ChatMessage] | None) -> list[dict[str, str]] | None:
    if history is None:
        return None
    return [
        item.model_dump() if hasattr(item, "model_dump") else item.dict()
        for item in history
    ]


async def _read_uploads(files: list[UploadFile] | None) -> list[UploadedFileData]:
    uploads: list[UploadedFileData] = []
    for file in files or []:
        content = await file.read()
        uploads.append(
            UploadedFileData(
                name=file.filename or "upload.bin",
                content=content,
                content_type=file.content_type or "",
            )
        )
    return uploads


def _handle_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="会话不存在。")
    if isinstance(exc, TravelServiceError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Travel Assistant API",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "knowledge_base": service.knowledge_status(),
    }


@app.get("/api/models")
def models() -> dict:
    return {
        "default_model": DEFAULT_MODEL,
        "models": config.MODEL_LIST,
    }


@app.get("/api/sessions")
def list_sessions() -> dict:
    return {"sessions": service.list_sessions()}


@app.post("/api/sessions")
def create_session(payload: SessionCreateRequest) -> dict:
    return {"session": service.create_session(payload.title)}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    try:
        return {"session": service.get_session(session_id)}
    except Exception as exc:
        raise _handle_error(exc)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    try:
        return service.delete_session(session_id)
    except Exception as exc:
        raise _handle_error(exc)


@app.delete("/api/sessions/{session_id}/messages")
def clear_session(session_id: str) -> dict:
    try:
        return {"session": service.clear_session(session_id)}
    except Exception as exc:
        raise _handle_error(exc)


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    try:
        return service.chat(
            message=payload.message,
            session_id=payload.session_id,
            model=payload.model,
            history=_as_history(payload.history),
            persist=payload.save_to_session,
        )
    except Exception as exc:
        raise _handle_error(exc)


@app.post("/api/chat/files")
async def chat_with_files(
    message: str = Form(default=""),
    session_id: str | None = Form(default=None),
    model: str = Form(default=DEFAULT_MODEL),
    save_to_session: bool = Form(default=True),
    files: list[UploadFile] | None = File(default=None),
) -> dict:
    try:
        uploads = await _read_uploads(files)
        return service.chat(
            message=message,
            session_id=session_id,
            model=model,
            uploads=uploads,
            persist=save_to_session,
        )
    except Exception as exc:
        raise _handle_error(exc)


@app.post("/api/chat/stream")
def stream_chat(payload: ChatRequest) -> StreamingResponse:
    def event_source():
        try:
            for event in service.stream_chat(
                message=payload.message,
                session_id=payload.session_id,
                model=payload.model,
                history=_as_history(payload.history),
                persist=payload.save_to_session,
            ):
                yield _sse(event["type"], event)
        except Exception as exc:
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/knowledge-base/status")
def knowledge_status() -> dict:
    return service.knowledge_status()


@app.post("/api/knowledge-base/files")
async def ingest_knowledge_base(
    files: list[UploadFile] = File(...),
    model: str = Form(default=DEFAULT_MODEL),
) -> dict:
    try:
        uploads = await _read_uploads(files)
        return service.ingest_knowledge(uploads, selected_model=model)
    except Exception as exc:
        raise _handle_error(exc)


@app.delete("/api/knowledge-base")
def clear_knowledge_base() -> dict:
    try:
        return service.clear_knowledge()
    except Exception as exc:
        raise _handle_error(exc)
