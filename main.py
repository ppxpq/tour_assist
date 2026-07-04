from __future__ import annotations

import json
import os
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.auth_store import AuthError, AuthStore
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


class RegenerateRequest(BaseModel):
    session_id: str | None = Field(default=None, description="后端会话 ID")
    model: str = Field(default=DEFAULT_MODEL, description="Planner 使用的模型")
    supplement: str = Field(default="", description="用户本次补充要求")


class SessionCreateRequest(BaseModel):
    title: str | None = None


class AuthRequest(BaseModel):
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., description="刷新令牌")


class LogoutRequest(BaseModel):
    refresh_token: str = Field(default="", description="刷新令牌")


class XhsUrlRequest(BaseModel):
    url: str = Field(..., description="小红书笔记 URL")
    model: str = Field(default=DEFAULT_MODEL, description="入库使用的模型配置")


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
auth_store = AuthStore()


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
    if isinstance(exc, AuthError):
        return HTTPException(status_code=401, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="会话不存在。")
    if isinstance(exc, TravelServiceError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=500, detail=str(exc))


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _bearer_token(authorization: str | None) -> str:
    value = (authorization or "").strip()
    prefix = "Bearer "
    if not value.startswith(prefix):
        raise AuthError("请先登录。")
    token = value[len(prefix) :].strip()
    if not token:
        raise AuthError("请先登录。")
    return token


def _access_token_from_header(authorization: str | None = Header(default=None)) -> str:
    try:
        return _bearer_token(authorization)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def current_user(access_token: str = Depends(_access_token_from_header)) -> dict:
    try:
        return auth_store.authenticate_access_token(access_token)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except Exception as exc:
        raise _handle_error(exc) from exc


def _user_agent(request: Request) -> str:
    return request.headers.get("user-agent", "")


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


@app.post("/api/auth/register")
def register(payload: AuthRequest, request: Request) -> dict:
    try:
        result = auth_store.register(payload.username, payload.password, _user_agent(request))
        service._ensure_default_session(str(result["user"]["id"]))
        return result
    except Exception as exc:
        raise _handle_error(exc)


@app.post("/api/auth/login")
def login(payload: AuthRequest, request: Request) -> dict:
    try:
        result = auth_store.login(payload.username, payload.password, _user_agent(request))
        service._ensure_default_session(str(result["user"]["id"]))
        return result
    except Exception as exc:
        raise _handle_error(exc)


@app.post("/api/auth/refresh")
def refresh_token(payload: RefreshRequest, request: Request) -> dict:
    try:
        return auth_store.refresh(payload.refresh_token, _user_agent(request))
    except Exception as exc:
        raise _handle_error(exc)


@app.post("/api/auth/logout")
def logout(
    payload: LogoutRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    try:
        access_token = ""
        if authorization:
            try:
                access_token = _bearer_token(authorization)
            except AuthError:
                access_token = ""
        auth_store.logout(payload.refresh_token, access_token or None)
        return {"success": True, "message": "已退出登录。"}
    except Exception as exc:
        raise _handle_error(exc)


@app.get("/api/auth/me")
def me(user: dict = Depends(current_user)) -> dict:
    return {"user": user}


@app.get("/api/sessions")
def list_sessions(user: dict = Depends(current_user)) -> dict:
    return {"sessions": service.list_sessions(str(user["id"]))}


@app.post("/api/sessions")
def create_session(payload: SessionCreateRequest, user: dict = Depends(current_user)) -> dict:
    return {"session": service.create_session(payload.title, user_id=str(user["id"]))}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str, user: dict = Depends(current_user)) -> dict:
    try:
        return {"session": service.get_session(session_id, user_id=str(user["id"]))}
    except Exception as exc:
        raise _handle_error(exc)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str, user: dict = Depends(current_user)) -> dict:
    try:
        return service.delete_session(session_id, user_id=str(user["id"]))
    except Exception as exc:
        raise _handle_error(exc)


@app.delete("/api/sessions/{session_id}/messages")
def clear_session(session_id: str, user: dict = Depends(current_user)) -> dict:
    try:
        return {"session": service.clear_session(session_id, user_id=str(user["id"]))}
    except Exception as exc:
        raise _handle_error(exc)


@app.post("/api/chat")
def chat(payload: ChatRequest, user: dict = Depends(current_user)) -> dict:
    try:
        return service.chat(
            message=payload.message,
            session_id=payload.session_id,
            model=payload.model,
            history=_as_history(payload.history),
            persist=payload.save_to_session,
            user_id=str(user["id"]),
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
    user: dict = Depends(current_user),
) -> dict:
    try:
        uploads = await _read_uploads(files)
        return service.chat(
            message=message,
            session_id=session_id,
            model=model,
            uploads=uploads,
            persist=save_to_session,
            user_id=str(user["id"]),
        )
    except Exception as exc:
        raise _handle_error(exc)


@app.post("/api/chat/stream")
def stream_chat(payload: ChatRequest, user: dict = Depends(current_user)) -> StreamingResponse:
    user_id = str(user["id"])

    def event_source():
        try:
            for event in service.stream_chat(
                message=payload.message,
                session_id=payload.session_id,
                model=payload.model,
                history=_as_history(payload.history),
                persist=payload.save_to_session,
                user_id=user_id,
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


@app.post("/api/chat/regenerate/stream")
def regenerate_chat(payload: RegenerateRequest, user: dict = Depends(current_user)) -> StreamingResponse:
    user_id = str(user["id"])
    prompt, display_message = service.build_regenerate_prompt(payload.supplement)

    def event_source():
        try:
            for event in service.stream_chat(
                message=prompt,
                session_id=payload.session_id,
                model=payload.model,
                persist=True,
                user_id=user_id,
                display_message=display_message,
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
def knowledge_status(user: dict = Depends(current_user)) -> dict:
    return service.knowledge_status(str(user["id"]))


@app.post("/api/knowledge-base/files")
async def ingest_knowledge_base(
    files: list[UploadFile] = File(...),
    model: str = Form(default=DEFAULT_MODEL),
    user: dict = Depends(current_user),
) -> dict:
    try:
        uploads = await _read_uploads(files)
        return service.ingest_knowledge(uploads, selected_model=model, user_id=str(user["id"]))
    except Exception as exc:
        raise _handle_error(exc)


@app.post("/api/knowledge-base/xhs-url")
def ingest_xhs_url(payload: XhsUrlRequest, user: dict = Depends(current_user)) -> dict:
    try:
        return service.ingest_xhs_url(payload.url, selected_model=payload.model, user_id=str(user["id"]))
    except Exception as exc:
        raise _handle_error(exc)


@app.delete("/api/knowledge-base")
def clear_knowledge_base(user: dict = Depends(current_user)) -> dict:
    try:
        return service.clear_knowledge(str(user["id"]))
    except Exception as exc:
        raise _handle_error(exc)
