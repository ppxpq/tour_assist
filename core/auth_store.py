from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

from utils import config


AUTH_DB_FILE = Path(config.BASE_DIR) / "data" / "auth.sqlite3"
ACCESS_TOKEN_SECONDS = 30 * 60
REFRESH_TOKEN_SECONDS = 7 * 24 * 60 * 60
PASSWORD_ITERATIONS = 210_000
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{3,32}$")


class AuthError(RuntimeError):
    pass


def _now_ts() -> int:
    return int(time.time())


def _now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _normalize_username(username: str) -> str:
    value = (username or "").strip()
    if not USERNAME_RE.fullmatch(value):
        raise AuthError("用户名需为 3-32 位，仅支持字母、数字和下划线。")
    return value


def _normalize_password(password: str) -> str:
    value = password or ""
    if not 8 <= len(value) <= 64:
        raise AuthError("密码长度需要在 8 到 64 位之间。")
    if any(char.isspace() for char in value):
        raise AuthError("密码不能包含空格。")
    if not re.search(r"[A-Za-z]", value) or not re.search(r"\d", value):
        raise AuthError("密码需要同时包含字母和数字。")
    return value


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)
    return salt.hex(), digest.hex()


def _verify_password(password: str, salt_hex: str, hash_hex: str) -> bool:
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    _, candidate = _hash_password(password, salt)
    return hmac.compare_digest(candidate, hash_hex)


def _new_token() -> str:
    return secrets.token_urlsafe(40)


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class AuthStore:
    def __init__(self, db_path: Path = AUTH_DB_FILE) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_schema(self) -> None:
        with closing(self._connect()) as conn:
            with conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id TEXT PRIMARY KEY,
                        username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                        password_salt TEXT NOT NULL,
                        password_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS auth_tokens (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        token_hash TEXT NOT NULL UNIQUE,
                        token_type TEXT NOT NULL CHECK (token_type IN ('access', 'refresh')),
                        expires_at INTEGER NOT NULL,
                        revoked_at INTEGER,
                        created_at TEXT NOT NULL,
                        user_agent TEXT NOT NULL DEFAULT '',
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );

                    CREATE INDEX IF NOT EXISTS idx_auth_tokens_hash
                        ON auth_tokens(token_hash);

                    CREATE INDEX IF NOT EXISTS idx_auth_tokens_user_type
                        ON auth_tokens(user_id, token_type, revoked_at);
                    """
                )

    def register(self, username: str, password: str, user_agent: str = "") -> dict[str, Any]:
        username = _normalize_username(username)
        password = _normalize_password(password)
        salt, password_hash = _hash_password(password)
        user_id = uuid.uuid4().hex
        now = _now_text()

        try:
            with closing(self._connect()) as conn:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO users(id, username, password_salt, password_hash, created_at)
                        VALUES(?, ?, ?, ?, ?)
                        """,
                        (user_id, username, salt, password_hash, now),
                    )
        except sqlite3.IntegrityError as exc:
            raise AuthError("用户名已存在。") from exc

        return self._issue_token_pair(user_id, username, user_agent)

    def login(self, username: str, password: str, user_agent: str = "") -> dict[str, Any]:
        username = _normalize_username(username)
        password = _normalize_password(password)

        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT id, username, password_salt, password_hash
                FROM users
                WHERE username = ?
                """,
                (username,),
            ).fetchone()

        if not row or not _verify_password(password, str(row["password_salt"]), str(row["password_hash"])):
            raise AuthError("用户名或密码不正确。")

        return self._issue_token_pair(str(row["id"]), str(row["username"]), user_agent)

    def refresh(self, refresh_token: str, user_agent: str = "") -> dict[str, Any]:
        token_hash = _hash_token(refresh_token or "")
        now_ts = _now_ts()

        with closing(self._connect()) as conn:
            with conn:
                row = conn.execute(
                    """
                    SELECT t.id, t.user_id, t.expires_at, t.revoked_at, u.username
                    FROM auth_tokens t
                    JOIN users u ON u.id = t.user_id
                    WHERE t.token_hash = ? AND t.token_type = 'refresh'
                    """,
                    (token_hash,),
                ).fetchone()

                if not row or row["revoked_at"] or int(row["expires_at"]) <= now_ts:
                    raise AuthError("登录状态已过期，请重新登录。")

                conn.execute(
                    "UPDATE auth_tokens SET revoked_at = ? WHERE id = ?",
                    (now_ts, str(row["id"])),
                )

        return self._issue_token_pair(str(row["user_id"]), str(row["username"]), user_agent)

    def authenticate_access_token(self, access_token: str) -> dict[str, Any]:
        token_hash = _hash_token(access_token or "")
        now_ts = _now_ts()
        with closing(self._connect()) as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.created_at
                FROM auth_tokens t
                JOIN users u ON u.id = t.user_id
                WHERE t.token_hash = ?
                  AND t.token_type = 'access'
                  AND t.revoked_at IS NULL
                  AND t.expires_at > ?
                """,
                (token_hash, now_ts),
            ).fetchone()

        if not row:
            raise AuthError("登录状态已过期，请重新登录。")

        return {
            "id": str(row["id"]),
            "username": str(row["username"]),
            "created_at": str(row["created_at"] or ""),
        }

    def logout(self, refresh_token: str, access_token: str | None = None) -> None:
        now_ts = _now_ts()
        token_hashes = [_hash_token(refresh_token or "")]
        if access_token:
            token_hashes.append(_hash_token(access_token))

        with closing(self._connect()) as conn:
            with conn:
                conn.executemany(
                    "UPDATE auth_tokens SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
                    [(now_ts, token_hash) for token_hash in token_hashes],
                )

    def _issue_token_pair(self, user_id: str, username: str, user_agent: str = "") -> dict[str, Any]:
        access_token = _new_token()
        refresh_token = _new_token()
        now_ts = _now_ts()
        now = _now_text()
        access_expires_at = now_ts + ACCESS_TOKEN_SECONDS
        refresh_expires_at = now_ts + REFRESH_TOKEN_SECONDS

        with closing(self._connect()) as conn:
            with conn:
                conn.executemany(
                    """
                    INSERT INTO auth_tokens(
                        id, user_id, token_hash, token_type, expires_at, created_at, user_agent
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            uuid.uuid4().hex,
                            user_id,
                            _hash_token(access_token),
                            "access",
                            access_expires_at,
                            now,
                            user_agent[:240],
                        ),
                        (
                            uuid.uuid4().hex,
                            user_id,
                            _hash_token(refresh_token),
                            "refresh",
                            refresh_expires_at,
                            now,
                            user_agent[:240],
                        ),
                    ],
                )

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "access_expires_in": ACCESS_TOKEN_SECONDS,
            "refresh_expires_in": REFRESH_TOKEN_SECONDS,
            "user": {
                "id": user_id,
                "username": username,
            },
        }
