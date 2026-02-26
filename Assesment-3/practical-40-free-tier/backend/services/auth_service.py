import hashlib
import hmac
import os
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from db import get_conn


JWT_SECRET = os.getenv("JWT_SECRET", "dev-insecure-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "120"))


def generate_api_key() -> str:
    return secrets.token_urlsafe(24)


def generate_webhook_secret() -> str:
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def verify_api_key(raw_api_key: str, hashed_api_key: str) -> bool:
    return hmac.compare_digest(hash_api_key(raw_api_key), hashed_api_key)


def create_access_token(tenant_id: str) -> str:
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": tenant_id,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_EXPIRE_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def fetch_tenant_auth_record(tenant_id: str) -> dict[str, str] | None:
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, api_key, webhook_secret FROM tenants WHERE id = ?",
            (tenant_id,),
        )
        row = cursor.fetchone()

    if not row:
        return None

    return {
        "tenant_id": row["id"],
        "api_key": row["api_key"] or "",
        "webhook_secret": row["webhook_secret"] or "",
    }
