import uuid
from contextlib import asynccontextmanager
from typing import Literal
import hmac
import hashlib

import jwt
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Query

from db import get_conn, init_db
from models import (
    AuthRequest,
    AuthResponse,
    CRMOutboxResponse,
    LeadResponse,
    MessageStatusResponse,
    QueuePublishResponse,
    TenantCreate,
    TenantResponse,
    WebhookMessage,
)
from services.auth_service import (
    create_access_token,
    decode_access_token,
    fetch_tenant_auth_record,
    generate_api_key,
    generate_webhook_secret,
    hash_api_key,
    verify_api_key,
)
from services.pipeline_service import process_message
from services.queue_service import QueueClient


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="Assessment 3 Practical 40%", version="0.1.0", lifespan=lifespan)
queue_client = QueueClient()


def _require_tenant_token(authorization: str | None, expected_tenant_id: str) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")

    token = authorization.split(" ", 1)[1].strip()

    try:
        payload = decode_access_token(token)
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

    token_tenant_id = payload.get("sub")
    if token_tenant_id != expected_tenant_id:
        raise HTTPException(status_code=403, detail="Token tenant mismatch")


def _signature_payload(
    tenant_id: str,
    platform: str,
    payload: WebhookMessage,
) -> str:
    return "|".join(
        [
            tenant_id,
            platform,
            payload.external_user_id,
            payload.external_message_id or "",
            payload.text,
        ]
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/tenants", response_model=TenantResponse)
def create_tenant(payload: TenantCreate) -> TenantResponse:
    tenant_id = f"tenant_{uuid.uuid4().hex[:10]}"
    raw_api_key = generate_api_key()
    raw_webhook_secret = generate_webhook_secret()

    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO tenants (id, name, api_key, webhook_secret) VALUES (?, ?, ?, ?)",
            (tenant_id, payload.name, hash_api_key(raw_api_key), raw_webhook_secret),
        )

    return TenantResponse(
        id=tenant_id,
        name=payload.name,
        api_key=raw_api_key,
        webhook_secret=raw_webhook_secret,
    )


@app.post("/auth/token", response_model=AuthResponse)
def issue_token(payload: AuthRequest) -> AuthResponse:
    record = fetch_tenant_auth_record(payload.tenant_id)
    if not record:
        raise HTTPException(status_code=404, detail="Tenant not found")

    if not verify_api_key(payload.api_key, record["api_key"]):
        raise HTTPException(status_code=401, detail="Invalid API key")

    token = create_access_token(payload.tenant_id)
    return AuthResponse(access_token=token)


@app.post("/webhooks/{platform}", response_model=QueuePublishResponse)
async def ingest_webhook(
    platform: Literal["facebook", "twitter", "linkedin"],
    payload: WebhookMessage,
    background_tasks: BackgroundTasks,
    tenant_id: str = Query(..., min_length=6),
    authorization: str | None = Header(default=None),
    x_webhook_signature: str | None = Header(default=None),
) -> QueuePublishResponse:
    _require_tenant_token(authorization, tenant_id)

    expected_signature = None

    with get_conn() as conn:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, webhook_secret FROM tenants WHERE id = ?",
            (tenant_id,),
        )
        tenant_row = cursor.fetchone()
        if not tenant_row:
            raise HTTPException(status_code=404, detail="Tenant not found")

        webhook_secret = tenant_row["webhook_secret"] or ""
        expected_signature = hmac.new(
            webhook_secret.encode("utf-8"),
            _signature_payload(tenant_id, platform, payload).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        if not x_webhook_signature or not hmac.compare_digest(x_webhook_signature, expected_signature):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

        cursor.execute(
            """
            INSERT INTO inbound_messages (
                tenant_id, platform, external_user_id, external_message_id, text, status
            ) VALUES (?, ?, ?, ?, ?, 'received')
            """,
            (
                tenant_id,
                platform,
                payload.external_user_id,
                payload.external_message_id,
                payload.text,
            ),
        )
        if cursor.lastrowid is None:
            raise HTTPException(status_code=500, detail="Failed to persist message")
        message_id = int(cursor.lastrowid)

    queued = queue_client.enqueue_message(message_id)
    if not queued:
        background_tasks.add_task(process_message, message_id)

    return QueuePublishResponse(
        message_id=message_id,
        queued=queued,
    )


@app.get("/messages/{message_id}", response_model=MessageStatusResponse)
def get_message_status(
    message_id: int,
    tenant_id: str = Query(..., min_length=6),
    authorization: str | None = Header(default=None),
) -> MessageStatusResponse:
    _require_tenant_token(authorization, tenant_id)

    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM inbound_messages WHERE id = ? AND tenant_id = ?",
            (message_id, tenant_id),
        )
        row = cursor.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Message not found")

    return MessageStatusResponse(
        message_id=row["id"],
        tenant_id=row["tenant_id"],
        platform=row["platform"],
        status=row["status"],
        intent=row["intent"],
        ai_reply=row["ai_reply"],
        lead_score=row["lead_score"],
    )


@app.get("/tenants/{tenant_id}/leads", response_model=list[LeadResponse])
def list_leads(
    tenant_id: str,
    authorization: str | None = Header(default=None),
) -> list[LeadResponse]:
    _require_tenant_token(authorization, tenant_id)

    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, message_id, intent, score, status
            FROM leads
            WHERE tenant_id = ?
            ORDER BY id DESC
            """,
            (tenant_id,),
        )
        rows = cursor.fetchall()

    return [
        LeadResponse(
            id=row["id"],
            message_id=row["message_id"],
            intent=row["intent"],
            score=row["score"],
            status=row["status"],
        )
        for row in rows
    ]


@app.get("/tenants/{tenant_id}/crm/outbox", response_model=list[CRMOutboxResponse])
def list_crm_outbox(
    tenant_id: str,
    authorization: str | None = Header(default=None),
) -> list[CRMOutboxResponse]:
    _require_tenant_token(authorization, tenant_id)

    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, lead_id, crm_target, status
            FROM crm_outbox
            WHERE tenant_id = ?
            ORDER BY id DESC
            """,
            (tenant_id,),
        )
        rows = cursor.fetchall()

    return [
        CRMOutboxResponse(
            id=row["id"],
            lead_id=row["lead_id"],
            crm_target=row["crm_target"],
            status=row["status"],
        )
        for row in rows
    ]
