import os
import sys
import hmac
import hashlib
from pathlib import Path

os.environ["DATABASE_PATH"] = str(Path(__file__).parent / "test.db")
os.environ["LLM_API_KEY"] = ""
os.environ["USE_REDIS_QUEUE"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from db import init_db
from main import app


init_db()


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_end_to_end_pipeline() -> None:
    tenant = client.post("/tenants", json={"name": "Acme Ltd"})
    assert tenant.status_code == 200
    tenant_id = tenant.json()["id"]
    api_key = tenant.json()["api_key"]
    webhook_secret = tenant.json()["webhook_secret"]

    token_response = client.post(
        "/auth/token",
        json={"tenant_id": tenant_id, "api_key": api_key},
    )
    assert token_response.status_code == 200
    access_token = token_response.json()["access_token"]
    auth_headers = {"Authorization": f"Bearer {access_token}"}

    external_user_id = "user_123"
    external_message_id = "m_001"
    text = "Hi, I need pricing and a demo for enterprise plan."
    platform = "facebook"
    signature_payload = f"{tenant_id}|{platform}|{external_user_id}|{external_message_id}|{text}"
    signature = hmac.new(
        webhook_secret.encode("utf-8"),
        signature_payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    webhook = client.post(
        f"/webhooks/{platform}?tenant_id={tenant_id}",
        headers={
            **auth_headers,
            "X-Webhook-Signature": signature,
        },
        json={
            "external_user_id": external_user_id,
            "external_message_id": external_message_id,
            "text": text,
        },
    )
    assert webhook.status_code == 200
    message_id = webhook.json()["message_id"]

    message = client.get(
        f"/messages/{message_id}?tenant_id={tenant_id}",
        headers=auth_headers,
    )
    assert message.status_code == 200
    payload = message.json()
    assert payload["status"] in ["received", "processed"]

    leads = client.get(f"/tenants/{tenant_id}/leads", headers=auth_headers)
    assert leads.status_code == 200
    assert isinstance(leads.json(), list)

    outbox = client.get(f"/tenants/{tenant_id}/crm/outbox", headers=auth_headers)
    assert outbox.status_code == 200
    assert isinstance(outbox.json(), list)
