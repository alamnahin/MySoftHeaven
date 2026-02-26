# Assessment 3 Practical Implementation (Free-Tier, ~55%)

This folder contains a practical, interview-ready implementation of approximately **50–60%** of the Assessment 3 architecture using only free-tier/local components.

## What This Implements (Completed Scope)

### 1) Multi-tenant base layer
- Tenant onboarding endpoint (`POST /tenants`)
- Tenant-scoped data model in SQLite
- Isolation by `tenant_id` across all flows

### 2) Social webhook ingestion (core)
- Platform endpoints:
  - `POST /webhooks/facebook`
  - `POST /webhooks/twitter`
  - `POST /webhooks/linkedin`
- Message persistence with processing status

### 3) AI pipeline (simplified but real)
- Intent classification (`sales`, `support`, `spam`)
- AI reply generation
- Lead scoring (0–100)
- Free-tier mode:
  - Uses Gemini when `LLM_API_KEY` is provided
  - Falls back to deterministic heuristics if key is missing

### 4) CRM sync simulation (outbox pattern)
- Converts processed leads into CRM outbox events
- Routes leads to a target CRM (`hubspot`/`salesforce`) based on score
- Exposes queue visibility for demo

### 5) Observability basics
- Health endpoint (`GET /health`)
- Query endpoints to inspect processing outputs

### 6) Security baseline (added)
- Tenant API key issuance at tenant creation
- JWT token endpoint (`POST /auth/token`)
- Tenant-scoped authorization on data APIs
- HMAC webhook signature validation (`X-Webhook-Signature`)

### 7) Worker split with Redis (added)
- API service ingests and enqueues message IDs into Redis
- Separate worker service dequeues and processes messages
- Safe fallback to in-process background task if queue is disabled/unavailable

## What This Intentionally Leaves for Later (Remaining 60–70%)

- Production broker stack (Kafka/Celery with retries and DLQ)
- Real CRM API connectors (OAuth + retries + DLQ)
- Full RBAC and role policies beyond tenant-level JWT
- Advanced monitoring (Prometheus/Grafana)
- Production-grade idempotency keys and replay protection
- Distributed deployment with API gateway policies

## Architecture Mapping

This practical implementation proves the most discussion-worthy vertical slice:

`Webhook -> Persist -> AI classify/reply/score -> Lead -> CRM outbox`

That is enough to demonstrate system thinking, implementation discipline, and phased delivery strategy in interview.

## Folder Structure

- `backend/main.py` — FastAPI app and APIs
- `backend/db.py` — SQLite schema + connection helpers
- `backend/models.py` — Pydantic request/response models
- `backend/services/llm_service.py` — Gemini + fallback logic
- `backend/services/pipeline_service.py` — processing orchestration
- `backend/services/crm_service.py` — CRM outbox queueing
- `backend/tests/test_api.py` — basic API + pipeline tests
- `docker-compose.yml` — local run stack
- `.env.example` — configuration template

## Quick Run (Local)

```bash
cd Assesment-3/practical-40-free-tier
cp .env.example .env

cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8010
```

Open docs: `http://localhost:8010/docs`

## Quick Run (Docker)

```bash
cd Assesment-3/practical-40-free-tier
docker-compose up --build
```

App URL: `http://localhost:8010`

## Demo Flow (Interview)

1. Create tenant:
```bash
curl -X POST http://localhost:8010/tenants \
  -H "Content-Type: application/json" \
  -d '{"name":"Demo Tenant"}'
```

2. Generate JWT token:
```bash
curl -X POST http://localhost:8010/auth/token \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"tenant_xxx","api_key":"TENANT_API_KEY"}'
```

3. Build webhook signature (example):
```bash
python - <<'PY'
import hmac, hashlib
tenant_id = "tenant_xxx"
platform = "facebook"
external_user_id = "fb_user_9"
external_message_id = "msg_101"
text = "We need enterprise pricing and a quick product demo"
webhook_secret = "TENANT_WEBHOOK_SECRET"
payload = f"{tenant_id}|{platform}|{external_user_id}|{external_message_id}|{text}"
print(hmac.new(webhook_secret.encode(), payload.encode(), hashlib.sha256).hexdigest())
PY
```

4. Ingest signed message:
```bash
curl -X POST "http://localhost:8010/webhooks/facebook?tenant_id=tenant_xxx" \
  -H "Authorization: Bearer JWT_TOKEN" \
  -H "X-Webhook-Signature: GENERATED_SIGNATURE" \
  -H "Content-Type: application/json" \
  -d '{
    "external_user_id":"fb_user_9",
    "external_message_id":"msg_101",
    "text":"We need enterprise pricing and a quick product demo"
  }'
```

5. Check processed message:
```bash
curl "http://localhost:8010/messages/1?tenant_id=tenant_xxx" \
  -H "Authorization: Bearer JWT_TOKEN"
```

6. Check leads:
```bash
curl http://localhost:8010/tenants/tenant_xxx/leads \
  -H "Authorization: Bearer JWT_TOKEN"
```

7. Check CRM outbox:
```bash
curl http://localhost:8010/tenants/tenant_xxx/crm/outbox \
  -H "Authorization: Bearer JWT_TOKEN"
```

## Free-Tier Notes

- No paid infra required.
- Runs on local SQLite and optional Gemini free API usage.
- Works even without API key due to fallback logic.

## Suggested Next Iteration

If you want, next I can extend this toward ~70% by adding:
- Retries + dead-letter queue semantics
- Webhook replay protection / idempotency keys
- Role-based access (admin/operator/viewer)
- Real CRM connector adapters with OAuth refresh
