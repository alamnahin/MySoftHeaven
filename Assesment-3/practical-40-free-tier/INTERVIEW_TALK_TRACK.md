# Interview Talk Track (2–3 Minutes)

Use this while running `./demo_interview.sh`.

## 0) Opening (15–20 sec)

"I implemented a practical free-tier slice of my Assessment 3 architecture. Instead of only diagrams, I built a working vertical flow: webhook intake, tenant auth, AI classification/reply/lead scoring, and CRM outbox routing. This demonstrates both architecture thinking and executable delivery."

## 1) Scope Framing (20–30 sec)

"This is intentionally an MVP slice, roughly 50–60% of the proposed core runtime path. I prioritized high signal features for production realism:
- Tenant isolation
- JWT-based API security
- Webhook signature validation
- Queue/worker split via Redis
- AI enrichment and CRM-ready events"

## 2) What You’re Running (15–20 sec)

"I’m running one command script that performs end-to-end verification:
1. create tenant
2. issue JWT token
3. sign webhook payload
4. enqueue message
5. worker processes it
6. confirm leads and CRM outbox records"

Command:

```bash
./demo_interview.sh
```

## 3) Architecture Walkthrough (40–50 sec)

"The API service is the ingestion boundary. It validates tenant token and HMAC signature before accepting data.
Then it stores raw message state and pushes the message ID to Redis.
A separate worker consumes from queue and performs AI enrichment.
That worker writes enriched output to `inbound_messages`, creates a `leads` row, and publishes CRM sync intent to an outbox table.
This gives us separation of concerns and a clean path to scale each component independently."

## 4) Security & Reliability (25–35 sec)

"For security, each tenant gets an API key and webhook secret at onboarding.
Token is short-lived JWT and all read endpoints are tenant-scoped.
Webhook data is accepted only with valid HMAC.
For reliability, queueing decouples ingestion latency from processing latency, and outbox pattern keeps external sync integration controlled and retryable."

## 5) Free-Tier/Practicality (15–20 sec)

"The stack is fully free-tier friendly: FastAPI, SQLite, Redis, Docker, and optional Gemini free API.
If Gemini key is missing, deterministic fallback logic still keeps the pipeline testable and demo-safe."

## 6) What I’d Build Next (20–30 sec)

"Next production steps would be:
- idempotency and replay protection,
- dead-letter queue and retry policies,
- role-based access controls,
- real CRM OAuth connectors,
- metrics dashboards and alerting."

## 7) Closing (10–15 sec)

"So this implementation shows I can move from architecture documents into secure, testable, and scalable service foundations with phased delivery."

---

## Optional Q&A Quick Answers

### Why queue + worker now?
"To prevent webhook latency spikes and isolate AI workload from ingestion SLA."

### Why outbox pattern?
"It decouples transaction success from external API availability and supports safe retries."

### How is multi-tenancy handled here?
"Tenant-scoped keys, JWT subject checks, tenant-filtered queries, and tenant-specific webhook secrets."

### How would this scale to enterprise?
"Replace SQLite with Postgres, move to managed Redis/Kafka, add autoscaling workers, and observability + governance controls."
