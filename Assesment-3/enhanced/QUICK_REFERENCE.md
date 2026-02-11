# Assessment 3: Quick Reference Card

## System Components

```
┌─────────────────────────────────────────────────────────────┐
│  CLIENTS: Web App | Mobile | Chrome Ext | REST API          │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  API GATEWAY: Kong (Auth, Rate Limit, Routing)               │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  SERVICES:                                                  │
│  • Social (Webhook handling)                                │
│  • AI (Intent → Reply → Score)                              │
│  • CRM (Sync to Salesforce/HubSpot)                         │
│  • Auth (JWT, OAuth, RBAC)                                  │
│  • Webhook (Event routing, DLQ)                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  DATA: PostgreSQL | Redis | Kafka | Pinecone | MinIO        │
└─────────────────────────────────────────────────────────────┘
```

## Key Metrics to Monitor

| Metric | Target | Alert If |
|--------|--------|----------|
| Message latency (p95) | <500ms | >1s |
| LLM API error rate | <1% | >5% |
| CRM sync lag | <30s | >5min |
| Database CPU | <70% | >85% |
| Cost per message | <$0.02 | >$0.05 |

## Failure Response Playbook

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| High latency | LLM slow | Enable GPT-3.5 fallback |
| CRM sync failing | Token expired | Manual re-auth in dashboard |
| Message loss | Webhook timeout | Check DLQ, retry manually |
| AI hallucinations | Low confidence | Increase threshold to 0.8 |

## Scaling Triggers

```
Messages/day    Action
─────────────────────────────────
< 1,000        Current setup
1,000-10,000   Add Redis cache
10,000-100,000 Enable Kafka partitioning
> 100,000      Shard by tenant_id
```

## Security Checklist

- [ ] Encrypt social tokens at rest (AES-256)
- [ ] TLS 1.3 for all communications
- [ ] JWT with RS256 signing
- [ ] Rate limiting per tenant
- [ ] Field-level encryption for PII
- [ ] Audit logs for all CRM syncs
- [ ] MFA for admin accounts

## Cost Optimization Quick Wins

1. **Use GPT-3.5 for 80% of replies** (save 70%)
2. **Cache common queries in Redis** (save 30%)
3. **Spot instances for AI workers** (save 60%)
4. **Batch CRM updates** (save API calls)
5. **Compress old conversations to S3** (save storage)

## API Endpoints

```
POST   /v1/webhook/{platform}     # Receive messages
GET    /v1/conversations         # List conversations
POST   /v1/conversations/{id}/reply  # Send reply
GET    /v1/leads                 # List leads
POST   /v1/leads/{id}/sync       # Force CRM sync
GET    /v1/analytics/dashboard   # Metrics
```

## Environment Variables

```bash
# Critical
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
KAFKA_BROKERS=kafka:9092
JWT_PRIVATE_KEY=-----BEGIN...

# AI
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...

# Social (encrypted in production)
FACEBOOK_APP_SECRET=...
TWITTER_API_SECRET=...
```
