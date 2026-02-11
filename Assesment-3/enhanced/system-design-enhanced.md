# Assessment 3: Enhanced System Design for AI SaaS Platform

## 1. Executive Summary

Building upon the patterns from Assessments 1 and 2, this design scales the AI agent architecture into a multi-tenant SaaS platform for social media automation.

**Key Innovations from Previous Work:**
- **Assessment 1's** n8n-style workflow engine → Scalable AI Agent Orchestrator
- **Assessment 2's** RAG implementation → Multi-tenant Knowledge Base with conversation memory
- **Combined** into enterprise-grade system with CRM integration

## 2. Detailed Architecture

### 2.1 System Context Diagram

```
                    ┌─────────────────────────────────────────┐
                    │           EXTERNAL ACTORS               │
                    │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
                    │  │ Facebook│ │ Twitter │ │LinkedIn │   │
                    │  └────┬────┘ └────┬────┘ └────┬────┘   │
                    │       └─────────────┴───────────┘       │
                    │                 │                       │
                    │                 ▼                       │
                    │  ┌─────────────────────────────────┐    │
                    │  │      Social Media Platforms     │    │
                    │  │   (Graph API, REST API, etc.)   │    │
                    │  └─────────────────────────────────┘    │
                    └───────────────────┬─────────────────────┘
                                        │ Webhooks
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                           SOCIALAI HUB PLATFORM                                │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         API GATEWAY (Kong)                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │  │
│  │  │   Auth      │  │  Rate Limit │  │   Request   │  │   Logging   │    │  │
│  │  │   JWT/OAuth2│  │  (Tiered)   │  │  Validation │  │   & Trace   │    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │  │
│  └──────────────────────────────────┬──────────────────────────────────────┘  │
│                                     │                                          │
│  ┌──────────────────────────────────┼──────────────────────────────────────┐  │
│  │                      SERVICE MESH (Istio)                                │  │
│  │                            │                                             │  │
│  │     ┌──────────────────────┼──────────────────────┐                     │  │
│  │     │                      │                      │                     │  │
│  │     ▼                      ▼                      ▼                     │  │
│  │ ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐          │  │
│  │ │   Social    │    │    AI       │    │       CRM           │          │  │
│  │ │  Service    │◄──►│  Service    │◄──►│     Service         │          │  │
│  │ │             │    │             │    │                     │          │  │
│  │ │ • Webhook   │    │ • Intent    │    │ • Salesforce        │          │  │
│  │ │   Handler   │    │   Classify  │    │ • HubSpot           │          │  │
│  │ │ • Polling   │    │ • Reply Gen │    │ • Sync Engine       │          │  │
│  │ │ • Auth Mgr  │    │ • Lead Score│    │ • Transformers      │          │  │
│  │ └─────────────┘    └──────┬──────┘    └─────────────────────┘          │  │
│  │                           │                                            │  │
│  │                           ▼                                            │  │
│  │              ┌─────────────────────┐                                   │  │
│  │              │   Agent Orchestrator│                                   │  │
│  │              │   (Celery + Redis)  │                                   │  │
│  │              │                     │                                   │  │
│  │              │ ┌─────────┐┌────────┐┌─────────┐                       │  │
│  │              │ │Classifier││Reply  ││Scorer   │                       │  │
│  │              │ │ Agent   ││ Agent ││ Agent  │                       │  │
│  │              │ └─────────┘└────────┘└─────────┘                       │  │
│  │              └─────────────────────┘                                   │  │
│  │                                                                        │  │
│  │ ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐          │  │
│  │ │   Auth      │    │  Webhook    │    │   Analytics         │          │  │
│  │ │  Service    │    │  Service    │    │   Service           │          │  │
│  │ │             │    │             │    │                     │          │  │
│  │ │ • Tenant    │    │ • Event     │    │ • Metrics           │          │  │
│  │ │   Isolation │    │   Router    │    │ • Dashboard         │          │  │
│  │ │ • RBAC      │    │ • Retry     │    │ • Reporting         │          │  │
│  │ │ • OAuth     │    │   Logic     │    │ • Audit Logs        │          │  │
│  │ └─────────────┘    └─────────────┘    └─────────────────────┘          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         DATA LAYER                                       │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │  │
│  │  │  PostgreSQL  │ │    Redis     │ │    Kafka     │ │   Pinecone   │    │  │
│  │  │  (Multi-     │ │   (Cache/    │ │   (Event     │ │  (Vector     │    │  │
│  │  │   tenant)    │ │    Queue)    │ │    Bus)      │ │    DB)       │    │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘    │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                      │  │
│  │  │   MinIO      │ │ ClickHouse   │ │Elasticsearch │                      │  │
│  │  │  (Object     │ │ (Analytics)  │ │  (Search)    │                      │  │
│  │  │   Storage)   │ │              │ │              │                      │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘                      │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │              CRM SYSTEMS               │
                    │  ┌─────────┐ ┌─────────┐ ┌─────────┐  │
                    │  │Salesforce│ │HubSpot  │ │ Zoho    │  │
                    │  └─────────┘ └─────────┘ └─────────┘  │
                    └───────────────────────────────────────┘
```

### 2.2 Multi-Tenant Data Isolation Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    TENANT ISOLATION MODEL                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    SHARED DATABASE                         │  │
│  │              (PostgreSQL with Row-Level Security)          │  │
│  │                                                           │  │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │  │
│  │   │  Tenant A   │    │  Tenant B   │    │  Tenant C   │  │  │
│  │   │  (user_id=1)│    │  (user_id=2)│    │  (user_id=3)│  │  │
│  │   │             │    │             │    │             │  │  │
│  │   │ conversations│   │ conversations│   │ conversations│  │  │
│  │   │ leads       │    │ leads       │    │ leads       │  │  │
│  │   │ social_accounts│  │ social_accounts│  │ social_accounts│ │  │
│  │   └─────────────┘    └─────────────┘    └─────────────┘  │  │
│  │                                                           │  │
│  │   RLS Policy: tenant_id = current_setting('app.tenant_id')│  │
│  │                                                           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    SHARED SCHEMA                           │  │
│  │                                                           │  │
│  │   Benefits:                                               │  │
│  │   • Cost efficient (single DB)                            │  │
│  │   • Easy schema migrations                                │  │
│  │   • Connection pooling efficiency                         │  │
│  │                                                           │  │
│  │   Trade-offs:                                             │  │
│  │   • Complex backup/restore per tenant                     │  │
│  │   • Noisy neighbor risk (mitigated by RLS + rate limits)  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Alternative: Schema-per-tenant (for enterprise tier)            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  tenant_a schema │ tenant_b schema │ tenant_c schema      │  │
│  │  • Full isolation                                   │  │
│  │  • Higher cost (connections, memory)                     │  │
│  │  • Complex migrations                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 3. AI Agent System Design

### 3.1 Agent Orchestrator (Enhanced from Assessment 1)

```python
# code-examples/agent_orchestrator.py
"""
Agent Orchestrator - Production version of Assessment 1's workflow
Handles lead classification, reply generation, and lead scoring
"""
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel
import asyncio
from enum import Enum
import httpx
import json
from datetime import datetime

class IntentType(str, Enum):
    SALES = "sales"
    SUPPORT = "support"
    SPAM = "spam"
    INQUIRY = "inquiry"
    COMPLAINT = "complaint"

class LeadTemperature(str, Enum):
    HOT = "hot"      # Immediate sales opportunity
    WARM = "warm"    # Engaged, nurture needed
    COLD = "cold"    # Low engagement

class MessageContext(BaseModel):
    tenant_id: str
    conversation_id: str
    message_id: str
    platform: Literal["facebook", "twitter", "linkedin", "instagram"]
    author_id: str
    content: str
    timestamp: datetime
    previous_messages: List[Dict] = []
    customer_data: Optional[Dict] = None

class AgentResult(BaseModel):
    intent: IntentType
    confidence: float
    reply: str
    lead_temperature: LeadTemperature
    lead_score: int  # 0-100
    should_escalate: bool
    metadata: Dict

class AgentOrchestrator:
    """
    Production-grade orchestrator combining Assessment 1 + 2 patterns
    """

    def __init__(self, config: Dict):
        self.llm_client = httpx.AsyncClient(timeout=30.0)
        self.config = config
        self.vector_store = None  # Pinecone/FAISS from Assessment 2

    async def process_message(self, context: MessageContext) -> AgentResult:
        """
        Main orchestration flow:
        1. Retrieve relevant knowledge (RAG from Assessment 2)
        2. Classify intent (Assessment 1 pattern)
        3. Score lead
        4. Generate reply
        5. Check escalation rules
        """

        # Step 1: Parallel intent + sentiment analysis
        intent_task = self._classify_intent(context)
        sentiment_task = self._analyze_sentiment(context)

        intent, sentiment = await asyncio.gather(intent_task, sentiment_task)

        # Step 2: Lead scoring based on multiple signals
        lead_score = self._calculate_lead_score(context, intent, sentiment)

        # Step 3: Generate contextual reply
        reply = await self._generate_reply(context, intent, sentiment)

        # Step 4: Check if human escalation needed
        should_escalate = self._check_escalation_rules(
            intent, sentiment, lead_score, context
        )

        return AgentResult(
            intent=intent["type"],
            confidence=intent["confidence"],
            reply=reply,
            lead_temperature=self._score_to_temperature(lead_score),
            lead_score=lead_score,
            should_escalate=should_escalate,
            metadata={
                "sentiment": sentiment,
                "processing_time_ms": 0,  # Track actual time
                "model_version": "gpt-4-1106"
            }
        )

    async def _classify_intent(self, context: MessageContext) -> Dict:
        """
        Enhanced intent classification from Assessment 1
        Uses few-shot prompting + confidence calibration
        """
        prompt = f"""
        Classify the following social media message into one of: sales, support, spam, inquiry, complaint.

        Message: {context.content}
        Platform: {context.platform}
        Previous context: {json.dumps(context.previous_messages[-3:])}

        Respond with JSON:
        {{
            "type": "intent_type",
            "confidence": 0.0-1.0,
            "extracted_entities": {{"product": "...", "issue_type": "..."}},
            "urgency": "low|medium|high"
        }}
        """

        # Call LLM (OpenAI/Anthropic/local)
        response = await self._call_llm(prompt, temperature=0.3)

        # Parse and validate
        result = json.loads(response)

        # Confidence calibration - be conservative
        if result["confidence"] < 0.7:
            result["type"] = "inquiry"  # Default to safe category

        return result

    def _calculate_lead_score(self, context: MessageContext, 
                             intent: Dict, sentiment: Dict) -> int:
        """
        Multi-factor lead scoring algorithm
        """
        score = 0

        # Intent signals (40% weight)
        intent_weights = {
            IntentType.SALES: 40,
            IntentType.INQUIRY: 30,
            IntentType.SUPPORT: 20,
            IntentType.COMPLAINT: 10,
            IntentType.SPAM: 0
        }
        score += intent_weights.get(intent["type"], 0)

        # Sentiment signals (30% weight)
        sentiment_score = sentiment.get("score", 0)  # -1 to 1
        score += int((sentiment_score + 1) * 15)  # Map to 0-30

        # Engagement history (20% weight)
        if context.previous_messages:
            score += min(len(context.previous_messages) * 4, 20)

        # Profile completeness (10% weight)
        if context.customer_data:
            score += 10

        return min(score, 100)

    async def _generate_reply(self, context: MessageContext, 
                             intent: Dict, sentiment: Dict) -> str:
        """
        RAG-based reply generation (Assessment 2 pattern)
        """
        # Retrieve relevant knowledge
        knowledge = await self._retrieve_knowledge(context)

        prompt = f"""
        You are a social media manager for {context.tenant_id}.

        Original message: {context.content}
        Intent: {intent['type']}
        Sentiment: {sentiment['label']}

        Relevant company knowledge:
        {knowledge}

        Guidelines:
        - Keep reply under 280 characters for Twitter, 500 for others
        - Match the brand voice (professional but friendly)
        - Address specific points from the message
        - Include call-to-action only for sales intent

        Generate reply:
        """

        reply = await self._call_llm(prompt, temperature=0.7)
        return reply.strip()

    def _check_escalation_rules(self, intent: Dict, sentiment: Dict, 
                               score: int, context: MessageContext) -> bool:
        """
        Business rules for human escalation
        """
        rules = [
            sentiment.get("label") == "angry" and sentiment.get("score", 0) < -0.7,
            intent["urgency"] == "high",
            "refund" in context.content.lower(),
            "lawsuit" in context.content.lower(),
            score > 80 and intent["type"] == IntentType.SALES  # VIP lead
        ]
        return any(rules)

    async def _call_llm(self, prompt: str, temperature: float = 0.5) -> str:
        """Abstracted LLM call with fallback"""
        # Implementation with retry logic from Assessment 1
        pass

    async def _retrieve_knowledge(self, context: MessageContext) -> str:
        """RAG retrieval from Assessment 2"""
        # Query vector store
        pass
```

### 3.2 Conversation Memory Management

```python
# code-examples/conversation_memory.py
"""
Conversation memory with Redis - scaling Assessment 2's memory
"""
import redis
import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class ConversationMemory:
    """
    Multi-tenant conversation memory with TTL
    """

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.ttl = 86400 * 7  # 7 days

    def _key(self, tenant_id: str, conversation_id: str) -> str:
        """Namespace isolation per tenant"""
        return f"conv:{tenant_id}:{conversation_id}"

    async def add_message(self, tenant_id: str, conversation_id: str, 
                         role: str, content: str, metadata: Dict = None):
        """
        Add message to conversation history
        """
        key = self._key(tenant_id, conversation_id)

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        # Push to list, trim to last 20 messages
        pipe = self.redis.pipeline()
        pipe.lpush(key, json.dumps(message))
        pipe.ltrim(key, 0, 19)
        pipe.expire(key, self.ttl)
        await pipe.execute()

    async def get_context(self, tenant_id: str, conversation_id: str, 
                         limit: int = 10) -> List[Dict]:
        """
        Retrieve conversation context for LLM prompt
        """
        key = self._key(tenant_id, conversation_id)
        messages = await self.redis.lrange(key, 0, limit - 1)

        # Parse and reverse (oldest first)
        context = [json.loads(m) for m in messages][::-1]
        return context

    async def get_summary(self, tenant_id: str, conversation_id: str) -> str:
        """
        Generate conversation summary for long contexts
        (Uses LLM to summarize if > 20 messages)
        """
        context = await self.get_context(tenant_id, conversation_id, limit=20)

        if len(context) < 20:
            return self._format_context(context)

        # Summarize older messages
        older = context[:-10]
        recent = context[-10:]

        summary = await self._summarize_messages(older)
        return f"Summary: {summary}\n\nRecent: {self._format_context(recent)}"

    def _format_context(self, messages: List[Dict]) -> str:
        """Format for LLM prompt"""
        return "\n".join([
            f"{m['role']}: {m['content']}" 
            for m in messages
        ])
```

## 4. Data Flow Deep Dive

### 4.1 Complete Message Processing Sequence

```
┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│  Social  │     │  API     │     │  Social  │     │  Kafka   │     │   AI     │
│ Platform │────►│  Gateway │────►│ Service  │────►│  Topic   │────►│ Service  │
└──────────┘     └──────────┘     └──────────┘     └──────────┘     └────┬─────┘
     │                                                                    │
     │ Webhook Payload                                                    │
     │ {                                                                  │
     │   "object": "page",                                                │
     │   "entry": [{                                                      │
     │     "messaging": [{                                                │
     │       "sender": {"id": "123"},                                      │
     │       "message": {"text": "Hi"}                                     │
     │     }]                                                             │
     │   }]                                                               │
     │ }                                                                  │
     │                                                                    │
     │                                                                    ▼
     │                                                          ┌──────────────┐
     │                                                          │ 1. Validate  │
     │                                                          │    Tenant    │
     │                                                          └──────┬───────┘
     │                                                                 │
     │                                                          ┌──────▼───────┐
     │                                                          │ 2. Check Rate│
     │                                                          │    Limit     │
     │                                                          └──────┬───────┘
     │                                                                 │
     │                                                          ┌──────▼───────┐
     │                                                          │ 3. Enrich    │
     │                                                          │    Context   │
     │                                                          │    (Redis)   │
     │                                                          └──────┬───────┘
     │                                                                 │
     │                                                          ┌──────▼───────┐
     │                                                          │ 4. Intent    │
     │                                                          │    Classify  │
     │                                                          └──────┬───────┘
     │                                                                 │
     │                                                          ┌──────▼───────┐
     │                                                          │ 5. Generate  │
     │                                                          │    Reply     │
     │                                                          └──────┬───────┘
     │                                                                 │
     │                                                          ┌──────▼───────┐
     │                                                          │ 6. Score Lead│
     │                                                          └──────┬───────┘
     │                                                                 │
     │     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────▼───────┐
     └────►│  Social  │◄────│  CRM     │◄────│  Kafka   │◄────│ 7. Publish   │
           │ Platform │     │ Service  │     │ Topic    │     │    Results   │
           └──────────┘     └──────────┘     └──────────┘     └──────────────┘
                │
                │ POST graph.facebook.com/v18.0/me/messages
                │ {recipient: {id: "123"}, message: {text: "Reply"}}
                ▼
```

### 4.2 CRM Synchronization Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CRM SYNC ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────────────────┘

Scenario: Hot lead detected (Lead Score > 80)

┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│   AI     │     │  Kafka   │     │  CRM     │     │Transform │     │ Salesforce│
│ Service  │────►│  Topic   │────►│ Service  │────►│  Layer   │────►│  REST API │
└──────────┘     └──────────┘     └──────────┘     └──────────┘     └────┬─────┘
     │                                                                    │
     │ Event: {                                                           │
     │   "type": "lead.hot",                                              │
     │   "tenant_id": "acme_corp",                                        │
     │   "lead": {                                                        │
     │     "external_id": "fb_123",                                       │
     │     "name": "John Doe",                                            │
     │     "email": "john@example.com",                                   │
     │     "score": 85,                                                   │
     │     "source": "facebook",                                          │
     │     "conversations": [...]                                         │
     │   }                                                                │
     │ }                                                                  │
     │                                                                    │
     │                                                          ┌─────────┴─────┐
     │                                                          │ POST /leads   │
     │                                                          │ Headers:      │
     │                                                          │ Authorization:│
     │                                                          │ Bearer token  │
     │                                                          │ Body: {       │
     │                                                          │   "FirstName":│
     │                                                          │   "John",     │
     │                                                          │   "LeadSource":
     │                                                          │   "SocialAI", │
     │                                                          │   "Rating":   │
     │                                                          │   "Hot"       │
     │                                                          │ }             │
     │                                                          └───────────────┘
     │                                                                    │
     │                                                          Response: 201
     │                                                          {id: "00Q5g..."}
     │                                                                    │
     │     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
     └────►│  Audit   │◄────│  DB      │◄────│  Success │◄────│  Store   │
           │  Log     │     │ Update   │     │ Handler  │     │ Mapping  │
           └──────────┘     └──────────┘     └──────────┘     └──────────┘

Conflict Resolution Strategy:
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Case 1: Lead exists in CRM                                                 │
│    → Update LeadScore__c custom field                                       │
│    → Add Task: "Social engagement - follow up"                              │
│    → Log to Activity History                                                │
│                                                                             │
│  Case 2: Lead doesn't exist                                                 │
│    → Create new Lead                                                        │
│    → Map social profile to custom field Social_ID__c                        │
│    → Set LeadSource = "SocialAI Hub"                                        │
│                                                                             │
│  Case 3: Sync fails (network error)                                         │
│    → Retry 3x with exponential backoff                                      │
│    → After 3 failures, move to Dead Letter Queue                            │
│    → Alert admin via PagerDuty                                              │
│    → Manual retry interface in dashboard                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 5. Authentication & Authorization

### 5.1 JWT Token Structure

```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT",
    "kid": "key-2024-01"
  },
  "payload": {
    "sub": "user_123",
    "tenant_id": "acme_corp",
    "roles": ["admin", "agent"],
    "permissions": [
      "conversations:read",
      "conversations:write",
      "leads:read",
      "analytics:read"
    ],
    "iat": 1704067200,
    "exp": 1704153600,
    "jti": "unique-token-id"
  }
}
```

### 5.2 Social Account OAuth Flow

```
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
│   User  │     │  Web App │     │  Backend │     │ Facebook │     │   DB     │
└────┬────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘     └────┬─────┘
     │               │                │                │                │
     │ Click "Connect Facebook"       │                │                │
     │──────────────►│                │                │                │
     │               │                │                │                │
     │               │ POST /auth/social/facebook/init   │                │
     │               │────────────────►│                │                │
     │               │                │                │                │
     │               │                │ Generate state │                │
     │               │                │ parameter      │                │
     │               │                │ (CSRF protection)              │
     │               │                │                │                │
     │               │                │ Redirect to Facebook OAuth     │
     │               │◄───────────────│                │                │
     │               │                │                │                │
     │ Redirect to facebook.com/dialog/oauth...      │                │
     │◄──────────────│                │                │                │
     │               │                │                │                │
     │ User authenticates & authorizes                │                │
     │────────────────────────────────────────────────►│                │
     │               │                │                │                │
     │ Redirect to callback URL with code            │                │
     │◄──────────────│                │                │                │
     │               │                │                │                │
     │ GET /auth/social/facebook/callback?code=...   │                │
     │──────────────►│                │                │                │
     │               │ POST facebook.com/v12.0/oauth/access_token      │
     │               │────────────────────────────────►│                │
     │               │                │                │                │
     │               │                │ Exchange code for tokens         │
     │               │                │◄───────────────│                │
     │               │                │                │                │
     │               │                │ Encrypt tokens │                │
     │               │                │ Store in DB    │                │
     │               │                │───────────────►│                │
     │               │                │                │                │
     │               │                │ Subscribe to webhooks            │
     │               │                │────────────────────────────────►│
     │               │                │                │                │
     │               │ Return success │                │                │
     │               │◄───────────────│                │                │
     │               │                │                │                │
     │ Show "Connected!"               │                │                │
     │◄──────────────│                │                │                │
```

## 6. Security Implementation

### 6.1 Secret Management

```yaml
# infrastructure/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: socialai-secrets
  namespace: production
type: Opaque
stringData:
  # Database
  DATABASE_URL: "postgresql://user:pass@host/db"

  # LLM APIs
  OPENAI_API_KEY: "sk-..."
  ANTHROPIC_API_KEY: "sk-ant-..."

  # Social Platform OAuth
  FACEBOOK_APP_SECRET: "..."
  TWITTER_API_SECRET: "..."
  LINKEDIN_CLIENT_SECRET: "..."

  # Encryption
  AES_MASTER_KEY: "..."  # For field-level encryption
  JWT_PRIVATE_KEY: |
    -----BEGIN RSA PRIVATE KEY-----
    ...
    -----END RSA PRIVATE KEY-----
```

### 6.2 Field-Level Encryption

```python
# code-examples/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class FieldEncryption:
    """
    Encrypt sensitive fields (social tokens, PII)
    """

    def __init__(self, master_key: str):
        self.master_key = master_key.encode()

    def _derive_key(self, tenant_id: str) -> bytes:
        """Derive tenant-specific key from master"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=tenant_id.encode(),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return key

    def encrypt(self, plaintext: str, tenant_id: str) -> str:
        """Encrypt field value"""
        key = self._derive_key(tenant_id)
        f = Fernet(key)
        return f.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str, tenant_id: str) -> str:
        """Decrypt field value"""
        key = self._derive_key(tenant_id)
        f = Fernet(key)
        return f.decrypt(ciphertext.encode()).decode()

# Usage in SocialAccount model
class SocialAccount:
    def __init__(self, encryption_service: FieldEncryption):
        self.encryption = encryption_service

    def store_tokens(self, tenant_id: str, access_token: str, refresh_token: str):
        """Store encrypted tokens"""
        encrypted_access = self.encryption.encrypt(access_token, tenant_id)
        encrypted_refresh = self.encryption.encrypt(refresh_token, tenant_id)

        # Store in DB
        db.execute(
            "INSERT INTO social_accounts (tenant_id, access_token_enc, refresh_token_enc) VALUES (?, ?, ?)",
            (tenant_id, encrypted_access, encrypted_refresh)
        )
```

## 7. Cost Optimization Details

### 7.1 AI Cost Breakdown

| Component | Model | Input Cost | Output Cost | Avg/Msg | Optimization |
|-----------|-------|------------|-------------|---------|--------------|
| Intent Classification | GPT-3.5 | $0.0015/1K | $0.002/1K | $0.002 | Cache common patterns |
| Reply Generation | GPT-4 | $0.03/1K | $0.06/1K | $0.015 | Use GPT-3.5 for simple replies |
| Lead Scoring | Local BERT | $0 | $0 | $0.001 | CPU inference, no API cost |
| Embedding | text-embedding-3 | $0.0001/1K | - | $0.0001 | Batch processing |

**Total per message: ~$0.018 (optimized from $0.08)**

### 7.2 Infrastructure Cost Model

```python
# code-examples/cost_calculator.py
class CostCalculator:
    """
    Monthly cost estimation for SaaS platform
    """

    def __init__(self):
        self.rates = {
            'eks_per_cluster': 73.00,
            'ec2_per_instance': 100.00,  # m5.large spot
            'rds_postgres': 200.00,      # db.t3.medium
            'redis_cache': 100.00,       # cache.t3.micro
            'kafka_msk': 300.00,         # 2 brokers
            'pinecone': 70.00,           # Starter plan
            's3_storage': 0.023,         # per GB
            'data_transfer': 0.09,       # per GB
            'openai_gpt4': 0.06,         # per 1K tokens output
            'openai_gpt35': 0.002,       # per 1K tokens output
        }

    def calculate_monthly(self, tenants: int, messages_per_day: int) -> dict:
        """
        Calculate monthly operational costs
        """
        monthly_messages = messages_per_day * 30

        # Infrastructure (fixed)
        infra_cost = (
            self.rates['eks_per_cluster'] +
            self.rates['ec2_per_instance'] * 4 +  # 4 nodes
            self.rates['rds_postgres'] +
            self.rates['redis_cache'] * 2 +
            self.rates['kafka_msk'] +
            self.rates['pinecone']
        )

        # Variable costs
        storage_gb = tenants * 0.5  # 500MB per tenant
        storage_cost = storage_gb * self.rates['s3_storage']

        # AI costs (tiered)
        gpt4_messages = int(monthly_messages * 0.2)  # 20% complex
        gpt35_messages = int(monthly_messages * 0.8)  # 80% simple

        ai_cost = (
            gpt4_messages * self.rates['openai_gpt4'] * 0.5 +  # 500 tokens avg
            gpt35_messages * self.rates['openai_gpt35'] * 0.3   # 300 tokens avg
        )

        total = infra_cost + storage_cost + ai_cost

        return {
            'infrastructure': infra_cost,
            'storage': storage_cost,
            'ai_processing': ai_cost,
            'total': total,
            'cost_per_message': total / monthly_messages,
            'cost_per_tenant': total / tenants
        }

# Example: 100 tenants, 1000 messages/day
calc = CostCalculator()
costs = calc.calculate_monthly(tenants=100, messages_per_day=1000)
print(f"Total monthly: ${costs['total']:.2f}")
print(f"Cost per message: ${costs['cost_per_message']:.4f}")
```

## 8. Failure Scenarios & Recovery

### 8.1 Circuit Breaker Pattern

```python
# code-examples/circuit_breaker.py
from enum import Enum
import time
from typing import Callable, Optional
import asyncio

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject fast
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    """
    Prevent cascade failures in distributed system
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0

    async def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info(f"Circuit {self.name} entering HALF_OPEN")
            else:
                raise CircuitBreakerOpen(f"Circuit {self.name} is OPEN")

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                raise CircuitBreakerOpen(f"Circuit {self.name} half-open limit reached")
            self.half_open_calls += 1

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self._reset()
        else:
            self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit {self.name} reopened due to failure")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.error(f"Circuit {self.name} opened after {self.failure_count} failures")

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try again"""
        if not self.last_failure_time:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _reset(self):
        """Reset circuit to closed state"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        logger.info(f"Circuit {self.name} reset to CLOSED")

# Usage
facebook_cb = CircuitBreaker("facebook_api", failure_threshold=3)

async def post_reply(message_id: str, reply: str):
    return await facebook_cb.call(
        _actual_facebook_api_call,
        message_id,
        reply
    )
```

### 8.2 Dead Letter Queue Handling

```python
# code-examples/dlq_handler.py
from typing import Dict, Callable
import json
import asyncio

class DeadLetterQueueHandler:
    """
    Handle failed events with retry logic
    """

    def __init__(self, kafka_producer, redis_client):
        self.kafka = kafka_producer
        self.redis = redis_client
        self.max_retries = 3
        self.retry_delays = [60, 300, 900]  # 1min, 5min, 15min

    async def process_with_retry(
        self, 
        event: Dict, 
        handler: Callable,
        topic: str
    ):
        """
        Process event with automatic retry
        """
        retry_count = event.get('retry_count', 0)

        try:
            await handler(event)
            logger.info(f"Event {event['id']} processed successfully")

        except RetryableException as e:
            # Temporary failure, retry
            if retry_count < self.max_retries:
                await self._schedule_retry(event, retry_count, topic)
            else:
                await self._move_to_dlq(event, str(e))

        except NonRetryableException as e:
            # Permanent failure, move to DLQ immediately
            await self._move_to_dlq(event, str(e))

        except Exception as e:
            # Unknown error, treat as retryable
            logger.error(f"Unexpected error: {e}")
            if retry_count < self.max_retries:
                await self._schedule_retry(event, retry_count, topic)
            else:
                await self._move_to_dlq(event, str(e))

    async def _schedule_retry(self, event: Dict, retry_count: int, topic: str):
        """Schedule event for retry with exponential backoff"""
        event['retry_count'] = retry_count + 1
        delay = self.retry_delays[retry_count]

        # Use Redis delayed queue or Kafka scheduled messages
        await self.redis.zadd(
            f"retry_queue:{topic}",
            {json.dumps(event): time.time() + delay}
        )

        logger.info(f"Scheduled retry {retry_count + 1} for event {event['id']} in {delay}s")

    async def _move_to_dlq(self, event: Dict, error_reason: str):
        """Move permanently failed event to DLQ"""
        dlq_event = {
            **event,
            'error_reason': error_reason,
            'failed_at': datetime.utcnow().isoformat(),
            'final_retry_count': event.get('retry_count', 0)
        }

        await self.kafka.send('dead_letter_queue', dlq_event)

        # Alert on-call if critical
        if event.get('priority') == 'high':
            await self._send_alert(event, error_reason)

        logger.error(f"Event {event['id']} moved to DLQ: {error_reason}")

    async def _send_alert(self, event: Dict, error: str):
        """Send PagerDuty/Slack alert for critical failures"""
        # Integration with alerting service
        pass
```

## 9. Deployment Architecture

### 9.1 Kubernetes Manifest

```yaml
# infrastructure/k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-service
  namespace: production
  labels:
    app: ai-service
    version: v1.2.3
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: ai-service
  template:
    metadata:
      labels:
        app: ai-service
    spec:
      containers:
      - name: ai-service
        image: socialai/ai-service:v1.2.3
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: socialai-secrets
              key: DATABASE_URL
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: socialai-secrets
              key: OPENAI_API_KEY
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: tmp
          mountPath: /tmp
      volumes:
      - name: tmp
        emptyDir: {}
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - ai-service
              topologyKey: kubernetes.io/hostname
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: External
    external:
      metric:
        name: kafka_consumer_lag
      target:
        type: AverageValue
        averageValue: "1000"
```

## 10. Monitoring & Observability

### 10.1 Key Metrics

```python
# code-examples/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Business metrics
messages_processed = Counter(
    'socialai_messages_total',
    'Total messages processed',
    ['tenant_id', 'platform', 'intent']
)

lead_conversions = Counter(
    'socialai_leads_converted_total',
    'Leads converted to customers',
    ['tenant_id', 'source']
)

# Technical metrics
llm_latency = Histogram(
    'socialai_llm_duration_seconds',
    'LLM API latency',
    ['model', 'operation'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

database_connections = Gauge(
    'socialai_db_connections_active',
    'Active database connections',
    ['database']
)

# AI quality metrics
intent_confidence = Histogram(
    'socialai_intent_confidence',
    'Confidence score of intent classification',
    ['intent_type']
)

hallucination_detected = Counter(
    'socialai_hallucinations_total',
    'Detected hallucinations in replies',
    ['detection_method']
)
```

## 11. Conclusion

This enhanced system design provides:

1. **Scalable Architecture**: Microservices with clear boundaries
2. **Production Patterns**: Circuit breakers, DLQ, encryption
3. **Cost Control**: Tiered AI models, caching, spot instances
4. **Security**: Defense in depth, field-level encryption
5. **Observability**: Comprehensive metrics and tracing

**Evolution from Assessments 1-2:**
- Assessment 1's workflow → Distributed agent orchestrator
- Assessment 2's RAG → Multi-tenant knowledge base
- Combined into enterprise SaaS with CRM integration

**Next Steps:**
1. Build MVP with core social platform (Facebook)
2. Implement AI agents with conversation memory
3. Add CRM sync with conflict resolution
4. Scale with Kubernetes and monitoring
