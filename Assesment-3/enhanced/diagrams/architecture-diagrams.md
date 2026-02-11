# System Architecture Diagrams

## 1. Message Flow Sequence

```mermaid
sequenceDiagram
    participant User as Social Media User
    participant FB as Facebook/Twitter/LinkedIn
    participant Kong as API Gateway
    participant Social as Social Service
    participant Kafka as Kafka Queue
    participant AI as AI Orchestrator
    participant LLM as Gemini/OpenAI
    participant Vector as Pinecone Vector DB
    participant CRM as CRM Service
    participant SF as Salesforce/HubSpot

    User->>FB: Sends message
    FB->>Kong: Webhook POST
    Kong->>Social: Verify & Route
    Social->>Social: Verify signature
    Social->>Kafka: Publish message event
    Social-->>FB: 200 OK (fast response)
    
    Kafka->>AI: Consume message
    
    par Intent Classification
        AI->>LLM: Classify intent
        LLM-->>AI: {intent: sales, confidence: 0.9}
    and Lead Scoring
        AI->>AI: Extract signals
        AI->>AI: Calculate score
        Note over AI: hot/warm/cold
    end
    
    AI->>Vector: Query context (RAG)
    Vector-->>AI: Relevant knowledge
    
    AI->>LLM: Generate reply with context
    LLM-->>AI: Personalized response
    
    AI->>Social: Post reply
    Social->>FB: Send message via API
    FB-->>User: Deliver reply
    
    par CRM Sync
        AI->>CRM: Sync lead data
        CRM->>SF: Create/Update lead
        SF-->>CRM: Success
        CRM->>SF: Add activity log
    end
    
    Note over AI,CRM: Total latency: <2s
```

## 2. Tenant Isolation Architecture

```mermaid
graph TB
    subgraph " "
        Kong[API Gateway<br/>Tenant ID from JWT]
    end
    
    Kong --> |tenant_123| T1[Tenant 123 Namespace]
    Kong --> |tenant_456| T2[Tenant 456 Namespace]
    Kong --> |tenant_789| T3[Tenant 789 Namespace]
    
    subgraph T1[Tenant 123 Resources]
        DB1[(PostgreSQL<br/>Schema: tenant_123)]
        Cache1[(Redis<br/>Namespace: t123:*)]
        Vector1[(Pinecone<br/>Index: prod-t123)]
        CRM1[CRM Config<br/>Encrypted]
    end
    
    subgraph T2[Tenant 456 Resources]
        DB2[(PostgreSQL<br/>Schema: tenant_456)]
        Cache2[(Redis<br/>Namespace: t456:*)]
        Vector2[(Pinecone<br/>Index: prod-t456)]
        CRM2[CRM Config<br/>Encrypted]
    end
    
    subgraph T3[Tenant 789 Resources]
        DB3[(PostgreSQL<br/>Schema: tenant_789)]
        Cache3[(Redis<br/>Namespace: t789:*)]
        Vector3[(Pinecone<br/>Index: prod-t789)]
        CRM3[CRM Config<br/>Encrypted]
    end
    
    style T1 fill:#e1f5ff
    style T2 fill:#fff3e0
    style T3 fill:#f3e5f5
```

## 3. AI Agent Pipeline Detailed

```mermaid
flowchart TD
    Start([Message Received]) --> Parse[Parse & Normalize]
    Parse --> Store[(Store in PostgreSQL)]
    Store --> Queue{Priority Queue<br/>Based on Score}
    
    Queue -->|High Priority| Fast[Fast Lane<br/>Dedicated Workers]
    Queue -->|Normal| Standard[Standard Lane<br/>Shared Workers]
    
    Fast --> Agent1
    Standard --> Agent1
    
    subgraph "AI Agent Pipeline"
        Agent1[Agent 1:<br/>Intent Classifier]
        Agent2[Agent 2:<br/>Entity Extractor]
        Agent3[Agent 3:<br/>Lead Scorer]
        Agent4[Agent 4:<br/>Reply Generator]
        
        Agent1 -->|intent, confidence| Agent2
        Agent2 -->|entities| Agent3
        Agent3 -->|score: hot/warm/cold| Agent4
        
        Agent4 --> Cache{Cache Hit?}
        Cache -->|Yes| UseCached[Use Cached Reply]
        Cache -->|No| RAG[RAG: Query Vector DB]
        
        RAG --> LLM[Call LLM with Context]
        LLM --> CacheStore[Store in Cache]
        CacheStore --> UseCached
    end
    
    UseCached --> Validate{Confidence<br/>>= Threshold?}
    Validate -->|Yes| Post[Post Reply]
    Validate -->|No| Manual[Queue for<br/>Manual Review]
    
    Post --> CRMSync[Sync to CRM]
    Manual --> Alert[Alert Team]
    
    CRMSync --> End([Complete])
    Alert --> End
    
    style Agent1 fill:#bbdefb
    style Agent2 fill:#c5cae9
    style Agent3 fill:#d1c4e9
    style Agent4 fill:#f8bbd0
```

## 4. Failure Handling & Recovery

```mermaid
flowchart TD
    Start([Request]) --> Gateway[API Gateway]
    Gateway --> Circuit{Circuit<br/>Breaker<br/>Open?}
    
    Circuit -->|Closed| Service[Target Service]
    Circuit -->|Open| Fallback[Fallback Response]
    
    Service --> Try{Success?}
    Try -->|Yes| Success[Return Result]
    Try -->|No| Error[Log Error]
    
    Error --> Retry{Retry<br/>Count<br/>< Max?}
    Retry -->|Yes| Backoff[Exponential<br/>Backoff]
    Backoff --> Service
    
    Retry -->|No| DLQ[(Dead Letter<br/>Queue)]
    DLQ --> Alert[Alert<br/>On-Call Team]
    Alert --> Manual[Manual<br/>Intervention]
    
    Success --> End([Done])
    Fallback --> End
    Manual --> Replay[Replay from DLQ]
    Replay --> Service
    
    style Error fill:#ffcdd2
    style DLQ fill:#ff8a80
    style Alert fill:#ff5252
    style Success fill:#c8e6c9
```

## 5. Cost Optimization Flow

```mermaid
graph TD
    Query[User Query] --> Cache{Redis<br/>Cache?}
    Cache -->|Hit| Return[Return Cached<br/>Cost: $0]
    Cache -->|Miss| Classify[Classify<br/>Complexity]
    
    Classify --> Simple{Simple<br/>Query?}
    Simple -->|Yes| GPT35[GPT-3.5 Turbo<br/>Cost: $0.0015/1K tokens]
    Simple -->|No| Complex{Complex<br/>Reasoning?}
    
    Complex -->|Yes| GPT4[GPT-4<br/>Cost: $0.03/1K tokens]
    Complex -->|No| Gemini[Gemini Flash<br/>Cost: $0.0005/1K tokens]
    
    GPT35 --> Store[Store in Cache<br/>TTL: 1h]
    Gemini --> Store
    GPT4 --> Store
    
    Store --> Monitor{Usage<br/>> Budget?}
    Monitor -->|Yes| Throttle[Apply Rate Limits]
    Monitor -->|No| Allow[Allow Request]
    
    Throttle --> Return
    Allow --> Return
    
    style Return fill:#c8e6c9
    style Gemini fill:#81c784
    style GPT35 fill:#fff176
    style GPT4 fill:#ffb74d
```

## 6. Scaling Strategy

```mermaid
graph LR
    subgraph "Load Levels"
        L1[< 1K msgs/day<br/>Single Instance]
        L2[1K-10K msgs/day<br/>+Redis Cache]
        L3[10K-100K msgs/day<br/>+Kafka Partitions]
        L4[100K-1M msgs/day<br/>+Database Sharding]
        L5[> 1M msgs/day<br/>+Multi-Region]
    end
    
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> L5
    
    L1 -.->|Metrics| M1[Latency p95 > 500ms]
    L2 -.->|Metrics| M2[Cache hit < 60%]
    L3 -.->|Metrics| M3[Queue lag > 1min]
    L4 -.->|Metrics| M4[DB CPU > 80%]
    
    M1 --> L2
    M2 --> L3
    M3 --> L4
    M4 --> L5
    
    style L1 fill:#e8f5e9
    style L2 fill:#c8e6c9
    style L3 fill:#a5d6a7
    style L4 fill:#81c784
    style L5 fill:#66bb6a
```

## 7. Security Layers

```mermaid
graph TD
    Internet[Internet<br/>Untrusted] --> WAF[WAF<br/>DDoS Protection]
    WAF --> TLS[TLS 1.3<br/>Certificate Validation]
    TLS --> Kong[API Gateway<br/>JWT Verification]
    
    Kong --> RBAC{RBAC<br/>Check}
    RBAC -->|Authorized| RateLimit[Rate Limiting<br/>Per Tenant]
    RBAC -->|Denied| Block403[403 Forbidden]
    
    RateLimit --> Encrypt{Data<br/>Contains PII?}
    Encrypt -->|Yes| FLE[Field-Level<br/>Encryption]
    Encrypt -->|No| Service[Service Layer]
    
    FLE --> Service
    Service --> DB[(Database<br/>Encrypted at Rest)]
    
    DB --> Audit[Audit Log<br/>All Access]
    Audit --> SIEM[SIEM<br/>Monitoring]
    
    style WAF fill:#ef5350
    style RBAC fill:#ff7043
    style FLE fill:#ffa726
    style DB fill:#66bb6a
```
