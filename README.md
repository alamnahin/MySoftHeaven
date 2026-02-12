# 🎯 MySoftHeaven - AI-Powered Solutions Portfolio

Complete assessment portfolio showcasing production-ready AI/ML implementations with modern DevOps practices.

## 📋 Overview
- **Demo Video**: [View here](https://drive.google.com/drive/folders/1CpiRAq4TC9cOV-l9MD3exgQ71J18h_H_?usp=drive_link)

This repository contains three comprehensive assessments demonstrating:
- ✅ AI-powered lead processing with multi-provider LLM support
- ✅ Production RAG (Retrieval-Augmented Generation) chatbot
- ✅ Enterprise-grade system design for multi-tenant SaaS platform

All assessments use **free Gemini API** (no credit card required) and include complete Docker deployments.

## 🗂️ Project Structure

```
MySoftHeaven/
├── Assesment-1/          # AI Lead Processing System
│   ├── backend/          # FastAPI + SQLite + Gemini
│   ├── docs/             # Documentation
│   ├── n8n-workflows/    # Workflow automation
│   └── docker-compose.yml
│
├── Assesment-2/          # RAG Chatbot
│   ├── backend/          # FAISS + Sentence Transformers
│   ├── frontend/         # HTML/CSS/JS interface
│   ├── data/             # Company knowledge base
│   └── docker-compose.yml
│
├── Assesment-3/          # System Design Documentation
│   └── enhanced/
│       ├── code-examples/     # Production code samples
│       ├── diagrams/          # Architecture diagrams
│       └── infrastructure/    # Deployment configs
│
├── DOCKER.md            # Complete Docker guide
└── README.md            # This file
```

## 🚀 Quick Start

### Prerequisites

- **Docker Desktop** (macOS/Windows) or **Docker Engine** (Linux)
- **Gemini API Key** (free): [Get one here](https://makersuite.google.com/app/apikey)
- **Python 3.11+** (for local development)

### 1️⃣ Assessment-1: Lead Processing System

**What it does:**
- Classifies incoming leads as qualified/unqualified using AI
- Generates personalized responses based on lead characteristics
- Stores leads in SQLite with full CRUD operations
- Supports both Gemini and OpenAI LLMs

**Run with Docker:**
```bash
cd Assesment-1
cp .env.example .env
# Edit .env and add your Gemini API key

docker-compose up -d
```

**Access:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

**Test:**
```bash
# Run automated tests
cd backend
pytest -v

# All 5 tests should pass ✅
```

### 2️⃣ Assessment-2: RAG Chatbot

**What it does:**
- Answers questions about your company using uploaded documents
- Uses FAISS for fast vector similarity search
- Generates contextual responses with Gemini LLM
- Provides confidence scores for answer quality

**Run with Docker:**
```bash
cd Assesment-2
cp .env.example .env
# Edit .env and add your Gemini API key

docker-compose up -d
```

**Access:**
- Frontend: http://localhost:8080
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### 3️⃣ Assessment-3: System Design

**What it contains:**
- Complete architecture for multi-tenant SaaS platform
- Production-ready code examples (AI orchestrator, webhooks, CRM sync)
- 7+ architecture diagrams (Mermaid format)
- Complete database ERD with multi-tenant isolation

**Explore:**
```bash
cd Assesment-3
cat enhanced/system-design-enhanced.md
```

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [DOCKER.md](DOCKER.md) | Complete Docker deployment guide |
| [Assesment-1/README.md](Assesment-1/README.md) | Lead processing API documentation |
| [Assesment-2/README.md](Assesment-2/README.md) | RAG chatbot setup and usage |
| [Assesment-3/README.md](Assesment-3/README.md) | System design overview |

## 🛠️ Technology Stack

### Assessment-1
- **Backend:** FastAPI, SQLite, Pydantic v2
- **LLM:** Gemini 3-flash-preview (free tier)
- **Testing:** pytest with 5 test cases
- **Deployment:** Docker + Docker Compose

### Assessment-2
- **Backend:** FastAPI, FAISS, Sentence Transformers 2.7.0
- **Frontend:** HTML/CSS/JS (vanilla)
- **LLM:** Gemini 3-flash-preview
- **Embeddings:** all-MiniLM-L6-v2
- **Testing:** pytest-asyncio with 8 test suites
- **Deployment:** Multi-container Docker

### Assessment-3
- **Architecture:** Microservices, Event-Driven
- **Task Queue:** Celery + Redis
- **API Gateway:** Kong (JWT auth, rate limiting)
- **Databases:** PostgreSQL, Pinecone
- **Observability:** Prometheus, Grafana, ELK

## 🧪 Testing

All assessments include comprehensive tests:

```bash
# Assessment-1: Unit tests
cd Assesment-1/backend
pytest -v
# ✅ 5/5 tests passing

# Assessment-2: Integration tests
cd Assesment-2/backend
pytest test_integration.py -v
# ✅ 8 test classes, all passing
```

## 🐳 Docker Deployment

All services are containerized with production-ready configurations:

- ✅ Health checks and automatic restarts
- ✅ Resource limits (CPU/memory)
- ✅ Volume persistence
- ✅ Network isolation
- ✅ Environment-based configuration

See [DOCKER.md](DOCKER.md) for complete deployment guide.

## 📊 API Examples

### Assessment-1: Create Lead

```bash
curl -X POST http://localhost:8000/leads \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "message": "Interested in enterprise plan",
    "source": "website"
  }'
```

### Assessment-2: Chat Query

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What services does your company offer?"
  }'
```

## 🎯 Assessment Highlights

### Assessment-1: Production Best Practices
- ✅ Environment-based configuration
- ✅ Comprehensive error handling
- ✅ Input validation with Pydantic v2
- ✅ Automated testing with pytest
- ✅ API documentation with OpenAPI
- ✅ Dual LLM provider support (Gemini/OpenAI)

### Assessment-2: Advanced RAG Implementation
- ✅ Efficient document chunking strategy
- ✅ Vector similarity search (FAISS)
- ✅ Context-aware answer generation
- ✅ Confidence scoring system
- ✅ Persistent vector storage
- ✅ Integration test coverage

### Assessment-3: Enterprise Architecture
- ✅ Multi-tenant isolation (schema-per-tenant)
- ✅ Distributed task processing (Celery)
- ✅ Event-driven architecture (Kafka)
- ✅ API Gateway patterns (Kong)
- ✅ Observability stack (Prometheus, ELK)
- ✅ Cost optimization strategies

## 💰 Cost Analysis

All assessments use **free-tier services**:

| Service | Tier | Monthly Cost |
|---------|------|--------------|
| Gemini API | Free | $0 (15 RPM, 1M tokens/day) |
| FAISS | Local | $0 (in-memory) |
| SQLite | Local | $0 (file-based) |
| Docker | CE | $0 (Community Edition) |
| **Total** | | **$0** |

For production scaling (Assessment-3):
- **100 tenants:** ~$2,847/month
- **Per tenant:** $28.47/month
- **Revenue:** $9,900/month (at $99/month)
- **Profit margin:** 71.5%

## 🔐 Security

All assessments implement security best practices:

- ✅ API key management via environment variables
- ✅ Input validation with Pydantic models
- ✅ SQL injection prevention
- ✅ CORS configuration
- ✅ No secrets in version control
- ✅ Health check endpoints

## 📈 Performance Benchmarks

**Assessment-1:**
- Lead classification: ~1-2s
- Database operations: <100ms
- Throughput: 50-100 req/min

**Assessment-2:**
- Document embedding: ~30s (one-time)
- Vector search: ~50-100ms
- Answer generation: ~2-3s
- Total query: ~3-4s

**Assessment-3 (Design):**
- Message throughput: 15K/day per tenant
- AI pipeline: 3.2s (p95)
- Uptime: 99.95%

## 🐛 Troubleshooting

### Common Issues

**1. Module not found:**
```bash
source env/bin/activate
pip install -r requirements.txt
```

**2. Docker won't start:**
```bash
docker-compose logs -f
docker-compose down -v
docker-compose up --build
```

**3. Gemini API errors:**
- Verify API key is valid
- Check rate limits (15 RPM free tier)

**4. FAISS not loading:**
```bash
cd Assesment-2/backend
rm -rf vector_store/
# Restart service to rebuild
```

## 🎓 Learning Resources

**Concepts Demonstrated:**
- RAG (Retrieval-Augmented Generation)
- Vector embeddings and similarity search
- Multi-provider LLM integration
- Multi-tenant architecture patterns
- Event-driven microservices
- API Gateway patterns
- Container orchestration

## 📞 Support

For questions or issues:

1. **Check Documentation:** Review README files
2. **Check Logs:** `docker-compose logs -f`
3. **API Docs:** Visit `/docs` endpoint
4. **Run Tests:** `pytest -v`

## 📝 License

This is portfolio/assessment work provided as reference material.

## 🙏 Acknowledgments

- **LLM Provider:** Google Gemini (free tier)
- **Embedding Model:** Sentence Transformers (Hugging Face)
- **Vector Store:** FAISS (Meta AI)
- **Framework:** FastAPI

---

**Built with ❤️ for demonstrating production-ready AI/ML engineering skills**

Last Updated: 2026
 