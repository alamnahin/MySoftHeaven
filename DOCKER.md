# 🐳 Docker Deployment Guide

Complete Docker setup for all three assessments.

## Prerequisites

- Docker Desktop installed (macOS/Windows) or Docker Engine (Linux)
- Docker Compose v2.0+
- Gemini API key (free tier available)

## Quick Start

### Assessment 1: Lead Processing System

```bash
cd Assesment-1

# Create environment file
cp .env.example .env
# Edit .env and add your Gemini API key

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Test the API
curl http://localhost:8000/health

# Stop the service
docker-compose down
```

**Access Points:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Database Viewer (dev): http://localhost:8080 (run with `docker-compose --profile dev up`)

### Assessment 2: RAG Chatbot

```bash
cd Assesment-2

# Create environment file
cp .env.example .env
# Edit .env and add your Gemini API key

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Test the API
curl http://localhost:8000/health

# Stop all services
docker-compose down
```

**Access Points:**
- Frontend: http://localhost:8080
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Assessment 3: System Design Reference

Assessment 3 contains system design documentation and code examples. To run the reference implementation:

```bash
cd Assesment-3/enhanced/code-examples

# Install dependencies
pip install -r requirements.txt

# Run individual services (requires Redis, PostgreSQL, etc.)
python ai_orchestrator.py
python social_service.py
python crm_service.py
```

For full production deployment, see [infrastructure/README.md](Assesment-3/enhanced/infrastructure/README.md).

## Configuration

### Environment Variables

Both Assessment-1 and Assessment-2 use the following key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM provider (gemini/openai) | gemini |
| `LLM_API_KEY` | API key for LLM service | (required) |
| `LLM_MODEL` | Model name | gemini-3-flash-preview |
| `LOG_LEVEL` | Logging level | INFO |

Assessment-2 additional variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | Sentence transformer model | all-MiniLM-L6-v2 |
| `CHUNK_SIZE` | Document chunk size | 500 |
| `CHUNK_OVERLAP` | Chunk overlap | 100 |
| `TOP_K` | Number of results to retrieve | 3 |
| `CONFIDENCE_THRESHOLD` | Minimum similarity score | 0.3 |

### Getting a Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and add it to your `.env` file

## Docker Commands Reference

### Build and Start

```bash
# Build images
docker-compose build

# Start in foreground
docker-compose up

# Start in background
docker-compose up -d

# Rebuild and start
docker-compose up --build
```

### Monitoring

```bash
# View logs (all services)
docker-compose logs -f

# View logs (specific service)
docker-compose logs -f backend

# View container status
docker-compose ps

# Execute command in container
docker-compose exec backend /bin/bash
```

### Maintenance

```bash
# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Restart services
docker-compose restart
```

### Troubleshooting

```bash
# Check container health
docker-compose ps

# View container logs
docker-compose logs backend

# Enter container shell
docker-compose exec backend sh

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

## Production Considerations

### Assessment 1 (Lead Processing)

**Before Production:**
- [ ] Replace SQLite with PostgreSQL or MySQL
- [ ] Add proper logging and monitoring
- [ ] Implement rate limiting
- [ ] Set up backup strategy for database
- [ ] Configure reverse proxy (nginx/Traefik)
- [ ] Enable HTTPS with valid certificates
- [ ] Set up automated testing in CI/CD

**Scaling:**
- Horizontal: Multiple API instances behind load balancer
- Database: Read replicas for heavy read workloads
- Caching: Redis for frequently accessed data

### Assessment 2 (RAG Chatbot)

**Before Production:**
- [ ] Pre-build vector embeddings at deployment time
- [ ] Use managed vector database (Pinecone, Weaviate)
- [ ] Implement request queuing for heavy loads
- [ ] Add authentication for chat endpoint
- [ ] Set up CDN for frontend assets
- [ ] Monitor embedding model performance
- [ ] Implement feedback loop for RAG quality

**Scaling:**
- Vector Store: Move to Pinecone/Weaviate for distributed search
- Embedding: Cache embeddings in Redis
- Backend: Multiple instances with sticky sessions
- Frontend: Serve from CDN (CloudFront, Cloudflare)

### Security Checklist

- [ ] Never commit `.env` files with real API keys
- [ ] Rotate API keys regularly
- [ ] Use Docker secrets for sensitive data in production
- [ ] Enable CORS only for trusted domains
- [ ] Implement rate limiting per IP/user
- [ ] Add input validation and sanitization
- [ ] Enable security headers (HSTS, CSP, etc.)
- [ ] Regular dependency updates (Dependabot)
- [ ] Container vulnerability scanning

## Resource Requirements

### Assessment 1
- **CPU:** 0.5-1 core
- **RAM:** 512MB-1GB
- **Storage:** 1GB (including database)
- **Network:** Minimal (LLM API calls)

### Assessment 2
- **CPU:** 1-2 cores (for embedding model)
- **RAM:** 1-2GB (sentence transformers + FAISS)
- **Storage:** 2GB (models + vector store)
- **Network:** Moderate (LLM API calls)

### Assessment 3 (Full System)
- **CPU:** 4-8 cores
- **RAM:** 8-16GB
- **Storage:** 50GB+ (databases + logs)
- **Network:** High (external API integrations)

## Performance Benchmarks

### Assessment 1
- Lead classification: ~1-2s per request (LLM latency)
- Throughput: 50-100 requests/minute (single instance)
- Database writes: <100ms

### Assessment 2
- Document embedding: ~30s for 50 pages (one-time)
- Similarity search: ~50-100ms (FAISS)
- Answer generation: ~2-3s (LLM latency)
- End-to-end query: ~3-4s

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Verify environment variables: `docker-compose config`
3. Test health endpoints: `curl http://localhost:8000/health`
4. Review API docs: http://localhost:8000/docs

## License

See individual assessment READMEs for specific licensing information.
