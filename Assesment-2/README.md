# Assessment 2: RAG Chatbot for Mysoft Heaven (BD) Ltd.

## Overview
A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly based on company-provided documents. Built with human-like pragmatism and clean architecture.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Frontend  │────▶│  FastAPI     │────▶│   FAISS     │
│  (HTML/JS)  │     │   Backend    │     │   Index     │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Sentence    │
                     │ Transformers │
                     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  LLM API     │
                     └──────────────┘
```

## Key Design Decisions

### 1. Document Chunking Strategy
**Approach**: Recursive sentence-based chunking with overlap
- **Why**: Preserves semantic coherence better than fixed-size chunks
- **Chunk size**: 500 characters (tunable via env var)
- **Overlap**: 100 characters (maintains context between chunks)
- **Strategy**: Sentence boundaries respected, prevents mid-sentence cuts

**Alternative considered**: Fixed-size token chunking - rejected because it breaks semantic flow.

### 2. Embedding Model Choice
**Selected**: `all-MiniLM-L6-v2` (Sentence Transformers)

**Rationale**:
- **Performance**: Good balance of speed and accuracy
- **Size**: 22MB (lightweight, fast loading)
- **Dimension**: 384 (efficient for FAISS)
- **Multilingual**: Supports English well (company docs are English)
- **Proven**: Industry standard, well-tested

**Alternatives considered**:
- `all-mpnet-base-v2`: Better accuracy but slower (420MB)
- OpenAI embeddings: API dependency, latency issues
- Custom training: Overkill for this use case

### 3. Irrelevant Query Handling
**Multi-layer defense**:
1. **Retrieval scoring**: FAISS returns similarity scores
2. **Confidence threshold**: 0.7 (tunable)
3. **Explicit guardrails**: System prompt instructs LLM to reject out-of-scope questions
4. **Fallback response**: Polite message directing user to company-related questions

**Example rejection**:
> "I'm sorry, but I can only answer questions about Mysoft Heaven (BD) Ltd..."

### 4. Multi-Company Support
The architecture supports multiple companies through:
- **Namespace isolation**: Each company gets separate FAISS index
- **Document routing**: API endpoint accepts `company_id` parameter (easy to add)
- **Configurable prompts**: System prompts loaded per-company
- **Shared infrastructure**: Same backend, different vector stores

**Implementation path**:
```python
# Add to ChatRequest
company_id: str = Field(default="mysoft")

# Route to company-specific index
index_path = f"vector_stores/{company_id}"
```

## Project Structure

```
Assesment-2/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── .env.example         # Configuration template
├── frontend/
│   └── index.html           # Simple web UI
├── data/
│   └── company_profile.txt  # Company knowledge base
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your OpenAI API key or local LLM endpoint
```

### 3. Run Backend
```bash
uvicorn main:app --reload
```

The server will:
- Load company documents from `data/`
- Chunk and embed them
- Build FAISS index (saved to `vector_store/`)

### 4. Open Frontend
Open `frontend/index.html` in your browser (or serve with any static server).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Main chat endpoint |
| `/health` | GET | Health check with index status |
| `/stats` | GET | Document store statistics |

### Example API Call
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What services does Mysoft Heaven provide?",
    "session_id": "user_123"
  }'
```

## Conversation Memory
- **Storage**: In-memory dict (per-session)
- **Retention**: Last 10 exchanges (20 messages)
- **Persistence**: None (resets on server restart)
- **Production upgrade**: Add Redis backend

## Confidence Scoring
- **High (≥0.8)**: Green badge - High relevance
- **Medium (0.6-0.8)**: Orange badge - Moderate relevance
- **Low (<0.6)**: Red badge - May be off-topic (rejected)

## Testing the System

### Valid Questions (Should answer well)
1. "What is SEBA ERP?"
2. "List your government clients"
3. "How long has Mysoft Heaven been in business?"
4. "What is your contact information?"

### Invalid Questions (Should reject)
1. "What is the weather today?"
2. "Tell me about Microsoft"
3. "How do I cook pasta?"
4. "Who won the World Cup?"

## Bonus Features Implemented

✅ **Conversation Memory**: Maintains context across messages
✅ **Confidence-based Responses**: Visual indicators + filtering
✅ **Source Attribution**: Shows which documents were used
✅ **Streaming Ready**: Endpoint structure supports SSE
✅ **Session Management**: Unique session IDs for users

## Production Considerations

1. **Vector Store**: Migrate to Pinecone/Weaviate for persistence
2. **Memory**: Replace in-memory history with Redis
3. **Monitoring**: Add structured logging and metrics
4. **Rate Limiting**: Prevent API abuse
5. **Authentication**: Add user auth for multi-tenancy
6. **Caching**: Cache frequent queries

## Cost Optimization

- **Embedding model**: Local (free) vs API ($)
- **LLM**: GPT-3.5 (cheap) with short prompts
- **Chunk size**: Optimized to reduce token usage
- **Top-K**: Only retrieve 3 most relevant chunks

## Demo Video Script

1. **Setup** (1m): Show dependency installation and index building
2. **Valid Query** (1m): Ask about SEBA ERP, show sources and confidence
3. **Invalid Query** (1m): Ask about weather, show rejection
4. **Conversation** (1m): Show memory with follow-up questions
5. **Architecture** (1m): Explain chunking and embedding choices
