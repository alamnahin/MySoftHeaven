## Overview
A production-ready lead processing system with FastAPI backend and n8n workflow integration. Built with human-like code structure - iterative, pragmatic, and well-commented.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Webhook   │────▶│  FastAPI     │────▶│   SQLite    │
│   / Form    │     │  Backend     │     │   Database  │
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  LLM API     │
                     │  (OpenAI)    │
                     └──────────────┘
```

## Quick Start

### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2. Configuration
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run Application
```bash
uvicorn main:app --reload
```

### 4. Test
```bash
pytest test_main.py -v
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/webhook/lead` | POST | Submit new lead |
| `/leads/{id}` | GET | Get lead details |
| `/health` | GET | Health check |

## Design Decisions

### Prompt Strategy
1. **Structured Output**: Force JSON format with explicit schema
2. **Few-shot examples**: Implicit in prompt structure
3. **Temperature 0.3**: Balance creativity with consistency
4. **Separate prompts**: Classification and response generation decoupled

### Hallucination Reduction
- **Confidence thresholding**: < 0.6 confidence → "Unclear"
- **Validation layer**: Pydantic models enforce data types
- **Conservative extraction**: Null rather than guess
- **Structured prompts**: Reduce free-form generation

### Error Handling
- **Retry with backoff**: 3 attempts, exponential delay
- **Graceful degradation**: Fallback responses if LLM fails
- **Background processing**: Webhook returns immediately
- **Status tracking**: Database tracks processing state

## n8n Integration

Import `n8n-workflows/lead-processing.json` into your n8n instance.

**Note**: The backend works standalone; n8n is optional for visual workflow management.

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=main --cov-report=html

# Specific test
pytest test_main.py::test_receive_lead_success -v
```

## Production Considerations

1. **Database**: Migrate to PostgreSQL for concurrency
2. **Queue**: Add Redis/RabbitMQ for background tasks
3. **Monitoring**: Implement structured logging (JSON)
4. **Rate limiting**: Add middleware for webhook protection
5. **Secrets**: Use Vault or AWS Secrets Manager

## Demo Video Script

1. **Setup** (30s): Show installation and env setup
2. **API Test** (1m): Submit lead via curl/Postman
3. **Database** (30s): Show SQLite entry
4. **Error handling** (1m): Disconnect internet, show retry
5. **n8n** (1m): Import workflow, show visual editor