from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import sqlite3
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lead Processing API (n8n Integration)")

# --- Database Helper ---
def get_db_path():
    # Handles paths for Docker vs Local
    url = os.getenv("DATABASE_URL", "leads.db")
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "")
    return url

def init_db():
    """Ensure table exists"""
    db_path = get_db_path()
    # Ensure directory exists
    if "/" in db_path:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
    with sqlite3.connect(db_path) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_message TEXT,
                source TEXT,
                intent TEXT,
                confidence_score REAL,
                ai_response TEXT,
                metadata TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
    logger.info(f"Database initialized at {db_path}")

@app.on_event("startup")
async def startup_event():
    init_db()

# --- Data Models ---
class N8NPayload(BaseModel):
    """Data coming from n8n"""
    message: str
    intent: str
    confidence: float
    response: str
    source: str = "n8n"

# --- Endpoints ---
@app.post("/leads/store")
async def store_lead(payload: N8NPayload):
    """
    n8n calls this endpoint to save the finished result.
    We do NOT call the LLM here. We just save.
    """
    try:
        with sqlite3.connect(get_db_path()) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO leads 
                (raw_message, source, intent, confidence_score, ai_response, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                payload.message,
                payload.source,
                payload.intent,
                payload.confidence,
                payload.response,
                "completed",
                datetime.now()
            ))
            lead_id = cursor.lastrowid
            conn.commit()
            
        logger.info(f"Saved lead {lead_id} from n8n")
        return {"status": "success", "lead_id": lead_id}
        
    except Exception as e:
        logger.error(f"Error saving lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "active", "mode": "database-service-only"}