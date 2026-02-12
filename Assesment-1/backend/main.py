from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from datetime import datetime
import sqlite3
import json
import os
import httpx
import asyncio
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database setup
def _sqlite_db_path(database_url: str) -> str:
    if database_url.startswith("sqlite:////"):
        return "/" + database_url[len("sqlite:////") :]
    if database_url.startswith("sqlite:///"):
        return database_url[len("sqlite:///") :]
    return "leads.db"

def _get_db_path() -> str:
    database_url = os.getenv("DATABASE_URL", "sqlite:///./leads.db")
    return _sqlite_db_path(database_url)

def init_db():
    """Initialize SQLite database with proper schema"""
    conn = sqlite3.connect(_get_db_path())
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_message TEXT NOT NULL,
            source TEXT,
            metadata TEXT,
            intent TEXT CHECK(intent IN ('Sales', 'Support', 'Spam', 'Unclear')),
            name TEXT,
            company TEXT,
            requirement TEXT,
            email TEXT,
            phone TEXT,
            confidence_score REAL,
            ai_response TEXT,
            status TEXT DEFAULT 'pending',
            retry_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    init_db()
    yield

app = FastAPI(
    title="Lead Processing API",
    description="AI-powered lead classification and response system",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === CONFIGURATION & CLIENT SETUP ===
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash")

# Initialize Gemini Client
gemini_client = None
if LLM_PROVIDER == "gemini":
    if not LLM_API_KEY:
        logger.warning("Gemini API Key is missing! Check your .env file.")
    gemini_client = genai.Client(api_key=LLM_API_KEY)

# OpenAI Config (Legacy support)
OPENAI_ENDPOINT = os.getenv("LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")

class LeadMessage(BaseModel):
    """Lead message model with validation"""
    message: str = Field(..., min_length=10, max_length=5000)
    source: Optional[str] = Field(default="web", description="Source of the lead")
    metadata: Optional[dict] = Field(default={}, description="Additional metadata")
    
    @field_validator('message')
    def validate_message(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Message too short')
        return v.strip()

class LeadResponse(BaseModel):
    """Response model for processed leads"""
    lead_id: int
    intent: str
    extracted_data: dict
    confidence: float
    auto_response: str
    status: str

class Database:
    """Simple database wrapper"""
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
    
    def get_conn(self):
        conn = sqlite3.connect(self.db_path or _get_db_path())
        conn.row_factory = sqlite3.Row
        return conn
    
    def save_lead(self, data: dict) -> int:
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO leads (raw_message, source, metadata, status, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            data['message'],
            data.get('source', 'web'),
            json.dumps(data.get('metadata') or {}),
            'processing',
            datetime.now()
        ))
        lead_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return lead_id
    
    def update_lead(self, lead_id: int, data: dict):
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE leads 
            SET intent = ?, name = ?, company = ?, requirement = ?,
                email = ?, phone = ?, confidence_score = ?, 
                ai_response = ?, status = ?, processed_at = ?
            WHERE id = ?
        ''', (
            data.get('intent'), data.get('name'), data.get('company'),
            data.get('requirement'), data.get('email'), data.get('phone'),
            data.get('confidence'), data.get('ai_response'),
            data.get('status'), datetime.now(), lead_id
        ))
        conn.commit()
        conn.close()
    
    def increment_retry(self, lead_id: int):
        conn = self.get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE leads SET retry_count = retry_count + 1 WHERE id = ?
        ''', (lead_id,))
        conn.commit()
        conn.close()

db = Database()

# === PROMPTS ===
CLASSIFICATION_PROMPT = """You are a lead classification assistant. Analyze the following message and extract information.

Message: {message}

Respond ONLY with a JSON object in this exact format:
{{
    "intent": "Sales|Support|Spam|Unclear",
    "confidence": 0.0-1.0,
    "extracted_fields": {{
        "name": "person name or null",
        "company": "company name or null", 
        "requirement": "brief summary of need or null",
        "email": "email or null",
        "phone": "phone or null"
    }},
    "reasoning": "brief explanation"
}}

Rules:
- Intent "Sales" for purchase inquiries, demos, pricing questions
- Intent "Support" for technical issues, help requests
- Intent "Spam" for unsolicited marketing, irrelevant content
- Intent "Unclear" if unable to determine
- Be conservative with confidence scores
- If information is missing, use null, don't guess"""

RESPONSE_PROMPT = """You are a helpful assistant responding to a lead inquiry.

Original Message: {message}
Intent: {intent}
Extracted Info: {extracted}

Write a brief, professional response (2-3 sentences max).
- If Sales: Acknowledge interest, suggest next steps
- If Support: Acknowledge issue, provide immediate help or escalation
- If Spam: Polite decline
- If Unclear: Ask clarifying questions

Response must be concise and natural."""

async def call_llm(prompt: str, max_retries: int = 3) -> dict:
    """
    Call LLM with retry logic.
    Uses google-genai SDK for Gemini and httpx for OpenAI.
    """
    for attempt in range(max_retries):
        try:
            content = ""
            
            if LLM_PROVIDER == "gemini":
                # === NEW GEMINI SDK IMPLEMENTATION ===
                if not gemini_client:
                    return {"success": False, "error": "Gemini client not initialized"}
                
                response = await gemini_client.aio.models.generate_content(
                    model=LLM_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=2000, 
                        response_mime_type="application/json"
                    )
                )
                content = response.text
                
            else:
                headers = {
                    "Authorization": f"Bearer {LLM_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1000 
                }
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        OPENAI_ENDPOINT, 
                        headers=headers, 
                        json=payload
                    )
                    response.raise_for_status()
                    result = response.json()
                    content = result['choices'][0]['message']['content']

            try:
                cleaned_content = content.strip()
                if '```json' in cleaned_content:
                    cleaned_content = cleaned_content.split('```json')[1].split('```')[0]
                elif '```' in cleaned_content:
                    cleaned_content = cleaned_content.split('```')[1].split('```')[0]
                
                parsed = json.loads(cleaned_content.strip())
                return {"success": True, "data": parsed}
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}, content: {content}")
                if attempt == max_retries - 1:
                    return {"success": False, "error": "Invalid JSON from LLM"}

        except Exception as e:
            logger.error(f"LLM call error: {e}")
            if attempt == max_retries - 1:
                return {"success": False, "error": str(e)}
            
            # Exponential backoff 
            wait_time = (2 ** attempt) + 1
            logger.info(f"Retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
    
    return {"success": False, "error": "Max retries exceeded"}

async def process_lead_async(lead_id: int, message: str):
    """Background task to process lead with LLM"""
    try:
        # Step 1: Classification
        classification_result = await call_llm(
            CLASSIFICATION_PROMPT.format(message=message)
        )
        
        if not classification_result["success"]:
            logger.error(f"Classification failed: {classification_result.get('error')}")
            db.update_lead(lead_id, {
                "status": "failed_classification",
                "ai_response": "We encountered an issue processing your request. Our team will follow up manually."
            })
            return
        
        classification = classification_result["data"]
        
        # Validation layer
        intent = classification.get("intent", "Unclear")
        confidence = float(classification.get("confidence", 0))
        extracted = classification.get("extracted_fields", {})
        
        # Hallucination reduction: confidence threshold
        if confidence < 0.6:
            intent = "Unclear"
            logger.info(f"Low confidence ({confidence}), marking as Unclear")
        
        # Step 2: Generate response
        response_result = await call_llm(
            RESPONSE_PROMPT.format(
                message=message,
                intent=intent,
                extracted=json.dumps(extracted)
            )
        )
        
        if response_result["success"]:
            auto_response = response_result["data"]
            if isinstance(auto_response, dict):
                auto_response = auto_response.get("response", str(auto_response))
        else:
            # Fallback response
            auto_response = "Thank you for your message. Our team will review and respond shortly."
        
        # Update database
        db.update_lead(lead_id, {
            "intent": intent,
            "name": extracted.get("name"),
            "company": extracted.get("company"),
            "requirement": extracted.get("requirement"),
            "email": extracted.get("email"),
            "phone": extracted.get("phone"),
            "confidence": confidence,
            "ai_response": auto_response,
            "status": "completed"
        })
        
        logger.info(f"Lead {lead_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Processing error for lead {lead_id}: {e}")
        db.increment_retry(lead_id)
        db.update_lead(lead_id, {
            "status": "failed",
            "ai_response": "Processing error occurred."
        })

@app.post("/webhook/lead", response_model=LeadResponse)
async def receive_lead(
    lead: LeadMessage,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    Webhook endpoint to receive new leads.
    Returns immediately, processes asynchronously.
    """
    try:
        # Save to database immediately
        lead_id = db.save_lead({
            "message": lead.message,
            "source": lead.source,
            "metadata": lead.metadata
        })
        
        # Queue background processing
        background_tasks.add_task(process_lead_async, lead_id, lead.message)
        
        return LeadResponse(
            lead_id=lead_id,
            intent="processing",
            extracted_data={},
            confidence=0.0,
            auto_response="Processing...",
            status="received"
        )
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.get("/leads/{lead_id}")
async def get_lead(lead_id: int):
    """Get lead status and details"""
    conn = db.get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM leads WHERE id = ?", (lead_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Lead not found")
    
    return dict(row)

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)