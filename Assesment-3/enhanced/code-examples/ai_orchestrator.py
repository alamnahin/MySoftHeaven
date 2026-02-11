"""
AI Agent Orchestrator - Core agent task routing and execution
Built on Celery for distributed task processing with retry logic
"""
from celery import Celery, Task
from typing import Dict, Optional, List
import logging
from dataclasses import dataclass
from datetime import datetime
import httpx
import os

# Celery app configuration
celery_app = Celery(
    'ai_orchestrator',
    broker=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
    backend=os.getenv('REDIS_URL', 'redis://localhost:6379/0')
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_acks_late=True,  # Reliability: acknowledge only after success
    worker_prefetch_multiplier=1,  # Process one at a time for AI tasks
)

logger = logging.getLogger(__name__)

@dataclass
class Message:
    id: str
    tenant_id: str
    platform: str  # facebook, twitter, linkedin
    content: str
    sender_id: str
    conversation_id: str
    timestamp: datetime

@dataclass
class AgentResponse:
    intent: str  # sales, support, general
    confidence: float
    reply_text: str
    lead_score: str  # hot, warm, cold
    extracted_entities: Dict

class AIAgentTask(Task):
    """Base task with automatic retry and error handling"""
    autoretry_for = (httpx.TimeoutException, ConnectionError)
    retry_kwargs = {'max_retries': 3}
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True

@celery_app.task(base=AIAgentTask, bind=True)
def classify_intent(self, message: Dict) -> Dict:
    """
    Agent 1: Intent Classification
    Determines the purpose of the incoming message
    """
    try:
        msg = Message(**message)
        logger.info(f"Classifying intent for message {msg.id}")
        
        # Call LLM for classification
        prompt = f"""Classify this message intent:
Message: {msg.content}
Platform: {msg.platform}

Reply with JSON:
{{"intent": "sales|support|general", "confidence": 0.0-1.0, "reasoning": "..."}}"""

        response = _call_llm(prompt, tenant_id=msg.tenant_id)
        
        return {
            "message_id": msg.id,
            "intent": response["intent"],
            "confidence": response["confidence"],
            "reasoning": response.get("reasoning")
        }
        
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        # Fallback: default to general with low confidence
        return {
            "message_id": message["id"],
            "intent": "general",
            "confidence": 0.3,
            "reasoning": "Classification failed, using fallback"
        }

@celery_app.task(base=AIAgentTask, bind=True)
def generate_reply(self, message: Dict, intent_result: Dict) -> Dict:
    """
    Agent 2: Reply Generation
    Creates contextually appropriate responses based on intent
    """
    try:
        msg = Message(**message)
        intent = intent_result["intent"]
        
        logger.info(f"Generating reply for {msg.id} with intent {intent}")
        
        # Load conversation history from cache
        history = _get_conversation_history(msg.conversation_id)
        
        # Retrieve relevant context from vector DB
        context = _retrieve_context(msg.content, tenant_id=msg.tenant_id)
        
        prompt = f"""Generate a reply for this message:

Message: {msg.content}
Intent: {intent}
Platform: {msg.platform}

Context: {context}
History: {history[-3:] if history else 'None'}

Rules:
- Be helpful and professional
- Match platform tone (casual for Twitter, formal for LinkedIn)
- Keep under 280 chars for Twitter
- Include CTA if sales intent"""

        response = _call_llm(prompt, tenant_id=msg.tenant_id, max_tokens=200)
        
        return {
            "message_id": msg.id,
            "reply_text": response["text"],
            "confidence": response.get("confidence", 0.8)
        }
        
    except Exception as e:
        logger.error(f"Reply generation failed: {e}")
        return {
            "message_id": message["id"],
            "reply_text": "Thank you for your message. Our team will respond shortly.",
            "confidence": 0.5
        }

@celery_app.task(base=AIAgentTask, bind=True)
def score_lead(self, message: Dict, intent_result: Dict) -> Dict:
    """
    Agent 3: Lead Scoring
    Evaluates lead quality and assigns hot/warm/cold tag
    """
    try:
        msg = Message(**message)
        
        logger.info(f"Scoring lead for message {msg.id}")
        
        # Extract signals
        signals = {
            "contains_budget_mention": any(word in msg.content.lower() for word in ['budget', 'price', 'cost', '$']),
            "contains_timeline": any(word in msg.content.lower() for word in ['asap', 'urgent', 'soon', 'when']),
            "intent_confidence": intent_result["confidence"],
            "is_sales_intent": intent_result["intent"] == "sales",
            "message_length": len(msg.content),
            "has_company_mention": '@' in msg.content or 'company' in msg.content.lower()
        }
        
        # Simple scoring logic (can be replaced with ML model)
        score = 0
        if signals["is_sales_intent"] and signals["intent_confidence"] > 0.7:
            score += 40
        if signals["contains_budget_mention"]:
            score += 30
        if signals["contains_timeline"]:
            score += 20
        if signals["has_company_mention"]:
            score += 10
            
        if score >= 70:
            lead_score = "hot"
        elif score >= 40:
            lead_score = "warm"
        else:
            lead_score = "cold"
            
        return {
            "message_id": msg.id,
            "lead_score": lead_score,
            "score_value": score,
            "signals": signals
        }
        
    except Exception as e:
        logger.error(f"Lead scoring failed: {e}")
        return {
            "message_id": message["id"],
            "lead_score": "cold",
            "score_value": 0,
            "signals": {}
        }

@celery_app.task(bind=True)
def orchestrate_pipeline(self, message: Dict):
    """
    Master orchestration: Chains all agents together
    Uses Celery's workflow primitives for reliability
    """
    from celery import chain
    
    # Build the pipeline: classify → generate → score (parallel) → sync CRM
    workflow = chain(
        classify_intent.s(message),
        # Fan out to parallel tasks
        (
            generate_reply.s(message) | 
            score_lead.s(message)
        ),
        # Collect results and sync
        sync_to_crm.s(message)
    )
    
    return workflow.apply_async()

# Helper functions

def _call_llm(prompt: str, tenant_id: str, max_tokens: int = 500) -> Dict:
    """Call LLM with tenant-specific settings and caching"""
    # Implementation would call Gemini/OpenAI
    # Includes caching, rate limiting per tenant
    pass

def _get_conversation_history(conversation_id: str, limit: int = 10) -> List[Dict]:
    """Retrieve recent conversation from Redis/DB"""
    pass

def _retrieve_context(query: str, tenant_id: str, k: int = 3) -> str:
    """Retrieve relevant context from vector DB (RAG)"""
    pass

@celery_app.task
def sync_to_crm(results: Dict, message: Dict):
    """Sync lead data to configured CRM"""
    # Implementation in crm_service.py
    pass

if __name__ == "__main__":
    # For local testing
    celery_app.start()
