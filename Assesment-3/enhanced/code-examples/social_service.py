"""
Social Media Service - Handles webhook ingestion and message posting
Supports Facebook, Twitter (X), LinkedIn with unified interface
"""
from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime
import hashlib
import hmac
import httpx
import logging
from enum import Enum

app = FastAPI(title="Social Media Service")
logger = logging.getLogger(__name__)

class Platform(str, Enum):
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    LINKEDIN = "linkedin"

class IncomingMessage(BaseModel):
    platform: Platform
    message_id: str
    conversation_id: str
    sender_id: str
    content: str
    timestamp: datetime
    tenant_id: str
    metadata: Optional[Dict] = {}

class WebhookVerification(BaseModel):
    """For platform-specific webhook verification"""
    mode: str
    challenge: str
    verify_token: str

# Platform-specific webhook handlers

@app.get("/webhook/facebook")
async def verify_facebook_webhook(
    hub_mode: str = None,
    hub_challenge: str = None,
    hub_verify_token: str = None
):
    """Facebook webhook verification"""
    if hub_mode == "subscribe" and hub_verify_token == get_verify_token("facebook"):
        logger.info("Facebook webhook verified")
        return int(hub_challenge)
    raise HTTPException(status_code=403, detail="Verification failed")

@app.post("/webhook/facebook")
async def receive_facebook_message(
    request: Request,
    x_hub_signature_256: str = Header(None),
    background_tasks: BackgroundTasks
):
    """
    Facebook Messenger webhook handler
    https://developers.facebook.com/docs/messenger-platform/webhooks
    """
    body = await request.body()
    
    # Verify signature
    if not verify_signature(body, x_hub_signature_256, "facebook"):
        raise HTTPException(status_code=403, detail="Invalid signature")
    
    payload = await request.json()
    
    # Extract messages from Facebook payload structure
    for entry in payload.get("entry", []):
        for messaging in entry.get("messaging", []):
            if "message" in messaging:
                message = parse_facebook_message(messaging, entry)
                
                # Queue for processing
                background_tasks.add_task(process_message, message)
                
    return {"status": "ok"}

@app.post("/webhook/twitter")
async def receive_twitter_message(
    request: Request,
    x_twitter_webhooks_signature: str = Header(None),
    background_tasks: BackgroundTasks
):
    """
    Twitter/X webhook handler with Account Activity API
    https://developer.twitter.com/en/docs/twitter-api/enterprise/account-activity-api
    """
    body = await request.body()
    
    if not verify_signature(body, x_twitter_webhooks_signature, "twitter"):
        raise HTTPException(status_code=403, detail="Invalid signature")
    
    payload = await request.json()
    
    # Handle different event types
    if "direct_message_events" in payload:
        for dm_event in payload["direct_message_events"]:
            if dm_event["type"] == "message_create":
                message = parse_twitter_dm(dm_event)
                background_tasks.add_task(process_message, message)
                
    elif "tweet_create_events" in payload:
        # Mentions or replies
        for tweet in payload["tweet_create_events"]:
            message = parse_twitter_mention(tweet)
            background_tasks.add_task(process_message, message)
    
    return {"status": "ok"}

@app.post("/webhook/linkedin")
async def receive_linkedin_message(
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    LinkedIn webhook handler
    https://docs.microsoft.com/en-us/linkedin/marketing/integrations/webhooks
    """
    payload = await request.json()
    
    # LinkedIn uses different verification method
    if not await verify_linkedin_webhook(request):
        raise HTTPException(status_code=403, detail="Invalid webhook")
    
    for event in payload.get("events", []):
        if event.get("eventType") == "MESSAGE_RECEIVED":
            message = parse_linkedin_message(event)
            background_tasks.add_task(process_message, message)
    
    return {"status": "ok"}

# Message parsing functions

def parse_facebook_message(messaging: Dict, entry: Dict) -> IncomingMessage:
    """Extract unified message format from Facebook payload"""
    message_data = messaging["message"]
    sender = messaging["sender"]["id"]
    
    return IncomingMessage(
        platform=Platform.FACEBOOK,
        message_id=message_data["mid"],
        conversation_id=f"fb_{messaging['recipient']['id']}_{sender}",
        sender_id=sender,
        content=message_data.get("text", ""),
        timestamp=datetime.fromtimestamp(messaging["timestamp"] / 1000),
        tenant_id=get_tenant_for_page(entry["id"]),
        metadata={
            "page_id": entry["id"],
            "has_attachments": "attachments" in message_data
        }
    )

def parse_twitter_dm(event: Dict) -> IncomingMessage:
    """Extract from Twitter DM event"""
    msg = event["message_create"]
    return IncomingMessage(
        platform=Platform.TWITTER,
        message_id=event["id"],
        conversation_id=f"tw_{msg['sender_id']}",
        sender_id=msg["sender_id"],
        content=msg["message_data"]["text"],
        timestamp=datetime.fromtimestamp(int(event["created_timestamp"]) / 1000),
        tenant_id=get_tenant_for_twitter_account(msg["target"]["recipient_id"]),
        metadata={}
    )

def parse_twitter_mention(tweet: Dict) -> IncomingMessage:
    """Extract from Twitter mention"""
    return IncomingMessage(
        platform=Platform.TWITTER,
        message_id=tweet["id_str"],
        conversation_id=f"tw_mention_{tweet['in_reply_to_status_id_str'] or tweet['id_str']}",
        sender_id=tweet["user"]["id_str"],
        content=tweet["text"],
        timestamp=datetime.strptime(tweet["created_at"], "%a %b %d %H:%M:%S %z %Y"),
        tenant_id=get_tenant_for_twitter_account(tweet["in_reply_to_user_id_str"]),
        metadata={"is_reply": tweet["in_reply_to_status_id_str"] is not None}
    )

def parse_linkedin_message(event: Dict) -> IncomingMessage:
    """Extract from LinkedIn message event"""
    return IncomingMessage(
        platform=Platform.LINKEDIN,
        message_id=event["messageId"],
        conversation_id=event["conversationId"],
        sender_id=event["from"]["member"],
        content=event["message"]["text"],
        timestamp=datetime.fromtimestamp(event["createdAt"] / 1000),
        tenant_id=get_tenant_for_linkedin_page(event["to"]["organization"]),
        metadata={}
    )

# Security and validation

def verify_signature(payload: bytes, signature: str, platform: str) -> bool:
    """Verify webhook signature from platform"""
    secret = get_webhook_secret(platform)
    
    if platform == "facebook":
        expected = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(f"sha256={expected}", signature)
    
    elif platform == "twitter":
        expected = hmac.new(secret.encode(), payload, hashlib.sha256).digest()
        import base64
        return hmac.compare_digest(base64.b64encode(expected).decode(), signature.split("=")[1])
    
    return False

async def verify_linkedin_webhook(request: Request) -> bool:
    """LinkedIn uses OAuth verification"""
    # Implementation depends on LinkedIn's auth flow
    return True

# Message processing pipeline

async def process_message(message: IncomingMessage):
    """
    Queue message for AI processing
    Publishes to Kafka or directly invokes Celery task
    """
    try:
        logger.info(f"Processing message {message.message_id} from {message.platform}")
        
        # Store in database
        await save_message(message)
        
        # Publish to Kafka for processing
        await publish_to_kafka(
            topic="incoming-messages",
            key=message.tenant_id,
            value=message.dict()
        )
        
        logger.info(f"Message {message.message_id} queued successfully")
        
    except Exception as e:
        logger.error(f"Failed to process message {message.message_id}: {e}")
        # Send to DLQ for manual review
        await send_to_dlq(message, error=str(e))

# Reply posting functions

@app.post("/reply")
async def post_reply(
    message_id: str,
    reply_text: str,
    platform: Platform,
    tenant_id: str
):
    """Post AI-generated reply back to platform"""
    
    credentials = get_platform_credentials(tenant_id, platform)
    
    if platform == Platform.FACEBOOK:
        return await post_facebook_reply(message_id, reply_text, credentials)
    elif platform == Platform.TWITTER:
        return await post_twitter_reply(message_id, reply_text, credentials)
    elif platform == Platform.LINKEDIN:
        return await post_linkedin_reply(message_id, reply_text, credentials)

async def post_facebook_reply(message_id: str, text: str, credentials: Dict):
    """Post reply via Facebook Graph API"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://graph.facebook.com/v18.0/me/messages",
            params={"access_token": credentials["access_token"]},
            json={
                "recipient": {"id": credentials["sender_id"]},
                "message": {"text": text}
            }
        )
        return response.json()

# Helper functions (stub implementations)

def get_verify_token(platform: str) -> str:
    """Get verification token from secure storage"""
    pass

def get_webhook_secret(platform: str) -> str:
    """Get webhook secret from secure storage"""
    pass

def get_tenant_for_page(page_id: str) -> str:
    """Map social page to tenant"""
    pass

def get_tenant_for_twitter_account(account_id: str) -> str:
    pass

def get_tenant_for_linkedin_page(org_id: str) -> str:
    pass

async def save_message(message: IncomingMessage):
    """Persist message to database"""
    pass

async def publish_to_kafka(topic: str, key: str, value: Dict):
    """Publish to Kafka topic"""
    pass

async def send_to_dlq(message: IncomingMessage, error: str):
    """Send failed message to Dead Letter Queue"""
    pass

def get_platform_credentials(tenant_id: str, platform: Platform) -> Dict:
    """Retrieve encrypted credentials for tenant"""
    pass
