from typing import Literal, Optional

from pydantic import BaseModel, Field


IntentType = Literal["sales", "support", "spam"]
PlatformType = Literal["facebook", "twitter", "linkedin"]


class TenantCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)


class TenantResponse(BaseModel):
    id: str
    name: str
    api_key: str
    webhook_secret: str


class AuthRequest(BaseModel):
    tenant_id: str = Field(min_length=6)
    api_key: str = Field(min_length=16)


class AuthResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"


class WebhookMessage(BaseModel):
    external_user_id: str = Field(min_length=1, max_length=100)
    external_message_id: Optional[str] = Field(default=None, max_length=120)
    text: str = Field(min_length=1, max_length=5000)


class MessageStatusResponse(BaseModel):
    message_id: int
    tenant_id: str
    platform: PlatformType
    status: str
    intent: Optional[IntentType] = None
    ai_reply: Optional[str] = None
    lead_score: Optional[int] = None


class LeadResponse(BaseModel):
    id: int
    message_id: int
    intent: IntentType
    score: int
    status: str


class CRMOutboxResponse(BaseModel):
    id: int
    lead_id: int
    crm_target: str
    status: str


class QueuePublishResponse(BaseModel):
    message_id: int
    queued: bool
