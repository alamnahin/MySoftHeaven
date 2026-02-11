"""
CRM Sync Service - Bidirectional sync with Salesforce, HubSpot, Zoho
Handles lead creation, update, and status synchronization
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Literal
from datetime import datetime
from enum import Enum
import httpx
import logging
import asyncio
from dataclasses import dataclass

app = FastAPI(title="CRM Sync Service")
logger = logging.getLogger(__name__)

class CRMProvider(str, Enum):
    SALESFORCE = "salesforce"
    HUBSPOT = "hubspot"
    ZOHO = "zoho"

class LeadStatus(str, Enum):
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    CONVERTED = "converted"
    CLOSED_LOST = "closed_lost"

@dataclass
class Lead:
    id: str
    tenant_id: str
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    company: Optional[str]
    score: Literal["hot", "warm", "cold"]
    status: LeadStatus
    source_platform: str
    messages: List[Dict]
    created_at: datetime
    metadata: Dict

class SyncRequest(BaseModel):
    lead_id: str
    tenant_id: str
    action: Literal["create", "update", "status_change"]
    force: bool = False

# Salesforce Integration

class SalesforceSync:
    """Salesforce CRM integration using REST API and Bulk API"""
    
    def __init__(self, credentials: Dict):
        self.instance_url = credentials["instance_url"]
        self.access_token = credentials["access_token"]
        self.api_version = "v59.0"
    
    async def create_lead(self, lead: Lead) -> str:
        """Create new lead in Salesforce"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.instance_url}/services/data/{self.api_version}/sobjects/Lead",
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "FirstName": lead.name.split()[0] if lead.name else "",
                    "LastName": lead.name.split()[-1] if lead.name else "Unknown",
                    "Email": lead.email,
                    "Phone": lead.phone,
                    "Company": lead.company or "Unknown",
                    "LeadSource": f"Social_{lead.source_platform}",
                    "Rating": self._map_score_to_rating(lead.score),
                    "Status": self._map_status_to_sf(lead.status),
                    "Description": self._build_description(lead),
                    # Custom fields
                    "Social_Platform__c": lead.source_platform,
                    "AI_Lead_Score__c": lead.score,
                    "Message_Count__c": len(lead.messages)
                }
            )
            
            if response.status_code == 201:
                result = response.json()
                logger.info(f"Created Salesforce lead {result['id']} for {lead.id}")
                return result["id"]
            else:
                logger.error(f"Salesforce create failed: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=response.text)
    
    async def update_lead(self, sf_lead_id: str, lead: Lead) -> bool:
        """Update existing lead"""
        async with httpx.AsyncClient() as client:
            response = await client.patch(
                f"{self.instance_url}/services/data/{self.api_version}/sobjects/Lead/{sf_lead_id}",
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "Rating": self._map_score_to_rating(lead.score),
                    "Status": self._map_status_to_sf(lead.status),
                    "AI_Lead_Score__c": lead.score,
                    "Message_Count__c": len(lead.messages),
                    "Last_Interaction__c": datetime.now().isoformat()
                }
            )
            
            return response.status_code == 204
    
    async def add_activity(self, sf_lead_id: str, message: Dict):
        """Log message as Task/Activity"""
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.instance_url}/services/data/{self.api_version}/sobjects/Task",
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                },
                json={
                    "WhoId": sf_lead_id,
                   "Subject": f"Social message from {message['platform']}",
                    "Description": message["content"],
                    "ActivityDate": message["timestamp"],
                    "Status": "Completed",
                    "Type": "Social_Message"
                }
            )
    
    def _map_score_to_rating(self, score: str) -> str:
        return {"hot": "Hot", "warm": "Warm", "cold": "Cold"}[score]
    
    def _map_status_to_sf(self, status: LeadStatus) -> str:
        mapping = {
            LeadStatus.NEW: "Open - Not Contacted",
            LeadStatus.CONTACTED: "Working - Contacted",
            LeadStatus.QUALIFIED: "Qualified",
            LeadStatus.CONVERTED: "Closed - Converted",
            LeadStatus.CLOSED_LOST: "Closed - Not Converted"
        }
        return mapping[status]
    
    def _build_description(self, lead: Lead) -> str:
        """Build comprehensive description from message history"""
        desc = f"AI-generated lead from {lead.source_platform}\n\n"
        desc += f"Score: {lead.score.upper()}\n"
        desc += f"Total messages: {len(lead.messages)}\n\n"
        desc += "Recent conversation:\n"
        for msg in lead.messages[-3:]:
            desc += f"- {msg['timestamp']}: {msg['content'][:100]}\n"
        return desc

# HubSpot Integration

class HubSpotSync:
    """HubSpot CRM integration"""
    
    def __init__(self, credentials: Dict):
        self.api_key = credentials["api_key"]
        self.base_url = "https://api.hubapi.com"
    
    async def create_contact(self, lead: Lead) -> str:
        """Create contact in HubSpot"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/crm/v3/objects/contacts",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "properties": {
                        "email": lead.email,
                        "firstname": lead.name.split()[0] if lead.name else "",
                        "lastname": lead.name.split()[-1] if lead.name else "",
                        "phone": lead.phone,
                        "company": lead.company,
                        "hs_lead_status": self._map_status_to_hs(lead.status),
                        "lead_source": lead.source_platform,
                        "lead_score": self._score_to_number(lead.score)
                    }
                }
            )
            
            if response.status_code == 201:
                result = response.json()
                return result["id"]
            else:
                raise HTTPException(status_code=response.status_code, detail=response.text)
    
    async def create_engagement(self, contact_id: str, message: Dict):
        """Log message as engagement (note)"""
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.base_url}/engagements/v1/engagements",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "engagement": {
                        "type": "NOTE",
                        "timestamp": int(message["timestamp"].timestamp() * 1000)
                    },
                    "associations": {
                        "contactIds": [contact_id]
                    },
                    "metadata": {
                        "body": f"{message['platform']}: {message['content']}"
                    }
                }
            )
    
    def _map_status_to_hs(self, status: LeadStatus) -> str:
        mapping = {
            LeadStatus.NEW: "NEW",
            LeadStatus.CONTACTED: "OPEN",
            LeadStatus.QUALIFIED: "IN_PROGRESS",
            LeadStatus.CONVERTED: "CONVERTED",
            LeadStatus.CLOSED_LOST: "UNQUALIFIED"
        }
        return mapping[status]
    
    def _score_to_number(self, score: str) -> int:
        return {"hot": 90, "warm": 60, "cold": 30}[score]

# Unified sync orchestrator

@app.post("/sync/lead")
async def sync_lead(request: SyncRequest, background_tasks: BackgroundTasks):
    """
    Sync lead to all configured CRMs for tenant
    Handles create, update, and status changes
    """
    try:
        # Fetch lead data
        lead = await get_lead(request.lead_id)
        
        # Get tenant CRM configurations
        crm_configs = await get_tenant_crm_configs(request.tenant_id)
        
        results = {}
        
        for config in crm_configs:
            if config["enabled"]:
                # Queue sync task for each CRM
                background_tasks.add_task(
                    sync_to_crm,
                    lead=lead,
                    crm_config=config,
                    action=request.action
                )
                results[config["provider"]] = "queued"
        
        return {"status": "success", "results": results}
        
    except Exception as e:
        logger.error(f"Sync failed for lead {request.lead_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def sync_to_crm(lead: Lead, crm_config: Dict, action: str):
    """Execute sync to specific CRM with retry logic"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            if crm_config["provider"] == CRMProvider.SALESFORCE:
                sync_client = SalesforceSync(crm_config["credentials"])
            elif crm_config["provider"] == CRMProvider.HUBSPOT:
                sync_client = HubSpotSync(crm_config["credentials"])
            else:
                logger.warning(f"Unsupported CRM: {crm_config['provider']}")
                return
            
            # Check if lead already exists
            external_id = await get_external_lead_id(lead.id, crm_config["provider"])
            
            if action == "create" or not external_id:
                # Create new lead
                if crm_config["provider"] == CRMProvider.SALESFORCE:
                    external_id = await sync_client.create_lead(lead)
                elif crm_config["provider"] == CRMProvider.HUBSPOT:
                    external_id = await sync_client.create_contact(lead)
                
                # Save mapping
                await save_external_lead_mapping(lead.id, crm_config["provider"], external_id)
                
            elif action == "update":
                # Update existing lead
                if crm_config["provider"] == CRMProvider.SALESFORCE:
                    await sync_client.update_lead(external_id, lead)
                # HubSpot update similar
            
            # Sync recent messages as activities
            for message in lead.messages[-5:]:  # Last 5 messages
                if crm_config["provider"] == CRMProvider.SALESFORCE:
                    await sync_client.add_activity(external_id, message)
                elif crm_config["provider"] == CRMProvider.HUBSPOT:
                    await sync_client.create_engagement(external_id, message)
            
            logger.info(f"Successfully synced lead {lead.id} to {crm_config['provider']}")
            break
            
        except Exception as e:
            logger.error(f"Sync attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (attempt + 1))
            else:
                # Final failure: send alert
                await send_sync_failure_alert(lead, crm_config["provider"], str(e))

@app.get("/sync/status/{lead_id}")
async def get_sync_status(lead_id: str, tenant_id: str):
    """Check sync status across all CRMs"""
    mappings = await get_all_external_mappings(lead_id)
    
    status = {}
    for mapping in mappings:
        status[mapping["provider"]] = {
            "synced": True,
            "external_id": mapping["external_id"],
            "last_sync": mapping["last_sync"],
            "sync_count": mapping["sync_count"]
        }
    
    return {"lead_id": lead_id, "crm_status": status}

# Webhook handlers for reverse sync (CRM → Platform)

@app.post("/webhook/salesforce")
async def salesforce_webhook(request: Dict):
    """Handle Salesforce outbound messages or platform events"""
    # When lead status changes in Salesforce, update internal status
    pass

@app.post("/webhook/hubspot")
async def hubspot_webhook(request: Dict):
    """Handle HubSpot webhook events"""
    pass

# Helper functions

async def get_lead(lead_id: str) -> Lead:
    """Fetch lead from database"""
    pass

async def get_tenant_crm_configs(tenant_id: str) -> List[Dict]:
    """Get all CRM configurations for tenant"""
    pass

async def get_external_lead_id(lead_id: str, provider: str) -> Optional[str]:
    """Get CRM-specific lead ID"""
    pass

async def save_external_lead_mapping(lead_id: str, provider: str, external_id: str):
    """Save lead ID mapping"""
    pass

async def get_all_external_mappings(lead_id: str) -> List[Dict]:
    """Get all CRM mappings for lead"""
    pass

async def send_sync_failure_alert(lead: Lead, provider: str, error: str):
    """Alert admin of sync failure"""
    pass
