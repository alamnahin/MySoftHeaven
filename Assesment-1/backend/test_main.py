import pytest
from fastapi.testclient import TestClient
from main import app, init_db
import os
import json

# Use test database
os.environ["DATABASE_URL"] = "sqlite:///./test_leads.db"

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_db():
    """Setup fresh database for each test"""
    init_db()
    yield
    # Cleanup
    if os.path.exists("test_leads.db"):
        os.remove("test_leads.db")

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_receive_lead_success():
    """Test successful lead submission"""
    payload = {
        "message": "Hi, I'm John from Acme Corp. We're interested in your enterprise pricing.",
        "source": "website"
    }
    
    response = client.post("/webhook/lead", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "lead_id" in data
    assert data["status"] == "received"

def test_receive_lead_validation_error():
    """Test validation catches short messages"""
    payload = {"message": "Hi"}  # Too short
    
    response = client.post("/webhook/lead", json=payload)
    assert response.status_code == 422  # Validation error

def test_get_lead_not_found():
    """Test 404 for non-existent lead"""
    response = client.get("/leads/99999")
    assert response.status_code == 404

def test_receive_lead_with_metadata():
    """Test lead with metadata"""
    payload = {
        "message": "Hello, this is Sarah from TechStart. We need support with integration.",
        "source": "api",
        "metadata": {"campaign": "q4_2024", "referrer": "linkedin"}
    }
    
    response = client.post("/webhook/lead", json=payload)
    assert response.status_code == 200
    
    lead_id = response.json()["lead_id"]
    
    # Check lead exists
    get_response = client.get(f"/leads/{lead_id}")
    assert get_response.status_code == 200
    
    lead_data = get_response.json()
    assert lead_data["raw_message"] == payload["message"]
    assert lead_data["source"] == "api"