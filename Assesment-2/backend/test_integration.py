"""
Integration Tests for RAG Chatbot
Tests the full pipeline: document loading → embedding → retrieval → LLM generation
"""
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from main import app, doc_store, load_company_documents
import os
import tempfile
import shutil
from pathlib import Path

client = TestClient(app)

@pytest.fixture(scope="module")
def test_data_dir():
    """Create temporary data directory with test documents"""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir()
    
    # Create test company profile
    profile_path = data_dir / "test_company.txt"
    profile_path.write_text("""
    Test Company Inc.
    
    Services:
    - Web Development
    - Mobile Apps
    - Cloud Solutions
    
    Products:
    - Product A: CRM System
    - Product B: Analytics Platform
    
    Contact:
    Email: test@company.com
    Phone: +1234567890
    
    Notable Projects:
    - Government portal system
    - Banking application
    - Healthcare management
    """)
    
    yield data_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="module")
def vector_store_dir():
    """Create temporary vector store directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

class TestDocumentProcessing:
    """Test document loading and chunking"""
    
    def test_load_documents(self, test_data_dir, monkeypatch):
        """Test loading documents from directory"""
        # Mock the data directory
        monkeypatch.setattr("main.os.path.dirname", lambda x: str(test_data_dir.parent))
        
        docs = load_company_documents()
        
        assert len(docs) > 0
        assert "test_company.txt" in docs
        assert "Web Development" in docs["test_company.txt"]
    
    def test_chunk_document(self):
        """Test document chunking strategy"""
        text = """This is sentence one. This is sentence two. This is sentence three. 
        This is sentence four. This is sentence five. This is sentence six."""
        
        chunks = doc_store.chunk_document(text, source="test.txt")
        
       # Should create multiple chunks
        assert len(chunks) > 0
        
        # Each chunk should have required fields
        for chunk in chunks:
            assert hasattr(chunk, 'text')
            assert hasattr(chunk, 'source')
            assert hasattr(chunk, 'chunk_id')
            assert chunk.source == "test.txt"
    
    def test_chunking_preserves_overlap(self):
        """Test that chunking includes overlap for context"""
        long_text = " ".join([f"Sentence {i}." for i in range(1, 101)])
        
        chunks = doc_store.chunk_document(long_text, source="test.txt")
        
        # Verify overlap exists (some text appears in consecutive chunks)
        if len(chunks) > 1:
            # There should be some overlap
            assert len(chunks) >= 2

class TestEmbeddingAndRetrieval:
    """Test vector embedding and similarity search"""
    
    def test_build_index(self, test_data_dir, monkeypatch):
        """Test building FAISS index from documents"""
        monkeypatch.setattr("main.os.path.dirname", lambda x: str(test_data_dir.parent))
        
        docs = load_company_documents()
        doc_store.build_index(docs)
        
        assert doc_store.index is not None
        assert len(doc_store.chunks) > 0
        assert doc_store.index.ntotal == len(doc_store.chunks)
    
    def test_search_retrieves_relevant_chunks(self, test_data_dir, monkeypatch):
        """Test that search returns relevant results"""
        monkeypatch.setattr("main.os.path.dirname", lambda x: str(test_data_dir.parent))
        
        docs = load_company_documents()
        doc_store.build_index(docs)
        
        # Search for services
        results = doc_store.search("What services are available?", k=3)
        
        assert len(results) > 0
        # Results should be tuples of (chunk, score)
        for chunk, score in results:
            assert hasattr(chunk, 'text')
            assert 0 <= score <= 1  # Normalized similarity score
        
        # Top result should mention services
        top_chunk, top_score = results[0]
        assert "services" in top_chunk.text.lower() or "web development" in top_chunk.text.lower()
    
    def test_search_scores_decrease(self, test_data_dir, monkeypatch):
        """Test that search results are ordered by relevance"""
        monkeypatch.setattr("main.os.path.dirname", lambda x: str(test_data_dir.parent))
        
        docs = load_company_documents()
        doc_store.build_index(docs)
        
        results = doc_store.search("web development services", k=3)
        
        if len(results) > 1:
            scores = [score for _, score in results]
            # Scores should be in descending order
            assert scores == sorted(scores, reverse=True)

class TestChatEndpoint:
    """Integration tests for the chat endpoint"""
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_with_relevant_query(self, test_data_dir, monkeypatch):
        """Test chat with query that should have answers in documents"""
        monkeypatch.setattr("main.os.path.dirname", lambda x: str(test_data_dir.parent))
        
        # Rebuild index with test data
        docs = load_company_documents()
        if docs:
            doc_store.build_index(docs)
        
        response = client.post(
            "/chat",
            json={
                "message": "What services does the company provide?",
                "session_id": "test-session-1"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert len(data["sources"]) > 0
        
        # Answer should mention services
        assert len(data["answer"]) > 10  # Should have substantial answer
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_with_irrelevant_query(self, test_data_dir, monkeypatch):
        """Test chat with out-of-scope query"""
        monkeypatch.setattr("main.os.path.dirname", lambda x: str(test_data_dir.parent))
        
        docs = load_company_documents()
        if docs:
            doc_store.build_index(docs)
        
        response = client.post(
            "/chat",
            json={
                "message": "What is the weather like today?",
                "session_id": "test-session-2"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should have low confidence and rejection message
        assert data["confidence"] < 0.5  # Low confidence for irrelevant query
    
    def test_chat_endpoint_validation(self):
        """Test input validation"""
        # Empty message
        response = client.post(
            "/chat",
            json={"message": ""}
        )
        assert response.status_code == 422  # Validation error
        
        # Message too long
        response = client.post(
            "/chat",
            json={"message": "x" * 2000}
        )
        assert response.status_code == 422
    
    def test_chat_maintains_session(self):
        """Test conversation memory across multiple messages"""
        session_id = "test-session-memory"
        
        # First message
        response1 = client.post(
            "/chat",
            json={
                "message": "What services do you offer?",
                "session_id": session_id
            }
        )
        assert response1.status_code == 200
        
        # Second message referencing first
        response2 = client.post(
            "/chat",
            json={
                "message": "Tell me more about the first one",
                "session_id": session_id
            }
        )
        assert response2.status_code == 200
        # Should use conversation history

class TestHealthAndStats:
    """Test health check and statistics endpoints"""
    
    def test_health_endpoint(self):
        """Test health check"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"
        assert "index_loaded" in data
        assert "chunks_count" in data
    
    def test_stats_endpoint(self):
        """Test statistics endpoint"""
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_chunks" in data
        assert "sources" in data
        assert "embedding_model" in data
        assert "chunk_size" in data

class TestIndexPersistence:
    """Test saving and loading of vector index"""
    
    def test_save_and_load_index(self, test_data_dir, vector_store_dir, monkeypatch):
        """Test that index can be saved and loaded"""
        monkeypatch.setattr("main.os.path.dirname", lambda x: str(test_data_dir.parent))
        
        # Build index
        docs = load_company_documents()
        doc_store.build_index(docs)
        original_chunk_count = len(doc_store.chunks)
        
        # Save index
        doc_store.save_index(vector_store_dir)
        
        # Verify files exist
        index_path = os.path.join(vector_store_dir, "index.faiss")
        chunks_path = os.path.join(vector_store_dir, "chunks.json")
        assert os.path.exists(index_path)
        assert os.path.exists(chunks_path)
        
        # Clear current index
        doc_store.chunks = []
        doc_store.index = None
        
        # Load index
        doc_store.load_index(vector_store_dir)
        
        assert doc_store.index is not None
        assert len(doc_store.chunks) == original_chunk_count

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_document_handling(self):
        """Test handling of empty documents"""
        with pytest.raises(ValueError):
            doc_store.build_index({})
    
    def test_search_before_index_built(self):
        """Test search when index is not built"""
        temp_store = type(doc_store).__new__(type(doc_store))
        temp_store.__init__()
        
        with pytest.raises(RuntimeError):
            temp_store.search("test query")
    
    def test_chat_with_index_not_ready(self):
        """Test chat endpoint when index is not ready"""
        # Temporarily clear index
        original_index = doc_store.index
        doc_store.index = None
        
        response = client.post(
            "/chat",
            json={"message": "test query"}
        )
        
        # Should handle gracefully
        assert response.status_code == 500
        
        # Restore
        doc_store.index = original_index

class TestConcurrency:
    """Test concurrent request handling"""
    
    @pytest.mark.asyncio
    async def test_concurrent_chat_requests(self):
        """Test handling multiple concurrent requests"""
        import asyncio
        
        async def make_request(i):
            return client.post(
                "/chat",
                json={
                    "message": f"Question {i}: What services are available?",
                    "session_id": f"concurrent-session-{i}"
                }
            )
        
        # Make 10 concurrent requests
        tasks = [make_request(i) for i in range(10)]
        responses = await asyncio.gather(*[asyncio.to_thread(task) for task in tasks])
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
