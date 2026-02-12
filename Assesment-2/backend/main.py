from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import os
import json
import numpy as np
import faiss
import httpx
import asyncio
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration with sensible defaults
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "openai" or "gemini"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_API_KEY = os.getenv("LLM_API_KEY", "your-api-key")

# Provider-specific configuration
if LLM_PROVIDER == "gemini":
    LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent")
    LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3-flash-preview")
else:  # openai
    LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "3"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))

@dataclass
class DocumentChunk:
    """Represents a chunk of text with metadata"""
    text: str
    source: str
    chunk_id: int
    start_pos: int
    end_pos: int

class DocumentStore:
    """
    Manages document chunking, embedding, and retrieval.
    """
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)
        self.chunks: List[DocumentChunk] = []
        self.index: Optional[faiss.Index] = None
        self.dimension: int = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: {model_name} (dim={self.dimension})")

    def chunk_document(self, text: str, source: str = "unknown") -> List[DocumentChunk]:
        """
        Chunking strategy: Recursive character splitting with overlap.
        """
        chunks = []
        start = 0
        chunk_id = 0

        # Simple sentence-based chunking
        sentences = text.replace('\n', ' ').split('. ')
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Add period back if removed
            if not sentence.endswith('.'):
                sentence += '.'

            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > CHUNK_SIZE and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    source=source,
                    chunk_id=chunk_id,
                    start_pos=start,
                    end_pos=start + len(chunk_text)
                ))

                # Overlap: keep last sentence for context continuity
                if len(current_chunk) > 1:
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0

                chunk_id += 1
                start += len(chunk_text) - (CHUNK_OVERLAP if current_chunk else 0)

            current_chunk.append(sentence)
            current_length += sentence_length

        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(DocumentChunk(
                text=chunk_text,
                source=source,
                chunk_id=chunk_id,
                start_pos=start,
                end_pos=start + len(chunk_text)
            ))

        logger.info(f"Created {len(chunks)} chunks from {source}")
        return chunks

    def build_index(self, documents: Dict[str, str]):
        """Build FAISS index from documents"""
        all_chunks = []

        for source, text in documents.items():
            chunks = self.chunk_document(text, source)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError("No documents to index")

        self.chunks = all_chunks

        # Create embeddings
        texts = [chunk.text for chunk in all_chunks]
        logger.info(f"Encoding {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)

        # Normalize for cosine similarity (L2 norm)
        faiss.normalize_L2(embeddings)

        # Create FAISS index
        if len(embeddings) < 100:
            # Small dataset: use flat index (exact search)
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product = cosine for normalized vectors
        else:
            # Larger dataset: use IVF for faster search
            nlist = min(int(np.sqrt(len(embeddings))), 100)  # Rule of thumb for nlist
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)

        self.index.add(embeddings)
        logger.info(f"Built index with {self.index.ntotal} vectors")

    def search(self, query: str, k: int = TOP_K) -> List[tuple]:
        """
        Search for relevant chunks.
        Returns: List of (chunk, score) tuples
        """
        if self.index is None:
            raise RuntimeError("Index not built yet")

        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):  # Valid index
                results.append((self.chunks[idx], float(score)))

        return results

    def save_index(self, path: str = "faiss_index"):
        """Save FAISS index and chunks metadata"""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        # Save chunks metadata
        chunks_data = [
            {
                "text": chunk.text,
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos
            }
            for chunk in self.chunks
        ]
        with open(os.path.join(path, "chunks.json"), 'w') as f:
            json.dump(chunks_data, f)

        logger.info(f"Saved index to {path}")

    def load_index(self, path: str = "faiss_index"):
        """Load FAISS index and chunks metadata"""
        index_path = os.path.join(path, "index.faiss")
        chunks_path = os.path.join(path, "chunks.json")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Index files not found in {path}")

        self.index = faiss.read_index(index_path)

        with open(chunks_path, 'r') as f:
            chunks_data = json.load(f)

        self.chunks = [DocumentChunk(**data) for data in chunks_data]
        logger.info(f"Loaded index with {len(self.chunks)} chunks")

# Global document store
doc_store = DocumentStore()

def load_company_documents():
    """Load company documents from data directory"""
    docs = {}
    # Look in parent directory since backend is in subdirectory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} not found")
        return docs

    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    docs[filename] = f.read()
                logger.info(f"Loaded {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")

    return docs

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: Build or load index"""
    try:
        # Try to load existing index
        doc_store.load_index("vector_store")
        logger.info("Loaded existing index")
    except FileNotFoundError:
        # Build new index from documents
        logger.info("Building new index from documents...")
        documents = load_company_documents()
        if documents:
            doc_store.build_index(documents)
            doc_store.save_index("vector_store")
        else:
            logger.warning("No documents found, creating empty index")

    yield

    # Shutdown cleanup if needed
    logger.info("Shutting down...")

app = FastAPI(
    title="Mysoft Heaven RAG Chatbot",
    description="Retrieval-Augmented Generation chatbot for company information",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Conversation memory - simple in-memory store
conversation_history: Dict[str, List[dict]] = {}

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: Optional[str] = Field(default=None, description="For conversation memory")
    stream: bool = Field(default=False, description="Stream response")

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float
    session_id: str
    timestamp: str

def create_system_prompt(context: str, conversation_history: List[dict] = None) -> str:
    """
    Prompt engineering for RAG.
    """
    base_prompt = f"""You are a helpful assistant for Mysoft Heaven (BD) Ltd. 
Answer questions STRICTLY based on the provided context. 

CRITICAL RULES:
1. If the answer is not in the context, say "I don't have information about that in my knowledge base."
2. Do not make up information or use outside knowledge.
3. Be concise but complete (2-4 sentences).
4. If asked about competitors or unrelated companies, decline politely.
5. Always maintain professional tone.

Context from company documents:
---
{context}
---
"""

    if conversation_history:
        base_prompt += "\n\nPrevious conversation:\n"
        for msg in conversation_history[-3:]:  # Last 3 messages for context
            role = "User" if msg["role"] == "user" else "Assistant"
            base_prompt += f"{role}: {msg['content']}\n"

    return base_prompt

async def generate_response(system_prompt: str, user_question: str, max_retries: int = 2) -> str:
    """Call LLM with retry logic, supports both OpenAI and Gemini"""
    
    if LLM_PROVIDER == "gemini":
        # Gemini API format - combine system prompt and question
        url = f"{LLM_ENDPOINT}?key={LLM_API_KEY}"
        headers = {"Content-Type": "application/json"}
        full_prompt = f"{system_prompt}\n\nUser Question: {user_question}\n\nAssistant:"
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 8192
            }
        }
    else:
        # OpenAI API format
        url = LLM_ENDPOINT
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            "temperature": 0.3,
            "max_tokens": 4096
        }

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                
                # Extract content based on provider
                if LLM_PROVIDER == "gemini":
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:  # openai
                    return result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"LLM error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return "I apologize, but I'm having trouble generating a response right now. Please try again later."
            await asyncio.sleep(2 ** attempt)

    return "Service temporarily unavailable."

def is_query_relevant(query: str, chunks: List[tuple]) -> tuple:
    """
    Determine if query is relevant based on retrieval scores.
    Returns: (is_relevant, confidence, best_score)
    """
    if not chunks:
        return False, 0.0, 0.0

    best_score = chunks[0][1]
    avg_score = sum(score for _, score in chunks) / len(chunks)

    # Improved confidence: weighted average of best and mean scores
    confidence = (best_score * 0.7) + (avg_score * 0.3)
    
    # More nuanced relevance check
    # Accept if best score is above threshold OR if avg is decent
    is_relevant = (best_score >= CONFIDENCE_THRESHOLD) or (avg_score >= CONFIDENCE_THRESHOLD * 0.8)

    return is_relevant, confidence, best_score

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Implements RAG: Retrieve -> Filter -> Generate
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{datetime.now().timestamp()}"

        # Initialize conversation history
        if session_id not in conversation_history:
            conversation_history[session_id] = []

        # Step 1: Retrieve relevant chunks
        results = doc_store.search(request.message, k=TOP_K)

        # Step 2: Check relevance (guard against off-topic questions)
        is_relevant, confidence, best_score = is_query_relevant(request.message, results)

        if not is_relevant:
            # Out-of-scope question handling
            answer = ("I'm sorry, but I can only answer questions about Mysoft Heaven (BD) Ltd. "
                     "based on our company documents. I don't have information about that topic. "
                     "Please ask about our services, products, or company information.")

            return ChatResponse(
                answer=answer,
                sources=[],
                confidence=confidence,
                session_id=session_id,
                timestamp=datetime.now().isoformat()
            )

        # Step 3: Build context from retrieved chunks
        context = "\n\n".join([f"[Source: {chunk.source}]\n{chunk.text}" for chunk, score in results])

        # Step 4: Create prompt with conversation history
        history = conversation_history[session_id]
        system_prompt = create_system_prompt(context, history)

        # Step 5: Generate response with the actual question
        answer = await generate_response(system_prompt, request.message)

        # Step 6: Update conversation history
        conversation_history[session_id].append({"role": "user", "content": request.message})
        conversation_history[session_id].append({"role": "assistant", "content": answer})

        # Keep only last 10 exchanges to prevent context bloat
        if len(conversation_history[session_id]) > 20:
            conversation_history[session_id] = conversation_history[session_id][-20:]

        # Prepare sources
        sources = [
            {
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "source": chunk.source,
                "relevance_score": round(score, 3)
            }
            for chunk, score in results
        ]

        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence=round(confidence, 3),
            session_id=session_id,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint for better UX"""
    # Implementation would use SSE or WebSocket
    # For now, return non-streaming response
    response = await chat(request)
    return response

@app.get("/health")
async def health_check():
    """Health check with index status"""
    return {
        "status": "healthy",
        "index_loaded": doc_store.index is not None,
        "chunks_count": len(doc_store.chunks),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats")
async def get_stats():
    """Get document store statistics"""
    return {
        "total_chunks": len(doc_store.chunks),
        "sources": list(set(chunk.source for chunk in doc_store.chunks)),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
