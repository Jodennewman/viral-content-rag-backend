#!/usr/bin/env python3
"""
Python API server for the viral content RAG system.
Exposes RAG functionality via REST API for MCP server consumption.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add pipeline to path
sys.path.append(str(Path(__file__).parent / "pipeline"))

from rag_query import RAGQueryInterface

# Initialize FastAPI app
app = FastAPI(
    title="Viral Content RAG API",
    description="RAG backend for the MCP server",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG interface
rag_interface = None

class QueryRequest(BaseModel):
    question: str
    collection: str = "auto"
    avoid_linkedin: bool = False

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    collection_used: str
    metadata: Dict

@app.on_event("startup")
async def startup_event():
    """Initialize RAG interface on startup."""
    global rag_interface
    try:
        rag_interface = RAGQueryInterface()
        print("✅ RAG interface initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG interface: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Viral Content RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/query", "/health"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "rag_ready": rag_interface is not None}

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system."""
    if rag_interface is None:
        raise HTTPException(status_code=503, detail="RAG interface not initialized")
    
    try:
        # Query the RAG system
        result = rag_interface.query(
            question=request.question,
            collection=request.collection,
            avoid_linkedin=request.avoid_linkedin
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            collection_used=result.get("collection", request.collection),
            metadata=result.get("metadata", {})
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.get("/collections")
async def get_collections():
    """Get available collections."""
    return {
        "collections": [
            {
                "name": "auto",
                "description": "Automatically selects the best collection based on query"
            },
            {
                "name": "framework",
                "description": "General viral content framework"
            },
            {
                "name": "logic",
                "description": "Tactical systems and delegation logic"
            },
            {
                "name": "linkedin",
                "description": "LinkedIn-specific content optimization"
            }
        ]
    }

if __name__ == "__main__":
    # Check if environment is properly configured
    required_env = ["OPENAI_API_KEY"]
    missing_env = [var for var in required_env if not os.getenv(var)]
    
    if missing_env:
        print(f"❌ Missing required environment variables: {', '.join(missing_env)}")
        print("Please set these variables before starting the server.")
        sys.exit(1)
    
    print("🚀 Starting Viral Content RAG API server...")
    print("📚 Make sure Qdrant is running and the vector database is populated")
    print("🔗 API will be available at http://localhost:8000")
    print("📖 Documentation at http://localhost:8000/docs")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["./"]
    )
