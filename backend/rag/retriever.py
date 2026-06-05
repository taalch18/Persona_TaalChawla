import asyncio
import os
from typing import List, Optional
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

class LocalSearchClient:
    def __init__(self) -> None:
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(os.getenv("PINECONE_INDEX_NAME", "taal-persona"))
        # Initializes the model directly into local RAM/CPU memory once
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    async def retrieve(self, query: str, top_k: int = 5, source_filter: Optional[str] = None) -> List[str]:
        # 1. Generate the 384-dimensional vector embedding on the local CPU thread pool
        vector_embeddings = await asyncio.to_thread(self.model.encode, [query])
        vector = vector_embeddings[0].tolist()

        # 2. Build explicit parameters
        params = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": True,
        }
        if source_filter:
            params["filter"] = {"source": {"$eq": source_filter}}

        # 3. Offload blocking synchronous Pinecone network I/O to a background thread
        results = await asyncio.to_thread(self.index.query, **params)

        # 4. Extract and return the raw text chunks cleanly
        return [
            match["metadata"]["text"] 
            for match in results.get("matches", []) 
            if match.get("metadata") and "text" in match["metadata"]
        ]

# Concurrency-safe lazy initialization state
_client: Optional[LocalSearchClient] = None
_lock = asyncio.Lock()

async def retrieve(query: str, top_k: int = 5, source_filter: Optional[str] = None) -> List[str]:
    global _client
    if _client is None:
        async with _lock:
            if _client is None:
                _client = LocalSearchClient()
                
    return await _client.retrieve(query, top_k, source_filter)