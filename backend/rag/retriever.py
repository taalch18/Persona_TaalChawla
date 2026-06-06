import asyncio
import os
from typing import Optional
from functools import lru_cache

import httpx
from pinecone import Pinecone

HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    return pc.Index(os.getenv("PINECONE_INDEX_NAME", "taal-persona"))


async def _embed(text: str) -> list[float]:
    headers = {}
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": text, "options": {"wait_for_model": True}},
        )
        response.raise_for_status()
        result = response.json()

    if isinstance(result[0], list):
        return result[0]
    return result


async def retrieve(
    query: str,
    top_k: int = 3,
    source_filter: Optional[str] = None,
) -> list[str]:
    
    index = _get_index()

    # 1. Embed the query
    try:
        vector = await _embed(query)
    except Exception as e:
        print(f"[RETRIEVER] Embedding failed: {e}")
        return []

    # 2. Build Pinecone query
    query_params = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True,
    }
    if source_filter:
        query_params["filter"] = {"source": {"$eq": source_filter}}

    results = await asyncio.to_thread(index.query, **query_params)

    # 3. Extract chunk texts
    chunks = [
        match["metadata"]["text"]
        for match in results.get("matches", [])
        if match.get("metadata") and "text" in match["metadata"]
    ]

    print(f"[RAG RETRIEVAL] Extracted {len(chunks)} valid context chunks for filter: '{source_filter}'")
    return chunks