"""
Run once to populate Pinecone with embeddings from all source PDFs.
Uses HuggingFace Inference API for embeddings — no torch required.

Usage (from project root): python -m backend.rag.ingest
"""

import os
import sys
import time
from pathlib import Path

import httpx
import pdfplumber
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

from backend.rag.chunker import chunk_text

HF_API_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"

def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Sends text batches directly to the updated Hugging Face feature-extraction pipeline router.
    """
    import httpx
    import time
    
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
        "Content-Type": "application/json"
    }
    
    # Structure the payload array to meet the updated pipeline configuration
    payload = {"inputs": texts}
    
    # Built-in network resilience wrapper loop
    for attempt in range(3):
        try:
            response = httpx.post(
                HF_API_URL,
                json=payload,
                headers=headers,
                timeout=30.0
            )
            
            # Catch upstream model loading states (503) or rate limits gracefully
            if response.status_code == 503:
                print(f"[HF INFRA] Model is warming up on the cluster. Retrying in 10s...")
                time.sleep(10)
                continue
                
            response.raise_for_status()
            return response.json()
            
        except httpx.ConnectError as conn_err:
            print(f"[DNS WARNING] Attempt {attempt + 1} failed to resolve router domain. Checking connection...")
            time.sleep(5)
        except Exception as e:
            print(f"[HF CRITICAL] Pipeline execution exception: {str(e)}")
            raise e
            
    raise ConnectionError("Unable to establish a stable DNS handshake with Hugging Face router endpoints.")


def extract_pdf(path: Path) -> str:
    print(f"  Extracting: {path.name}")
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        print(f"  ERROR: {e}")
        return ""


def main():
    pc_key = os.getenv("PINECONE_API_KEY")
    idx_name = os.getenv("PINECONE_INDEX_NAME", "taal-persona")

    if not pc_key:
        print("ERROR: PINECONE_API_KEY missing from .env")
        sys.exit(1)

    pc = Pinecone(api_key=pc_key)

    # Recreate index at 384-dim if needed
    existing = [i.name for i in pc.list_indexes()]
    if idx_name in existing:
        existing_dim = pc.describe_index(idx_name).dimension
        if existing_dim != 384:
            print(f"Index dimension mismatch ({existing_dim} != 384). Deleting and recreating...")
            pc.delete_index(idx_name)
            existing = []

    if idx_name not in existing:
        print(f"Creating index '{idx_name}' at 384-dim cosine...")
        pc.create_index(
            name=idx_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(idx_name).status["ready"]:
            time.sleep(2)
        print("Index ready.")

    index = pc.Index(idx_name)

    # Source PDFs
    sources_dir = BASE_DIR / "ingestion_sources"
    sources = {
        "resume":      sources_dir / "resume.pdf",
        "nexusops":    sources_dir / "nexusops_tech_ref.pdf",
        "brain_tumor": sources_dir / "neuro_confidence.pdf",
    }

    all_chunks = []
    for source_id, pdf_path in sources.items():
        if not pdf_path.exists():
            print(f"  WARNING: Not found — {pdf_path.name}")
            continue
        text = extract_pdf(pdf_path)
        chunks = chunk_text(text, source_id)
        print(f"  {source_id}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No content to ingest.")
        sys.exit(0)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Batch embed and upsert — HF API handles up to 64 texts per request safely
    batch_size = 32
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]

        try:
            vectors_data = embed_batch(texts)
        except Exception as e:
            print(f"  Embedding batch {i//batch_size + 1} failed: {e}")
            print("  Retrying after 10s...")
            time.sleep(10)
            vectors_data = embed_batch(texts)

        vectors = [
            {
                "id": f"{c['source']}_chunk_{c['chunk_index']}",
                "values": vec,
                "metadata": {"text": c["text"], "source": c["source"]},
            }
            for c, vec in zip(batch, vectors_data)
        ]

        index.upsert(vectors=vectors)
        print(f"  Upserted batch {i//batch_size + 1} ({len(vectors)} vectors)")

        # Small delay to respect HF free tier rate limits
        time.sleep(1)

    print("\nIngestion complete.")
    stats = index.describe_index_stats()
    print(f"Total vectors in index: {stats['total_vector_count']}")


if __name__ == "__main__":
    main()