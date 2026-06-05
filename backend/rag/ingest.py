import os
import sys
import time
from pathlib import Path
from typing import Iterable, List

import pdfplumber
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# 1. Absolute Workspace Path Configuration for Taal's Desktop Environment
workspace_root = Path(r"C:\Users\Taal\OneDrive\Desktop\Persona_TaalChawla")
load_dotenv(workspace_root / ".env")

# Ensure internal backend modules are discoverable by Python's search path
if str(workspace_root) not in sys.path:
    sys.path.append(str(workspace_root))

# The ONLY local utility module import required to tokenize document data arrays
from backend.rag.chunker import chunk_text


def extract_pdf_content(path: Path) -> str:
    """Safely extracts raw text layout strings out of technical target PDF documents."""
    print(f"  Extracting PDF: {path.name}")
    text = ""
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"  ERROR processing extraction on PDF asset {path.name}: {e}")
    return text


def extract_md_content(path: Path) -> str:
    """Safely reads raw plain text payloads straight from localized repository Markdown logs."""
    print(f"  Extracting Markdown: {path.name}")
    try:
        if path.exists():
            return path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  ERROR processing read on Markdown layout {path.name}: {e}")
    return ""


def get_batches(items: List, size: int) -> Iterable[List]:
    """Protects active computing resource limits by yielding sub-arrays sequentially."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main():
    # 2. Extract and Validate Essential Platform Operational Tokens
    pc_key = os.getenv("PINECONE_API_KEY")
    idx_name = os.getenv("PINECONE_INDEX_NAME", "taal-persona")

    if not pc_key:
        print("ERROR: PINECONE_API_KEY must be defined inside your local workspace .env file.")
        sys.exit(1)

    pc = Pinecone(api_key=pc_key)

    # 3. Idempotent Index Life-Cycle Handling (Configured strictly for 384-dim open-source vectors)
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if idx_name not in existing_indexes:
        print(f"Provisioning fresh 384-dimensional serverless vector index space '{idx_name}'...")
        pc.create_index(
            name=idx_name,
            dimension=384,  # Perfect structural match footprint for local SentenceTransformers
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Waiting for cloud routing records to come online across global nodes...")
        while not pc.describe_index(idx_name).status["ready"]:
            time.sleep(2)
        print("Vector storage framework initialized.")
    else:
        print(f"Target index space '{idx_name}' verified active in connected infrastructure.")

    index = pc.Index(idx_name)

    # 4. Initialize Local Embedding Neural Network Framework
    print("Loading sentence-transformers/all-MiniLM-L6-v2 directly into host memory context...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 5. Build Blueprint Target Documentation File Traversal Points
    sources_dir = workspace_root / "ingestion_sources"
    files_to_process = {
        "resume": sources_dir / "resume.pdf",
        "nexusops": sources_dir / "nexusops_tech_ref.pdf",
        "brain_tumor": sources_dir / "neuro_confidence.pdf",
        "github_readme": workspace_root / "README.md"  # Directly ingests design tradeoffs from the root repository
    }

    all_chunks = []
    for sid, path in files_to_process.items():
        if not path.exists():
            print(f"WARNING: Resource target not found, bypassing operational pipeline leg — {path.name}")
            continue

        # Polymorphic extraction parser assignment routing
        if path.suffix == ".pdf":
            text = extract_pdf_content(path)
        elif path.suffix in [".md", ".txt"]:
            text = extract_md_content(path)
        else:
            continue

        chunks = chunk_text(text, sid)
        print(f"  {sid}: Generated {len(chunks)} contextual text slices.")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("ERROR: Zero source data points generated. Please inspect files inside ingestion_sources/ directory location.")
        sys.exit(0)

    print(f"\nTotal vector blocks compiled for automated indexing: {len(all_chunks)}")

    # 6. Bulk Load Index Execution Loop (Processing safe data sets of 100 entries at a time)
    for batch_idx, batch in enumerate(get_batches(all_chunks, 100)):
        texts = [item["text"] for item in batch]
        
        # Computes semantic array math loops right on your native CPU thread layer without any API costs
        embeddings = model.encode(texts).tolist()

        vectors = []
        for i, emb in enumerate(embeddings):
            chunk = batch[i]
            vectors.append({
                "id": f"{chunk['source']}_chunk_{chunk['chunk_index']}",
                "values": emb,
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_index": chunk["chunk_index"],
                },
            })

        index.upsert(vectors=vectors)
        print(f"  Successfully dispatched batch sequence {batch_idx + 1} ({len(vectors)} payloads synced)")

    print("\nVector space architecture successfully updated. Pipeline has finished processing.")


if __name__ == "__main__":
    main()