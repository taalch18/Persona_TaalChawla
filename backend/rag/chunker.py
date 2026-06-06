import tiktoken
from typing import List, Dict

def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:

    text = text.strip()
    if not text:
        return []

    enc = tiktoken.get_encoding("cl100k_base") 
    tokens = enc.encode(text)

    chunks = []
    stride = max(chunk_size - overlap, 1) # Prevent infinite loops if overlap >= size

    for i in range(0, len(tokens), stride):
        batch = tokens[i : i + chunk_size]
        chunks.append({
            "text": enc.decode(batch),
            "source": source,
            "chunk_index": len(chunks),
        })
        if i + chunk_size >= len(tokens):
            break

    return chunks