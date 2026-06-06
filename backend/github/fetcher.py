"""
GitHub live fetcher — fetches README content and file tree at query time.
Repos covered:
  nexusops     -> taalch18/NexusOps
  brain_tumor  -> taalch18/neuro_confidence_aware
"""

import os
import asyncio  # Moved to the top to prevent runtime NameError dropouts
from typing import Optional
import httpx

GITHUB_API = "https://api.github.com"

REPO_MAP = {
    "nexusops": "taalch18/NexusOps",
    "brain_tumor": "taalch18/neuro_confidence_aware",
}

REPO_KEYWORDS = {
    "nexusops": [
        "nexusops", "nexus ops", "kubernetes", "k8s", "langgraph",
        "governor", "rrf", "reciprocal rank", "hitl", "sre", "pinecone",
        "minilm", "agentic rag", "oomkill", "crashloopbackoff",
    ],
    "brain_tumor": [
        "brain tumor", "neuro", "resnet", "temperature scaling",
        "ece", "calibration", "grad-cam", "gradcam", "mri",
        "meningioma", "glioma", "pituitary", "abstention",
        "hallucination rate", "confidence aware",
    ],
}


def detect_repo_from_query(query: str) -> Optional[str]:
    """
    Returns 'nexusops' or 'brain_tumor' if the query is about a specific repo.
    """
    query_lower = query.lower()
    for repo_id, keywords in REPO_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return repo_id
    return None


def _github_headers() -> dict:
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def fetch_readme(repo_id: str) -> str:
    
    repo = REPO_MAP.get(repo_id)
    if not repo:
        return ""

    url = f"{GITHUB_API}/repos/{repo}/readme"

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get(
                url,
                headers={**_github_headers(), "Accept": "application/vnd.github.raw+json"},
            )
            if response.status_code == 200:
                return response.text
            
            # Local fallback snapshot to preserve RAG flows during high GitHub api throttle limits
            return _get_offline_readme_fallback(repo_id)
    except Exception:
        return _get_offline_readme_fallback(repo_id)


async def fetch_file_tree(repo_id: str) -> str:
    repo = REPO_MAP.get(repo_id)
    if not repo:
        return ""

    url = f"{GITHUB_API}/repos/{repo}/contents/"

    try:
        async with httpx.AsyncClient(timeout=8) as client:
            response = await client.get(url, headers=_github_headers())
            if response.status_code != 200:
                return ""

            items = response.json()
            lines = [f"Repository: {repo}", "Top-level structure:"]
            for item in items:
                icon = "📁" if item["type"] == "dir" else "📄"
                lines.append(f"  {icon} {item['name']}")
            return "\n".join(lines)
    except Exception:
        return ""


async def fetch_repo_context(repo_id: str) -> str:

    readme, tree = await asyncio.gather(
        fetch_readme(repo_id),
        fetch_file_tree(repo_id),
    )

    parts = []
    if tree:
        parts.append(f"[LIVE FROM GITHUB — {REPO_MAP[repo_id]}]\n{tree}")
    if readme:
        parts.append(f"[README.md — {REPO_MAP[repo_id]}]\n{readme}")

    return "\n\n---\n\n".join(parts) if parts else ""


def _get_offline_readme_fallback(repo_id: str) -> str:
    if repo_id == "nexusops":
        return (
            "Architecture: Modular, asynchronous RAG infrastructure pipeline built over FastAPI.\n"
            "Core Stack: Pinecone 384-dimension vector DB, LangGraph State Machines, Groq LLaMA models.\n"
            "Optimizations: Reciprocal Rank Fusion (RRF) hybrid search, Governor Pattern Human-in-the-loop gates."
        )
    elif repo_id == "brain_tumor":
        return (
            "Architecture: Confidence-Aware Brain Tumor Classification built on top of ResNet18 core layers.\n"
            "Features: Temperature Scaling calibration adjustment (ECE reduction down to 0.031), patient-wise data splitting logic, and Grad-CAM interpretability visualizations."
        )
    return ""