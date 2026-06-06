import os
import sys
import traceback
from pathlib import Path
from contextlib import asynccontextmanager

# Path resolution — works locally and on Railway
if os.getenv("RAILWAY_ENVIRONMENT"):
    WORKSPACE_ROOT = Path(os.getcwd())
else:
    WORKSPACE_ROOT = Path(r"C:\Users\Taal\OneDrive\Desktop\Persona_TaalChawla")

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from dotenv import load_dotenv
load_dotenv(WORKSPACE_ROOT / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import AsyncGroq

from backend.calendar.calcom import create_booking, get_available_slots
from backend.github.fetcher import fetch_repo_context
from backend.models.schemas import (
    AvailabilityRequest,
    BookingRequest,
    ChatRequest,
    ChatResponse,
)
from backend.persona.system_prompt import build_system_prompt
from backend.rag.retriever import retrieve

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_PINECONE_CHARS = 3000
MAX_GITHUB_CHARS   = 800
MAX_OUTPUT_TOKENS  = 400
HISTORY_TURNS      = 4
GROQ_MODEL         = "llama-3.1-8b-instant"

# ── GitHub cache — populated once at boot ──────────────────────────────────────
REPO_CACHE: dict[str, str] = {"nexusops": "", "brain_tumor": ""}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[BOOT] Pre-fetching GitHub repository contexts...")
    try:
        REPO_CACHE["nexusops"]    = await fetch_repo_context("nexusops") or ""
        REPO_CACHE["brain_tumor"] = await fetch_repo_context("brain_tumor") or ""
        print("[BOOT] GitHub caches loaded.")
    except Exception as e:
        print(f"[BOOT WARNING] GitHub cache failed: {e}")
    yield


app = FastAPI(
    title="Taal Chawla AI Persona",
    description="AI representative for Taal Chawla — RAG-grounded over resume and GitHub repos.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))


# ── Source routing helpers ─────────────────────────────────────────────────────

def detect_source_filter(message: str) -> str | None:
    m = message.lower()
    if any(k in m for k in [
        "nexusops", "nexus ops", "langgraph", "governor", "rrf",
        "reciprocal rank", "hitl", "oomkill", "crashloop",
        "latency", "kubernetes", "k8s", "sre", "playbook",
        "embedding model", "minilm",
    ]):
        return "nexusops"
    if any(k in m for k in [
        "tumor", "brain", "resnet", "temperature scaling", "ece",
        "calibration", "grad-cam", "gradcam", "mri", "meningioma",
        "glioma", "pituitary", "abstention", "hallucination rate",
        "confidence", "patient", "splitting",
    ]):
        return "brain_tumor"
    if any(k in m for k in [
        "study", "college", "university", "cgpa", "gpa",
        "certification", "certif", "drdo", "internship",
        "resume", "education", "mait", "ggsipu",
    ]):
        return "resume"
    return None


def detect_github_repo(message: str) -> str | None:
    m = message.lower()
    if any(k in m for k in [
        "nexusops", "nexus ops", "langgraph", "governor",
        "rrf", "hitl", "kubernetes", "k8s", "latency",
    ]):
        return "nexusops"
    if any(k in m for k in [
        "tumor", "brain", "resnet", "temperature scaling",
        "ece", "calibration", "grad-cam", "patient splitting",
    ]):
        return "brain_tumor"
    return None


# ── /chat ──────────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):

    source_filter = detect_source_filter(request.message)

    # 1. Pinecone retrieval
    try:
        chunks = await retrieve(request.message, top_k=3, source_filter=source_filter)
        if not chunks and source_filter:
            chunks = await retrieve(request.message, top_k=3, source_filter=None)
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        chunks = []

    pinecone_context = "\n\n---\n\n".join(chunks) if chunks else ""
    if len(pinecone_context) > MAX_PINECONE_CHARS:
        pinecone_context = pinecone_context[:MAX_PINECONE_CHARS] + "\n...[trimmed]"

    # 2. GitHub context — only injected when question is repo-specific
    github_context = ""
    repo_id = detect_github_repo(request.message)
    if repo_id and REPO_CACHE.get(repo_id):
        github_context = REPO_CACHE[repo_id][:MAX_GITHUB_CHARS]

    # 3. Assemble full context
    context_parts = []
    if github_context:
        context_parts.append(f"[LIVE GITHUB — {repo_id}]\n{github_context}")
    if pinecone_context:
        context_parts.append(f"[KNOWLEDGE BASE]\n{pinecone_context}")
    full_context = "\n\n---\n\n".join(context_parts) if context_parts else "No context retrieved."

    # 4. Build system prompt
    sys_prompt = build_system_prompt(full_context, is_voice=request.is_voice)

    # 5. Build messages for Groq (OpenAI-compatible format)
    messages = [{"role": "system", "content": sys_prompt}]
    if request.conversation_history:
        try:
            for msg in request.conversation_history[-HISTORY_TURNS:]:
                role = "assistant" if msg.role.lower() == "assistant" else "user"
                messages.append({"role": role, "content": msg.content})
        except Exception as e:
            print(f"[WARNING] History parse error: {e}")

    messages.append({"role": "user", "content": request.message})

    # 6. Groq async call
    try:
        completion = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
        response_text = completion.choices[0].message.content

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")

    sources_used = []
    if github_context:
        sources_used.append(f"github_{repo_id}")
    if chunks:
        sources_used.append(source_filter or "general")

    return ChatResponse(response=response_text, sources_used=sources_used)


# ── Calendar routes ────────────────────────────────────────────────────────────

@app.post("/check-availability")
async def check_availability_endpoint(request: AvailabilityRequest):
    event_type_id = os.getenv("CALCOM_EVENT_TYPE_ID")
    if not event_type_id:
        raise HTTPException(status_code=500, detail="CALCOM_EVENT_TYPE_ID not set.")
    try:
        slots = await get_available_slots(
            event_type_id=event_type_id,
            user_timezone=request.timezone,
        )
        return {"slots": slots[:3]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Cal.com error: {str(e)}")


@app.post("/book-slot")
async def book_slot_endpoint(request: BookingRequest):
    event_type_id = os.getenv("CALCOM_EVENT_TYPE_ID")
    if not event_type_id:
        raise HTTPException(status_code=500, detail="CALCOM_EVENT_TYPE_ID not set.")
    try:
        booking = await create_booking(
            event_type_id=event_type_id,
            start=request.slot_start,
            attendee_name=request.attendee_name,
            attendee_email=request.attendee_email,
            user_timezone=request.timezone,
        )
        uid = booking.get("uid") or booking.get("id") or "confirmed"
        return {
            "confirmed": True,
            "booking_uid": str(uid),
            "confirmation_message": (
                f"Done! A meeting has been confirmed. "
                f"{request.attendee_name} will receive a confirmation at {request.attendee_email}."
            ),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"Booking failed: {str(e)}")


# ── Infra routes ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok", "persona": "Taal Chawla AI Representative"}


@app.get("/")
async def serve_ui():
    index_path = WORKSPACE_ROOT / "frontend" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Taal Chawla AI Persona API", "docs": "/docs"}


frontend_path = WORKSPACE_ROOT / "frontend"
if frontend_path.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_path)), name="frontend")