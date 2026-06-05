import os
import sys
from pathlib import Path

# 1. DYNAMIC SYSTEM PATH RESOLUTION (Cross-Platform Resilient)
if os.getenv("RAILWAY_ENVIRONMENT"):
    WORKSPACE_ROOT = Path(os.getcwd())
else:
    WORKSPACE_ROOT = Path(r"C:\Users\Taal\OneDrive\Desktop\Persona_TaalChawla")

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from groq import AsyncGroq
from dotenv import load_dotenv

# Load environment configurations safely
load_dotenv(WORKSPACE_ROOT / ".env")

# Synchronized Type & Package Imports
from backend.models.schemas import ChatRequest, ChatResponse, AvailabilityRequest, BookingRequest
from backend.rag.retriever import retrieve
from backend.persona.system_prompt import build_system_prompt
from backend.calendar.calcom import get_available_slots, create_booking
from backend.github.fetcher import detect_repo_from_query, fetch_repo_context

# 2. Application Setup
app = FastAPI(
    title="Taal Chawla AI Persona Platform",
    description="Production-grade autonomous representative gateway running over Groq & Local Transformers.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))


# ─── Core API Chat Routing Operations ─────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main Chat Routing Engine: Triages keywords specifically from the *current* message 
    to prevent conversational history context pollution, collects vector chunks, 
    injects live GitHub contexts, and runs fast inference cycles via Groq.
    """
    # CRITICAL FIX: Only evaluate the CURRENT user message for metadata filtering.
    # Evaluating query_lower against history causes strict retrieval dropouts!
    current_message_lower = request.message.lower()
    source_filter = None
    
    if "nexusops" in current_message_lower:
        source_filter = "nexusops"
    elif "tumor" in current_message_lower or "brain" in current_message_lower or "scaling" in current_message_lower:
        source_filter = "brain_tumor"
    elif "study" in current_message_lower or "college" in current_message_lower or "certifications" in current_message_lower or "resume" in current_message_lower:
        source_filter = "resume"

    # Step 1: Extract matching contexts from Pinecone Vector space
    try:
        chunks = await retrieve(request.message, top_k=5, source_filter=source_filter)
        
        # Cross-document search fallback if targeted namespace filter comes up empty
        if not chunks and source_filter:
            chunks = await retrieve(request.message, top_k=5, source_filter=None)
            
        context_string = "\n\n---\n\n".join(chunks) if chunks else "No specific portfolio document context retrieved."
        
    except Exception as e:
        print(f"[ERROR] RAG Extraction breakdown: {str(e)}")
        context_string = "No specific portfolio document context retrieved."

    # Step 2: Intercept repository queries and pull hot live context from GitHub
    repo_id = detect_repo_from_query(request.message)
    if repo_id:
        try:
            github_context = await fetch_repo_context(repo_id)
            if github_context:
                context_string = f"{github_context}\n\n---\n\n{context_string}"
        except Exception as e:
            print(f"[WARNING] GitHub Live integration path failure: {str(e)}")

    # Step 3: Build System Prompt using our corrected female pronoun profile (she/her)
    sys_prompt = build_system_prompt(context_string, is_voice=request.is_voice)

    messages = [{"role": "system", "content": sys_prompt}]
    
    # Step 4: Map history safely using strict Pydantic parsing loops
    for msg in request.conversation_history[-6:]:
        # Standardize structural model indicators cleanly to prevent Groq validation faults
        role_map = "assistant" if msg.role.lower() == "assistant" else "user"
        messages.append({"role": role_map, "content": msg.content})
        
    messages.append({"role": "user", "content": request.message})

    # Step 5: Execute fast inference processing over Groq
    try:
        completion = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.2,
            max_tokens=500
        )
        response_text = completion.choices[0].message.content
    except Exception as e:
        print(f"[CRITICAL] Groq compilation failure: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Groq runtime thread exception: {str(e)}")

    # Step 6: Package response schemas safely
    sources_used = []
    if source_filter:
        sources_used.append(source_filter)
    if repo_id:
        sources_used.append(f"github_live:{repo_id}")
    if not sources_used:
        sources_used = ["vector_index"]

    return ChatResponse(
        response=response_text,
        sources_used=sources_used
    )


# ─── Core Calendar Integration Routes ─────────────────────────────────────────

@app.post("/check-availability")
async def check_availability_endpoint(request: AvailabilityRequest):
    """Parses linked Cal.com profile openings dynamically and isolates the next 3 available slots."""
    username = os.getenv("CALCOM_USERNAME", "taalchawla")
    event_type_id = os.getenv("CALCOM_EVENT_TYPE_ID")
    
    if not event_type_id:
        raise HTTPException(status_code=500, detail="Environmental variable CALCOM_EVENT_TYPE_ID is unconfigured.")

    try:
        slots = await get_available_slots(
            event_type_id=event_type_id,
            timezone=request.timezone,
            username=username
        )
        return {"slots": slots[:3]}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream calendar server lookup failed: {str(e)}")


@app.post("/book-slot")
async def book_slot_endpoint(request: BookingRequest):
    """Commits and locks a confirmed schedule block on your calendar."""
    event_type_id = os.getenv("CALCOM_EVENT_TYPE_ID")
    if not event_type_id:
        raise HTTPException(status_code=500, detail="Environmental variable CALCOM_EVENT_TYPE_ID is unconfigured.")

    try:
        booking = await create_booking(
            event_type_id=event_type_id,
            start=request.slot_start,
            name=request.attendee_name,
            email=request.attendee_email,
            timezone=request.timezone
        )
        
        booking_uid = booking.get("uid") or booking.get("id") or "confirmed_fallback_uid"
        return {
            "confirmed": True,
            "booking_uid": str(booking_uid),
            "confirmation_message": f"Done! A meeting has been confirmed. {request.attendee_name} will receive an invitation at {request.attendee_email}."
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream verification failed to finalize appointment seat: {str(e)}")


# ─── Infrastructure Configuration Routes ──────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok", "persona": "Taal Chawla AI Representative"}


@app.get("/")
async def serve_ui():
    index_path = WORKSPACE_ROOT / "frontend" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Taal Chawla AI Persona API Operational", "docs": "/docs"}


frontend_path = WORKSPACE_ROOT / "frontend"
if frontend_path.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_path)), name="frontend")