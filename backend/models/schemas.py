from pydantic import BaseModel, EmailStr
from typing import List, Optional

class ChatMessage(BaseModel):
    role: str  # "user", "system", or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage] = []
    is_voice: bool = False  # Set to True when Vapi hits the endpoint to force brief text responses

class ChatResponse(BaseModel):
    response: str
    sources_used: List[str]  # Tracking array identifying source documents used by the vector lookup

class AvailabilityRequest(BaseModel):
    timezone: str = "Asia/Kolkata"

class SlotDisplay(BaseModel):
    start: str        # ISO 8601 Timestamp used for scheduling transactions
    display: str      # Clean, cross-platform human-readable date presentation

class AvailabilityResponse(BaseModel):
    slots: List[SlotDisplay]

class BookingRequest(BaseModel):
    slot_start: str           # ISO 8601 Target Timestamp
    attendee_name: str
    attendee_email: EmailStr  # Enforces formal RFC validation on input email strings
    timezone: str = "Asia/Kolkata"

class BookingResponse(BaseModel):
    confirmed: bool
    booking_uid: Optional[str] = None
    meeting_url: Optional[str] = None
    confirmation_message: str