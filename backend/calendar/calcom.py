"""
Cal.com v2 API wrapper configured for zero-downtime integration with FastAPI.
Docs: https://cal.com/docs/api-reference/v2
"""

import os
import sys
from datetime import datetime, timedelta, timezone
import httpx

CALCOM_BASE = "https://api.cal.com/v2"


def _headers() -> dict:
    """Compiles unified validation authorization headers for upstream Cal.com connectivity."""
    api_key = os.getenv("CALCOM_API_KEY")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "cal-api-version": "2024-08-13",
    }


def _format_slot_display(iso_time: str, user_timezone: str) -> str:
    """
    Converts an ISO 8601 UTC time string into a highly readable voice/UI display string.
    Example: '2026-06-07T09:00:00.000Z' -> 'Saturday, Jun 7 at 9:00 AM IST'
    """
    try:
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        
        # Simple IST offset calculation helper for seamless terminal responses
        if "Asia/Kolkata" in user_timezone or "kolkata" in user_timezone.lower():
            ist_offset = timedelta(hours=5, minutes=30)
            dt = dt + ist_offset
            
        # Cross-platform safe day/hour padding format stripping (# for Windows, - for Linux)
        fmt_flag = "#" if sys.platform.startswith("win") else "-"
        return dt.strftime(f"%A, %b %{fmt_flag}d at %{fmt_flag}I:%M %p")
    except Exception:
        return iso_time


async def get_available_slots(
    event_type_id: str,
    timezone: str = "Asia/Kolkata",
    username: str = None,  # Gracefully accepts the parameter passed by main.py
) -> list[dict]:
    """
    Queries Cal.com for openings and packages the immediate next 3 valid slots.
    Each entry matches: {"start": ISO_string, "display": human_readable_string}
    """
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=7)

    params = {
        "startTime": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eventTypeId": event_type_id,
        "timeZone": timezone,
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            f"{CALCOM_BASE}/slots/available",
            params=params,
            headers=_headers(),
        )
        response.raise_for_status()
        data = response.json()

    slots = []
    raw_slots_data = data.get("data", {}).get("slots", {}) if "data" in data else data.get("slots", {})
    
    for date_key in sorted(raw_slots_data.keys()):
        for slot in raw_slots_data[date_key]:
            # Accommodates variation variations in API v2 response variants safely
            slot_time = slot.get("time") or slot.get("start")
            if slot_time:
                slots.append({
                    "start": slot_time,
                    "display": _format_slot_display(slot_time, timezone),
                })
            if len(slots) >= 3:
                return slots

    return slots


async def create_booking(
    event_type_id: str,
    start: str,
    name: str,       # Maps directly to request parameter from main.py
    email: str,      # Maps directly to request parameter from main.py
    timezone: str = "Asia/Kolkata",
) -> dict:
    """
    Commits a confirmed calendar seat booking immediately via Cal.com transactional endpoints.
    """
    payload = {
        "eventTypeId": int(event_type_id),
        "start": start,
        "attendee": {
            "name": name,
            "email": email,
            "timeZone": timezone,
            "language": "en",
        },
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            f"{CALCOM_BASE}/bookings",
            json=payload,
            headers=_headers(),
        )
        response.raise_for_status()
        data = response.json()

    # Normalize response data layout variations between staging variants safely
    return data.get("data", data)