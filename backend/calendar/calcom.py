import os
from datetime import datetime, timedelta, timezone

import httpx

CALCOM_BASE = "https://api.cal.com/v2"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.getenv('CALCOM_API_KEY')}",
        "Content-Type": "application/json",
        "cal-api-version": "2024-08-13",
    }


def _format_display(iso_time: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        day = dt.strftime("%A, %b")
        date_num = str(dt.day)
        hour = dt.strftime("%I").lstrip("0") or "12"
        minute = dt.strftime("%M")
        ampm = dt.strftime("%p")
        return f"{day} {date_num} at {hour}:{minute} {ampm} IST"
    except Exception:
        return iso_time


async def get_available_slots(
    event_type_id: str,
    user_timezone: str = "Asia/Kolkata",
) -> list[dict]:
    
    from datetime import timezone as tz
    import zoneinfo

    ist = zoneinfo.ZoneInfo("Asia/Kolkata")
    now = datetime.now(tz.utc)
    now_ist = now.astimezone(ist)
    end = now + timedelta(days=14)
    end_ist = end.astimezone(ist)

    params = {
        "startTime": now_ist.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime": end_ist.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "eventTypeId": event_type_id,
        "timeZone": user_timezone,
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
    for date_key in sorted(data.get("data", {}).get("slots", {}).keys()):
        for slot in data["data"]["slots"][date_key]:
            iso_time = slot["time"]
            slots.append({
                "start": iso_time,          # pass this back as-is for booking
                "display": _format_display(iso_time),
            })
            if len(slots) >= 3:
                return slots

    return slots


async def create_booking(
    event_type_id: str,
    start: str,
    attendee_name: str,
    attendee_email: str,
    user_timezone: str = "Asia/Kolkata",
) -> dict:
    """
    Creates a confirmed booking on Cal.com.
    start: ISO 8601 string exactly as returned by get_available_slots
    """
    payload = {
        "eventTypeId": int(event_type_id),
        "start": start,
        "attendee": {
            "name": attendee_name,
            "email": attendee_email,
            "timeZone": user_timezone,
            "language": "en",
        },
    }

    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(
            f"{CALCOM_BASE}/bookings",
            json=payload,
            headers=_headers(),
        )

        if response.status_code not in (200, 201):
            # Print full response for debugging
            print(f"[CALCOM ERROR] Status {response.status_code}: {response.text}")
            response.raise_for_status()

        data = response.json()

    return data.get("data", {})