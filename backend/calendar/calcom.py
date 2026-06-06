import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx

CALCOM_BASE = "https://api.cal.com/v2"
IST = ZoneInfo("Asia/Kolkata")


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


async def _fetch_slots(start_dt: datetime, end_dt: datetime, event_type_id: str, user_timezone: str) -> list[dict]:
    """Raw slot fetch between two UTC datetimes."""
    params = {
        "startTime": start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "endTime":   end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
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
            slots.append({
                "start":   slot["time"],
                "display": _format_display(slot["time"]),
                "date":    date_key,
            })
    return slots


async def get_available_slots(
    event_type_id: str,
    user_timezone: str = "Asia/Kolkata",
) -> list[dict]:
    """
    Returns 3 slots spread across different days.
    If all 3 happen to be on the same day, still returns them.
    """
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=14)

    all_slots = await _fetch_slots(now, end, event_type_id, user_timezone)

    # Pick one slot per day until we have 3
    seen_dates = set()
    selected = []
    for slot in all_slots:
        if slot["date"] not in seen_dates:
            seen_dates.add(slot["date"])
            selected.append(slot)
        if len(selected) == 3:
            break

    # Fallback: if fewer than 3 unique days, just take first 3 slots
    if len(selected) < 3:
        selected = all_slots[:3]

    return [{"start": s["start"], "display": s["display"]} for s in selected]


async def get_slots_for_date(
    event_type_id: str,
    target_date: str,
    user_timezone: str = "Asia/Kolkata",
) -> list[dict]:
    """
    Returns up to 3 slots on a specific date.
    target_date format: 'YYYY-MM-DD'
    """
    # Build UTC window covering the full IST day
    date_obj = datetime.strptime(target_date, "%Y-%m-%d")
    # Start of that day in IST = 00:00 IST = previous day 18:30 UTC
    start_ist = datetime(date_obj.year, date_obj.month, date_obj.day, 0, 0, 0, tzinfo=IST)
    end_ist   = datetime(date_obj.year, date_obj.month, date_obj.day, 23, 59, 0, tzinfo=IST)

    start_utc = start_ist.astimezone(timezone.utc)
    end_utc   = end_ist.astimezone(timezone.utc)

    all_slots = await _fetch_slots(start_utc, end_utc, event_type_id, user_timezone)

    # Filter to only slots on the target date
    day_slots = [s for s in all_slots if s["date"] == target_date]

    # Return first 3
    return [{"start": s["start"], "display": s["display"]} for s in day_slots[:3]]


async def create_booking(
    event_type_id: str,
    start: str,
    attendee_name: str,
    attendee_email: str,
    user_timezone: str = "Asia/Kolkata",
) -> dict:
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
            print(f"[CALCOM ERROR] {response.status_code}: {response.text}")
            response.raise_for_status()
        data = response.json()
    return data.get("data", {})