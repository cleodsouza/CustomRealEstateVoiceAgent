"""providers/calendar/google_calendar.py — Google Calendar integration.

Handles creating site-visit appointments on a tenant's Google Calendar.
Credentials and calendar_id are stored in the agent's tool_config.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials as SACredentials

log = logging.getLogger("providers.calendar.google_calendar")


class GoogleCalendarClient:
    """Create events on a tenant's Google Calendar using service account auth."""

    def __init__(self, service_account_key: dict, calendar_id: str):
        """
        Args:
            service_account_key: The full service account JSON (parsed dict).
            calendar_id: The calendar ID or "primary" for the account's main calendar.
        """
        self.calendar_id = calendar_id
        self._credentials = SACredentials.from_service_account_info(
            service_account_key,
            scopes=["https://www.googleapis.com/auth/calendar"]
        )
        self._token: str | None = None

    async def _ensure_token(self) -> str:
        """Get a fresh access token (synchronous auth call wrapped in a thread)."""
        if self._token is None or self._credentials.expired:
            import asyncio
            await asyncio.to_thread(self._credentials.refresh, Request())
        return self._credentials.token

    async def create_event(self, title: str, start_time: datetime,
                           duration_minutes: int = 30,
                           description: str = "",
                           attendee_email: str = "") -> dict:
        """
        Create a calendar event.

        Args:
            title: Event title (e.g., "Site Visit - Northern Heights")
            start_time: Event start as datetime (timezone-aware recommended)
            duration_minutes: Event duration
            description: Event description
            attendee_email: Attendee email to invite (optional)

        Returns:
            Event dict from Google Calendar API
        """
        token = await self._ensure_token()

        end_time = start_time + timedelta(minutes=duration_minutes)

        event = {
            "summary": title,
            "description": description,
            "start": {
                "dateTime": start_time.isoformat(),
                "timeZone": "Asia/Kolkata",  # Priya's timezone
            },
            "end": {
                "dateTime": end_time.isoformat(),
                "timeZone": "Asia/Kolkata",
            },
        }

        if attendee_email:
            event["attendees"] = [{"email": attendee_email}]

        url = f"https://www.googleapis.com/calendar/v3/calendars/{self.calendar_id}/events"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(url, json=event, headers=headers)
            response.raise_for_status()
            return response.json()
