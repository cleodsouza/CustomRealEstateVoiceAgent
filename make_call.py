"""make_call.py

Place an outbound call through Vobiz.

Examples:

    python make_call.py --to 9845102114
    python make_call.py --to +919845102114
    python make_call.py --to 9845102114 --agent priya
    python make_call.py --to 9845102114 --from +917971443000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx

import config


FROM_NUMBER = os.getenv("FROM_NUMBER", "")


def normalize_indian_number(number: str) -> str:
    """Normalize and validate an Indian 10-digit mobile number."""

    if not number:
        raise ValueError("Phone number cannot be empty.")

    # Remove common formatting.
    number = (
        number
        .strip()
        .replace(" ", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
    )

    # 0091XXXXXXXXXX -> +91XXXXXXXXXX
    if number.startswith("00"):
        number = "+" + number[2:]

    # +91XXXXXXXXXX
    if number.startswith("+91"):
        digits = number[3:]

        if len(digits) != 10 or not digits.isdigit():
            raise ValueError(
                f"Invalid Indian mobile number: {number!r}. "
                "Expected +91 followed by exactly 10 digits."
            )

        # Indian mobile numbers normally start with 6-9.
        if digits[0] not in "6789":
            raise ValueError(
                f"Invalid Indian mobile number: {number!r}. "
                "Mobile number should start with 6, 7, 8, or 9."
            )

        return "+91" + digits

    # 91XXXXXXXXXX
    if number.startswith("91"):
        digits = number[2:]

        if len(digits) != 10 or not digits.isdigit():
            raise ValueError(
                f"Invalid Indian mobile number: {number!r}. "
                "Expected 91 followed by exactly 10 digits."
            )

        if digits[0] not in "6789":
            raise ValueError(
                f"Invalid Indian mobile number: {number!r}."
            )

        return "+91" + digits

    # 10-digit local number.
    if len(number) == 10 and number.isdigit():

        if number[0] not in "6789":
            raise ValueError(
                f"Invalid Indian mobile number: {number!r}. "
                "Mobile number should start with 6, 7, 8, or 9."
            )

        return "+91" + number

    raise ValueError(
        f"Invalid Indian phone number: {number!r}. "
        "Use 10 digits, 91XXXXXXXXXX, or +91XXXXXXXXXX."
    )


def make_call(
    to: str,
    from_: str,
    agent: str | None = None,
) -> None:
    """Create an outbound Vobiz call."""

    if not config.VOBIZ_AUTH_ID:
        sys.exit("Missing VOBIZ_AUTH_ID in .env")

    if not config.VOBIZ_AUTH_TOKEN:
        sys.exit("Missing VOBIZ_AUTH_TOKEN in .env")

    try:
        to = normalize_indian_number(to)
        from_ = normalize_indian_number(from_)

    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    # Build the answer webhook.
    answer_url = (
        f"https://{config.PUBLIC_HOST}"
        f"/answer?token={config.WS_AUTH_TOKEN}"
    )

    if agent:
        answer_url += f"&agent={agent}"

    # Build hangup webhook.
    hangup_url = (
        f"https://{config.PUBLIC_HOST}"
        f"/hangup?token={config.WS_AUTH_TOKEN}"
    )

    # Vobiz outbound-call endpoint.
    url = (
        f"{config.VOBIZ_API_BASE}"
        f"/Account/{config.VOBIZ_AUTH_ID}/Call/"
    )

    headers = {
        "Content-Type": "application/json",
        "X-Auth-ID": config.VOBIZ_AUTH_ID,
        "X-Auth-Token": config.VOBIZ_AUTH_TOKEN,
    }

    payload: dict[str, Any] = {
        "from": from_,
        "to": to,
        "answer_url": answer_url,
        "answer_method": "POST",
        "hangup_url": hangup_url,
        "hangup_method": "POST",
    }

    print()
    print("=" * 72)
    print("VOBIZ OUTBOUND CALL")
    print("=" * 72)
    print(f"API:          {url}")
    print(f"FROM:         {from_}")
    print(f"TO:           {to}")
    print(f"ANSWER URL:   {answer_url}")
    print(f"HANGUP URL:   {hangup_url}")
    print(f"AGENT:        {agent or 'default'}")
    print("=" * 72)
    print()

    try:
        response = httpx.post(
            url,
            json=payload,
            headers=headers,
            timeout=20.0,
        )

    except httpx.RequestError as exc:
        print("VOBIZ HTTP REQUEST FAILED")
        print(str(exc))
        raise SystemExit(1)

    print(f"HTTP STATUS: {response.status_code}")
    print()

    print("RAW VOBIZ RESPONSE:")
    print(response.text)
    print()

    try:
        data = response.json()
    except ValueError:
        data = None

    if isinstance(data, dict):
        print("PARSED RESPONSE:")
        print(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            )
        )
        print()

    if response.is_error:
        print("VOBIZ API REQUEST FAILED")
        response.raise_for_status()

    print("CALL REQUEST ACCEPTED BY VOBIZ")
    print()

    if isinstance(data, dict):
        request_uuid = data.get("request_uuid")
        api_id = data.get("api_id")

        if request_uuid:
            print(f"REQUEST UUID: {request_uuid}")

        if api_id:
            print(f"API ID:       {api_id}")

        print()

    print(
        "The call is queued. A later CALL_REJECTED event means "
        "the actual outbound call leg was rejected after creation."
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Place an outbound Vobiz call."
    )

    parser.add_argument(
        "--to",
        required=True,
        help=(
            "Destination Indian mobile number. "
            "Examples: 9845102114 or +919845102114"
        ),
    )

    parser.add_argument(
        "--from",
        dest="from_",
        default=FROM_NUMBER,
        help="Your Vobiz DID.",
    )

    parser.add_argument(
        "--agent",
        default=None,
        help="Agent ID, e.g. priya.",
    )

    args = parser.parse_args()

    if not args.from_:
        sys.exit(
            "No FROM number provided. "
            "Use --from or set FROM_NUMBER in .env."
        )

    make_call(
        to=args.to,
        from_=args.from_,
        agent=args.agent,
    )


if __name__ == "__main__":
    main()