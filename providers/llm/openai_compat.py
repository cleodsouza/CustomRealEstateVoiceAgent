"""
providers/llm/openai_compat.py

OpenAI-compatible streaming LLM adapter.

Supports:
- Gemini OpenAI-compatible endpoint
- OpenAI
- Ollama
- vLLM
- Other /chat/completions-compatible providers

Realtime V2 behavior:
- Streams tokens immediately.
- Keeps temperature/max_tokens simple.
- Preserves reasoning_effort when explicitly configured.
- Removes Gemini thinking_config so it cannot conflict with
  reasoning_effort.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from runtime.types import LLMDelta, ToolCallRequest

log = logging.getLogger("providers.llm.openai_compat")


class OpenAICompatLLM:
    """OpenAI-compatible streaming LLM adapter."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int = 160,
        extra_body: dict | None = None,
        client: Any | None = None,
    ) -> None:
        self._client = (
            client
            if client is not None
            else AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
            )
        )

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Keep provider-specific options compatible with config.py,
        # but sanitize Gemini thinking_config so it cannot conflict
        # with reasoning_effort="none".
        self._extra_body = self._sanitize_extra_body(
            extra_body
        )

    @staticmethod
    def _sanitize_extra_body(
        extra_body: dict | None,
    ) -> dict | None:
        """
        Preserve reasoning_effort.

        For Gemini 2.5, the golden V2 configuration can use:

            {"reasoning_effort": "none"}

        to disable reasoning/thinking and reduce time-to-first-token.

        The old failure happened when both:

            reasoning_effort
            thinking_config

        were sent simultaneously.

        Therefore:
        - KEEP reasoning_effort
        - REMOVE thinking_config
        """

        if not extra_body:
            return None

        body = dict(extra_body)

        # --------------------------------------------------------------
        # Top-level Gemini/OpenAI reasoning settings
        # --------------------------------------------------------------

        # IMPORTANT:
        # Do NOT remove reasoning_effort.
        #
        # The intended golden V2 config is:
        #
        #     reasoning_effort = "none"
        #
        # which is useful for latency.
        body.pop("thinking_config", None)

        # --------------------------------------------------------------
        # Nested Google config
        # --------------------------------------------------------------

        google = body.get("google")

        if isinstance(google, dict):
            google = dict(google)

            google.pop(
                "thinking_config",
                None,
            )

            if google:
                body["google"] = google
            else:
                body.pop("google", None)

        # --------------------------------------------------------------
        # Older nested structure:
        #
        # {
        #   "extra_body": {
        #       "google": {
        #           "thinking_config": ...
        #       }
        #   }
        # }
        # --------------------------------------------------------------

        nested = body.get("extra_body")

        if isinstance(nested, dict):
            nested = dict(nested)

            # Keep reasoning_effort if somebody placed it here.
            nested.pop(
                "thinking_config",
                None,
            )

            nested_google = nested.get(
                "google"
            )

            if isinstance(
                nested_google,
                dict,
            ):
                nested_google = dict(
                    nested_google
                )

                nested_google.pop(
                    "thinking_config",
                    None,
                )

                if nested_google:
                    nested["google"] = (
                        nested_google
                    )
                else:
                    nested.pop(
                        "google",
                        None,
                    )

            if nested:
                body["extra_body"] = nested
            else:
                body.pop(
                    "extra_body",
                    None,
                )

        return body or None

    async def healthy(self) -> bool:
        """Lightweight provider health probe."""

        try:
            await self._client.models.list()
            return True

        except Exception as exc:
            log.warning(
                "LLM health check failed: %s",
                str(exc) or type(exc).__name__,
            )
            return False

    async def stream(
        self,
        messages: list[Any],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[LLMDelta]:
        """
        Stream text and tool-call fragments.

        Text is yielded immediately so the realtime TTS pipeline
        can begin speaking before the entire response is generated.
        """

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "stream": True,
            "max_tokens": self._max_tokens,
        }

        if tools:
            kwargs["tools"] = tools

        if self._extra_body:
            kwargs["extra_body"] = self._extra_body

        log.debug(
            "LLM request model=%s temperature=%s "
            "max_tokens=%s extra_body=%s",
            self._model,
            self._temperature,
            self._max_tokens,
            self._extra_body,
        )

        stream = await (
            self._client
            .chat
            .completions
            .create(**kwargs)
        )

        # Native tool calls are streamed as fragments.
        pending: dict[
            int,
            dict[str, str],
        ] = {}

        def flush(
            indices: list[int],
        ) -> list[LLMDelta]:
            output: list[LLMDelta] = []

            for index in indices:
                slot = pending.pop(
                    index,
                    None,
                )

                if slot is None:
                    continue

                request = _assemble_tool_call(
                    name=slot["name"],
                    args_json=slot["args"],
                    call_id=slot["id"],
                )

                if request is not None:
                    output.append(
                        LLMDelta(
                            tool_call=request
                        )
                    )

            return output

        async for part in stream:

            if not part.choices:
                continue

            delta = part.choices[0].delta

            # ----------------------------------------------------------
            # Native tool call fragments
            # ----------------------------------------------------------

            for tool_call in (
                getattr(
                    delta,
                    "tool_calls",
                    None,
                )
                or []
            ):
                current_index = (
                    tool_call.index
                )

                # Flush any other tool-call indices first.
                other_indices = [
                    idx
                    for idx in pending
                    if idx != current_index
                ]

                for event in flush(
                    other_indices
                ):
                    yield event

                slot = pending.setdefault(
                    current_index,
                    {
                        "name": "",
                        "args": "",
                        "id": "",
                    },
                )

                call_id = getattr(
                    tool_call,
                    "id",
                    None,
                )

                if call_id:
                    slot["id"] = call_id

                function = getattr(
                    tool_call,
                    "function",
                    None,
                )

                if function is not None:

                    name = getattr(
                        function,
                        "name",
                        None,
                    )

                    if name:
                        slot["name"] = name

                    arguments = getattr(
                        function,
                        "arguments",
                        None,
                    )

                    if arguments:
                        slot["args"] += (
                            arguments
                        )

            # ----------------------------------------------------------
            # Normal text
            # ----------------------------------------------------------

            if delta.content:

                # If text resumes after a tool call,
                # flush any pending tool fragments first.
                for event in flush(
                    list(pending)
                ):
                    yield event

                yield LLMDelta(
                    text=delta.content
                )

        # --------------------------------------------------------------
        # End-of-stream tool flush
        # --------------------------------------------------------------

        for event in flush(
            list(pending)
        ):
            yield event


def _assemble_tool_call(
    name: str,
    args_json: str,
    call_id: str = "",
) -> ToolCallRequest | None:
    """Turn streamed tool-call fragments into one request."""

    if not name:
        return None

    try:
        args = (
            json.loads(args_json)
            if args_json.strip()
            else {}
        )

    except (
        ValueError,
        TypeError,
    ) as exc:
        log.warning(
            "Malformed tool-call arguments for %s: %s",
            name,
            exc,
        )
        return None

    if not isinstance(
        args,
        dict,
    ):
        log.warning(
            "Tool-call arguments for %s "
            "are not an object: %r",
            name,
            args,
        )
        return None

    return ToolCallRequest(
        name=name,
        args=args,
        id=call_id,
    )