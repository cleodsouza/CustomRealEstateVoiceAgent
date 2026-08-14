"""providers/llm/openai_compat.py — adapter for any OpenAI-compatible endpoint.

Ollama, vLLM, OpenAI, Gemini's compat endpoint, Sarvam — anything speaking
/chat/completions. Which one is a config decision at the composition root,
never a code change.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from runtime.types import LLMDelta, ToolCallRequest

log = logging.getLogger("providers.llm.openai_compat")


class OpenAICompatLLM:
    """Implements runtime.interfaces.LLM.

    Errors are allowed to propagate: runtime.clauses.stream_clauses owns the
    speak-what-we-have fallback, so behavior on stream failure is unchanged
    from the old module-global client.
    """

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
        # `client` is injectable for tests; typed Any so fakes qualify.
        self._client = client if client is not None else AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # Deployment-tuned request extras (S7), e.g. {"reasoning_effort":
        # "none"} to switch Gemini 2.5's thinking off. Self-healing: a 4xx
        # with extras present drops them for the adapter's lifetime and
        # retries once — a capability probe, not a hard dependency.
        self._extra_body = dict(extra_body) if extra_body else None

    async def healthy(self) -> bool:
        """SupportsHealth probe: GET /models — validates reachability and,
        where the endpoint enforces it, the API key."""
        try:
            await self._client.models.list()
            return True
        except Exception:  # noqa: BLE001
            return False

    async def stream(self, messages: list[Any],
                     tools: list[dict] | None = None) -> AsyncIterator[LLMDelta]:
        kwargs: dict[str, Any] = dict(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            stream=True,
            max_tokens=self._max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
        if self._extra_body:
            kwargs["extra_body"] = self._extra_body
        try:
            stream = await self._client.chat.completions.create(**kwargs)
        except Exception as e:  # noqa: BLE001
            status = getattr(e, "status_code", None)
            if self._extra_body and status is not None and 400 <= status < 500:
                log.warning("LLM rejected extra body %r (%s); disabling it",
                            self._extra_body, status)
                self._extra_body = None
                kwargs.pop("extra_body", None)
                stream = await self._client.chat.completions.create(**kwargs)
            else:
                raise

        # Native tool calls arrive as fragments (index-correlated name +
        # argument-JSON pieces). Accumulate per index and yield each call as
        # ONE assembled delta — the runtime never sees fragments. Assembled
        # calls are yielded EAGERLY, the moment a slot is provably complete
        # (a fragment for a different index starts, or text resumes), not
        # held to stream end: the pipeline's halt/resume tool loop needs the
        # call as soon as it exists. Slots still open at stream end flush
        # then, preserving the old behavior for the common single-call case.
        pending: dict[int, dict[str, str]] = {}

        def _flush(indices: list[int]) -> list[LLMDelta]:
            out = []
            for i in indices:
                slot = pending.pop(i)
                request = _assemble_tool_call(slot["name"], slot["args"],
                                              slot["id"])
                if request is not None:
                    out.append(LLMDelta(tool_call=request))
            return out

        async for part in stream:
            if not part.choices:
                continue
            delta = part.choices[0].delta
            for tc in getattr(delta, "tool_calls", None) or []:
                # A fragment for a new index closes every other open slot.
                for d in _flush([i for i in pending if i != tc.index]):
                    yield d
                slot = pending.setdefault(tc.index,
                                          {"name": "", "args": "", "id": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments
            if delta.content:
                # Text resuming means any open call finished (models that
                # interleave speech around calls).
                for d in _flush(list(pending)):
                    yield d
                yield LLMDelta(text=delta.content)
        for d in _flush(list(pending)):
            yield d


def _assemble_tool_call(name: str, args_json: str,
                        call_id: str = "") -> ToolCallRequest | None:
    """Parse an accumulated tool call; malformed output from small models
    is dropped with a log — the marker strategy exists for exactly them."""
    if not name:
        return None
    try:
        args = json.loads(args_json) if args_json.strip() else {}
    except ValueError:
        log.warning("Malformed tool-call arguments for %s: %r", name, args_json)
        return None
    if not isinstance(args, dict):
        log.warning("Tool-call arguments for %s not an object: %r", name, args)
        return None
    return ToolCallRequest(name=name, args=args, id=call_id)
