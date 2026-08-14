"""runtime/metrics.py — minimal metrics registry + the turn-metrics subscriber.

Purpose
    Measure the README's latency budget for real instead of asserting it:
    commit→first clause, commit→first audio, barge-in reaction. Exposed in
    Prometheus text format at /metrics.

Why hand-rolled instead of prometheus_client
    Five counters and three histograms. Rolling them keeps the registry an
    injectable instance (no process-global default registry), costs ~100
    lines, and drops a dependency. No label support — nothing here needs
    labels yet; if that changes, this registry is the single swap point
    for the real client library.

The registry is pure data — loop-free, I/O-free. TurnMetrics is a bus
subscriber projecting conversation events into it; it holds no cross-event
state because TurnCompleted carries its latencies precomputed.
"""
from __future__ import annotations

from runtime import events

DEFAULT_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
REACTION_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


class Counter:
    def __init__(self, name: str, help_text: str) -> None:
        self.name = name
        self.help_text = help_text
        self.value = 0.0

    def inc(self, n: float = 1.0) -> None:
        self.value += n

    def render(self) -> str:
        return (f"# HELP {self.name} {self.help_text}\n"
                f"# TYPE {self.name} counter\n"
                f"{self.name} {_fmt(self.value)}\n")


class Histogram:
    def __init__(self, name: str, help_text: str,
                 buckets: tuple[float, ...] = DEFAULT_BUCKETS) -> None:
        self.name = name
        self.help_text = help_text
        self.buckets = tuple(sorted(buckets))
        # Cumulative counts, Prometheus-style: counts[i] = observations <= buckets[i]
        self.counts = [0] * len(self.buckets)
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.sum += value
        self.count += 1
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help_text}",
                 f"# TYPE {self.name} histogram"]
        for bound, count in zip(self.buckets, self.counts):
            lines.append(f'{self.name}_bucket{{le="{_fmt(bound)}"}} {count}')
        lines.append(f'{self.name}_bucket{{le="+Inf"}} {self.count}')
        lines.append(f"{self.name}_sum {_fmt(self.sum)}")
        lines.append(f"{self.name}_count {self.count}")
        return "\n".join(lines) + "\n"


def _fmt(value: float) -> str:
    """Render 2.0 as '2' (Prometheus convention) but keep real fractions."""
    return str(int(value)) if value == int(value) else repr(value)


class MetricsRegistry:
    def __init__(self) -> None:
        self._metrics: list[Counter | Histogram] = []

    def counter(self, name: str, help_text: str) -> Counter:
        c = Counter(name, help_text)
        self._metrics.append(c)
        return c

    def histogram(self, name: str, help_text: str,
                  buckets: tuple[float, ...] = DEFAULT_BUCKETS) -> Histogram:
        h = Histogram(name, help_text, buckets)
        self._metrics.append(h)
        return h

    def render(self) -> str:
        """Prometheus text exposition format 0.0.4."""
        return "\n".join(m.render() for m in self._metrics)


# -------------------------------------------------------------- subscriber
class TurnMetrics:
    """Bus subscriber: conversation events → registry. Stateless across
    events by design — TurnCompleted carries its latencies precomputed."""

    def __init__(self, registry: MetricsRegistry) -> None:
        self.calls = registry.counter(
            "calls_total", "Calls started")
        self.turns = registry.counter(
            "turns_total", "User turns a reply pipeline ran for")
        self.interruptions = registry.counter(
            "interruptions_total",
            "Agent output cancelled (barge-in or superseding turn)")
        self.tool_calls = registry.counter(
            "tool_calls_total", "Tool executions started")
        self.tool_failures = registry.counter(
            "tool_failures_total", "Tool executions that raised")
        self.provider_failures = registry.counter(
            "provider_failures_total",
            "Providers lost beyond their retry budget (alarm-grade)")
        self.fallbacks = registry.counter(
            "fallbacks_total",
            "Turns that spoke the scripted fallback line instead of a reply")
        self.thinking = registry.histogram(
            "turn_thinking_seconds",
            "Turn commit to first speakable clause (LLM TTFT + clause accumulation)")
        self.first_token = registry.histogram(
            "turn_first_token_seconds",
            "Turn commit to first LLM delta on the token stream "
            "(raw TTFT, before clause accumulation)")
        self.first_audio = registry.histogram(
            "turn_first_audio_seconds",
            "Turn commit to first audio frame handed to the transport (filler included)")
        self.bargein_reaction = registry.histogram(
            "bargein_reaction_seconds",
            "CancelOutput intent to pipeline unwound and carrier buffer cleared",
            buckets=REACTION_BUCKETS)

    async def __call__(self, event: events.Event) -> None:
        if isinstance(event, events.CallStarted):
            self.calls.inc()
        elif isinstance(event, events.TurnCompleted):
            self.turns.inc()
            if event.thinking_s is not None:
                self.thinking.observe(event.thinking_s)
            if event.first_audio_s is not None:
                self.first_audio.observe(event.first_audio_s)
        elif isinstance(event, events.AgentInterrupted):
            self.interruptions.inc()
            self.bargein_reaction.observe(event.reaction_s)
        elif isinstance(event, events.ToolCalled):
            self.tool_calls.inc()
        elif isinstance(event, events.ToolFailed):
            self.tool_failures.inc()
        elif isinstance(event, events.ProviderFailed):
            self.provider_failures.inc()
        elif isinstance(event, events.FallbackSpoken):
            self.fallbacks.inc()
        elif isinstance(event, events.TokenStreamEnded):
            if event.first_token_s is not None:
                self.first_token.observe(event.first_token_s)
