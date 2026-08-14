"""Metrics registry + TurnMetrics subscriber tests: counting, histogram
bucketing, and Prometheus text exposition shape."""
from runtime import events
from runtime.metrics import Histogram, MetricsRegistry, TurnMetrics


def test_counter_renders_prometheus_text():
    reg = MetricsRegistry()
    c = reg.counter("calls_total", "Calls started")
    c.inc()
    c.inc(2)
    out = reg.render()
    assert "# HELP calls_total Calls started" in out
    assert "# TYPE calls_total counter" in out
    assert "\ncalls_total 3\n" in out


def test_histogram_buckets_are_cumulative():
    h = Histogram("t_seconds", "t", buckets=(0.5, 1.0, 2.0))
    h.observe(0.3)
    h.observe(1.0)   # boundary lands in its own bucket (le is inclusive)
    h.observe(9.9)   # beyond all bounds: only +Inf

    assert h.counts == [1, 2, 2]
    assert h.count == 3
    assert abs(h.sum - 11.2) < 1e-9

    out = h.render()
    assert 't_seconds_bucket{le="0.5"} 1' in out
    assert 't_seconds_bucket{le="1"} 2' in out
    assert 't_seconds_bucket{le="2"} 2' in out
    assert 't_seconds_bucket{le="+Inf"} 3' in out
    assert "t_seconds_count 3" in out


async def test_turn_metrics_projects_events():
    reg = MetricsRegistry()
    tm = TurnMetrics(reg)

    await tm(events.CallStarted(call_id="c1", caller="+91", agent_id="priya"))
    await tm(events.TurnCompleted(
        call_id="c1", turn_seq=1, user_text="u", agent_text="a",
        thinking_s=0.4, first_audio_s=0.7, interrupted=False))
    await tm(events.TurnCompleted(  # cut short before any latency existed
        call_id="c1", turn_seq=2, user_text="u", agent_text="",
        thinking_s=None, first_audio_s=None, interrupted=True))
    await tm(events.AgentInterrupted(call_id="c1", turn_seq=2, reaction_s=0.05))
    await tm(events.ToolCalled(call_id="c1", tool="save_booking"))
    await tm(events.ToolFailed(call_id="c1", tool="save_booking", error="x"))

    assert tm.calls.value == 1
    assert tm.turns.value == 2
    assert tm.interruptions.value == 1
    assert tm.tool_calls.value == 1
    assert tm.tool_failures.value == 1
    assert tm.thinking.count == 1          # None latencies never observed
    assert tm.first_audio.count == 1
    assert tm.bargein_reaction.count == 1


async def test_token_stream_ended_feeds_first_token_histogram():
    """S3: raw TTFT lands in turn_first_token_seconds; a turn with no
    tokens (tool-only or failed stream) observes nothing."""
    from runtime.metrics import MetricsRegistry, TurnMetrics

    tm = TurnMetrics(MetricsRegistry())
    await tm(events.TokenStreamEnded(call_id="c1", turn_seq=1,
                                     tokens=7, first_token_s=0.12))
    assert tm.first_token.count == 1 and tm.first_token.sum == 0.12
    await tm(events.TokenStreamEnded(call_id="c1", turn_seq=2,
                                     tokens=0, first_token_s=None))
    assert tm.first_token.count == 1
