"""Per-request chat token accounting; no payload text is logged."""

from contextlib import contextmanager
from contextvars import ContextVar

usage_context = ContextVar("chat_usage", default=None)


@contextmanager
def collect_usage():
    counters = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "model_calls": 0}
    token = usage_context.set(counters)
    try:
        yield counters
    finally:
        usage_context.reset(token)


def record_usage(message):
    counters = usage_context.get()
    if counters is not None:
        counters["model_calls"] += 1
        for key in ["input_tokens", "output_tokens", "total_tokens"]:
            counters[key] += (message.usage_metadata or {}).get(key, 0)
