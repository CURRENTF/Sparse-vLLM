"""Opt-in HTTP recording in the synchronous MiniSWE instance worker.

The context variable binds HTTP calls (including retries) to an instance, without
guessing identity from prompt prefixes. No credentials or HTTP headers are saved.
"""
from contextvars import ContextVar
import functools
import json
from pathlib import Path
import time


SCHEMA = "miniswe_http_trace_v1"
CURRENT_TRACE = ContextVar("miniswe_agent_trace", default=None)


class AgentTrace:
    def __init__(self, path: Path, instance_id: str):
        self.stream = path.open("x", encoding="utf-8")
        self.instance_id = instance_id
        self.index = 0
        self.previous_end = None
        self.write({"event": "instance_start", "unix_s": time.time()})

    def write(self, row):
        self.stream.write(json.dumps({"schema": SCHEMA, "instance_id": self.instance_id,
                                      **row}, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()

    def finish(self, error):
        try:
            self.write({"event": "instance_end", "uncaught_exception": error,
                        "request_count": self.index, "unix_s": time.time()})
        finally:
            self.stream.close()

    def send(self, original, client, request, **kwargs):
        body = json.loads(request.content)
        if body.get("stream") or kwargs.get("stream"):
            raise ValueError("Agent trace capture supports non-streaming HTTP only")
        # Persist the input first: a killed worker leaves an explicit unfinished turn.
        index = self.index
        self.index += 1
        self.write({"event": "request", "turn": index, "request": body})
        started = time.perf_counter()
        timing = {"started_monotonic_s": started,
                  "think_time_s": None if self.previous_end is None else started - self.previous_end}
        try:
            response = original(client, request, **kwargs)
            response.read()
        except BaseException as exc:
            ended = time.perf_counter()
            self.previous_end = ended
            self.write({"event": "response", "turn": index, "status": "model_failed",
                        "error": type(exc).__name__, "ended_monotonic_s": ended, **timing})
            raise
        ended = time.perf_counter()
        self.previous_end = ended
        row = {"event": "response", "turn": index,
               "status": "success" if response.status_code == 200 else "model_failed",
               "http_status": response.status_code, "ended_monotonic_s": ended, **timing}
        try:
            row["response"] = response.json()
        except ValueError:
            row.update(status="parse_failed", response_raw=response.text)
        self.write(row)
        return response


def install_http_recorder():
    """Install once before MiniSWE starts its thread pool; unbound calls pass through."""
    import httpx

    if getattr(httpx.Client.send, "_miniswe_trace", False):
        return
    original = httpx.Client.send

    @functools.wraps(original)
    def send(client, request, **kwargs):
        trace = CURRENT_TRACE.get()
        if trace is None or not request.url.path.rstrip("/").endswith("/chat/completions"):
            return original(client, request, **kwargs)
        return trace.send(original, client, request, **kwargs)

    send._miniswe_trace = True
    httpx.Client.send = send
