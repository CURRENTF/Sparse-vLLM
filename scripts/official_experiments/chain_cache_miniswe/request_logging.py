"""Record upstream vLLM non-streaming HTTP requests for the shared collector.

This ASGI middleware forwards messages unchanged; it adds no CUDA synchronization.
Elapsed time is the server HTTP boundary, including queueing and response delivery.
"""
import json
import os
from pathlib import Path
import time
import uuid


class RequestLoggingMiddleware:
    def __init__(self, app):
        self.app = app
        self.directory = Path(os.environ["MINISWE_REQUEST_LOG_DIR"])
        self.directory.mkdir(parents=True, exist_ok=True)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/v1/chat/completions":
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        request_body, response_body = bytearray(), bytearray()
        status = None
        error = None

        async def logged_receive():
            message = await receive()
            if message["type"] == "http.request":
                request_body.extend(message.get("body", b""))
            return message

        async def logged_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body":
                response_body.extend(message.get("body", b""))
            await send(message)

        try:
            await self.app(scope, logged_receive, logged_send)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed = time.perf_counter() - started
            record = {"status": "success" if status == 200 and error is None else "failed",
                      "http_status": status, "elapsed_s": elapsed, "error": error,
                      "timing_boundary": "server_http_asgi", "backend": "vllm"}
            for name, body in (("request", request_body), ("response", response_body)):
                try:
                    record[name] = json.loads(body)
                except (ValueError, UnicodeDecodeError):
                    record[name + "_raw"] = body.decode("utf-8", errors="replace")
                    record["status"] = "failed"
            record["request_id"] = record.get("response", {}).get("id") or uuid.uuid4().hex
            path = self.directory / f"{time.time_ns()}_{uuid.uuid4().hex}.json"
            with path.open("x") as stream:
                json.dump(record, stream, ensure_ascii=False)
                stream.write("\n")
