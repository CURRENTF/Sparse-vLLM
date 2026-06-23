#!/usr/bin/env python3
"""Minimal OpenAI-compatible chat server for Claw-Eval.

This adapter intentionally supports only text chat completions. Unsupported
request features fail with HTTP 400 instead of being silently ignored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from sparsevllm import LLM, SamplingParams


def _load_json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("--engine-kwargs must resolve to a JSON object")
    return data


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for idx, part in enumerate(content):
            if not isinstance(part, dict):
                raise ValueError(f"message content part {idx} must be an object")
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text", "")
                if not isinstance(text, str):
                    raise ValueError(f"message content part {idx}.text must be a string")
                parts.append(text)
                continue
            raise ValueError(
                "Sparse-vLLM OpenAI shim only supports text content parts; "
                f"got part type {part_type!r}"
            )
        return "\n".join(parts)
    raise ValueError(f"message content must be a string or text-part list, got {type(content).__name__}")


def _normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"messages[{idx}] must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"messages[{idx}].role must be system/user/assistant/tool, got {role!r}")
        normalized.append({"role": role, "content": _message_content_to_text(message.get("content", ""))})
    return normalized


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _render_prompt(tokenizer: Any, messages: list[dict[str, str]], no_chat_template: bool, enable_thinking: bool) -> str:
    if not no_chat_template and getattr(tokenizer, "chat_template", None):
        return _apply_chat_template(tokenizer, messages, enable_thinking=enable_thinking)

    rendered = []
    for message in messages:
        rendered.append(f"{message['role']}:\n{message['content']}")
    rendered.append("assistant:\n")
    return "\n\n".join(rendered)


def _truncate_at_stop(text: str, stop: Any) -> tuple[str, str | None]:
    if stop is None:
        return text, None
    stops = [stop] if isinstance(stop, str) else stop
    if not isinstance(stops, list) or not all(isinstance(s, str) for s in stops):
        raise ValueError("stop must be a string or list of strings")
    best_idx = None
    best_stop = None
    for item in stops:
        if item == "":
            continue
        idx = text.find(item)
        if idx >= 0 and (best_idx is None or idx < best_idx):
            best_idx = idx
            best_stop = item
    if best_idx is None:
        return text, None
    return text[:best_idx], best_stop


class SparseVLLMOpenAIServer:
    def __init__(
        self,
        *,
        model_path: str,
        served_model_name: str,
        engine_kwargs: dict[str, Any],
        no_chat_template: bool,
        enable_thinking: bool,
        request_log_dir: str | None,
    ) -> None:
        self.served_model_name = served_model_name
        self.no_chat_template = no_chat_template
        self.enable_thinking = enable_thinking
        self.request_log_dir = Path(request_log_dir) if request_log_dir else None
        if self.request_log_dir:
            self.request_log_dir.mkdir(parents=True, exist_ok=True)

        self.llm = LLM(model_path, **engine_kwargs)
        self.tokenizer = self.llm.tokenizer
        self.lock = threading.Lock()

    def write_log(self, payload: dict[str, Any]) -> None:
        if not self.request_log_dir:
            return
        path = self.request_log_dir / f"{int(time.time() * 1000)}_{uuid.uuid4().hex}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def chat_completion(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("stream", False):
            raise ValueError("stream=true is not supported by Sparse-vLLM OpenAI shim")
        if int(request.get("n", 1)) != 1:
            raise ValueError("Only n=1 is supported")
        if request.get("tools") and request.get("tool_choice") not in (None, "none"):
            raise ValueError("OpenAI tool-call protocol is not supported; use text-only agent prompts")
        if request.get("functions") or request.get("function_call"):
            raise ValueError("OpenAI function-call protocol is not supported")
        if request.get("modalities") not in (None, ["text"]):
            raise ValueError("Only text modality is supported")

        messages = _normalize_messages(request.get("messages"))
        prompt = _render_prompt(
            self.tokenizer,
            messages,
            no_chat_template=self.no_chat_template,
            enable_thinking=self.enable_thinking,
        )
        max_tokens = int(request.get("max_tokens") or request.get("max_completion_tokens") or 1024)
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        temperature = float(request.get("temperature", 1.0))
        if temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {temperature}")
        top_p = float(request.get("top_p", 1.0))
        if not (0.0 < top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        with self.lock:
            outputs = self.llm.generate([prompt], sampling_params, use_tqdm=False)

        text = outputs[0]["text"]
        token_ids = outputs[0].get("token_ids", [])
        text, matched_stop = _truncate_at_stop(text, request.get("stop"))
        completion_tokens = len(token_ids) if isinstance(token_ids, list) else len(self.tokenizer.encode(text))
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        finish_reason = "stop" if matched_stop is not None or completion_tokens < max_tokens else "length"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }


def _make_handler(server_state: SparseVLLMOpenAIServer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SparseVLLMOpenAI/0.1"

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            body = self.rfile.read(length)
            try:
                payload = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON request body: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError("JSON request body must be an object")
            return payload

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._send_json(200, {"status": "ok", "model": server_state.served_model_name})
                return
            if self.path.rstrip("/") == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": server_state.served_model_name,
                                "object": "model",
                                "created": 0,
                                "owned_by": "sparsevllm",
                            }
                        ],
                    },
                )
                return
            self._send_json(404, {"error": {"message": f"Unknown endpoint: {self.path}"}})

        def do_POST(self) -> None:  # noqa: N802
            started = time.time()
            try:
                request = self._read_json_body()
                if self.path.rstrip("/") != "/v1/chat/completions":
                    raise FileNotFoundError(f"Unknown endpoint: {self.path}")
                response = server_state.chat_completion(request)
                server_state.write_log(
                    {
                        "status": "success",
                        "path": self.path,
                        "elapsed_s": round(time.time() - started, 6),
                        "request": request,
                        "response": response,
                    }
                )
                self._send_json(200, response)
            except FileNotFoundError as exc:
                self._send_json(404, {"error": {"message": str(exc)}})
            except Exception as exc:
                payload = {
                    "status": "model_failed",
                    "path": self.path,
                    "elapsed_s": round(time.time() - started, 6),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                try:
                    payload["request"] = request  # type: ignore[name-defined]
                except Exception:
                    pass
                server_state.write_log(payload)
                self._send_json(400, {"error": {"message": str(exc), "type": type(exc).__name__}})

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="sparsevllm-claw")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--engine-kwargs", default=None, help="JSON object or path to JSON file passed to sparsevllm.LLM")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--request-log-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine_kwargs = _load_json_arg(args.engine_kwargs)
    print(
        json.dumps(
            {
                "event": "starting_sparsevllm_openai_server",
                "model_path": args.model_path,
                "served_model_name": args.served_model_name,
                "host": args.host,
                "port": args.port,
                "engine_kwargs": engine_kwargs,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "request_log_dir": args.request_log_dir,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    state = SparseVLLMOpenAIServer(
        model_path=args.model_path,
        served_model_name=args.served_model_name,
        engine_kwargs=engine_kwargs,
        no_chat_template=args.no_chat_template,
        enable_thinking=not args.disable_thinking,
        request_log_dir=args.request_log_dir,
    )
    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(state))
    print(f"Serving OpenAI-compatible API on http://{args.host}:{args.port}/v1", flush=True)
    try:
        httpd.serve_forever()
    finally:
        state.llm.exit()


if __name__ == "__main__":
    main()
