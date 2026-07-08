import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from fastapi.responses import StreamingResponse

from sparsevllm.utils.log import logger


ROUTE_FIELDS = {
    "svllm_route_profile",
    "svllm_method_preference",
    "svllm_required_tags",
    "svllm_preferred_tags",
    "svllm_target_worker",
}


@dataclass
class WorkerState:
    url: str
    info: dict[str, Any]
    local_inflight: int = 0
    healthy: bool = True
    last_error: str | None = None


@dataclass(frozen=True)
class WorkerProbe:
    worker: WorkerState
    load: dict[str, Any]
    match: dict[str, Any]

    @property
    def load_value(self) -> int:
        return int(self.load.get("active_requests", 0) or 0) + int(self.worker.local_inflight)

    @property
    def match_ratio(self) -> float:
        if not (self.match.get("supported") and self.match.get("enabled")):
            return 0.0
        return float(self.match.get("match_ratio", 0.0) or 0.0)

    @property
    def matched_tokens(self) -> int:
        if not (self.match.get("supported") and self.match.get("enabled")):
            return 0
        return int(self.match.get("matched_tokens", 0) or 0)


@dataclass(frozen=True)
class UpstreamStream:
    response: Any
    headers: dict[str, str]


@dataclass(frozen=True)
class UpstreamError:
    status: int
    headers: dict[str, str]
    body: bytes


def create_app(
    worker_urls: list[str],
    *,
    request_timeout_s: float = 30.0,
    overload_load_factor: float = 1.5,
    load_abs_threshold: int = 1,
    profiles: dict[str, Any] | None = None,
    route_log_dir: str | None = None,
) -> FastAPI:
    router = SmartRouter(
        worker_urls=worker_urls,
        request_timeout_s=request_timeout_s,
        overload_load_factor=overload_load_factor,
        load_abs_threshold=load_abs_threshold,
        profiles=profiles or {},
        route_log_dir=Path(route_log_dir) if route_log_dir else None,
    )
    app = FastAPI(title="Sparse-vLLM OpenAI smart router")
    app.state.router = router

    @app.on_event("startup")
    async def _startup():
        await router.refresh_worker_info()

    @app.get("/health")
    async def health():
        await router.refresh_worker_info()
        healthy = [worker.url for worker in router.workers if worker.healthy]
        return {"status": "ok" if healthy else "unavailable", "healthy_workers": healthy}

    @app.get("/v1/models")
    async def models():
        await router.refresh_worker_info()
        created = int(time.time())
        model_ids = sorted(
            {
                str(worker.info.get("served_model_name"))
                for worker in router.workers
                if worker.healthy and worker.info.get("served_model_name")
            }
        )
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": created,
                    "owned_by": "sparsevllm-router",
                }
                for model_id in model_ids
            ],
        }

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await router.route_openai_request("/v1/completions", await request.json())

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await router.route_openai_request("/v1/chat/completions", await request.json())

    @app.post("/v1/prefix_cache/inspect")
    async def prefix_cache_inspect(request: Request):
        payload = await request.json()
        all_workers = bool(payload.pop("all", False))
        if all_workers:
            return JSONResponse(await router.broadcast_json("/v1/prefix_cache/inspect", payload))
        worker, payload, route = await router.select_worker("/v1/prefix_cache/inspect", payload)
        response = await router.forward_json(worker, "/v1/prefix_cache/inspect", payload)
        return _with_route_headers(response, route)

    @app.post("/v1/prefix_cache/delete_subtree")
    async def prefix_cache_delete_subtree(request: Request):
        return JSONResponse(await router.broadcast_json("/v1/prefix_cache/delete_subtree", await request.json()))

    @app.post("/v1/prefix_cache/set_eviction_priority")
    async def prefix_cache_set_eviction_priority(request: Request):
        return JSONResponse(await router.broadcast_json("/v1/prefix_cache/set_eviction_priority", await request.json()))

    return app


class SmartRouter:
    def __init__(
        self,
        *,
        worker_urls: list[str],
        request_timeout_s: float,
        overload_load_factor: float,
        load_abs_threshold: int,
        profiles: dict[str, Any],
        route_log_dir: Path | None,
    ):
        if not worker_urls:
            raise ValueError("At least one worker URL is required.")
        self.workers = [
            WorkerState(url=url.rstrip("/"), info={})
            for url in worker_urls
        ]
        self.request_timeout_s = float(request_timeout_s)
        self.overload_load_factor = float(overload_load_factor)
        self.load_abs_threshold = int(load_abs_threshold)
        self.profiles = profiles
        self.route_log_dir = route_log_dir
        if self.route_log_dir is not None:
            self.route_log_dir.mkdir(parents=True, exist_ok=True)

    async def refresh_worker_info(self):
        results = await asyncio.gather(
            *[asyncio.to_thread(_get_json, f"{worker.url}/v1/worker/info", self.request_timeout_s) for worker in self.workers],
            return_exceptions=True,
        )
        for worker, result in zip(self.workers, results):
            if isinstance(result, Exception):
                worker.healthy = False
                worker.last_error = f"{type(result).__name__}: {result}"
                continue
            worker.info = dict(result)
            worker.healthy = True
            worker.last_error = None

    async def route_openai_request(self, endpoint: str, payload: dict[str, Any]) -> Response:
        worker, forward_payload, route = await self.select_worker(endpoint, payload)
        stream = bool(forward_payload.get("stream", False))
        if stream:
            return await self.forward_stream(worker, endpoint, forward_payload, route)
        response = await self.forward_json(worker, endpoint, forward_payload)
        return _with_route_headers(response, route)

    async def select_worker(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> tuple[WorkerState, dict[str, Any], dict[str, Any]]:
        await self.refresh_worker_info()
        forward_payload, route_hints = strip_route_hints(payload)
        candidates = self._candidate_workers(endpoint, forward_payload, route_hints)
        if not candidates:
            raise HTTPException(status_code=503, detail="No healthy Sparse-vLLM worker matches this request.")
        if route_hints.get("target_worker"):
            target = str(route_hints["target_worker"])
            for worker in candidates:
                if worker.url == target or worker.url.endswith(target) or str(self.workers.index(worker)) == target:
                    return worker, forward_payload, self._route_record(endpoint, worker, "target_worker", [], route_hints)
            raise HTTPException(status_code=404, detail=f"Requested target worker {target!r} is not available.")

        match_payload = match_payload_for_request(endpoint, forward_payload)
        probes = await self._probe_workers(candidates, match_payload)
        if not probes:
            raise HTTPException(status_code=503, detail="No healthy Sparse-vLLM worker responded to route probes.")
        worker, reason = choose_worker(
            probes,
            overload_load_factor=self.overload_load_factor,
            load_abs_threshold=self.load_abs_threshold,
        )
        route = self._route_record(endpoint, worker, reason, probes, route_hints)
        self._write_route_log(route)
        return worker, forward_payload, route

    def _candidate_workers(
        self,
        endpoint: str,
        payload: dict[str, Any],
        route_hints: dict[str, Any],
    ) -> list[WorkerState]:
        profile_name = str(route_hints.get("route_profile") or infer_route_profile(endpoint, payload))
        profile = dict(self.profiles.get(profile_name, {}) or {})
        required_tags = set(_string_list(profile.get("required_tags")) + _string_list(route_hints.get("required_tags")))
        preferred_tags = set(_string_list(profile.get("preferred_tags")) + _string_list(route_hints.get("preferred_tags")))
        method_preference = _string_list(route_hints.get("method_preference")) or _string_list(profile.get("methods"))
        model = payload.get("model")
        candidates = []
        for worker in self.workers:
            if not worker.healthy:
                continue
            info = worker.info
            worker_model = info.get("served_model_name")
            if model is not None and worker_model is not None and str(worker_model) != str(model):
                continue
            tags = set(_string_list(info.get("tags")))
            if required_tags and not required_tags.issubset(tags):
                continue
            if method_preference:
                method = str(info.get("sparse_method", "") or "")
                if not _method_matches(method, method_preference):
                    continue
            candidates.append(worker)
        if preferred_tags:
            preferred = [
                worker
                for worker in candidates
                if preferred_tags.intersection(set(_string_list(worker.info.get("tags"))))
            ]
            if preferred:
                return preferred
        return candidates

    async def _probe_workers(
        self,
        candidates: list[WorkerState],
        match_payload: dict[str, Any] | None,
    ) -> list[WorkerProbe]:
        tasks = [
            self._probe_worker(worker, match_payload)
            for worker in candidates
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        probes = []
        for worker, result in zip(candidates, results):
            if isinstance(result, Exception):
                worker.healthy = False
                worker.last_error = f"{type(result).__name__}: {result}"
                continue
            probes.append(result)
        return probes

    async def _probe_worker(
        self,
        worker: WorkerState,
        match_payload: dict[str, Any] | None,
    ) -> WorkerProbe:
        load_task = asyncio.to_thread(_get_json, f"{worker.url}/v1/worker/load", self.request_timeout_s)
        if match_payload is None:
            load = await load_task
            match = {"supported": False, "enabled": False, "matched_tokens": 0, "match_ratio": 0.0}
        else:
            load, match = await asyncio.gather(
                load_task,
                asyncio.to_thread(_post_json, f"{worker.url}/v1/prefix_cache/match", match_payload, self.request_timeout_s),
            )
        return WorkerProbe(worker=worker, load=dict(load), match=dict(match))

    async def forward_json(self, worker: WorkerState, endpoint: str, payload: dict[str, Any]) -> Response:
        worker.local_inflight += 1
        try:
            status, headers, body = await asyncio.to_thread(
                _request_bytes,
                f"{worker.url}{endpoint}",
                "POST",
                payload,
                self.request_timeout_s,
            )
        finally:
            worker.local_inflight -= 1
        return Response(content=body, status_code=status, headers=_content_headers(headers))

    async def forward_stream(
        self,
        worker: WorkerState,
        endpoint: str,
        payload: dict[str, Any],
        route: dict[str, Any],
    ) -> Response:
        worker.local_inflight += 1
        stream_handoff = False
        opened_response = None
        try:
            upstream = await asyncio.to_thread(
                _open_stream_response,
                f"{worker.url}{endpoint}",
                payload,
                self.request_timeout_s,
            )
            if isinstance(upstream, UpstreamError):
                return Response(
                    content=upstream.body,
                    status_code=upstream.status,
                    headers={**_content_headers(upstream.headers), **route_headers(route)},
                )
            opened_response = upstream.response
            response_headers = {**_content_headers(upstream.headers), **route_headers(route)}

            async def _stream():
                try:
                    async for chunk in _stream_response_chunks(opened_response):
                        yield chunk
                finally:
                    try:
                        opened_response.close()
                    finally:
                        worker.local_inflight -= 1

            response = StreamingResponse(
                _stream(),
                media_type="text/event-stream",
                headers=response_headers,
            )
            stream_handoff = True
            return response
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            if not stream_handoff:
                if opened_response is not None:
                    try:
                        opened_response.close()
                    finally:
                        worker.local_inflight -= 1
                else:
                    worker.local_inflight -= 1

    async def broadcast_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        await self.refresh_worker_info()
        workers = [worker for worker in self.workers if worker.healthy]
        if not workers:
            raise HTTPException(status_code=503, detail="No healthy Sparse-vLLM worker is available.")
        results = await asyncio.gather(
            *[self._broadcast_one(worker, endpoint, payload) for worker in workers],
            return_exceptions=True,
        )
        items = []
        for worker, result in zip(workers, results):
            if isinstance(result, Exception):
                items.append({"worker_url": worker.url, "ok": False, "error": f"{type(result).__name__}: {result}"})
            else:
                items.append({"worker_url": worker.url, "ok": True, "result": result})
        return {"workers": items}

    async def _broadcast_one(self, worker: WorkerState, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        status, _headers, body = await asyncio.to_thread(
            _request_bytes,
            f"{worker.url}{endpoint}",
            "POST",
            payload,
            self.request_timeout_s,
        )
        decoded = json.loads(body.decode("utf-8")) if body else {}
        if status >= 400:
            raise RuntimeError(json.dumps(decoded, ensure_ascii=False))
        return decoded

    def _route_record(
        self,
        endpoint: str,
        worker: WorkerState,
        reason: str,
        probes: list[WorkerProbe],
        route_hints: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "id": f"route-{uuid.uuid4().hex}",
            "endpoint": endpoint,
            "selected_worker_url": worker.url,
            "selected_sparse_method": worker.info.get("sparse_method", ""),
            "reason": reason,
            "hints": route_hints,
            "probes": [
                {
                    "worker_url": probe.worker.url,
                    "sparse_method": probe.worker.info.get("sparse_method", ""),
                    "load": probe.load_value,
                    "match_ratio": probe.match_ratio,
                    "matched_tokens": probe.matched_tokens,
                }
                for probe in probes
            ],
        }

    def _write_route_log(self, route: dict[str, Any]):
        if self.route_log_dir is None:
            return
        path = self.route_log_dir / f"{int(time.time() * 1000)}_{route['id']}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(route, f, ensure_ascii=False, sort_keys=True)
            f.write("\n")


def choose_worker(
    probes: list[WorkerProbe],
    *,
    overload_load_factor: float,
    load_abs_threshold: int,
) -> tuple[WorkerState, str]:
    if not probes:
        raise ValueError("No worker probes supplied.")
    by_load = sorted(probes, key=lambda probe: (probe.load_value, probe.worker.url))
    best_load_probe = by_load[0]
    by_cache = sorted(
        probes,
        key=lambda probe: (-probe.matched_tokens, -probe.match_ratio, probe.load_value, probe.worker.url),
    )
    best_cache_probe = by_cache[0]
    if best_cache_probe.matched_tokens <= 0:
        return best_load_probe.worker, "lowest_load_no_prefix_match"
    min_load = best_load_probe.load_value
    best_load = best_cache_probe.load_value
    avg_load = sum(probe.load_value for probe in probes) / float(len(probes))
    overloaded = (
        best_load - min_load > int(load_abs_threshold)
        and best_load > max(avg_load * float(overload_load_factor), min_load)
    )
    if overloaded:
        return best_load_probe.worker, "prefix_match_overloaded_lowest_load"
    return best_cache_probe.worker, "best_prefix_match"


def strip_route_hints(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    forward_payload = dict(payload)
    hints = {
        "route_profile": forward_payload.pop("svllm_route_profile", None),
        "method_preference": forward_payload.pop("svllm_method_preference", None),
        "required_tags": forward_payload.pop("svllm_required_tags", None),
        "preferred_tags": forward_payload.pop("svllm_preferred_tags", None),
        "target_worker": forward_payload.pop("svllm_target_worker", None),
    }
    return forward_payload, {key: value for key, value in hints.items() if value is not None}


def infer_route_profile(endpoint: str, payload: dict[str, Any]) -> str:
    if endpoint == "/v1/chat/completions" and len(payload.get("messages") or []) > 2:
        return "conversation"
    return "default"


def match_payload_for_request(endpoint: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    if endpoint == "/v1/chat/completions":
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            return {"messages": messages}
    if endpoint in {"/v1/completions", "/v1/prefix_cache/inspect"}:
        prompt = payload.get("prompt")
        if "token_ids" in payload or "text" in payload:
            return {key: payload[key] for key in ("token_ids", "text") if key in payload}
        if isinstance(prompt, str):
            return {"text": prompt}
        if isinstance(prompt, list) and prompt:
            if all(isinstance(item, int) for item in prompt):
                return {"token_ids": prompt}
            if isinstance(prompt[0], str):
                return {"text": prompt[0]}
            if isinstance(prompt[0], list) and all(isinstance(item, int) for item in prompt[0]):
                return {"token_ids": prompt[0]}
    return None


def route_headers(route: dict[str, Any]) -> dict[str, str]:
    return {
        "X-SparseVLLM-Worker": str(route["selected_worker_url"]),
        "X-SparseVLLM-Route-Reason": str(route["reason"]),
        "X-SparseVLLM-Sparse-Method": str(route.get("selected_sparse_method", "")),
    }


def _with_route_headers(response: Response, route: dict[str, Any]) -> Response:
    for key, value in route_headers(route).items():
        response.headers[key] = value
    return response


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _canonical_method(value: str) -> str:
    method = str(value or "").strip()
    return "" if method == "vanilla" else method


def _method_matches(worker_method: str, preferences: list[str]) -> bool:
    worker = _canonical_method(worker_method)
    return any(worker == _canonical_method(item) for item in preferences)


def _get_json(url: str, timeout_s: float) -> dict[str, Any]:
    status, _headers, body = _request_bytes(url, "GET", None, timeout_s)
    if status >= 400:
        raise RuntimeError(body.decode("utf-8", errors="replace"))
    return json.loads(body.decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    status, _headers, body = _request_bytes(url, "POST", payload, timeout_s)
    if status >= 400:
        raise RuntimeError(body.decode("utf-8", errors="replace"))
    return json.loads(body.decode("utf-8"))


def _request_bytes(
    url: str,
    method: str,
    payload: dict[str, Any] | None,
    timeout_s: float,
) -> tuple[int, dict[str, str], bytes]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = UrlRequest(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return int(response.status), dict(response.headers.items()), response.read()
    except HTTPError as exc:
        return int(exc.code), dict(exc.headers.items()), exc.read()
    except URLError as exc:
        raise RuntimeError(str(exc)) from exc


def _open_stream_response(
    url: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> UpstreamStream | UpstreamError:
    data = json.dumps(payload).encode("utf-8")
    request = UrlRequest(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    try:
        response = urlopen(request, timeout=timeout_s)
        return UpstreamStream(response=response, headers=dict(response.headers.items()))
    except HTTPError as exc:
        return UpstreamError(status=int(exc.code), headers=dict(exc.headers.items()), body=exc.read())
    except URLError as exc:
        raise RuntimeError(f"Failed to open upstream stream {url}: {type(exc).__name__}: {exc}") from exc


async def _stream_response_chunks(response: Any):
    while True:
        chunk = await asyncio.to_thread(response.read, 8192)
        if not chunk:
            break
        yield chunk


def _content_headers(headers: dict[str, str]) -> dict[str, str]:
    content_type = headers.get("Content-Type") or headers.get("content-type")
    return {"Content-Type": content_type} if content_type else {}


def _load_profiles(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    path = Path(value)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sparse-vLLM OpenAI smart router")
    parser.add_argument("--worker-url", "--worker-urls", action="append", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--request-timeout-s", type=float, default=30.0)
    parser.add_argument("--overload-load-factor", type=float, default=1.5)
    parser.add_argument("--load-abs-threshold", type=int, default=1)
    parser.add_argument("--profiles-json", default=None, help="Inline JSON or a JSON file path for route profiles.")
    parser.add_argument("--route-log-dir", default=None)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    worker_urls = []
    for item in args.worker_url:
        worker_urls.extend(_string_list(item))
    app = create_app(
        worker_urls,
        request_timeout_s=args.request_timeout_s,
        overload_load_factor=args.overload_load_factor,
        load_abs_threshold=args.load_abs_threshold,
        profiles=_load_profiles(args.profiles_json),
        route_log_dir=args.route_log_dir,
    )
    logger.info("Starting Sparse-vLLM smart router on {}:{} for workers={}", args.host, args.port, worker_urls)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
