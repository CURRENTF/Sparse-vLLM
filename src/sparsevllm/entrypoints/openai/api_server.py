import argparse
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from sparsevllm.config import Config
from sparsevllm.entrypoints.openai.dispatcher import AsyncEngineDispatcher
from sparsevllm.entrypoints.openai.dispatcher import RequestHandle
from sparsevllm.entrypoints.openai.dispatcher import _ActiveRequest
from sparsevllm.entrypoints.openai.protocol.chat import ChatCompletionRequest
from sparsevllm.entrypoints.openai.protocol.chat import ChatContentPart
from sparsevllm.entrypoints.openai.protocol.chat import ChatMessage
from sparsevllm.entrypoints.openai.protocol.completion import CompletionRequest
from sparsevllm.entrypoints.openai.protocol.prefix_cache import PrefixCacheDeleteSubtreeRequest
from sparsevllm.entrypoints.openai.protocol.prefix_cache import PrefixCacheInspectRequest
from sparsevllm.entrypoints.openai.protocol.prefix_cache import PrefixCacheMatchRequest
from sparsevllm.entrypoints.openai.protocol.prefix_cache import PrefixCacheSetEvictionPriorityRequest
from sparsevllm.entrypoints.openai.protocol.responses import ResponseReasoning
from sparsevllm.entrypoints.openai.protocol.responses import ResponseRequest
from sparsevllm.entrypoints.openai.render import _chat_content_text
from sparsevllm.entrypoints.openai.render import _chat_prompt
from sparsevllm.entrypoints.openai.render import _chat_request_prompt
from sparsevllm.entrypoints.openai.render import _chat_template_role
from sparsevllm.entrypoints.openai.render import _response_prompt
from sparsevllm.entrypoints.openai.render import resolve_chat_template_kwargs
from sparsevllm.entrypoints.openai.render import resolve_chat_tools
from sparsevllm.entrypoints.openai.render import validate_chat_template_kwargs
from sparsevllm.entrypoints.openai.responses.reasoning import get_reasoning_parser
from sparsevllm.entrypoints.openai.routes.chat import router as chat_router
from sparsevllm.entrypoints.openai.routes.completion import router as completion_router
from sparsevllm.entrypoints.openai.routes.models import router as models_router
from sparsevllm.entrypoints.openai.routes.prefix_cache import router as prefix_cache_router
from sparsevllm.entrypoints.openai.routes.responses import router as responses_router
from sparsevllm.entrypoints.openai.routes.worker import router as worker_router
from sparsevllm.entrypoints.openai.sampling import _coerce_cli_value
from sparsevllm.entrypoints.openai.sampling import _field_was_set
from sparsevllm.entrypoints.openai.sampling import _find_stop_index
from sparsevllm.entrypoints.openai.sampling import _normalize_prompts
from sparsevllm.entrypoints.openai.sampling import _normalize_stop
from sparsevllm.entrypoints.openai.sampling import _safe_stream_text_len
from sparsevllm.entrypoints.openai.sampling import _sampling_params_from_request
from sparsevllm.entrypoints.openai.sampling import _sampling_params_from_response_request
from sparsevllm.entrypoints.openai.serving.base import _chat_logprobs
from sparsevllm.entrypoints.openai.serving.base import _completion_logprobs
from sparsevllm.entrypoints.openai.serving.base import _model_dump_json
from sparsevllm.entrypoints.openai.serving.base import _sse
from sparsevllm.entrypoints.openai.serving.base import _tokens_per_second
from sparsevllm.entrypoints.openai.serving.base import _wait_final
from sparsevllm.entrypoints.openai.serving.base import _write_request_log
from sparsevllm.entrypoints.openai.serving.chat import _chat_completion_response
from sparsevllm.entrypoints.openai.serving.chat import _chat_message
from sparsevllm.entrypoints.openai.serving.chat import _chat_completion_stream
from sparsevllm.entrypoints.openai.serving.chat import _stream_include_usage
from sparsevllm.entrypoints.openai.serving.chat import _validate_chat_request
from sparsevllm.entrypoints.openai.serving.chat_parsing import ParsedChatOutput
from sparsevllm.entrypoints.openai.serving.chat_parsing import parse_chat_output
from sparsevllm.entrypoints.openai.serving.completion import _completion_response
from sparsevllm.entrypoints.openai.serving.completion import _completion_stream
from sparsevllm.entrypoints.openai.serving.completion import _validate_request
from sparsevllm.entrypoints.openai.serving.prefix_cache import _encode_prefix_cache_text
from sparsevllm.entrypoints.openai.serving.prefix_cache import _prefix_cache_match_token_ids_from_request
from sparsevllm.entrypoints.openai.serving.prefix_cache import _prefix_cache_token_ids_from_request
from sparsevllm.entrypoints.openai.serving.prefix_cache import _run_prefix_cache_control
from sparsevllm.entrypoints.openai.serving.responses import _response_output_items
from sparsevllm.entrypoints.openai.serving.responses import _response_response
from sparsevllm.entrypoints.openai.serving.responses import _response_stream
from sparsevllm.entrypoints.openai.serving.responses import _validate_response_request
from sparsevllm.entrypoints.openai.serving.worker import _worker_tags
from sparsevllm.llm import LLM
from sparsevllm.method_registry import normalize_sparse_method
from sparsevllm.utils.log import logger


UNSUPPORTED_SERVING_METHOD_PREFIXES = ("deltakv",)
SEMANTIC_ENGINE_ARGS = {
    "sparse_method",
    "deltakv_checkpoint_path",
    "decode_keep_tokens",
    "prefill_keep_tokens",
    "sink_keep_tokens",
    "recent_keep_tokens",
    "full_attention_layers",
    "engine_prefill_chunk_size",
    "deltakv_neighbor_count",
}


def create_app(
    model: str,
    engine_kwargs: dict[str, Any] | None = None,
    *,
    served_model_name: str | None = None,
    engine: LLM | None = None,
    request_log_dir: str | None = None,
    reasoning_parser: str | None = None,
) -> FastAPI:
    served_model_name = served_model_name or model
    engine_kwargs = dict(engine_kwargs or {})
    if engine is None:
        engine_kwargs.setdefault("throughput_log_interval_s", 0.0)
    _validate_serving_method(engine_kwargs, engine)
    get_reasoning_parser(reasoning_parser)
    engine = engine or LLM(model, **engine_kwargs)
    dispatcher = AsyncEngineDispatcher(engine)
    request_log_path = Path(request_log_dir) if request_log_dir else None
    if request_log_path is not None:
        request_log_path.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            dispatcher.close()

    app = FastAPI(title="Sparse-vLLM OpenAI-compatible API", lifespan=lifespan)
    app.state.engine = engine
    app.state.dispatcher = dispatcher
    app.state.served_model_name = served_model_name
    app.state.request_log_dir = request_log_path
    app.state.reasoning_parser = reasoning_parser
    app.include_router(models_router)
    app.include_router(worker_router)
    app.include_router(prefix_cache_router)
    app.include_router(completion_router)
    app.include_router(chat_router)
    app.include_router(responses_router)
    return app


def _validate_serving_method(engine_kwargs: dict[str, Any], engine: LLM | None = None):
    method = (
        getattr(getattr(engine, "config", None), "vllm_sparse_method", "")
        if engine is not None
        else engine_kwargs.get("sparse_method", engine_kwargs.get("vllm_sparse_method", ""))
    )
    method = normalize_sparse_method(method)
    if any(method.startswith(prefix) for prefix in UNSUPPORTED_SERVING_METHOD_PREFIXES):
        raise ValueError(
            f"vllm_sparse_method={method!r} is not supported by the OpenAI API server yet. "
            "Run this method through the offline experiment entrypoints until serving support is validated."
        )


def _load_engine_kwargs_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    stripped = value.strip()
    if stripped.startswith("{"):
        data = json.loads(stripped)
    else:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"--engine-kwargs path does not exist: {value}")
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--engine-kwargs must resolve to a JSON object")
    return data


def _parse_engine_kwargs(raw_args: list[str]) -> dict[str, Any]:
    config_fields = {
        name: config_field
        for name, config_field in Config.__dataclass_fields__.items()
        if config_field.init
    }
    allowed_fields = set(config_fields) | SEMANTIC_ENGINE_ARGS
    kwargs: dict[str, Any] = {}
    idx = 0
    while idx < len(raw_args):
        key = raw_args[idx]
        if not key.startswith("--"):
            raise ValueError(f"Unexpected engine argument {key!r}; expected --name value.")
        name = key[2:].replace("-", "_")
        if name not in allowed_fields:
            raise ValueError(f"Unknown Sparse-vLLM engine argument {key!r}.")
        if idx + 1 >= len(raw_args) or raw_args[idx + 1].startswith("--"):
            if name not in config_fields or not isinstance(config_fields[name].default, bool):
                raise ValueError(f"Missing value for Sparse-vLLM engine argument {key!r}.")
            value: Any = True
            idx += 1
        else:
            value = _coerce_cli_value(raw_args[idx + 1])
            idx += 2
        kwargs[name] = value
    return kwargs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Sparse-vLLM with an OpenAI-compatible completions API.")
    parser.add_argument("--model", required=True, help="Local Hugging Face model path to load.")
    parser.add_argument("--served-model-name", default=None, help="Model name accepted by OpenAI-compatible endpoints.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--engine-kwargs", default=None, help="JSON object or JSON file with Sparse-vLLM engine kwargs.")
    parser.add_argument("--request-log-dir", default=None, help="Optional directory for per-request JSON logs.")
    parser.add_argument("--reasoning-parser", choices=["qwen3"], default=None)
    return parser


def main():
    parser = build_arg_parser()
    args, raw_engine_args = parser.parse_known_args()
    engine_kwargs = _load_engine_kwargs_arg(args.engine_kwargs)
    cli_engine_kwargs = _parse_engine_kwargs(raw_engine_args)
    duplicate_keys = sorted(set(engine_kwargs) & set(cli_engine_kwargs))
    if duplicate_keys:
        raise ValueError(
            "--engine-kwargs and CLI engine flags both set the same keys: "
            f"{duplicate_keys}"
        )
    engine_kwargs.update(cli_engine_kwargs)
    app = create_app(
        args.model,
        engine_kwargs,
        served_model_name=args.served_model_name,
        request_log_dir=args.request_log_dir,
        reasoning_parser=args.reasoning_parser,
    )

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
