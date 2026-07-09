from fastapi import APIRouter
from fastapi import Request

from sparsevllm.entrypoints.openai.protocol.prefix_cache import PrefixCacheDeleteSubtreeRequest
from sparsevllm.entrypoints.openai.protocol.prefix_cache import PrefixCacheInspectRequest
from sparsevllm.entrypoints.openai.protocol.prefix_cache import PrefixCacheMatchRequest
from sparsevllm.entrypoints.openai.protocol.prefix_cache import PrefixCacheSetEvictionPriorityRequest
from sparsevllm.entrypoints.openai.serving.prefix_cache import serve_prefix_cache_delete_subtree
from sparsevllm.entrypoints.openai.serving.prefix_cache import serve_prefix_cache_inspect
from sparsevllm.entrypoints.openai.serving.prefix_cache import serve_prefix_cache_match
from sparsevllm.entrypoints.openai.serving.prefix_cache import serve_prefix_cache_set_eviction_priority


router = APIRouter()


@router.post("/v1/prefix_cache/inspect")
async def prefix_cache_inspect(body: PrefixCacheInspectRequest, request: Request):
    return await serve_prefix_cache_inspect(
        body,
        request.app.state.dispatcher,
        request.app.state.engine.tokenizer,
    )


@router.post("/v1/prefix_cache/match")
async def prefix_cache_match(body: PrefixCacheMatchRequest, request: Request):
    return await serve_prefix_cache_match(
        body,
        request.app.state.dispatcher,
        request.app.state.engine.tokenizer,
    )


@router.post("/v1/prefix_cache/delete_subtree")
async def prefix_cache_delete_subtree(body: PrefixCacheDeleteSubtreeRequest, request: Request):
    return await serve_prefix_cache_delete_subtree(
        body,
        request.app.state.dispatcher,
        request.app.state.engine.tokenizer,
    )


@router.post("/v1/prefix_cache/set_eviction_priority")
async def prefix_cache_set_eviction_priority(body: PrefixCacheSetEvictionPriorityRequest, request: Request):
    return await serve_prefix_cache_set_eviction_priority(
        body,
        request.app.state.dispatcher,
        request.app.state.engine.tokenizer,
    )

