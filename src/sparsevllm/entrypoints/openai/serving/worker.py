import os
from typing import Any

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from sparsevllm.entrypoints.openai.dispatcher import AsyncEngineDispatcher


def serve_worker_info(engine: Any, served_model_name: str):
    return JSONResponse(engine.worker_info(served_model_name=served_model_name, tags=_worker_tags()))


async def serve_worker_load(dispatcher: AsyncEngineDispatcher):
    try:
        result = await dispatcher.control("worker_load")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail=f"Worker load returned non-object result: {type(result).__name__}.")
    return JSONResponse(result)


def _worker_tags() -> list[str]:
    raw = os.getenv("SPARSEVLLM_WORKER_TAGS", "")
    return [tag.strip() for tag in raw.split(",") if tag.strip()]

