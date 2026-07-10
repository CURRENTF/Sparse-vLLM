import time

from fastapi import APIRouter
from fastapi import Request


router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/v1/models")
def models(request: Request):
    served_model_name = request.app.state.served_model_name
    created = int(time.time())
    config = getattr(request.app.state.engine, "config", None)
    max_model_len = int(getattr(config, "max_model_len", 0) or 0)
    model = {
        "id": served_model_name,
        "object": "model",
        "created": created,
        "owned_by": "sparsevllm",
    }
    if max_model_len > 0:
        model["max_model_len"] = max_model_len
    return {
        "object": "list",
        "data": [model],
    }
