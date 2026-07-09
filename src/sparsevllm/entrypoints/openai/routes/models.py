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
    return {
        "object": "list",
        "data": [
            {
                "id": served_model_name,
                "object": "model",
                "created": created,
                "owned_by": "sparsevllm",
            }
        ],
    }

