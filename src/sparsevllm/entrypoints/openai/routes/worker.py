from fastapi import APIRouter
from fastapi import Request

from sparsevllm.entrypoints.openai.serving.worker import serve_worker_info
from sparsevllm.entrypoints.openai.serving.worker import serve_worker_load


router = APIRouter()


@router.get("/v1/worker/info")
def worker_info(request: Request):
    return serve_worker_info(
        request.app.state.engine,
        request.app.state.served_model_name,
    )


@router.get("/v1/worker/load")
async def worker_load(request: Request):
    return await serve_worker_load(request.app.state.dispatcher)

