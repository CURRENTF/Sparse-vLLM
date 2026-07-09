from fastapi import APIRouter
from fastapi import Request

from sparsevllm.entrypoints.openai.protocol.responses import ResponseRequest
from sparsevllm.entrypoints.openai.serving.responses import serve_response


router = APIRouter()


@router.post("/v1/responses")
async def responses(body: ResponseRequest, request: Request):
    return await serve_response(
        body,
        request.app.state.dispatcher,
        request.app.state.engine.tokenizer,
        request.app.state.served_model_name,
        request.app.state.request_log_dir,
        request.app.state.reasoning_parser,
    )

