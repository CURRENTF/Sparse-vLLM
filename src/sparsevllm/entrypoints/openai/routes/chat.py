from fastapi import APIRouter
from fastapi import Request

from sparsevllm.entrypoints.openai.protocol.chat import ChatCompletionRequest
from sparsevllm.entrypoints.openai.serving.chat import serve_chat_completion


router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request):
    return await serve_chat_completion(
        body,
        request.app.state.dispatcher,
        request.app.state.engine.tokenizer,
        request.app.state.served_model_name,
        request.app.state.request_log_dir,
        request.app.state.reasoning_parser,
    )
