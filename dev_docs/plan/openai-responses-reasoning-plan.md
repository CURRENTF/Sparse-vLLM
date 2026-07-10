# OpenAI Responses、工具调用与 Qwen3 思考控制实现计划

本文档描述 Sparse-vLLM OpenAI-compatible server 后续支持 Qwen3 思考控制、
Responses API、工具调用和 reasoning 解析器的实现计划。目标是先做可验证的
最小实现，保持当前服务端的研究代码风格：失败显式、行为可复现、不要静默
忽略未实现参数。

## 落地状态

已完成第一批最小实现：

- OpenAI worker server 已拆分为 `dispatcher.py`、`protocol/`、`routes/`、
  `serving/`、`render.py`、`sampling.py` 和 `responses/` 子模块；
  `api_server.py` 仅保留 app 装配、生命周期、CLI 和兼容 re-export。
- `/v1/chat/completions` 支持
  `chat_template_kwargs.enable_thinking`，未知 key、非 bool 值、无 chat
  template 时传入该参数都会显式失败。
- `/v1/completions` 保持 raw prompt 语义，不新增 thinking 控制字段。
- `/v1/responses` 支持非流式文本输入、text-only message input、
  `function_call_output` input、function tool schema、`reasoning.effort` 到
  Qwen3 thinking 开关的映射、usage、incomplete 状态组装和 Responses SSE
  streaming。
- 未实现完整语义的 Responses 控制字段会显式失败：`tool_choice` 仅接受
  `null`/`"auto"`，`parallel_tool_calls=false` 和 `reasoning.summary` 暂不支持。
- Responses streaming 使用 Responses 语义事件，不复用 Chat Completions chunk
  格式；支持文本 delta、Qwen3 reasoning delta 和 function call arguments
  delta/done。
- `--reasoning-parser qwen3` 支持把 `<think>...</think>` 拆成 Sparse-vLLM
  自有扩展 reasoning item 和 message/function_call item；parser 未启用时原样
  返回 `output_text`。
- `responses/tools.py` 支持 function tool schema 规范化和显式
  `<tool_call>`/`<tool_calls>` 输出解析；服务端不执行工具。
- Smart router 已透明接入 `/v1/responses`，prefix-cache match 使用
  `{"response": payload}` 交给 worker 复用 `_response_prompt()` 渲染。
- `tests/test_openai_api_server.py` 和 `tests/test_openai_smart_router.py` 已补充
  Chat thinking、Responses、reasoning parser、tool call、response selector 和
  router 转发覆盖。

仍明确延后：

- server-side conversation storage、`previous_response_id`、encrypted reasoning
  items、内置 web/file/computer/MCP 工具、图片/音频/文件输入。
- thinking budget；该能力需要推理过程中断和二段继续生成。

## 当前代码基线

当前 OpenAI-compatible server 主要集中在
`src/sparsevllm/entrypoints/openai/api_server.py`：

- `CompletionRequest` 服务 `/v1/completions`，直接接收 raw prompt。
- `ChatCompletionRequest` 服务 `/v1/chat/completions`，接收 `messages`，用
  `_chat_prompt()` 渲染为单个 prompt。
- `_chat_prompt()` 当前只调用
  `tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)`，
  没有透传 `chat_template_kwargs`、`tools` 或 Qwen3 的 `enable_thinking`。
- `AsyncEngineDispatcher` 是统一调度入口，只接收已经渲染好的
  `prompt` 和 `SamplingParams`，通过 `engine.add_request()`、`engine.step()`、
  `engine.abort_request()` 驱动推理。
- 非流式响应由 `_completion_response()` 和 `_chat_completion_response()` 组装。
  流式响应由 `_completion_stream()` 和 `_chat_completion_stream()` 组装。
- 当前 chat 请求模型已经允许 `tools`、`tool_choice`、`parallel_tool_calls`，
  但这些字段没有进入 chat template，也没有工具调用解析或执行流程。
- 当前测试集中在 `tests/test_openai_api_server.py`，已经覆盖请求校验、
  chat template fallback、prefix-cache chat selector、SSE、logprobs、取消和
  dispatcher 错误传播。
- `api_server.py` 目前还同时承担 app wiring、CLI、协议模型、prompt 渲染、
  sampling 参数转换、dispatcher、响应组装、worker/load/prefix-cache 控制等
  职责。文件规模已经过大，继续直接加入 Responses API 会把 reasoning parser、
  tool call、Responses stream event 和状态上下文继续堆进同一个文件，维护风险
  过高。

这意味着正式加入新能力前，应先做阶段零的模块拆分。拆分必须保持行为一致，
不改变现有 chat/completions、worker、prefix-cache 和 smart router 的外部语义。
后续阶段只能落到拆分后的协议、路由、serving、render、parser 模块中，不再把
新逻辑继续堆进 `api_server.py`。

## 外部 API 语义边界

OpenAI 官方文档的关键点：

- Reasoning 模型推荐使用 Responses API。Chat Completions 仍被支持，但
  Responses API 更适合 reasoning、tool calling 和多轮状态延续。
- OpenAI 官方 API 不暴露原始 reasoning tokens；reasoning tokens 会计入输出
  token 统计，可通过 usage 明细观察。可选的 reasoning summary 是摘要，不是
  原始思考链。
- Responses API 中 tool call 和 tool output 是不同 item，通过 `call_id` 关联。
  应用侧负责执行真实工具，然后把 `function_call_output` 传回模型。
- Responses API 的流式输出是语义化事件，而不是只把 chat/completions 的
  `delta.content` 搬过去。

Qwen3 / vLLM 生态的关键点：

- Qwen3 默认启用 thinking；在 chat template 支持的情况下，可用
  `chat_template_kwargs: {"enable_thinking": false}` 显式关闭。
- Qwen3 还支持 `/think` 和 `/no_think` 这类文本标记，但这是 prompt 层软控制，
  不应替代服务端显式参数。
- vLLM 的 reasoning parser 是 OpenAI-compatible 生态扩展：它把模型输出中的
  `<think>...</think>` 和正文拆分成结构化字段。这个行为不是 OpenAI 官方
  Chat Completions 标准字段，但对本地开源模型服务有工程价值。

## 总体分阶段

0. 先拆分 OpenAI server 模块边界，严格测试并保证现有行为一致。
1. 让现有 `/v1/chat/completions` 支持 Qwen3 思考显式开关。
2. `/v1/completions` 保持 raw prompt 语义，不提供服务端显式思考开关。
3. 新增 `/v1/responses` 最小可用实现，严格对照 OpenAI Responses API 的
   item 化输入输出模型。
4. 让 smart router 支持 `/v1/responses`，并保持 prefix-cache match 与真实
   Responses prompt 渲染一致。
5. 在 `/v1/responses` 中实现通用 reasoning/tool call 输出结构，并接入
   Qwen3 reasoning parser 和 thinking 开关控制。
6. 后续单独实现 thinking budget。它需要推理过程中断和二段继续生成，不应
   混入第一版 reasoning parser。

## 阶段零：OpenAI server 模块拆分

### 目标

阶段零只做结构调整，不新增 API 能力。目标是让 `api_server.py` 退回到真正的
server 装配层，为 Responses API 留出清晰落点。

拆分原则参考 vLLM 的 OpenAI server 组织方式，但不照搬其完整复杂度：

- `api_server.py` 只负责 FastAPI app 创建、router 注册、app state、生命周期和
  CLI 入口。
- endpoint 的 FastAPI route 保持很薄，只负责接收请求、调用 serving 对象、
  返回 JSON 或 StreamingResponse。
- Pydantic request/response schema 放在 protocol 模块。
- prompt 渲染、sampling 参数转换、响应组装和 endpoint 业务逻辑放在 serving
  或 render/helper 模块。
- dispatcher 作为模型执行调度层独立出来，供 chat、completions、Responses
  复用。
- reasoning parser、tool call helpers、Responses event/context 不进入
  `api_server.py`。
- `routes/` 只表示 worker OpenAI server 的 thin FastAPI route，不承载 smart
  router 的候选选择、负载探测、prefix-cache match 或 HTTP 转发策略。
- smart router 是独立 gateway 入口，后续若继续增长，应拆到独立
  `smart_router/` 包，而不是并入 worker server 的 `routes/`。

本仓库可对照 `reference/vllm/vllm/entrypoints/openai/`：

- `api_server.py`：app 装配、生命周期、state 初始化。
- `completion/api_router.py`、`chat_completion/api_router.py`、
  `responses/api_router.py`：thin route。
- `completion/protocol.py`、`chat_completion/protocol.py`、
  `responses/protocol.py`：请求/响应 schema。
- `completion/serving.py`、`chat_completion/serving.py`、
  `responses/serving.py`：endpoint 业务逻辑。
- `responses/context.py`、`responses/streaming_events.py`：Responses 独有状态
  和事件层。

### 建议目录

```text
src/sparsevllm/entrypoints/openai/
  api_server.py                 # app wiring / lifecycle / CLI only
  dispatcher.py                 # AsyncEngineDispatcher / RequestHandle

  protocol/
    completion.py               # /v1/completions schema
    chat.py                     # /v1/chat/completions schema
    prefix_cache.py             # /v1/prefix_cache/* schema
    worker.py                   # /v1/worker/* schema
    responses.py                # 阶段二新增

  routes/
    completion.py               # thin FastAPI routes
    chat.py
    prefix_cache.py
    worker.py
    models.py
    responses.py                # 阶段二新增

  serving/
    base.py                     # shared validation/model/sampling/logging helpers
    completion.py               # completion response/stream assembly
    chat.py                     # chat rendering/response/stream assembly
    prefix_cache.py
    worker.py
    responses.py                # 阶段二新增

  render.py                     # chat_template / response_prompt rendering
  sampling.py                   # SamplingParams / stop / logprobs helpers
  smart_router.py               # existing gateway entrypoint, not worker routes/
  smart_router/                 # 后续如需拆分 smart router，再建立该包
    app.py                      # gateway FastAPI app / thin forwarding routes
    policy.py                   # existing candidate filtering / route hints handling
    probes.py                   # worker info/load/prefix-cache probes
    forwarding.py               # HTTP/SSE forwarding and route headers
    payloads.py                 # route hints stripping and match payload building
  responses/
    events.py                   # Responses SSE events, 阶段二/三新增
    reasoning.py                # Qwen3 reasoning parser, 阶段三新增
    tools.py                    # generic tool call normalization/context
```

如果第一轮拆分发现 `routes/` 和 `serving/` 同时引入会造成过大改动，可以先
只抽 `dispatcher.py`、`protocol/`、`render.py`、`sampling.py`，然后再把
route 和 serving 分离。关键约束是：Responses 新代码不能继续进入
`api_server.py`。

不要在阶段零顺手重构 smart router。smart router 行为会影响实验调度和性能
结论，应作为后续单独重点处理；阶段零只保证现有 smart router 测试继续通过。

### 拆分顺序

1. 抽出 `dispatcher.py`
   - 移动 `RequestHandle`、队列状态对象和 `AsyncEngineDispatcher`。
   - 保持 submit/abort/shutdown 行为、异常传播和 token delta 发布逻辑不变。
   - 先跑 dispatcher 相关测试，再继续下一步。

2. 抽出协议模型
   - 移动 completion、chat、prefix-cache、worker 的 Pydantic models。
   - 保留当前字段、默认值、`extra="forbid"` 和 validator 行为。
   - 不在这一步新增 `chat_template_kwargs`。

3. 抽出 render/sampling/helper
   - 移动 `_chat_prompt()`、chat content normalization、sampling params、
     stop/logprobs/usage helper。
   - prefix-cache `messages` selector 继续复用同一份 chat render helper。

4. 抽出现有 endpoint serving 逻辑
   - completion/chat 的非流式和流式响应组装进入 `serving/`。
   - worker/load/prefix-cache 控制进入独立 serving 或 route helper。
   - FastAPI route 只做 request -> serving -> response 的薄封装。

5. 收敛 `api_server.py`
   - 保留 app factory、state 初始化、router 注册、CLI/main。
   - 删除已经迁出的业务 helper，避免出现两份实现。

### 行为一致性测试

阶段零必须以“无行为变化”为验收标准。每完成一个拆分小步都要运行现有测试，
不能等到全部拆完再统一修。

至少执行：

```bash
pytest tests/test_openai_api_server.py
pytest tests/test_openai_smart_router.py
```

如果局部拆分只影响某个区域，可以先跑更小的 `-k` 子集，但阶段零结束前必须
跑完整的 OpenAI server 和 smart router 测试。

需要重点保证：

- `/v1/completions` 请求校验、非流式、流式、logprobs、stop 行为不变。
- `/v1/chat/completions` chat template fallback、SSE、usage、logprobs 行为不变。
- dispatcher 取消、异常传播、shutdown、finished 状态不变。
- `/v1/worker/info`、`/v1/worker/load` 输出字段不变。
- `/v1/prefix_cache/*` selector、tokenization、match/inspect/delete 行为不变。
- smart router 的 route hints 剥离、profile/method/tags/target_worker、
  prefix-cache match 和 route headers 行为不变。

阶段零不允许把失败测试通过静默 fallback、吞异常或放宽校验来修掉。若拆分中
发现原测试覆盖不足，应补充等价性测试，而不是改变外部行为。

### 阶段零真实服务验证

阶段零完成后必须启动真实模型 worker 的 OpenAI API server 做回归验证，不能
只依赖 mock 和单元测试。

验证模型使用 Qwen3-4B-Thinking-2507；本地路径用占位符表示：

```text
<MODEL_ROOT>/Qwen3-4B-Thinking-2507
```

启动前必须按仓库任务规则检查设备空闲状态，选择空闲设备；如果没有空闲设备，
等待后仍不可用则跳过真实验证并记录原因。

示例启动命令可按实际空闲设备和显存调整：

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> sparsevllm-openai-server \
  --model <MODEL_ROOT>/Qwen3-4B-Thinking-2507 \
  --served-model-name Qwen3-4B-Thinking-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9
```

阶段零真实服务验证只验证已有行为，不验证尚未实现的新功能：

- `/health` 正常。
- `/v1/models` 返回 served model。
- `/v1/completions` 非流式和流式可用。
- `/v1/chat/completions` 非流式和流式可用。
- `/v1/worker/info` 和 `/v1/worker/load` 可用。
- 如果启动了 prefix cache，验证 `/v1/prefix_cache/match` 现有 selector 行为。

阶段零验收要求：模块拆分前后，上述真实服务请求的 HTTP 状态、基本响应结构、
usage 字段和 streaming 事件格式保持一致。

## 阶段一：Chat Completions 思考开关

### 请求字段

在 `protocol/chat.py` 的 `ChatCompletionRequest` 增加：

```python
chat_template_kwargs: dict[str, Any] | None = None
```

第一版只允许：

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

校验规则：

- `chat_template_kwargs` 必须是 JSON object。
- 当前只允许 `enable_thinking`。
- `enable_thinking` 必须是 bool。
- 未知 key 直接返回 400。
- 如果 tokenizer 没有 `chat_template`，但请求传了 `chat_template_kwargs`，
  返回 400，不静默忽略。

这样可以避免研究运行中以为关闭了 thinking，但实际 prompt 没有变化。

### Prompt 渲染

把 `render.py` 中的 `_chat_prompt()` 改为：

```python
def _chat_prompt(tokenizer, messages, chat_template_kwargs=None):
    ...
```

当 tokenizer 有 `chat_template` 时：

```python
kwargs = {
    "tokenize": False,
    "add_generation_prompt": True,
}
kwargs.update(chat_template_kwargs or {})
return tokenizer.apply_chat_template(chat, **kwargs)
```

调用点：

- chat serving 传入 `request.chat_template_kwargs`。
- prefix-cache `messages` selector 默认不传，保持旧行为。

### 输出语义

不改 chat serving 中的 response 和 stream 组装逻辑。

- 关闭 thinking 后，模型理论上不再输出 `<think>`。
- 如果模型仍输出 `<think>`，服务端仍原样放入 `message.content`。
- 解析 `<think>` 是 Responses 阶段或客户端职责，不在 chat 第一阶段做。

### 单测

新增测试：

- `chat_template_kwargs.enable_thinking=false` 会传入 tokenizer。
- 未知 `chat_template_kwargs` key 返回 400。
- 非 bool `enable_thinking` 返回 400。
- 没有 chat template 时传 `chat_template_kwargs` 返回 400。
- 不传 `chat_template_kwargs` 时现有 `_chat_prompt()` 行为不变。

## 阶段一补充：Completions 不提供思考开关

`/v1/completions` 是 raw prompt 接口，没有 `messages`，也不调用
`apply_chat_template()`。因此它不实现 Qwen3 的
`chat_template_kwargs.enable_thinking`，也不新增服务端显式思考开关。

要求：

- 不在 `/v1/completions` 中伪造 chat template。
- 不新增 `enable_thinking`、`chat_template_kwargs` 或其他会改写 raw prompt
  语义的思考控制字段。
- raw completions 如需控制 Qwen3 thinking，可以由客户端把 `/think` 或
  `/no_think` 写进 prompt；服务端只保证 raw prompt 原样进入模型。
- 如果用户向 `/v1/completions` 传 `chat_template_kwargs` 或
  `enable_thinking`，由于请求模型 `extra="forbid"`，应继续返回校验错误，
  而不是静默忽略。

## 阶段二：Responses API 最小实现

### 新 endpoint

新增：

```text
POST /v1/responses
```

不要把 Responses 语义塞进 `/v1/chat/completions`。Responses API 应该有独立
请求模型、响应模型、流式事件模型和测试。

阶段二建立在阶段零拆分后的模块边界上：

- `protocol/responses.py` 定义 request/response/event schema。
- `routes/responses.py` 注册 `/v1/responses` thin route。
- `serving/responses.py` 负责 validation、prompt 构造、dispatcher 调用和
  response assembly。
- `render.py` 提供 `_response_prompt()`，供 Responses endpoint 和
  prefix-cache match 复用。
- `api_server.py` 只负责 include router 和初始化 serving 对象。

### 请求模型第一版

在 `protocol/responses.py` 中先支持文本和函数调用所需的最小子集：

```python
class ResponseRequest(BaseModel):
    model: str
    input: str | list[ResponseInputItem]
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False
    tools: list[ResponseTool] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning: ResponseReasoning | None = None
    chat_template_kwargs: dict[str, Any] | None = None
```

第一版输入 item 支持：

- string input，等价于单条 user message。
- message item：`role` 为 `developer`、`system`、`user`、`assistant`；
  `content` 支持 text-only。
- function call output item：`type="function_call_output"`，包含
  `call_id` 和 `output`。
- 后续轮次可把上一轮的 `reasoning`、`function_call`、`message` output item
  放回 input，但第一版可以只保存/转写必要文本，不做 server-side 状态存储。

不支持的 OpenAI item 类型必须返回 400，包括图片、音频、文件、web_search、
computer_use、remote MCP、structured output 等。

### Prompt 构造

Responses 请求最后仍要落到当前 `AsyncEngineDispatcher.submit(prompt, sampling, ...)`。
因此需要在 `render.py` 中新增一个独立 helper：

```python
def _response_prompt(tokenizer, request: ResponseRequest) -> str:
    ...
```

职责：

- 把 `instructions` 放到 developer/system 层。
- 把 text input/message input 转成 chat template 所需的 messages。
- 把 `function_call_output` 转成 tool role message 或 Qwen template 可识别的
  工具结果结构。
- 如果传了 `tools`，优先通过 tokenizer chat template 的 `tools` kwarg 渲染。
- 如果 tokenizer 不支持 tools kwarg，但请求传了 tools，返回 400。

不要在第一版中引入新依赖；必要时用 `inspect.signature()` 判断 tokenizer
是否支持 `tools`、`chat_template_kwargs` 中的 key。

### 响应模型第一版

非流式响应至少包含：

```json
{
  "id": "resp_...",
  "object": "response",
  "created_at": 1234567890,
  "status": "completed",
  "model": "Qwen3-8B",
  "output": [
    {
      "id": "msg_...",
      "type": "message",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "最终回答",
          "annotations": []
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0
  }
}
```

注意：

- 不要把 SDK convenience 属性当作原始 HTTP 响应字段。
- `usage` 必须从 dispatcher final item 的 prompt/completion token 统计构造。
- 如果后续 reasoning parser 产生 reasoning token 计数，再增加
  `output_tokens_details.reasoning_tokens`。
- 如果达到 `max_output_tokens`，`status` 应为 `incomplete`，并提供
  `incomplete_details.reason="max_output_tokens"`。

### 流式响应第一版

Responses stream 不复用 chat chunk 格式。当前实现已在
`responses/events.py` 中定义基础事件：

- `response.created`
- `response.output_item.added`
- `response.content_part.added`
- `response.output_text.delta`
- `response.output_text.done`
- `response.content_part.done`
- `response.output_item.done`
- `response.reasoning_part.added`
- `response.reasoning_part.done`
- `response.reasoning_text.delta`
- `response.reasoning_text.done`
- `response.function_call_arguments.delta`
- `response.function_call_arguments.done`
- `response.completed`
- `response.failed`

`response.reasoning_text.delta` 是 Sparse-vLLM 对本地模型 raw reasoning 的扩展
事件，不声明等同于 OpenAI-hosted raw reasoning token 字段。`stream=true` 不再
返回未实现错误。

### 单测

新增测试集中覆盖：

- string input 转 user message。
- `instructions` 转 system/developer 层。
- 不支持 input item 类型返回 400。
- `max_output_tokens` 映射到 `SamplingParams.max_tokens`。
- 非流式 response object/output/usage 形状。
- `stream=true` 返回合法 Responses SSE frame。
- `api_server.py` 只 include Responses router，不包含 Responses 业务逻辑。

## 阶段三：Responses reasoning 通用结构与 Qwen3 parser

### 通用结构

Responses 的 reasoning 支持应分成两层：

1. 通用 Responses 输出结构
2. 模型协议相关 reasoning parser

通用层负责：

- 根据 parser 结果组装 Responses `reasoning` item 和 `message` item。
- 处理 `status`、`incomplete_details`、usage 和 output 顺序。
- 保持 parser 未启用时原样输出 `output_text`。
- 为后续其他模型 parser 预留统一接口。

模型特异层只负责把某个模型的原始文本协议解析成通用中间结果。例如 Qwen3
第一版解析 `<think>...</think>`；未来如果其他模型使用不同标签或不同协议，
只新增 parser，不改 Responses output 组装逻辑。

建议 parser 接口保持简单，例如：

```python
class ParsedReasoning(BaseModel):
    reasoning_text: str | None
    output_text: str
    incomplete_reasoning: bool = False

def parse_reasoning(text: str, finish_reason: str | None) -> ParsedReasoning:
    ...
```

`serving/responses.py` 只依赖这个通用结果，不直接写 Qwen3 标签规则。

### 开关和优先级

Responses 阶段支持两类控制：

1. `chat_template_kwargs.enable_thinking`
2. `reasoning.effort`

Qwen3 当前是二值 thinking 控制，不是 OpenAI 的 effort ladder。建议映射：

- `reasoning.effort == "none"`：`enable_thinking=false`
- `reasoning.effort in {"minimal", "low", "medium", "high", "xhigh"}`：
  `enable_thinking=true`
- 未设置 `reasoning.effort`：保持 Qwen3 默认，即 thinking enabled

冲突处理：

- 如果同时传 `chat_template_kwargs.enable_thinking=false` 和
  `reasoning.effort!="none"`，返回 400。
- 如果同时传 `chat_template_kwargs.enable_thinking=true` 和
  `reasoning.effort=="none"`，返回 400。
- 不要静默选择一个。

### Parser 注册

在阶段零预留的 Responses 子模块中新增一个很小的 parser 层，例如：

```text
src/sparsevllm/entrypoints/openai/responses/reasoning.py
```

第一版只注册：

```python
qwen3
```

推荐配置方式：

- server 启动参数：`--reasoning-parser qwen3`
- 或 engine kwargs 中的 OpenAI server 专属参数。注意不要混入
  `sparsevllm.Config`，因为它不是模型运行时参数。

如果没有启用 parser，Responses 仍把完整文本作为 `output_text` 返回。
如果启用了 parser，才把 `<think>` 拆成 reasoning item 和 message item。
parser 由 `serving/responses.py` 调用，不进入 `api_server.py`。
除 parser 以外，Responses reasoning 的输出结构和错误处理不应写死为 Qwen3。

### 非流式解析

Qwen3 parser 规则：

- 如果输出以可选空白后接 `<think>` 开始，查找第一个 `</think>`。
- 找到闭合标签：
  - `<think>` 内为 reasoning text。
  - `</think>` 后为正文或 tool call 文本。
- 没有闭合标签：
  - 若 finish_reason 是 length，返回 `status="incomplete"`，并把已生成部分作为
    reasoning item；正文为空。
  - 若 finish_reason 是 stop，则返回 parse error，避免把不完整 reasoning
    当正文。
- 如果输出不以 `<think>` 开始，全部当正文。

Responses output 结构：

```json
[
  {
    "id": "rs_...",
    "type": "reasoning",
    "summary": []
  },
  {
    "id": "msg_...",
    "type": "message",
    "role": "assistant",
    "content": [
      {"type": "output_text", "text": "正文", "annotations": []}
    ]
  }
]
```

对于本地 Qwen3，raw reasoning 是否放进 `reasoning` item 需要显式命名。建议
第一版使用 repo 自有扩展字段，例如：

```json
{
  "type": "reasoning",
  "text": "解析出的 <think> 内容",
  "summary": []
}
```

并在文档中注明：OpenAI 官方 API 不暴露 raw reasoning tokens；该字段是
Sparse-vLLM 对本地模型的兼容扩展。这样避免假装完全等同 OpenAI 内部
reasoning tokens。

### 流式解析

流式 parser 必须是状态机，不能用简单字符串替换：

- 状态 `content`：尚未进入 `<think>`。
- 状态 `maybe_think_start`：缓存可能被切开的 `<think>` 前缀。
- 状态 `reasoning`：输出 `response.reasoning_text.delta` 或 repo 自有扩展事件。
- 状态 `maybe_think_end`：缓存可能被切开的 `</think>`。
- 状态 `answer`：输出 `response.output_text.delta`。

Responses streaming 已实现流式 parser，并覆盖标签跨 chunk 被切开的情况。

### 单测

新增测试：

- 完整 `<think>...</think>正文` 拆成 reasoning item + message item。
- 没有 `<think>` 时只输出 message item。
- `<think>` 未闭合且 length finish 时返回 incomplete。
- `<think>` 未闭合且 stop finish 时 fail fast。
- 标签跨 chunk 的流式状态机测试。
- parser 未启用时原样输出。

## 阶段四：Responses API 与 Smart Router 兼容

commit `1340cdb` 引入了 OpenAI smart router。它目前对
`/v1/completions` 和 `/v1/chat/completions` 做透明转发，并根据 worker
info、worker load、prefix-cache match、method、tags、profile 和
`svllm_target_worker` 选择 worker。Responses API 接入时应复用这套控制面，
但不要让 router 理解 reasoning 或 tool call 语义。

Smart router 的行为必须谨慎演进。阶段四的默认目标是“让 `/v1/responses`
进入现有路由框架”，不是重新设计全局路由策略。任何会影响
`/v1/completions` 或 `/v1/chat/completions` 现有选择结果的改动，都应拆成
独立变更，并用回归测试证明行为变化是预期的。

### 模块边界

smart router 不放进 worker server 的 `routes/`。`routes/` 只服务
`api_server.py` 这个 worker OpenAI server；smart router 是独立 gateway。

阶段四可以继续在现有 `smart_router.py` 上做最小修改。若文件继续膨胀，再按
下面方式拆分：

```text
src/sparsevllm/entrypoints/openai/smart_router/
  app.py          # gateway FastAPI app and endpoint registration
  policy.py       # existing candidate filtering and route hints handling
  probes.py       # worker info/load/prefix-cache probes
  forwarding.py   # JSON/SSE forwarding and route headers
  payloads.py     # route hint stripping and prefix-match payload conversion
```

这个包的 `app.py` 可以有 gateway route，但不要和 worker server 的
`routes/responses.py` 混用。两者一个负责“转发到哪个 worker”，一个负责
“worker 如何处理请求”。

### Router endpoint

在 `src/sparsevllm/entrypoints/openai/smart_router.py` 中新增：

```python
@app.post("/v1/responses")
async def responses(request: Request):
    return await router.route_openai_request("/v1/responses", await request.json())
```

这样 `/v1/responses` 和现有 OpenAI endpoint 走同一个流程：

1. `strip_route_hints()` 剥离 `svllm_*` 路由字段。
2. `_candidate_workers()` 根据 model、method、tags、profile 过滤 worker。
3. `_probe_workers()` 探测 `/v1/worker/load` 和 `/v1/prefix_cache/match`。
4. `choose_worker()` 在 prefix 命中和负载之间做选择。
5. `forward_json()` 或 `forward_stream()` 透明转发到选中 worker。

### Prefix-cache match

Responses 的 `input` 不是简单的 `prompt` 或 `messages`。router 不应该自己
复刻 `_response_prompt()`，否则容易和 worker 实际推理 prompt 不一致，导致
prefix-cache match 结果不可信。

推荐设计：

- `match_payload_for_request("/v1/responses", payload)` 返回：

```python
{"response": payload}
```

- worker 的 `PrefixCacheMatchRequest` 增加 `response` selector。
- worker 内部用和 `/v1/responses` 完全相同的 `render.py::_response_prompt()` 渲染
  prompt，再通过 `_encode_prefix_cache_text()` tokenization 后调用
  `prefix_cache_match()`。

这样 `instructions`、`input` item、`tools`、`function_call_output`、
`reasoning`、`chat_template_kwargs` 都会参与同一套 prompt 渲染，router
看到的 prefix 命中和真实请求一致。

第一版不建议让 router 从 Responses `input` 中抽第一段 text 做近似 match。
近似 match 虽然容易实现，但会破坏研究结果可信度：worker 真实 prompt 可能
包含 instructions、tool schema 或 chat template 控制字段。

### 不变更路由算法

阶段四不实现新的 route profile 推断，不实现 Sparse-VLLM 方法感知路由，也
不新增 SnapKV、OmniKV 或其他 sparse method 的选择策略。

要求：

- 不修改现有 `/v1/completions` 和 `/v1/chat/completions` 的默认 worker 选择。
- 不根据 Responses 的 `tools`、`function_call_output`、`reasoning` 或输入长度
  自动选择新的 profile、method 或 tags。
- 不新增 bulk、agent、conversation 等默认 profile 推断。
- 不把任何新的 sparse method 偏好硬编码进 smart router。
- 仅复用当前 smart router 已有的 route hints、worker filtering、load probe、
  prefix-cache match 和 forwarding 流程。

阶段四的目标只有一个：让 smart router 能透明接收 `/v1/responses`，并把请求
按现有机制转发到 worker。

### Worker info 和 load

Responses 不需要新增 worker load 语义。继续使用现有：

- `/v1/worker/info`
- `/v1/worker/load`

后续如果有 worker 专门启用了 reasoning parser 或 tool call support，可以在
`worker_info()` 中增加 server 能力字段，例如：

```json
{
  "openai_endpoints": ["completions", "chat.completions", "responses"],
  "reasoning_parsers": ["qwen3"],
  "tool_parsers": ["qwen3"]
}
```

能力字段第一版仅作为观测信息，不参与 smart router 候选过滤，避免影响已有
worker 选择。是否基于能力字段做过滤，应作为后续独立变更讨论。

### Streaming

router 不解析 Responses streaming event。它应像现有 chat/completions stream
一样，转发上游 bytes，并附加 route headers：

- `X-SparseVLLM-Worker`
- `X-SparseVLLM-Route-Reason`
- `X-SparseVLLM-Sparse-Method`

如果上游 worker 返回 streaming HTTP error，router 只透传该错误，不伪造
Responses stream。

### 状态与 previous_response_id

第一版 Responses 不做 server-side conversation storage，也不做
`previous_response_id`。因此 router 可以保持无状态。

后续如果支持 `previous_response_id`，必须保证同一 response state 回到同一
worker。可选方案：

- router 维护 `response_id -> worker_url` 映射；
- 或要求客户端在后续请求中带 `svllm_target_worker`；
- 或把 response state 外置到共享存储。

在没有这些机制前，不应声称支持跨 worker 的 stateful Responses。

### 单测

新增 `tests/test_openai_smart_router.py` 覆盖：

- `/v1/responses` route 会调用 `route_openai_request("/v1/responses", payload)`。
- `match_payload_for_request("/v1/responses", payload)` 返回 `{"response": payload}`。
- route hints 会在转发前被剥离，Responses payload 不带 `svllm_*` 到 worker。
- `svllm_target_worker` 对 `/v1/responses` 生效。
- streaming 上游 HTTP error 能按现有逻辑透传，并释放 `local_inflight`。
- 现有 `/v1/completions` 和 `/v1/chat/completions` 的默认 worker 选择不因
  Responses 接入发生变化。
- Responses 的 `tools`、`function_call_output`、`reasoning` 或输入长度不会触发
  新的 profile/method/tag 推断。

新增 `tests/test_openai_api_server.py` 覆盖：

- `/v1/prefix_cache/match` 接受 `response` selector。
- `response` selector 复用 `_response_prompt()`，而不是单独拼接字符串。
- 同时设置多个 selector 时 fail fast。

## 阶段五：Responses 中的 tool call

### 服务端职责

第一版服务端只负责：

- 把 `tools` 传给 tokenizer chat template。
- 让模型生成工具调用文本。
- 用 `responses/tools.py` 中的通用 tool call 解析和规范化逻辑，把模型输出转成
  Responses `function_call` output item。
- 接收客户端下一轮传入的 `function_call_output` input item。

服务端不执行用户工具。真实工具执行仍在客户端或上层应用。

tool call 的 API 结构应尽量保持模型无关：

- OpenAI function tool schema 规范化是通用逻辑。
- `function_call` output item 组装是通用逻辑。
- `function_call_output` input item 转 prompt 的数据模型是通用逻辑。
- tokenizer/chat template 是否支持 `tools` 是能力校验，不应改变 API 结构。

第一版测试目标模型是 Qwen3，但不要把 Responses tool call 的公共结构命名为
Qwen3 专用实现。若未来发现某个模型的工具调用文本协议必须特异解析，应像
reasoning parser 一样通过独立 adapter 接入，而不是改 Responses 公共层。

### 请求工具 schema

先支持 OpenAI function tool 子集：

```json
{
  "type": "function",
  "name": "get_weather",
  "description": "查询天气",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {"type": "string"}
    },
    "required": ["city"]
  },
  "strict": false
}
```

兼容 Chat Completions 常见嵌套格式：

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "查询天气",
    "parameters": {}
  }
}
```

内部规范化为 Responses function tool 形态。未知工具类型返回 400。

### 输出工具调用

当 parser 识别出工具调用，Responses output 追加：

```json
{
  "id": "fc_...",
  "type": "function_call",
  "call_id": "call_...",
  "name": "get_weather",
  "arguments": "{\"city\":\"北京\"}",
  "status": "completed"
}
```

如果同时有 reasoning，则 output 顺序为：

1. reasoning item
2. function_call item

如果模型输出普通正文，则 output 为 message item。

### 下一轮工具结果

客户端把工具执行结果传回：

```json
{
  "type": "function_call_output",
  "call_id": "call_...",
  "output": "{\"city\":\"北京\",\"temperature\":\"28C\"}"
}
```

服务端在 `render.py::_response_prompt()` 中把它转换为 tokenizer chat
template 可识别的 tool result；第一版用 Qwen3 模板验证。若 tokenizer 不支持
工具模板，返回 400。

### 单测

新增测试：

- tools 正规化。
- tokenizer 支持 `tools` kwarg 时收到规范化工具。
- tokenizer 不支持工具但请求传 tools 时返回 400。
- 模型输出的 function call 文本解析为 `function_call` item。
- `function_call_output` input item 能进入下一轮 prompt。
- malformed tool call JSON 返回 parse_failed 风格的显式错误，不当作正文吞掉。

## 后续：Thinking budget

thinking budget 不是简单的输出后处理。Qwen3 技术报告描述的做法是在推理中
达到预算后强制结束 thinking，然后继续生成正文。实现需要新能力：

- reasoning parser 在生成过程中统计 reasoning token 或 reasoning 文本 token。
- `AsyncEngineDispatcher` 支持在某个 request 的生成过程中触发中断。
- 引擎支持用追加文本继续同一请求，或服务端能安全地二段生成。
- 插入结束文本，例如 Qwen3 报告中类似
  `Considering the limited time ... </think>` 的 early-stop 提示。
- usage 中区分 reasoning token 和 visible output token。

因此 budget 应单独做，不和 `enable_thinking` 或第一版 parser 合并。

## 文档和测试更新

每个阶段都需要同步：

- `docs/configuration/runtime-parameter-semantics.md` 的 OpenAI-compatible
  serving 字段表。
- `dev_docs/plan/openai-responses-reasoning-plan.md` 的完成状态。
- `tests/test_openai_api_server.py` 和 `tests/test_openai_smart_router.py` 的
  单元测试。
- 如果新增 parser 模块，补充 parser 级纯函数测试，避免必须启动模型。

真实模型验证要求：

2026-07-10 已使用空闲 GPU 0 和
`/data2/pretrain_models/Qwen3-4B-Thinking-2507` 完成最终真实服务验证，覆盖
Responses 非流式/流式 reasoning、工具调用及结果回填、Chat thinking 参数、
smart router SSE 转发和客户端取消后的资源释放。完整请求结果和 usage 记录见
`openai-responses-streaming-reasoning-fix-plan.md` 的“2026-07-10 真实模型验证
记录”。该 checkpoint 官方 README 声明仅支持 thinking mode，因此 thinking off
只验收 API 映射、输出结构和标签可见性，不将模型自然语言是否继续推导作为服务
端开关有效性的判据。

阶段零完成后和本计划全部完成后，都必须启动真实模型 worker 的 OpenAI API
server 验证。验证模型使用 Qwen3-4B-Thinking-2507；本地路径用占位符表示：

```text
<MODEL_ROOT>/Qwen3-4B-Thinking-2507
```

启动前必须按仓库任务规则检查设备空闲状态，选择空闲设备；没有空闲设备时
跳过真实验证并记录原因。

最终验收同样需要启动真实 OpenAI API server。示例命令：

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> sparsevllm-openai-server \
  --model <MODEL_ROOT>/Qwen3-4B-Thinking-2507 \
  --served-model-name Qwen3-4B-Thinking-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser qwen3
```

阶段零真实服务验证见“阶段零真实服务验证”小节。本计划全部完成后的真实服务
验证至少覆盖：

- Qwen3 thinking on：Responses 输出 reasoning item + message item。
- Qwen3 thinking off：Responses 只输出 message item。
- Chat thinking off：`/v1/chat/completions` 通过
  `chat_template_kwargs.enable_thinking=false` 后不再出现 `<think>`。
- 工具调用：第一轮输出 `function_call`，客户端回填 `function_call_output` 后
  第二轮输出最终 message。
- Smart router 透明转发 `/v1/responses`，route headers 存在，且不改变现有
  `/v1/completions`、`/v1/chat/completions` 的路由行为。

## 明确不做

第一批实现不做：

- 服务端执行用户工具。
- OpenAI 内置 web/file/computer/MCP 工具。
- 图片、音频、文件输入。
- conversation storage 或 `previous_response_id` 的完整状态管理。
- encrypted reasoning items。
- raw reasoning 与 OpenAI 官方不可见 reasoning token 的完全等价声明。
- thinking budget。

## 参考资料

- OpenAI Responses create API:
  https://developers.openai.com/api/reference/resources/responses/methods/create
- OpenAI reasoning models:
  https://developers.openai.com/api/docs/guides/reasoning
- OpenAI function calling:
  https://developers.openai.com/api/docs/guides/function-calling
- OpenAI migrate to Responses:
  https://developers.openai.com/api/docs/guides/migrate-to-responses
- OpenAI streaming Responses:
  https://developers.openai.com/api/docs/guides/streaming-responses
- vLLM reasoning outputs:
  https://docs.vllm.ai/en/latest/features/reasoning_outputs/
- Qwen vLLM deployment notes:
  https://qwen.readthedocs.io/en/latest/deployment/vllm.html
