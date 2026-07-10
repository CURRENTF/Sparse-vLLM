# Responses 流式 reasoning 实现记录

本文档记录 `/v1/responses` Responses streaming、流式 reasoning 状态机、
流式 tool call 输出，以及随实现删除的过时代码和测试逻辑。

## 落地状态

已完成第一版实现：

- `stream=true` 返回 Responses 语义 SSE frame，不再返回未实现错误。
- 新增 Responses SSE event 层。
- 新增 Qwen3 流式 reasoning parser 状态机，覆盖 `<think>` 与 `</think>` 跨
  chunk。
- 新增流式 tool call parser，输出 `function_call` item、
  `response.function_call_arguments.delta` 和
  `response.function_call_arguments.done`。
- streaming completed/incomplete/failed 生命周期携带 usage 或显式错误。
- streaming 取消会调用 dispatcher cancel。
- smart router 对 `/v1/responses` streaming 继续透明转发 upstream bytes。
- 过时的 `stream=true` 失败测试和文档描述已改写。

## 背景

当前代码已经支持 `/v1/responses` 非流式输出、Qwen3 非流式 reasoning parser、
function tool schema 规范化、非流式 tool call 解析、Responses streaming，以及
smart router 透明转发。

## 目标

`/v1/responses` 的流式输出支持 Qwen3 reasoning 文本、正文和 tool call
在 SSE 中以 Responses 语义事件输出。

第一版目标已落地：

- `stream=true` 不再返回 400。
- 输出合法 SSE frame。
- 用 Responses 语义事件，不复用 Chat Completions chunk 格式。
- Qwen3 `<think>...</think>` 流式解析必须使用状态机，支持标签跨 chunk。
- reasoning delta 和 answer delta 分开输出。
- tool call 必须在 streaming 中输出 `function_call` item 和 arguments 事件，
  不要求客户端为了工具调用回退到非流式请求。
- 连接断开或生成取消时释放 dispatcher request。
- smart router 只透明转发 upstream SSE bytes，不解析事件。

## 当前代码审查结论

当前实现的 `/v1/responses` 已经覆盖以下能力：

- `ResponseRequest` 请求模型和 unknown field 拒绝。
- `model` 校验。
- `max_output_tokens`、`temperature`、`top_p`、`top_k` 到 `SamplingParams`。
- `instructions`、string input、text-only message input。
- `function_call_output` 和 `function_call` input item 转 chat template message。
- function tool schema 规范化并传给 tokenizer `tools` kwarg。
- `reasoning.effort` 与 `chat_template_kwargs.enable_thinking` 冲突校验。
- Qwen3 非流式 `<think>...</think>` parser。
- parser 未启用时 raw text 原样作为 `output_text`。
- 非流式 `<tool_call>` / `<tool_calls>` 输出解析为 `function_call` item。
- `finish_reason=length` 时 response `status="incomplete"`。
- Responses SSE event encoder。
- 流式 Qwen3 reasoning parser 状态机。
- 流式 tool call parser 状态机。
- streaming usage/completed/incomplete/failure 生命周期。
- smart router 透明接入 `/v1/responses`，并把 prefix-cache match payload 交给
  worker 的 `response` selector。

## 流式与非流式对齐矩阵

streaming 必须和当前 non-streaming 的功能面保持一致：

| 能力 | 非流式现状 | 流式要求 |
| --- | --- | --- |
| 请求校验 | 进入 engine 前校验 model、tools、tool_choice、reasoning.summary 等 | 完全复用同一校验路径 |
| prompt 渲染 | `_response_prompt()` 处理 instructions/input/tools/function outputs | 完全复用 `_response_prompt()` |
| sampling | `_sampling_params_from_response_request()` | 完全复用同一转换 |
| thinking 开关 | `reasoning.effort` 映射 `enable_thinking` | 完全一致 |
| reasoning parser disabled | raw text 原样输出 | raw delta 原样作为 output text delta |
| Qwen3 reasoning parser enabled | reasoning item + message/function_call item | reasoning item events + message/function_call item events |
| tool call | `<tool_call>` / `<tool_calls>` 转 `function_call` item | 输出 `function_call` item 和 arguments delta/done |
| malformed reasoning/tool call | fail fast | 发送 failed event 或在 header 发送前 HTTP error，不吞正文 |
| `finish_reason=length` | `status="incomplete"` | 终止 `response.completed` 携带 `status="incomplete"`、`incomplete_details` 和 usage |
| usage | response body 中返回 input/output/total tokens | completed/incomplete 终止事件中包含 usage |
| cancellation | dispatcher cancel | streaming disconnect/cancel 调用 dispatcher cancel |
| request log | success/cancel/failure 记录 | 同样记录，且包含 stream=true |

如果某个字段在非流式中显式不支持，streaming 也必须用同样方式 fail fast，不允许
静默忽略。

## 非目标

本补充计划不做：

- server-side conversation storage。
- `previous_response_id`。
- encrypted reasoning items。
- OpenAI 内置 web/file/computer/MCP 工具。
- 服务端执行用户工具。
- 图片、音频、文件输入。
- thinking budget。
- `reasoning.summary`。
- `parallel_tool_calls=false`。
- 新的 smart router 路由算法、profile 推断或 sparse method 选择策略。

tool call 边界：

- 非流式 tool call 保持当前实现。
- 流式 tool call 必须和 reasoning/text 一并支持，因为客户端通常统一使用
  streaming consumer。
- 第一版支持 OpenAI function tool 子集，不支持内置 web/file/computer/MCP 工具。
- 流式事件至少包含 `function_call` output item、arguments delta、arguments done
  和 output item done。
- Qwen3 第一版可以基于 `<tool_call>` / `<tool_calls>` 协议实现；未来其他模型
  通过模型协议 parser adapter 接入。
- 如果无法可靠地区分普通正文和未闭合 tool call，必须 fail fast 或显式缓冲，
  不把不完整 tool call 当作正文发送。

## 模块设计

### 事件层

新增：

```text
src/sparsevllm/entrypoints/openai/responses/events.py
```

职责：

- 生成 SSE frame。
- 生成 Responses event payload。
- 统一 `event: <type>\ndata: <json>\n\n` 格式。
- 不依赖 Qwen3 parser。

第一版事件：

- `response.created`
- `response.output_item.added`
- `response.content_part.added`
- `response.content_part.done`
- `response.reasoning_part.added`
- `response.reasoning_part.done`
- `response.reasoning_text.delta`
- `response.reasoning_text.done`
- `response.output_text.delta`
- `response.output_text.done`
- `response.function_call_arguments.delta`
- `response.function_call_arguments.done`
- `response.output_item.done`
- `response.completed`
- `response.failed`

`response.reasoning_text.delta` 是 Sparse-VLLM 对本地模型 raw reasoning 的扩展
事件。文档中必须明确它不是 OpenAI 官方 raw reasoning token 字段。

`response.function_call_arguments.delta` 和
`response.function_call_arguments.done` 对齐 OpenAI Responses streaming 的
function calling 事件语义。

文本 message item 应按 OpenAI/vLLM 事件生命周期发送：

1. `response.output_item.added`
2. `response.content_part.added`
3. `response.output_text.delta`，可多次
4. `response.output_text.done`
5. `response.content_part.done`
6. `response.output_item.done`

reasoning item 使用同样的 item 生命周期，但 content part 使用
`response.reasoning_part.added/done`，正文 delta 使用
`response.reasoning_text.delta/done`。

function call item 生命周期：

1. `response.output_item.added`
2. `response.function_call_arguments.delta`，可多次
3. `response.function_call_arguments.done`
4. `response.output_item.done`

`finish_reason=length` 时，最终 `response.completed` 里的 response 对象应包含
`status="incomplete"` 和 `incomplete_details.reason="max_output_tokens"`，与
非流式 response body 对齐。

### 流式 reasoning 状态机

在：

```text
src/sparsevllm/entrypoints/openai/responses/reasoning.py
```

新增模型无关接口和 Qwen3 实现。

建议接口：

```python
class ReasoningStreamEvent(BaseModel):
    kind: Literal["reasoning_delta", "answer_delta", "reasoning_done"]
    text: str = ""

class ReasoningStreamParser:
    def feed(self, text_delta: str) -> list[ReasoningStreamEvent]:
        ...

    def finish(self, finish_reason: str | None) -> list[ReasoningStreamEvent]:
        ...
```

Qwen3 状态：

- `content`：还没确认是否进入 `<think>`。
- `maybe_think_start`：缓存可能被切开的 `<think>`。
- `reasoning`：输出 reasoning delta。
- `maybe_think_end`：缓存可能被切开的 `</think>`。
- `answer`：输出 answer delta。

要求：

- `<think>` 跨 chunk 必须正确识别。
- `</think>` 跨 chunk 必须正确识别。
- 若生成以 `<think>` 开始，`</think>` 前的内容只能作为 reasoning 输出。
- `</think>` 后的内容作为 answer 输出。
- 若 finish_reason 是 `length` 且仍在 reasoning 中，输出 incomplete。
- 若 finish_reason 是 `stop` 且仍在未闭合 reasoning 中，返回 parse error。
- parser 未启用时，所有 delta 直接作为 answer delta。

### 流式 tool call parser

tool call 的输出协议同样是模型相关职责，不应写死在
`serving/responses.py`。在：

```text
src/sparsevllm/entrypoints/openai/responses/tools.py
```

新增流式 parser 或 adapter 接口。

建议接口：

```python
class ToolCallStreamEvent(BaseModel):
    kind: Literal[
        "tool_call_started",
        "tool_call_arguments_delta",
        "tool_call_done",
        "answer_delta",
    ]
    call_id: str | None = None
    name: str | None = None
    arguments_delta: str = ""
    text: str = ""

class ToolCallStreamParser:
    def feed(self, text_delta: str) -> list[ToolCallStreamEvent]:
        ...

    def finish(self, finish_reason: str | None) -> list[ToolCallStreamEvent]:
        ...
```

Qwen3 第一版处理：

- `<tool_call>...</tool_call>`
- `<tool_calls>...</tool_calls>`
- 标签跨 chunk。
- JSON body 跨 chunk。
- malformed JSON fail fast。

输出要求：

- 一旦确认进入 tool call，后续内容不能作为普通 output text 发送。
- 能够确认 `name` 后，发送 `function_call` output item added。
- `arguments` 内容输出为 `response.function_call_arguments.delta`。
- 完成时发送 `response.function_call_arguments.done` 和
  `response.output_item.done`。
- 如果第一版不能安全地逐字段提取 arguments，可以先缓冲到完整 tool call JSON
  闭合后，一次性发送完整 arguments 作为单个 delta；但仍必须走 streaming 事件，
  不能要求客户端改用非流式请求。

reasoning parser 和 tool call parser 的组合顺序：

1. 原始 delta 先进入 reasoning parser。
2. reasoning parser 输出的 `answer_delta` 再进入 tool call parser。
3. tool call parser 输出 text delta 或 function_call events。

这样 `<think>...</think><tool_call>...</tool_call>` 可以稳定拆成 reasoning item
和 function_call item。

### serving 层

修改：

```text
src/sparsevllm/entrypoints/openai/serving/responses.py
```

新增 streaming 路径：

```python
if request.stream:
    return StreamingResponse(
        _response_stream(...),
        media_type="text/event-stream",
    )
```

`_response_stream()` 负责：

- 创建 response id 和 created_at。
- 渲染 prompt。
- submit dispatcher。
- 发送 `response.created`。
- 接收 dispatcher token delta。
- 把 token delta 送入 reasoning stream parser。
- 把 reasoning parser 输出的 answer delta 送入 tool call stream parser。
- 按 parser event 发送 reasoning/text/tool call delta。
- final 时发送 done/completed/incomplete。
- 异常时发送 failed 或抛出 HTTPException。
- 客户端取消时调用 `dispatcher.cancel(handle)`。
- 日志包含 prompt/completion/total tokens、elapsed、TPS。

不要复用 `_chat_completion_stream()` 或 Chat Completions chunk schema。

### dispatcher 边界

不改 engine 执行路径。继续使用当前 `AsyncEngineDispatcher` 的 token/final 输出。

如果现有 dispatcher token 事件不能提供 raw text delta，需要明确选择：

- 优先基于 cumulative raw token ids 解码得到 raw delta；
- 或扩展 dispatcher token event 增加 `raw_text` / `raw_text_delta`。

不能用 `skip_special_tokens=True` 的 visible text 去做 reasoning parser，否则
可能丢掉 `<think>`、`</think>` 或 Qwen special token，导致状态机失效。

### smart router 边界

smart router 只透明转发 `/v1/responses` 的 SSE bytes。

不做：

- 解析 Responses event。
- 解析 reasoning。
- 改 route profile。
- 根据 tools/reasoning/input length 改 worker 选择。
- 新增 sparse method 选择策略。

## 已删除的过时代码和逻辑

实现 streaming 时已删除或替换以下内容，未保留死逻辑：

- `serve_response()` 中 `request.stream` 返回 400 的分支。
- `test_response_stream_true_fails_explicitly`。
- 计划文档中 “Responses streaming 第一版未实现” 的落地状态。
- 计划文档中 “`stream=true` 若未实现，返回显式 400” 作为当前验收的描述。
- runtime docs 中 “`stream=true` fails explicitly until Responses SSE events
  are implemented” 的描述。
- 任何把 Responses streaming 映射成 Chat Completions chunk 的临时路径。
- 任何要求 tool call 客户端回退到非流式 Responses 的临时说明。

实现后不允许同时存在 “支持 Responses streaming” 和 “stream=true 返回 400” 两套
逻辑。

## 单元测试计划

继续使用当前 OpenAI server 测试集合，并新增 Responses streaming 重点覆盖。

必须运行：

```bash
.venv/bin/pytest -q tests/test_openai_api_server.py -k "response or reasoning or tool or chat_template_kwargs"
.venv/bin/pytest -q tests/test_openai_smart_router.py -k responses
.venv/bin/python -m py_compile \
  src/sparsevllm/entrypoints/openai/serving/responses.py \
  src/sparsevllm/entrypoints/openai/responses/reasoning.py \
  src/sparsevllm/entrypoints/openai/responses/tools.py \
  src/sparsevllm/entrypoints/openai/render.py \
  src/sparsevllm/entrypoints/openai/protocol/responses.py
```

新增 `tests/test_openai_api_server.py` 覆盖：

- `stream=true` 返回 `text/event-stream`，不再 400。
- 基础文本流：`response.created`、`response.output_item.added`、
  `response.content_part.added`、`response.output_text.delta`、
  `response.output_text.done`、`response.content_part.done`、
  `response.output_item.done`、`response.completed`。
- Qwen3 完整 `<think>...</think>answer` 流式拆成 reasoning delta 和 answer delta。
- `<think>` 跨 chunk。
- `</think>` 跨 chunk。
- `<think>` 未闭合且 finish_reason=`length` 时输出 incomplete。
- `<think>` 未闭合且 finish_reason=`stop` 时 fail fast。
- parser 未启用时 `<think>...</think>` 原样作为 output text delta。
- Qwen3 `<tool_call>` 跨 chunk 后输出 `function_call` item。
- Qwen3 `<tool_call>` 输出 `response.function_call_arguments.delta` 和
  `response.function_call_arguments.done`。
- `<think>...</think><tool_call>...</tool_call>` 同时输出 reasoning item 和
  function_call item。
- malformed streaming tool call JSON fail fast，不当作正文吞掉。
- `finish_reason=length` 的 streaming 终止事件包含 `status="incomplete"`、
  `incomplete_details.reason="max_output_tokens"` 和 usage。
- streaming completed 事件 usage 与非流式 usage 计算一致。
- 客户端取消 streaming 时调用 `dispatcher.cancel(handle)`。
- streaming 日志包含 prompt/completion/total tokens 和 TPS。

新增 `tests/test_openai_smart_router.py` 覆盖：

- smart router 对 `/v1/responses` streaming 只透明转发 bytes。
- upstream SSE event 原样透传。
- route headers 仍存在。
- upstream streaming HTTP error 仍按现有逻辑返回。
- 接入 Responses streaming 不改变 `/v1/completions` 和
  `/v1/chat/completions` 的默认 worker 选择。

## 真实模型验证计划

阶段完成后必须启动真实 OpenAI API server 验证。模型固定使用：

```text
/data2/pretrain_models/Qwen3-4B-Thinking-2507
```

启动前必须检查 GPU 是否空闲，选择空闲设备；如果所有设备都忙，先等待，等待
过长再报告情况，不要直接抢占忙碌设备。

示例命令：

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> sparsevllm-openai-server \
  --model /data2/pretrain_models/Qwen3-4B-Thinking-2507 \
  --served-model-name Qwen3-4B-Thinking-2507 \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --reasoning-parser qwen3
```

验证请求：

- `/health`
- `/v1/models`
- `/v1/responses` 非流式 thinking on
- `/v1/responses` 非流式 thinking off
- `/v1/responses` 流式 thinking on
- `/v1/responses` 流式 thinking off
- `/v1/responses` 流式 tool call
- `/v1/responses` 流式 reasoning + tool call
- `/v1/chat/completions` `chat_template_kwargs.enable_thinking=false`
- smart router 透明转发 `/v1/responses` streaming，route headers 存在

流式 thinking on 必须能观察到：

- reasoning delta 事件。
- output text delta 事件。
- completed 或 incomplete 终止事件。
- SSE 格式合法。

流式 tool call 必须能观察到：

- function_call output item。
- function call arguments delta/done 事件。
- output item done 事件。
- completed 或 incomplete 终止事件。

### 2026-07-10 真实模型验证记录

使用空闲的 GPU 0（NVIDIA H100 80GB）和
`/data2/pretrain_models/Qwen3-4B-Thinking-2507` 完成验证。worker 监听
`127.0.0.1:18080`，smart router 监听 `127.0.0.1:18081`；worker 使用
`--reasoning-parser qwen3`、`gpu_memory_utilization=0.9` 和
`max_model_len=8192`。验证结果如下：

- `/health` 和 `/v1/models` 返回 200，served model 正确。
- 非流式 thinking on 返回 completed response，output 为 reasoning item 和
  message item，usage 为 28/390/418。
- 非流式 thinking off 只返回 message item；64 token 上限触发显式
  `status="incomplete"`、`reason="max_output_tokens"` 和 24/64/88 usage。
- 确定性流式 thinking on 依次产生 reasoning item 事件、1631 字符 reasoning
  delta、message item 事件和正文 `323`，最终 completed usage 为 24/673/697。
- 流式 thinking off 只产生 message/output text 事件，不产生 reasoning 事件；
  达到 token 上限时正确携带 incomplete 状态和 usage。
- 流式工具请求产生 reasoning item 和 `function_call` item，包含
  `response.function_call_arguments.delta/done`、output item done，参数为
  `{"city":"Paris"}`，最终 completed usage 为 181/110/291。
- 回填对应的 `function_call` 与 `function_call_output` 后，第二轮返回 reasoning
  item 和最终天气 message，completed usage 为 211/206/417。
- Chat Completions 的 `chat_template_kwargs.enable_thinking=false` 请求返回 200，
  可见正文不包含 `<think>` 或 `</think>` 标签。
- smart router 对 Responses SSE 原样转发，返回
  `content-type: text/event-stream`、`x-sparsevllm-worker` 和
  `x-sparsevllm-route-reason`；Chat Completions 和 Completions 经同一 worker
  转发也均返回 200。
- 客户端在 streaming 中途断开后，worker 的 waiting、decoding 和 active
  request 数量均恢复为 0，KV cache 空闲槽恢复。

该 checkpoint 的 README 明确声明它只支持 thinking mode。因此
`reasoning.effort="none"` 和 `enable_thinking=false` 在本次验证中只能确认 API
映射、输出结构、标签可见性和终止状态，不能证明模型语义上停止自然语言推导。
特别是关闭 Responses reasoning parser 后的工具请求，模型仍生成思考正文、
`</think>` 和 `<tool_call>`，服务端按 fail-visible 原则将其保留为 message；真实
tool-call 事件验收使用该模型支持的默认 thinking mode 完成。

## 完成标准

- `stream=true` 不再返回 “Responses streaming is not implemented yet.”
- Responses streaming 不使用 Chat Completions chunk schema。
- Qwen3 reasoning stream parser 有跨 chunk 状态机测试。
- Qwen3 tool call stream parser 有跨 chunk 状态机测试。
- 流式 tool call 输出 function call arguments delta/done 事件。
- 过时代码、过时测试、过时文档描述已删除或改写。
- 单元测试、py_compile 和真实模型服务验证通过。
- 未实现能力仍 fail fast，不静默降级。
