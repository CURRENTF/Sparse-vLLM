# Experiment Records

## 2026-06-24 00:44 CST - qwen3_4b_vanilla_gpu5_t062_stream_smoke_20260624_004414

- Status: failed
- Goal: Smoke-test Qwen3-4B-Instruct-2507 through the Sparse-vLLM OpenAI shim on Claw-Eval T062 after adding streaming-compatible responses.
- Working dir: `/home/haojitai/projects/Sparse-vLLM`
- Command:

```bash
OPENROUTER_API_KEY=dummy \
RUN_NAME=qwen3_4b_vanilla_gpu5_t062_stream_smoke_20260624_004414 \
MODEL_PATH=/data2/haojitai/models/Qwen3-4B-Instruct-2507 \
SERVED_MODEL_NAME=qwen3-4b-vanilla \
CUDA_VISIBLE_DEVICES=5 \
SPARSEVLLM_CONTEXT_WINDOW=32768 \
ENGINE_KWARGS='{"tensor_parallel_size":1,"gpu_memory_utilization":0.80,"max_model_len":32768,"engine_prefill_chunk_size":2048,"sparse_method":"vanilla"}' \
CLAW_EVAL_ARGS='batch --config ${CLAW_EVAL_CONFIG} --filter T062 --trials 1 --parallel 1 --no-judge --trace-dir ${TRACE_DIR}' \
bash benchmark/claw_eval/run_sparsevllm_claw_eval.sh
```

- Code: `codex/import-deltakv-main` / `349002b`; worktree had relevant uncommitted Claw-Eval shim and runner changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; Sparse-vLLM env `/data2/haojitai/conda_envs/sparse-vllm-tf530`; Claw-Eval env `/data2/haojitai/conda_envs/claw-eval-py311`; requested GPU range was 4-7 but GPU4/GPU6 were occupied, so the smoke used GPU5.
- Model: `/data2/haojitai/models/Qwen3-4B-Instruct-2507`; Sparse-vLLM `sparse_method=vanilla`; `max_model_len=32768`; `engine_prefill_chunk_size=2048`; `tensor_parallel_size=1`.
- Data: Claw-Eval task `T062_finance_pltr_cagr`; 1 trial; non-sandbox; no judge.
- Logs: `/data2/haojitai/outputs/Sparse-vLLM/claw-eval/qwen3_4b_vanilla_gpu5_t062_stream_smoke_20260624_004414/logs/sparsevllm_openai_server.log`
- Results: no Claw-Eval result; server exited before `/health`.
- Notes: server failed at `torch.distributed.init_process_group` because default Sparse-vLLM port `2333` was already in use. Follow-up runs set `SPARSEVLLM_MASTER_PORT`.

## 2026-06-24 00:45 CST - qwen3_4b_vanilla_gpu5_t062_stream_smoke_20260624_004532

- Status: inconclusive
- Goal: Re-run T062 with a unique Sparse-vLLM distributed port to validate the OpenAI streaming path.
- Working dir: `/home/haojitai/projects/Sparse-vLLM`
- Command:

```bash
OPENROUTER_API_KEY=dummy \
RUN_NAME=qwen3_4b_vanilla_gpu5_t062_stream_smoke_20260624_004532 \
MODEL_PATH=/data2/haojitai/models/Qwen3-4B-Instruct-2507 \
SERVED_MODEL_NAME=qwen3-4b-vanilla \
CUDA_VISIBLE_DEVICES=5 \
SERVER_PORT=18005 \
SPARSEVLLM_MASTER_PORT=24335 \
SPARSEVLLM_CONTEXT_WINDOW=32768 \
ENGINE_KWARGS='{"tensor_parallel_size":1,"gpu_memory_utilization":0.80,"max_model_len":32768,"engine_prefill_chunk_size":2048,"sparse_method":"vanilla"}' \
CLAW_EVAL_ARGS='batch --config ${CLAW_EVAL_CONFIG} --filter T062 --trials 1 --parallel 1 --no-judge --trace-dir ${TRACE_DIR}' \
bash benchmark/claw_eval/run_sparsevllm_claw_eval.sh
```

- Code: `codex/import-deltakv-main` / `349002b`; worktree had relevant uncommitted Claw-Eval shim and runner changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU5; `SPARSEVLLM_MASTER_PORT=24335`; HTTP server `127.0.0.1:18005`.
- Model: `/data2/haojitai/models/Qwen3-4B-Instruct-2507`; Sparse-vLLM vanilla; `max_model_len=32768`; `engine_prefill_chunk_size=2048`.
- Data: Claw-Eval task `T062_finance_pltr_cagr`; 1 trial; non-sandbox; no judge.
- Logs: `/data2/haojitai/outputs/Sparse-vLLM/claw-eval/qwen3_4b_vanilla_gpu5_t062_stream_smoke_20260624_004532/logs/claw_eval.log`; server requests in `/data2/haojitai/outputs/Sparse-vLLM/claw-eval/qwen3_4b_vanilla_gpu5_t062_stream_smoke_20260624_004532/sparsevllm_requests`
- Results: model call succeeded with `806` total model tokens (`643` input / `163` output), but batch result errored `1/1` with `'NoneType' object has no attribute 'evaluate'`.
- Notes: request log confirms the Claw-Eval agent request used `stream=true` and tools. The failure was in the T062 grader because `--no-judge` sets `judge=None` while the task grader still calls `judge.evaluate(...)`.

## 2026-06-24 00:50 CST - qwen3_4b_vanilla_gpu5_t062_localjudge_stream_smoke_20260624_005017

- Status: completed
- Goal: Produce a complete Claw-Eval smoke result by routing both the evaluated model and the judge to the local Sparse-vLLM OpenAI shim.
- Working dir: `/home/haojitai/projects/Sparse-vLLM`
- Command:

```bash
OPENROUTER_API_KEY=dummy \
RUN_NAME=qwen3_4b_vanilla_gpu5_t062_localjudge_stream_smoke_20260624_005017 \
MODEL_PATH=/data2/haojitai/models/Qwen3-4B-Instruct-2507 \
SERVED_MODEL_NAME=qwen3-4b-vanilla \
CUDA_VISIBLE_DEVICES=5 \
SERVER_PORT=18006 \
SPARSEVLLM_MASTER_PORT=24336 \
SPARSEVLLM_CONTEXT_WINDOW=32768 \
CLAW_EVAL_JUDGE_BASE_URL=http://127.0.0.1:18006/v1 \
CLAW_EVAL_JUDGE_MODEL=qwen3-4b-vanilla \
ENGINE_KWARGS='{"tensor_parallel_size":1,"gpu_memory_utilization":0.80,"max_model_len":32768,"engine_prefill_chunk_size":2048,"sparse_method":"vanilla"}' \
CLAW_EVAL_ARGS='batch --config ${CLAW_EVAL_CONFIG} --filter T062 --trials 1 --parallel 1 --trace-dir ${TRACE_DIR}' \
bash benchmark/claw_eval/run_sparsevllm_claw_eval.sh
```

- Code: `codex/import-deltakv-main` / `349002b`; worktree had relevant uncommitted Claw-Eval shim, runner, config-template, and experiment-doc changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPU5; `SPARSEVLLM_MASTER_PORT=24336`; HTTP server `127.0.0.1:18006`; download proxy `http://127.0.0.1:7898`.
- Model: `/data2/haojitai/models/Qwen3-4B-Instruct-2507`; Sparse-vLLM `sparse_method=vanilla`; `max_model_len=32768`; `engine_prefill_chunk_size=2048`; `tensor_parallel_size=1`.
- Data: Claw-Eval task `T062_finance_pltr_cagr`; 1 trial; non-sandbox. The judge was `qwen3-4b-vanilla` served by the same local Sparse-vLLM shim, so this is a functional smoke, not an official OpenRouter-judge result.
- Logs: `/data2/haojitai/outputs/Sparse-vLLM/claw-eval/qwen3_4b_vanilla_gpu5_t062_localjudge_stream_smoke_20260624_005017/logs/claw_eval.log`; server log `/data2/haojitai/outputs/Sparse-vLLM/claw-eval/qwen3_4b_vanilla_gpu5_t062_localjudge_stream_smoke_20260624_005017/logs/sparsevllm_openai_server.log`
- Results: `avg_score=0.200`, `pass^1=0/1`, `pass@1=0/1`, `errored=0/1`; model tokens `806` (`643` input / `163` output); wall time `7.01s`; result file `/data2/haojitai/outputs/Sparse-vLLM/claw-eval/qwen3_4b_vanilla_gpu5_t062_localjudge_stream_smoke_20260624_005017/traces/qwen3-4b-vanilla_26-06-24-00-50/batch_results.json`.
- Notes: request logs contain two successful calls. The evaluated agent call had `stream=true` and tool schemas; the judge call was non-streaming. No server traceback/error was found in the server log.

## 2026-06-24 00:53 CST - qwen3_4b_vanilla_gpu4_7_tp4_t062_localjudge_stream_smoke_20260624_005350

- Status: aborted
- Goal: Check whether a 4-GPU tensor-parallel Qwen3-4B Claw-Eval smoke could replace the single-card run.
- Working dir: `/home/haojitai/projects/Sparse-vLLM`
- Command:

```bash
OPENROUTER_API_KEY=dummy \
RUN_NAME=qwen3_4b_vanilla_gpu4_7_tp4_t062_localjudge_stream_smoke_20260624_005350 \
MODEL_PATH=/data2/haojitai/models/Qwen3-4B-Instruct-2507 \
SERVED_MODEL_NAME=qwen3-4b-vanilla \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
SERVER_PORT=18007 \
SPARSEVLLM_MASTER_PORT=24347 \
SPARSEVLLM_CONTEXT_WINDOW=32768 \
CLAW_EVAL_JUDGE_BASE_URL=http://127.0.0.1:18007/v1 \
CLAW_EVAL_JUDGE_MODEL=qwen3-4b-vanilla \
ENGINE_KWARGS='{"tensor_parallel_size":4,"gpu_memory_utilization":0.80,"max_model_len":32768,"engine_prefill_chunk_size":2048,"sparse_method":"vanilla"}' \
CLAW_EVAL_ARGS='batch --config ${CLAW_EVAL_CONFIG} --filter T062 --trials 1 --parallel 1 --trace-dir ${TRACE_DIR}' \
bash benchmark/claw_eval/run_sparsevllm_claw_eval.sh
```

- Code: `codex/import-deltakv-main` / `349002b`; worktree had relevant uncommitted Claw-Eval shim, runner, config-template, and experiment-doc changes.
- Environment: host `guest-KR6288-X2-A0-R0-00`; GPUs `4,5,6,7`; `SPARSEVLLM_MASTER_PORT=24347`; HTTP server `127.0.0.1:18007`.
- Model: `/data2/haojitai/models/Qwen3-4B-Instruct-2507`; Sparse-vLLM `sparse_method=vanilla`; `tensor_parallel_size=4`.
- Data: Claw-Eval task `T062_finance_pltr_cagr`; 1 trial; non-sandbox; judge routed to the same local Sparse-vLLM server.
- Logs: `/data2/haojitai/outputs/Sparse-vLLM/claw-eval/qwen3_4b_vanilla_gpu4_7_tp4_t062_localjudge_stream_smoke_20260624_005350/logs/sparsevllm_openai_server.log`; request logs in `/data2/haojitai/outputs/Sparse-vLLM/claw-eval/qwen3_4b_vanilla_gpu4_7_tp4_t062_localjudge_stream_smoke_20260624_005350/sparsevllm_requests`.
- Results: no Claw-Eval score. The run was interrupted after repeated model failures.
- Notes: TP workers crashed in decode with `TypeError: 'NoneType' object is not subscriptable` from `decode_cuda_graph_runner.run_eager_static`, followed by repeated `TimeoutError("Timed out waiting for TP worker to read shared-memory RPC 'run' after 30.0s.")` request logs. GPU4-7 were released after abort. Per the follow-up decision, continue with single-card runs for now.
