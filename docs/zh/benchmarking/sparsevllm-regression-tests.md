# SparseVLLM 回归测试

## 目的

本文说明如何运行 `benchmark/sparsevllm_regression/` 下固定的 SparseVLLM regression harness。

harness 用于对以下层进行可复现的方法/模型检查：

- `quality`：LongBench-mini generation quality。
- `logits`：HF reference 与 SparseVLLM logits alignment。
- `perf`：prefill/decode throughput 和 memory accounting。
- `stress`：固定长度、高并发的 SparseVLLM admission/decode stress。
- `stress_v2`：带 shared-prefix 和 multi-turn workload、可变 prompt length，并对支持方法验证 prefix-cache hit 的 synthetic serving trace stress。
- `validate`：manifest 和 output artifact validation。

test plan 由 `benchmark/sparsevllm_regression/manifest.json` 控制。

## 前置条件

为运行 suite 的机器配置以下路径：

- Working directory：`<REPO_ROOT>`
- Conda env：`<CONDA_ENV>`
- Output root：`<OUTPUT_ROOT>`
- LongBench data：`<LONGBENCH_ROOT>`
- 模型：
  - `<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M`
  - `<MODEL_ROOT>/Qwen3-4B-Instruct-2507`
  - `<MODEL_ROOT>/Llama-3.1-8B-Instruct`
- Compressor checkpoint：
  - `<CHECKPOINT_ROOT>/Qwen2.5-7B-Instruct-1M-Compressor`
  - `<CHECKPOINT_ROOT>/Qwen3-4B-Instruct-2507-Compressor`
  - `<CHECKPOINT_ROOT>/Llama-3.1-8B-Instruct-Compressor`

运行 suite 前设置环境：

```bash
cd <REPO_ROOT>

export DELTAKV_OUTPUT_DIR=<OUTPUT_ROOT>
export DELTAKV_LONGBENCH_DATA_DIR=<LONGBENCH_ROOT>

export DELTAKV_MODEL_QWEN25_7B=<MODEL_ROOT>/Qwen2.5-7B-Instruct-1M
export DELTAKV_MODEL_QWEN3_4B=<MODEL_ROOT>/Qwen3-4B-Instruct-2507
export DELTAKV_MODEL_LLAMA31_8B=<MODEL_ROOT>/Llama-3.1-8B-Instruct

export DELTAKV_COMPRESSOR_QWEN25_7B=<CHECKPOINT_ROOT>/Qwen2.5-7B-Instruct-1M-Compressor
export DELTAKV_COMPRESSOR_QWEN3_4B=<CHECKPOINT_ROOT>/Qwen3-4B-Instruct-2507-Compressor
export DELTAKV_COMPRESSOR_LLAMA31_8B=<CHECKPOINT_ROOT>/Llama-3.1-8B-Instruct-Compressor

export PYTHONPATH=<REPO_ROOT>:<REPO_ROOT>/src:${PYTHONPATH:-}
```

manifest 还包含 `qwen25_32b`；除非 GPU memory 足够且设置了相应 model/checkpoint 环境变量，否则省略它。

## 快速 Unit Test

运行保护 regression harness、grading、manifest policy 和 OmniKV full-layer selector 的 unit test：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python -m unittest \
  tests.test_sparsevllm_regression_grading \
  tests.test_omnikv_full_layer_selector \
  -v
```

当前 harness 的预期结果：所有 test 通过。

## Manifest Validation

长 GPU run 前使用 `validate`。它解析 runtime path，写入 resolved manifest，并创建空的必需 artifact 文件。

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer validate \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id validate_omnikv_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

如果缺少 model/checkpoint path 时应使 run 失败，而不是记录为 skipped，请使用 `--no-allow_skipped_policy`。

## 常用运行命令

所有命令写入：

```text
<OUTPUT_ROOT>/sparsevllm_regression/<run_id>/
```

### Quality

Quality 使用 LongBench-mini：

- task：`qasper,hotpotqa,multi_news,trec,passage_retrieval_en,lcc`
- LongBench batch size：`100`
- SparseVLLM `max_num_seqs_in_batch`：`16`
- SparseVLLM `max_decoding_seqs`：`16`
- 每个 task 的 sample：`50`

将 OmniKV 与 vanilla baseline 对比：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,omnikv \
  --run_id omnikv_quality_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

运行不含 32B 的完整 quality：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,quest,deltakv,deltakv-less-memory \
  --run_id quality_3models_all_methods_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

TP decode CUDA Graph v1 quality validation 使用默认 LongBench data-worker parallelism，并通过 regression-suite override 传入 engine TP：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer quality \
  --models qwen25_7b \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,rkv,skipkv \
  --tensor_parallel_size 2 \
  --run_id tp2_graph_quality_v1_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

该 run 内 sparse method 与 TP vanilla 对比。记录 A/B/C grade；crash 或 D grade 使 TP graph quality gate 失败。

快速 TP prefix-cache + decode-graph regression gate 保持相同方法覆盖，但使用显式小 sample override 和 child-command timeout。它仍覆盖 LongBench quality、SCBench quality 和 stress，但属于 smoke/regression gate，不是完整的每 task 50 sample quality suite：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer quality \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --tensor_parallel_size 2 \
  --enable_prefix_caching \
  --prefix_cache_block_size 16 \
  --quality_tasks qasper,hotpotqa \
  --quality_batch_size 2 \
  --quality_samples_per_task 2 \
  --quality_min_required_samples 2 \
  --quality_sparsevllm_max_num_seqs_in_batch 2 \
  --quality_sparsevllm_max_decoding_seqs 2 \
  --command_timeout_s 600 \
  --run_id tp_prefix_graph_quality_quick_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

然后运行匹配的 SCBench quality 和 prefix-hit stress layer：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer scbench \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --tensor_parallel_size 2 \
  --enable_prefix_caching \
  --prefix_cache_block_size 16 \
  --scbench_decode_cuda_graph \
  --scbench_tasks scbench_kv \
  --scbench_num_eval_examples 1 \
  --scbench_max_turns 2 \
  --scbench_max_seq_length 1024 \
  --scbench_batch_size 1 \
  --command_timeout_s 600 \
  --run_id tp_prefix_graph_scbench_quick_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>

conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer stress \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --tensor_parallel_size 2 \
  --enable_prefix_caching \
  --prefix_cache_block_size 16 \
  --require_prefix_cache_hit \
  --stress_length 256 \
  --stress_request_counts 2 \
  --stress_output_len 2 \
  --stress_max_num_seqs_in_batch 2 \
  --stress_max_decoding_seqs 2 \
  --stress_max_decode_steps_after_full 1 \
  --stress_admission_wave_size 1 \
  --stress_wave_decode_gap_steps 1 \
  --command_timeout_s 600 \
  --run_id tp_prefix_graph_stress_quick_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

### Correctness / Logits

`logits` 对声明 `hf_logits_reference=true` 的方法比较 HF sparse reference output 与 SparseVLLM。没有 HF reference 的方法按 policy 评为 `N/A`。

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer logits \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_logits_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

### Performance

Performance 使用：

- prompt length：`16000,64000`
- batch size：`1,4`
- output token：`256`
- 方法支持时请求 decode CUDA Graph

对于 sparse method，benchmark 还会以相同 shape 运行 vanilla，以便 suite 计算 decode speedup。

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer perf \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_perf_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

TP decode CUDA Graph v1 performance validation：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer perf \
  --models qwen25_7b \
  --methods vanilla,streamingllm,snapkv,pyramidkv,omnikv,rkv,skipkv \
  --tensor_parallel_size 2 \
  --run_id tp2_graph_perf_v1_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

检查 `perf.jsonl` 中的 `decode_cuda_graph_expected=true` 和 `decode_cuda_graph_active=true`。

### Stress

Stress 当前使用：

- prompt length：`16000`
- request count / batch size：`80`
- output token：`64`
- `max_num_seqs_in_batch=80`
- `max_decoding_seqs=80`
- full admission 后最大 decode step：`32`

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer stress \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods omnikv \
  --run_id omnikv_stress80_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

### Stress V2

`stress_v2` 使用 `scripts/benchmarks/bench_prefix_cache.py` 作为 serving-like trace 的 regression layer。与固定 `stress` 不同，它运行有 seed 的 synthetic request：

- workload：`shared_prefix,multiturn`
- 支持的方法：`vanilla`、`omnikv`、`quest`
- `vanilla` case：`baseline_full,prefix_full`
- `omnikv` case：`prefix_omnikv`
- `quest` case：`prefix_quest`
- session / turn：`8 / 4`
- shared-prefix request：`8`
- output token：`64`
- 最大 active request：`8`
- 可变 multi-turn user length：`128..1024`
- 可变 session-prefix length：`1024..4096`
- 可变 shared suffix length：`512..4096`
- 最大 prompt length：multi-turn 约 `16.5k` token，shared-prefix 约 `12.3k`
- prefix-cache block size：`16`

启用 prefix cache 的 case 未观察到 cache hit，或实际 prompt length 没有变化时，gate 失败。不支持的方法记录为 `skipped_by_policy`，因为该 layer 专门验证 prefix-cache serving 行为。

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer stress_v2 \
  --models qwen3_4b \
  --methods vanilla,omnikv,quest \
  --run_id stress_v2_qwen3_serving_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

### 组合 Layer

`nightly` 运行 quality、logits 和 performance，不运行 stress。

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python benchmark/sparsevllm_regression/run_suite.py \
  --layer nightly \
  --models qwen25_7b,qwen3_4b,llama31_8b \
  --methods vanilla,omnikv \
  --run_id nightly_omnikv_$(date -u +%Y%m%d_%H%M%S) \
  --output_root <OUTPUT_ROOT>
```

`pre-refactor` 运行 quality、logits、performance 和 stress。

## 结果记录

将此文件作为稳定 regression runbook。不要在此添加按时间排列的实验记录或本地 result index。面向仓库的结果声明需要证据时，直接引用原始 run artifact path。

## OmniKV Full-Layer Selection

OmniKV full layer 与模型相关。发布新模型的 OmniKV 或与 OmniKV 对齐的 DeltaKV regression 数值前，使用 `scripts/analysis/select_omnikv_full_layers.py`。

selector 在 LongBench task 上运行离线 decode-attention coverage calibration，选择 `--num-full-layers` 个 layer，并把选中的 layer 字符串写入 `selected_full_layers.json`。这不是 online runtime mode：必须将该字符串作为 `full_attention_layers` 传回。

Qwen2.5-7B 选择六个 full layer 的示例：

```bash
conda run -n <CONDA_ENV> --no-capture-output \
  python scripts/analysis/select_omnikv_full_layers.py \
  --model-path <MODEL_ROOT>/Qwen2.5-7B-Instruct-1M \
  --longbench-root <LONGBENCH_ROOT> \
  --config-dir benchmark/long_bench/config \
  --dataset narrativeqa \
  --output-dir <OUTPUT_ROOT>/omnikv_full_layer_calibration_$(date -u +%Y%m%d)/qwen25_7b_full6 \
  --num-full-layers 6 \
  --num-samples 32 \
  --topk 2048 \
  --random-decode-points-per-sample 8 \
  --num-sink-tokens 0 \
  --num-recent-tokens 32 \
  --prefill-chunk-size 512 \
  --torch-dtype bfloat16 \
  --device cuda
```

主要输出：

- `selected_full_layers.json`：选中的 layer ID 和 runtime config 使用的 `full_attention_layers` 字符串。
- `per_sample_points.jsonl`：calibration 使用的 sampled decode point。
- `pair_scores.npy` 和 `segment_scores.npy`：用于审计的 raw coverage matrix。
- `run_info.json`：命令、Git state、model/data path 和 calibration setting。
- `top128_kl_metrics.json`：使用 `--top128-kl-only` 运行时的可选 validation output。

在临时 Sparse-VLLM run 中使用选中 layer 时，将 `full_attention_layers` 值复制到 `--hyper_params`：

```bash
PYTHONPATH=$PWD:$PWD/src python scripts/benchmarks/bench_sparse_vllm.py \
  --model_path <MODEL_DIR> \
  --methods omnikv \
  --lengths 131072 \
  --batch_sizes 4 \
  --output_len 128 \
  --hyper_params '{"sparse_method":"omnikv","full_attention_layers":"0,2,4,11,16,22","decode_keep_tokens":4096,"recent_keep_tokens":32,"sink_keep_tokens":0,"engine_prefill_chunk_size":512}'
```

对于 regression run，更新 `benchmark/sparsevllm_regression/manifest.json` 中的 `methods.omnikv.model_configs`。如果 DeltaKV regression config 有意与 OmniKV observation/full layer 对齐，在同一 manifest 中更新匹配的 DeltaKV model config，并在 run summary 中记录。当前 manifest 使用：

```text
qwen25_7b:  0,2,4,11,16,22
qwen3_4b:   0,1,3,9,13,16,21,28
llama31_8b: 0,2,7,13,16,26
```

更改这些 layer 后，运行 `validate`，并重新运行 OmniKV quality/logits/perf/stress。

## 输出

每次 run 写入：

- `resolved_manifest.json`：环境变量解析后的 manifest。
- `grade_summary.json`：command record、grade 和 final status。
- `metrics.json`：quality aggregate record。
- `logits_alignment.json`：logits comparison summary。
- `perf.jsonl`：展平的 performance row。
- `memory.json`：根据 performance row 得到的 memory grade。
- `stress.json`：stress row 和 stress grade。
- `stress_v2.json`：serving-trace stress row 和 stress_v2 grade。
- `raw_outputs.jsonl`、`parsed_outputs.jsonl`、`sample_results.jsonl`：运行 quality 时的 generation artifact。
- Layer-specific log：
  - `quality/<model>/<method>/run.log`
  - `logits/<model>/<method>/run.log`
  - `perf/<model>/<method>.log`
  - `stress/<model>/<method>.log`

快速 summary 命令：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("<OUTPUT_ROOT>/sparsevllm_regression/<run_id>")
data = json.loads((root / "grade_summary.json").read_text())
print("status:", data["status"])
print("worst_required_grade:", data.get("worst_required_grade"))
for grade in data.get("grades", []):
    print(grade.get("model"), grade.get("method"), grade["name"], grade["grade"], grade["status"], grade["metrics"])
PY
```

## Regression Rubric

可执行 gate rule 位于 `benchmark/sparsevllm_regression/grading.py`。稳定的人类可读 rubric 位于 `benchmark/sparsevllm_regression/rubrics.md`。

只有在稳定 ABCD rubric 定义改变时，才更新 `benchmark/sparsevllm_regression/rubrics.md`。不要向 rubric 文件添加带日期的 campaign result、open blocker、run ID 或 remote log path。

## 故障排查

- 缺少 model 或 compressor path：
  - 运行 `validate`。
  - 检查 `resolved_manifest.json`。
  - 如果缺少 path 时 run 应失败，传入 `--no-allow_skipped_policy`。
- Import error：
  - 确保 `PYTHONPATH=<REPO_ROOT>:<REPO_ROOT>/src:${PYTHONPATH:-}`。
  - 使用包含[快速开始](../getting_started/README.md)所列依赖的环境。
- Quality dataset error：
  - 设置 `DELTAKV_LONGBENCH_DATA_DIR=<LONGBENCH_ROOT>`。
- GPU memory failure：
  - 不要在 harness 内添加 fallback。
  - 在 issue note 中记录准确 run ID、model、method、layer、log path 和 error。
- 命令提前退出：
  - 检查 `<run_id>/grade_summary.json`；failed command 会记录 `returncode`、`cmd` 和 `log_path`。
  - 检查 command record 中对应 layer-specific log path。
