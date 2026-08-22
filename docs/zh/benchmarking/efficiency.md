# 效率与吞吐性能基准套件

[English](../../en/benchmarking/efficiency.md) | 简体中文

## 用途

效率套件使用匹配的 synthetic request trace，对比 Sparse-vLLM 方法与 vLLM
全注意力基线。覆盖的维度包括 prompt length、并发度、固定 batch、超额请求
churn、tensor parallelism、请求延迟、吞吐、GPU 活动率和峰值显存。

Synthetic workload 使用确定性的随机 token ID。相同 seed 和 case 下，不同系统
收到完全相同的 trace；每个实测 iteration 使用新的 trace；当并发度大于 1 时，
batch 内 prompt 长度不完全相同；prefix caching 固定关闭。churn 场景提交的请求数
大于最大并发数，因此会包含序列替换和 scheduler 行为。

先用 synthetic suite 定位可疑 setting；只有发现可疑 case 后，才使用单独的
Nsight 诊断。

## 前置条件

所有命令都从仓库根目录运行。开始前确认：

- 当前 Python 环境已经安装 Sparse-vLLM 和 benchmark 依赖。
- 请求 `vllm-vanilla` 时，同一个 `PYTHON_BIN` 能够导入 `vllm`。
- `nvidia-smi` 可用，所选物理 GPU 全部空闲。
- 模型路径或 Hugging Face model ID 可访问。
- 输出盘有足够空间保存 JSONL trace 和硬件 timeline。
- 如需直接使用下文的 JSON 验证命令，环境中需要有 `jq`。

建议先检查：

```bash
cd "<SPARSE_VLLM_REPO>"

PYTHON_BIN=python3
"${PYTHON_BIN}" -c 'import sparsevllm'
"${PYTHON_BIN}" -c 'import vllm'  # 仅运行 vLLM baseline 时需要。
nvidia-smi
```

Shell wrapper 会拒绝已有计算进程的 GPU。独立 Python CLI 不执行空卡预检，
使用它之前必须手动检查设备。

## 快速 Smoke Test

下面的小型 fixed-batch run 用 Qwen3-30B、TP=2 验证模型加载、推理、硬件采样
和 artifact 生成。它不是有代表性的吞吐结果。

```bash
cd "<SPARSE_VLLM_REPO>"

PROMPT_LENS=4096 \
OUTPUT_LENS=32 \
BATCH_SIZES=1 \
BENCH_SCENARIO=fixed \
NUM_WARMUPS=0 \
NUM_ITERS=1 \
SPARSEVLLM_OUTPUT_DIR="<OUTPUT_ROOT>" \
bash scripts/benchmarks/run_efficiency_probe.sh \
  "svllm-vanilla" \
  "qwen3_30b" \
  "0,1"
```

Wrapper 会在 `<OUTPUT_ROOT>` 下创建带时间戳的目录。成功的 run 包含
`svllm-vanilla/run_status.json`，其中 `status` 为 `success`。

## 跨系统匹配 Sweep

下面的命令在不同 prompt length 和并发度下对比 H2O、SnapKV、Sparse-vLLM
全注意力和 vLLM 全注意力。`BENCH_SCENARIO=all` 同时运行 fixed batch 和
oversubscribed churn。

```bash
cd "<SPARSE_VLLM_REPO>"

PROMPT_LENS="8192,16384,32768" \
OUTPUT_LENS=512 \
BATCH_SIZES="1,4,8" \
BENCH_SCENARIO=all \
BENCH_SEED=42 \
NUM_WARMUPS=1 \
NUM_ITERS=3 \
MAX_NUM_BATCHED_TOKENS=8192 \
SPARSE_PREFILL_SCORE_MODE=probability \
SPARSEVLLM_OUTPUT_DIR="<OUTPUT_ROOT>" \
bash scripts/benchmarks/run_efficiency_probe.sh \
  "svllm-h2o,svllm-snapkv,svllm-vanilla,vllm-vanilla" \
  "qwen3_30b" \
  "0,1"
```

这是一次较重的运行：每个系统都会执行所有 prompt length、output length、
并发度、场景、warmup 和实测 iteration 的组合。先运行 smoke test 或单个 prompt
length，再启动完整矩阵。512-token output 更适合有代表性的 decode 指标；更短的
output 只适合功能验证或受时间限制的探索。

Wrapper 支持以下系统名：

| 名称 | Engine | 方法 |
| --- | --- | --- |
| `svllm-vanilla` | Sparse-vLLM | `vanilla` |
| `svllm-h2o` | Sparse-vLLM | `h2o` |
| `svllm-snapkv` | Sparse-vLLM | `snapkv` |
| `svllm-omnikv` | Sparse-vLLM | `omnikv` |
| `svllm-deltakv` | Sparse-vLLM | `deltakv` |
| `vllm-vanilla` 或 `vllm` | vLLM | `vanilla` |

Probe wrapper 支持以下模型别名：

| 别名 | 默认模型 | Tensor parallel size |
| --- | --- | --- |
| `qwen3_30b` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 2 |
| `qwen3_8b` | `Qwen/Qwen3-8B` | 1 |
| `qwen25_7b` | `Qwen/Qwen2.5-7B-Instruct-1M` | 1 |

稀疏方法参数统一从 `benchmark/sparsevllm_regression/manifest.json` 解析，
其中包括 OmniKV 的模型专用 full-attention 层。若模型没有经过校准的
manifest 配置，OmniKV 会在加载模型前直接失败。已校准的自定义模型可通过
`BENCH_MANIFEST_MODEL_ID` 指定 manifest 条目；一次性的外部校准结果可通过
`OMNIKV_FULL_ATTENTION_LAYERS` 显式传入。单层 OmniKV 配置默认会被拒绝，
只有 standalone Python runner 的显式消融开关可以放行。

可以用 `MODEL_PATH` 覆盖别名对应的 model ID。当前 probe wrapper 按别名固定
TP，并对任意自定义模型路径使用 TP=2。需要显式测试不同 TP 时，使用独立 CLI。

## Tensor-Parallel Sweep

对每个系统和 TP setting 分别调用一次 `benchmark/efficiency/bench_probe.py`。
不同系统必须保持 seed、prompt/output length、并发度、scheduler token budget、
jitter、warmup 和 iteration 完全相同。每个 run 使用独立输出目录。

TP=1 Sparse-vLLM H2O 示例：

```bash
cd "<SPARSE_VLLM_REPO>"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD:$PWD/src" python3 \
  benchmark/efficiency/bench_probe.py \
  --engine sparsevllm \
  --sparse-method h2o \
  --model-path "<MODEL_PATH>" \
  --prompt-lens 8192,16384 \
  --output-lens 512 \
  --batch-sizes 1,4,8 \
  --scenario all \
  --seed 42 \
  --tensor-parallel-size 1 \
  --max-num-batched-tokens 8192 \
  --monitor-gpus 0 \
  --output-dir "<OUTPUT_ROOT>/tp1-svllm-h2o"
```

TP=2 vLLM baseline 示例：

```bash
cd "<SPARSE_VLLM_REPO>"

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH="$PWD:$PWD/src" python3 \
  benchmark/efficiency/bench_probe.py \
  --engine vllm \
  --sparse-method vanilla \
  --model-path "<MODEL_PATH>" \
  --prompt-lens 8192,16384 \
  --output-lens 512 \
  --batch-sizes 1,4,8 \
  --scenario all \
  --seed 42 \
  --tensor-parallel-size 2 \
  --max-num-batched-tokens 8192 \
  --monitor-gpus 0,1 \
  --output-dir "<OUTPUT_ROOT>/tp2-vllm-vanilla"
```

`--monitor-gpus` 使用物理 GPU ID。选择非零物理设备时，必须与
`CUDA_VISIBLE_DEVICES` 对齐；例如同时使用 `CUDA_VISIBLE_DEVICES=6,7` 和
`--monitor-gpus 6,7`。

完整 CLI 参见 `python3 benchmark/efficiency/bench_probe.py --help`。

## Unified Synthetic 与 LongBench 套件

`run_unified_efficiency_suite.sh` 先运行匹配的 synthetic suite，再运行匹配的
LongBench lifecycle workload。它会验证 stage 状态、synthetic trace 一致性、
prefix-cache policy、逐请求状态、硬件采样、LongBench 样本数量，以及各系统间
source ID coverage。

LongBench 数据默认位于 `data/LongBench`。其他路径通过
`SPARSEVLLM_LONGBENCH_DATA_DIR` 指定。

```bash
cd "<SPARSE_VLLM_REPO>"

SPARSEVLLM_LONGBENCH_DATA_DIR="<LONGBENCH_ROOT>" \
LONGBENCH_SAMPLES=10 \
PROMPT_LENS="8192,16384,32768" \
OUTPUT_LENS=512 \
BATCH_SIZES="1,4,8" \
SPARSEVLLM_OUTPUT_DIR="<OUTPUT_ROOT>" \
bash scripts/benchmarks/run_unified_efficiency_suite.sh \
  "0,1" \
  "svllm-h2o,svllm-snapkv,svllm-vanilla,vllm-vanilla" \
  "qwen3_30b"
```

注意它与 `run_efficiency_probe.sh` 的位置参数顺序不同：unified runner 依次接收
`GPUS`、`SYSTEMS`、`MODEL_NAME`。最终 validator 在带时间戳的 run root 写入
`suite_status.json`；验证失败时进程返回非零退出码。

## 主 Wrapper 参数

`run_efficiency_probe.sh` 接收三个位置参数：

```text
run_efficiency_probe.sh SYSTEMS MODEL_NAME_OR_PATH PHYSICAL_GPU_IDS
```

主要环境变量如下：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `PYTHON_BIN` | `python3` | 所有被测系统共用的 Python executable。 |
| `MODEL_PATH` | 由别名决定 | 覆盖已知别名解析出的模型。 |
| `SPARSEVLLM_OUTPUT_DIR` | `outputs` | 输出根目录；wrapper 会增加带时间戳的 run 目录。 |
| `PROMPT_LENS` | `8192,16384,32768` | 请求的最大 prompt length。 |
| `OUTPUT_LENS` | `512` | 请求的生成 token 数。 |
| `BATCH_SIZES` | `1,4,8` | Fixed-batch size 和 churn 最大并发度阶梯。 |
| `BENCH_SCENARIO` | `all` | `fixed`、`churn` 或 `all`。 |
| `BENCH_SEED` | `42` | 各 engine 共用的 base seed。 |
| `PROMPT_LENGTH_JITTER` | `0.10` | 从每个 prompt 上限向下生成变长序列的比例。 |
| `OUTPUT_LENGTH_JITTER` | `0.25` | churn output length 向下抖动的比例。 |
| `CHURN_REQUEST_MULTIPLIER` | `4` | Churn request count 与最大并发度的倍数。 |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | 两个 engine 匹配的 scheduler token budget。 |
| `NUM_WARMUPS` | `1` | 每个 synthetic case 的 warmup iteration。 |
| `NUM_ITERS` | `3` | 每个 synthetic case 的实测 iteration。 |
| `SPARSE_PREFILL_SCORE_MODE` | `probability` | SnapKV prefill score mode：`probability` 或 `logits`。 |
| `CUDA_HOME` | `/usr/local/cuda-13.0` | Probe wrapper 使用的 CUDA toolkit 根目录。 |

该套件始终关闭 prefix caching；这个入口不提供 prefix-caching benchmark 模式。

## 指标与解释

- Request、input-token、output-token 和 total-token throughput 由实测请求的直接
  计时计算。
- Observed sweep saturation 只相对于当前并发度阶梯中的最佳 output throughput，
  不能证明已经达到绝对硬件饱和。
- GPU compute activity 和 memory I/O activity 直接来自 `nvidia-smi` 采样，不是
  理论 MFU/MBU、achieved FLOP/s 或 achieved HBM GB/s。
- Coarse active duty 是 GPU utilization 大于 10% 的采样比例。它的补集不能把
  idle time 归因到 CPU scheduling 或 kernel launch。
- TTFT 和 TPOT 是端到端请求 wall-clock 指标，包含 host scheduling、同步和
  engine overhead。
- Churn 指标将 oversubscribed workload 与匹配的 fixed-batch setting 比较，包括
  throughput ratio 和 tail-TTFT 变化。

只有模型 checkpoint、benchmark trace/metric contract、engine config、TP、seed、
长度、scheduler token budget、warmup 和 iteration count 一致时，才能把结果视为
matched comparison。

成功的 vLLM sweep 应作为不可变 baseline artifact 保存。后续 Sparse-vLLM
改动直接复用该 baseline，不要每次重跑 vLLM。只有 GPU 型号、checkpoint、TP、
request trace、scheduler budget、graph/backend policy 或 metric contract 变化时才
生成新 baseline；package 版本只记录 provenance，不能静默覆盖旧 baseline。

## Artifact 与验证

每个独立 CLI 或 wrapper 的逐系统输出包含：

| Artifact | 含义 |
| --- | --- |
| `run_manifest.json` | 命令、Git 状态、参数、package/GPU 环境、模型元数据、workload contract 和最终状态。 |
| `run_status.json` | 终态成功或失败状态。 |
| `raw_samples.jsonl` | 每个实测 synthetic iteration 一条记录。 |
| `request_samples.jsonl` | 逐请求 trace 元数据和状态。 |
| `summary.json` | 聚合结果和终态状态。 |
| `comparison_report.md` | 便于阅读的指标表。 |
| `case_hardware/*.json` | 每个 case 的 GPU 采样 timeline 和 summary。 |
| `operator_runtime_stats.json` | Sparse fixed-batch 的 Provider 绑定、拒绝原因与实际 kernel path。 |

不能只根据终端输出声明 run 完成。至少检查：

```bash
jq -e '.status == "success"' "<RUN_DIR>/run_status.json"
jq -e '.status == "success"' "<RUN_DIR>/summary.json"
jq -e '.status == "success"' "<RUN_DIR>/run_manifest.json"
test -s "<RUN_DIR>/raw_samples.jsonl"
test -s "<RUN_DIR>/request_samples.jsonl"
```

Unified run 还必须检查：

```bash
jq -e '.status == "success"' "<UNIFIED_RUN_ROOT>/suite_status.json"
```

Probe 拒绝向已经包含 benchmark artifact 的目录写入。请选择新的输出目录，不要
混合或覆盖不同 run。

## Nsight 诊断

只有标准 sweep 定位到可疑 case 后才使用 profiling wrapper。当前支持
`svllm-vanilla`、`svllm-snapkv` 和 `vllm-vanilla`，并使用 probe CLI 默认的
TP=2。

```bash
cd "<SPARSE_VLLM_REPO>"

PROMPT_LEN=16384 \
OUTPUT_LEN=512 \
CONCURRENCY=8 \
SPARSEVLLM_OUTPUT_DIR="<OUTPUT_ROOT>" \
bash scripts/benchmarks/run_efficiency_profile.sh \
  svllm-vanilla \
  "<MODEL_PATH>" \
  "0,1"
```

该命令要求安装 Nsight Systems（`nsys`）并具有 NVIDIA performance counter
权限。counter 不可用时会直接失败，不会用估算硬件指标替代。主要输出为
`timeline.nsys-rep`。

## 故障排查

| 现象 | 检查方法 |
| --- | --- |
| Wrapper 报告 GPU busy | 等待列出的 PID 结束，或选择其他空闲物理 GPU ID。 |
| `sparsevllm` 或 `vllm` 出现 `ModuleNotFoundError` | 将 `PYTHON_BIN` 指向能导入所有被测 engine 的环境；source checkout 可设置 `PYTHONPATH="$PWD:$PWD/src"`。 |
| 无法加载模型配置 | 检查 `<MODEL_PATH>/config.json`，或确认 Hugging Face model ID 可访问。 |
| 长 context 或高并发 OOM | 减小 `BATCH_SIZES` 或 `PROMPT_LENS`；记录修改后的矩阵，不要静默删除失败行。 |
| 输出目录冲突 | 使用新的 `--output-dir` 或 output root，不要追加到旧 run。 |
| Hardware metric 状态为 `metric_failed` | 检查逐 case hardware JSON 是否存在缺失或失败的 `nvidia-smi` sample。 |
| Nsight 报告权限不足 | 在 host 开启 NVIDIA performance counter，或跳过诊断；不能把 coarse activity 重新解释为 hardware-counter 数据。 |
