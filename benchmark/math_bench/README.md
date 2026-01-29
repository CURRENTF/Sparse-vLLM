# MathBench (GSM8K + AIME 2024)

This benchmark follows the same inference path as `benchmark/long_bench/pred.py` (via `deltakv.get_chat_api.get_generate_api`), but evaluates **pass@1** only.

## Data format

By default, this runner loads datasets via HuggingFace Datasets:

- GSM8K: `load_dataset('openai/gsm8k', 'main', split='test')` (columns: `question`, `answer`)
- AIME 2024: `load_dataset('Maxwell-Jia/AIME_2024', split='train')` (columns: `Problem`, `Answer`)
- HMMT Nov 2025: `load_dataset('MathArena/hmmt_nov_2025', split='train')` (columns: `problem`, `answer`) (task: `hmmt_nov`)

You can also place local dataset files under `--data_dir` (default: `$DELTAKV_DATA_DIR` or `/root/autodl-fs/datasets`) or pass explicit paths:

- GSM8K: a `.jsonl` / `.json` file containing at least `question` and `answer` (official GSM8K uses `answer` with `#### <final>`).
- AIME 2024: a `.jsonl` / `.json` file containing at least `Problem`/`problem` (or `question`) and `Answer`/`answer` (integer).

## Run

```bash
python benchmark/math_bench/pred.py \
  --model my_model \
  --model_path /path/to/model \
  --compressor_path /path/to/compressor_or_none \
  --task gsm8k,aime2024,hmmt_nov \
  --split test \
  --data_dir /root/autodl-fs/datasets \
  --temperature 0.6 \
  --max_new_tokens 512 \
  --batch_size 1
```

Outputs:

- Predictions: `$DELTAKV_OUTPUT_DIR/benchmark/math_bench/pred/<model>/<compressor>_<time>/`
- Eval result: `result.json` in the output folder, plus `$DELTAKV_OUTPUT_DIR/mathbench_eval.log`

Notes:

- Evaluation extracts the final answer from the last `\boxed{...}` by default. Use `python benchmark/math_bench/eval.py --path ... --allow_unboxed` to enable a fallback that matches the last number when no `\boxed{}` is found.
- This runner uses sampling (`do_sample=True`) and enforces `--temperature` within `[0.5, 0.7]` (recommended `0.6`).
- It enforces outputs to start with `<think>\n` by default; disable with `--no_force_think_prefix`.
