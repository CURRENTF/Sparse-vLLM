# MathBench (GSM8K + AIME 2024 + MATH-500)

This benchmark follows the same inference path as `benchmark/long_bench/pred.py` (via `deltakv.get_chat_api.get_generate_api`), but evaluates **pass@1** only.

## Data format

By default, this runner loads datasets via HuggingFace Datasets:

- GSM8K: `load_dataset('openai/gsm8k', 'main', split='test')` (columns: `question`, `answer`)
- AIME 2024: `load_dataset('Maxwell-Jia/AIME_2024', split='train')` (columns: `Problem`, `Answer`)
- MATH-500: `load_dataset('HuggingFaceH4/MATH-500', split='test')` (columns: `problem`, `answer`) (task: `math500`)
- HMMT Nov 2025: `load_dataset('MathArena/hmmt_nov_2025', split='train')` (columns: `problem`, `answer`) (task: `hmmt_nov`)

You can also place local dataset files under `--data_dir` (default: `$DELTAKV_DATA_DIR` or `/root/autodl-fs/datasets`) or pass explicit paths:

- GSM8K: a `.jsonl` / `.json` file containing at least `question` and `answer` (official GSM8K uses `answer` with `#### <final>`).
- AIME 2024: a `.jsonl` / `.json` file containing at least `Problem`/`problem` (or `question`) and `Answer`/`answer` (integer).
- MATH-500: a `.jsonl` / `.json` file containing at least `problem` and `answer`.

## Run

```bash
python benchmark/math_bench/pred.py \
  --model my_model \
  --model_path /path/to/model \
  --sparse_method deltakv \
  --deltakv_checkpoint_path /path/to/compressor_or_none \
  --task gsm8k,aime2024,math500,hmmt_nov \
  --split test \
  --data_dir /root/autodl-fs/datasets \
  --temperature 0.6 \
  --max_new_tokens 512 \
  --batch_size 1
```

Outputs:

- Predictions: `$DELTAKV_OUTPUT_DIR/benchmark/math_bench/pred/<model>/<compressor>_<time>/`
- Eval result: `result.json` in the output folder, `*_parsed_outputs.jsonl`, `*_per_sample_results.jsonl`, plus `$DELTAKV_OUTPUT_DIR/mathbench_eval.log`

Notes:

- Evaluation uses `math-verify==0.9.0` for answer parsing and equivalence checking. Predictions use `ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)`; MATH-500 gold uses `solution` with `answer` fallback. The previous regex/string-equality scorer has been removed.
- This runner uses sampling (`do_sample=True`) and enforces `--temperature` within `[0.5, 0.7]` (recommended `0.6`).
- It enforces outputs to start with `<think>\n` by default; disable with `--no_force_think_prefix`.
