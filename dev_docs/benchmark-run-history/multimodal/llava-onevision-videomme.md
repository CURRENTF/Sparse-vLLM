# LLaVA-OneVision Video-MME Evaluation

This adapter evaluates LLaVA-OneVision vanilla and no-checkpoint DeltaKV
delta-quant paths on Video-MME multiple-choice video QA.

Primary script:

```bash
benchmark/multimodal/video_qa/videomme.py
```

Dataset source:

- Hugging Face dataset: `lmms-lab/Video-MME`
- Official project/code: <https://github.com/MME-Benchmarks/Video-MME>

Video-MME contains 900 videos and 2700 QA pairs. The Hugging Face mirror stores
one annotation parquet, `subtitle.zip`, and 20 video zip shards.

## Download

The local default root is:

```text
<DATA_ROOT>/Video-MME_hf
```

Download metadata only:

```bash
VIDEOMME_DOWNLOAD_SCOPE=metadata \
bash scripts/data/download_videomme_full.sh
```

Download the full dataset in the background:

```bash
VIDEOMME_ROOT=<DATA_ROOT>/Video-MME_hf \
HF_MAX_WORKERS=1 \
PROXY_URL=http://localhost:7890 \
bash scripts/data/tmux_download_videomme_full.sh
```

The script uses `hf download`, keeps Hugging Face cache under
`<HF_CACHE_ROOT>`, and unzips `subtitle.zip` plus `videos_chunked_*.zip` after
the download finishes.

## Dry Run

Validate annotation parsing without requiring videos:

```bash
PYTHONPATH=$PWD/src \
python benchmark/multimodal/video_qa/videomme.py \
  --dataset_dir <DATA_ROOT>/Video-MME_hf \
  --output_dir <DATA_ROOT>/llava_onevision_videomme_dry_run \
  --dry_run_metadata \
  --num_samples 5
```

This writes `videomme_metadata_dry_run.json` and reports duration, domain,
subcategory, and task-type counts.

## Smoke Command

After at least one video shard has been downloaded and unzipped:

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD/src \
python benchmark/multimodal/video_qa/videomme.py \
  --model_path <MODEL_ROOT>/llava-onevision-qwen2-0.5b-ov-hf \
  --dataset_dir <DATA_ROOT>/Video-MME_hf \
  --output_dir <DATA_ROOT>/llava_onevision_videomme_smoke \
  --methods vanilla,deltakv_delta_quant \
  --num_samples 8 \
  --batch_size 1 \
  --num_video_frames 32 \
  --frame_sampling_backend decord \
  --allow_missing_videos
```

`--allow_missing_videos` is only for shard-level smoke tests while the full
download is incomplete. Full benchmark runs should omit it so missing videos
fail fast.

## Outputs

The script follows the research-code reliability format used by the
StreamingBench adapter:

- `<method>_raw_outputs.jsonl`
- `<method>_parsed_outputs.jsonl`
- `<method>_per_sample_results.jsonl`
- `<method>_aggregate_metrics.json`
- `run_info.json`
- `last_videomme_result.json`

Aggregate metrics include overall accuracy plus grouped stats by Video-MME
duration, domain, subcategory, and task type.
