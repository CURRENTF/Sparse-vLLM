# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import json
import os
import sys
import time
import copy
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

# Where to append scbench_eval.log. Default to repo-local outputs unless overridden.
BASE_PATH = os.environ.get(
    "DELTAKV_OUTPUT_DIR",
    str(Path(__file__).resolve().parents[2] / "outputs"),
)

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

import torch
from deltakv.get_chat_api import get_generate_api
from args import parse_args
from compute_scores import compute_scores
from datasets import load_dataset
from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    GreedySearch,
    GreedySearch_InfLLM,
    GreedySearch_Mamba2,
    GreedySearch_RetrAttn,
    GreedySearch_RetrAttn_Legacy,
    GreedySearch_vLLM,
    DeltaKVGreedySearch,
    check_benchmark_availability,
    create_multiturn_prompt,
    create_scdq_prompt,
    dump_jsonl,
    get_compressed_examples,
    get_ground_truth,
    load_data,
)
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
    MambaForCausalLM,
    Qwen2ForCausalLM,
)
from transformers.cache_utils import SinkCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils.import_utils import _is_package_available

if _is_package_available("vllm"):
    from vllm import LLM, SamplingParams
if _is_package_available("lmcache_vllm"):
    from lmcache_vllm.vllm import LLM as LMCacheLLM
    import lmcache_vllm

import random

try:
    from minference import MInference
except ImportError:
    MInference = None


# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def _shorten_val(v: Any) -> str:
    v = str(v)
    if "/" in v:
        v = os.path.basename(v.rstrip("/"))
    if len(v) > 40:
        v = v[:20] + ".." + v[-15:]
    return v


def _build_result_dir(args, real_model_name: str, scdq_mode: bool) -> tuple[Path, str, str, str]:
    disable_golden_context = "_disable_golden_context" if args.disable_golden_context else ""
    use_scdq = "_scdq" if scdq_mode else "_multi_turn"
    use_llmlingua = "_lingua" if args.use_llmlingua else ""

    verbalize_hyper_param = (
        f"_{'-'.join([f'{k}={v}' for k, v in args.hyper_param.items() if k != 'best_pattern'])}"
        if args.hyper_param
        else ""
    )
    verbalize_hyper_param = _shorten_val(verbalize_hyper_param)

    result_dir = Path(
        args.output_dir,
        f"{real_model_name}_{args.attn_type}{disable_golden_context}_{args.kv_type}{verbalize_hyper_param}",
    )
    real_model_name_tag = (
        f"{real_model_name}_{args.attn_type}{use_scdq}{disable_golden_context}_{args.kv_type}{verbalize_hyper_param}"
    )
    return result_dir, real_model_name_tag, use_scdq, use_llmlingua


def _merge_rank_jsonl(dst_path: Path, src_paths: list[Path]):
    rows = []
    for p in src_paths:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    rows.sort(key=lambda x: (x.get("id", -1), x.get("turn_idx", -1)))
    dump_jsonl(rows, dst_path)


def _load_subset_indices(path: str | None) -> list[int] | None:
    if not path:
        return None

    subset_path = Path(path)
    if subset_path.suffix.lower() == ".json":
        with open(subset_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("indices", [])
        if not isinstance(payload, list):
            raise ValueError(f"Invalid subset index file: {path}")
        return [int(x) for x in payload]

    indices = []
    with open(subset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            indices.append(int(line))
    return indices


def _filter_examples(
    examples,
    tok,
    subset_indices: list[int] | None = None,
    context_min_tokens: int = -1,
    context_max_tokens: int = -1,
):
    selected_indices = list(range(len(examples)))

    if subset_indices is not None:
        subset_set = set(int(i) for i in subset_indices)
        selected_indices = [i for i in selected_indices if i in subset_set]

    if context_min_tokens >= 0 or context_max_tokens >= 0:
        filtered = []
        lengths = []
        for i in selected_indices:
            context = examples[i]["context"]
            n_tokens = len(tok.encode(context, add_special_tokens=False))
            if context_min_tokens >= 0 and n_tokens < context_min_tokens:
                continue
            if context_max_tokens >= 0 and n_tokens >= context_max_tokens:
                continue
            filtered.append(i)
            lengths.append(n_tokens)
        selected_indices = filtered
        if lengths:
            print(
                "[SCBench] Context length filter kept "
                f"{len(lengths)} examples | min={min(lengths)} avg={sum(lengths)/len(lengths):.1f} max={max(lengths)}"
            )
        else:
            print("[SCBench] Context length filter kept 0 examples")

    if isinstance(examples, list):
        return [examples[i] for i in selected_indices]
    return examples.select(selected_indices)


def _run_scbench_worker(
    rank: int,
    world_size: int,
    args,
    data_names: list[str],
    max_seq_length: int,
    scdq_mode: bool,
):
    # DeltaKVGreedySearch uses `input_ids.cuda()` without specifying device.
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    hyper_param = args.hyper_param.copy() if args.hyper_param else {}
    # Force single-GPU load per rank (avoid HF `device_map="auto"` sharding).
    hyper_param["cuda_device"] = rank

    model_name = args.model_name_or_path
    real_model_name = model_name.split("/")[-1]
    result_dir, _, use_scdq, use_llmlingua = _build_result_dir(args, real_model_name, scdq_mode)
    result_dir.mkdir(exist_ok=True, parents=True)

    model, tok = load_model(
        model_name,
        args.topk,
        args.starting_layer,
        args.topk_dims_file_path,
        args.use_sparq,
        attn_type=args.attn_type,
        max_seq_length=max_seq_length,
        is_search=args.is_search,
        kv_type=args.kv_type,
        trust_remote_code=args.trust_remote_code,
        kv_cache_cpu=args.kv_cache_cpu,
        kv_cache_cpu_device=args.kv_cache_cpu_device,
        tensor_parallel_size=args.tensor_parallel_size,
        hyper_param=hyper_param,
        copy_on_gpu=args.copy_on_gpu,
    )
    subset_indices = _load_subset_indices(args.subset_indices_file)

    for data_name in data_names:
        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        if isinstance(max_new_tokens, dict):
            assert (
                max(max_new_tokens.values()) <= max_seq_length
            ), "max_new_tokens must be less than max_seq_length"
        elif max_new_tokens >= max_seq_length:
            max_new_tokens = 500

        output_path = result_dir / f"prediction_{data_name}{use_scdq}{use_llmlingua}.rank{rank}.jsonl"
        examples = load_dataset("microsoft/SCBench", data_name, split="test")

        if args.use_llmlingua:
            compression_ratio = hyper_param.get("llmlingua_ratio", 3) if hyper_param else 3
            examples = get_compressed_examples(examples, data_name, args.data_dir, rate=1 / compression_ratio)

        examples = _filter_examples(
            examples,
            tok=tok,
            subset_indices=subset_indices,
            context_min_tokens=args.context_min_tokens,
            context_max_tokens=args.context_max_tokens,
        )

        max_turn_size = len(examples[0]["multi_turns"])
        if args.max_turns > 0 and args.max_turns < max_turn_size:
            examples = [{**eg, "multi_turns": eg["multi_turns"][: args.max_turns]} for eg in examples]
            max_turn_size = args.max_turns

        if args.num_eval_examples != -1:
            num_eval_examples = min(args.num_eval_examples, len(examples))
            if isinstance(examples, list):
                examples = examples[:num_eval_examples]
            else:
                examples = examples.select(range(num_eval_examples))

        preds = []
        for i in tqdm(range(len(examples)), desc=f"[Rank {rank}] {data_name}"):
            if i < args.start_example_id:
                continue
            if i % world_size != rank:
                continue

            eg = examples[i]

            if isinstance(eg, str):
                try:
                    eg = json.loads(eg)
                except:
                    pass

            if data_name in ["scbench_summary_with_needles", "scbench_repoqa_and_kv"]:
                tokens_to_sum = sum(list(max_new_tokens.values())) if isinstance(max_new_tokens, dict) else max_new_tokens
                max_input_length = max_seq_length - (tokens_to_sum * max_turn_size // 2)
            else:
                max_input_length = max_seq_length - max_new_tokens * max_turn_size

            pred = get_pred(
                model,
                eg,
                data_name,
                max_new_tokens,
                max_input_length=max_input_length,
                attn_type=args.attn_type,
                tok=tok,
                use_chat_template=args.use_chat_template,
                scdq_mode=scdq_mode,
                disable_golden_context=args.disable_golden_context,
            )
            gts = get_ground_truth(eg, data_name)
            for turn_idx, (ans, gt, turn) in enumerate(zip(pred["answers"], gts, eg["multi_turns"])):
                case = {
                    "id": i,
                    "turn_idx": turn_idx,
                    "prediction": ans,
                    "ground_truth": gt,
                }
                if "task" in pred:
                    case["task"] = pred["task"][turn_idx]
                if data_name == "scbench_repoqa":
                    case["lang"] = eg["lang"]
                    case["repo"] = eg["repo"]
                    case["func_name"] = turn["name"]
                if data_name == "scbench_repoqa_and_kv":
                    case["lang"] = eg["lang"]
                    case["repo"] = eg["repo"]
                    if turn["task"] == "scbench_repoqa":
                        case["func_name"] = turn["name"]
                if data_name == "scbench_kv_compressible":
                    case["task"] = eg["task"]
                preds.append(case)
            dump_jsonl(preds, output_path)
            torch.cuda.empty_cache()

    try:
        model.clear()
    except Exception:
        pass


def get_pred(
    model,
    eg,
    data_name,
    max_new_tokens,
    max_input_length: int,
    attn_type: str = "vllm",
    tok=None,
    use_chat_template=False,
    scdq_mode=False,
    disable_golden_context=False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    if scdq_mode:
        encoded_eg = create_scdq_prompt(
            eg,
            data_name=data_name,
            tok=tok,
            use_chat_template=use_chat_template,
            use_vllm=("vllm" in attn_type),
        )
    else:
        # multi-turn mode
        encoded_eg = create_multiturn_prompt(
            eg,
            data_name=data_name,
            tok=tok,
            use_chat_template=use_chat_template,
            use_vllm=("vllm" in attn_type),
            disable_golden_context=disable_golden_context,
        )
    context = truncate_by_tokens(
        encoded_eg["prompts"][0], model.tokenizer, max_input_length
    )
    encoded_eg["prompts"][0] = context
    if scdq_mode:
        # scdq mode has no action for disable_golden_context
        outputs = model.test_scdq(encoded_eg, max_length=max_new_tokens)
    else:
        # multi-turn mode test
        outputs = model.test(
            encoded_eg,
            max_length=max_new_tokens,
            disable_golden_context=disable_golden_context,
        )

    print("Chunked generation:", json.dumps(outputs, indent=2, ensure_ascii=False))
    return outputs


def load_model(
    model_name: str,
    topk: int = -1,
    starting_layer: int = -1,
    topk_dims_file_path: str = "",
    use_sparq: bool = False,
    attn_type: str = "vllm",
    max_seq_length: int = None,
    is_search: bool = False,
    kv_type: str = "",
    trust_remote_code: bool = False,
    kv_cache_cpu: bool = False,
    kv_cache_cpu_device: str = "cpu",
    tensor_parallel_size: int = 1,
    hyper_param: dict = None,
    copy_on_gpu: bool = False,
):
    if model_name == "THUDM/glm-4-9b-chat-1m":
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, revision="refs/pr/19"
        )
    else:
        tok = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    # tok.pad_token = tok.eos_token

    if attn_type in ["deltakv", "full_deltakv", "origin_residual_quant", "all_origin_residual_quant", "snapkv", "pyramidkv", "palu", "quest"]:
        compressor_path = hyper_param.get("compressor_path")
        
        infer_config = hyper_param.copy()
        model_cls = infer_config.pop("model_cls", attn_type)
        cuda_device = infer_config.pop("cuda_device", "auto")
        
        from deltakv.get_chat_api import get_generate_api

        _, model = get_generate_api(
            model_path=model_name,
            infer_config=infer_config,
            compressor_path=compressor_path,
            model_cls=model_cls,
            cuda_device=cuda_device,
            return_model=True
        )
        
        llm = DeltaKVGreedySearch(model, tok, copy_on_gpu=copy_on_gpu)
        return llm, tok

    if attn_type == "vllm_blend":
        llm = LMCacheLLM(
            model=model_name,
            enable_prefix_caching=True,
            max_model_len=max_seq_length,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=False,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=0.5,
            swap_space=64,
        )
        llm = GreedySearch_vLLM(llm, tok)
    elif attn_type == "vllm_kv":
        llm = LLM(
            model=model_name,
            max_model_len=max_seq_length,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=False,
            trust_remote_code=True,
            swap_space=64,
            enforce_eager=True,
            enable_kvcompress=True,
            block_size=16,
            kv_head_bias_path=None,
            kv_head_bias_weight=0,
            disable_log_stats=True,
            prefill_metric_collection_window_size=32,
            prefill_metric_collection_block_size=4096,
            max_kv_per_compression=50_000_000,
            metric_aggregation="L2-sum",
            maxpool_metrics=True,
        )
        llm = GreedySearch_vLLM(
            llm,
            tok,
            is_kv_compress=True,
        )
    elif "vllm" in attn_type:
        # num_gpus
        llm = LLM(
            model=model_name,
            enable_prefix_caching="Jamba" not in model_name,
            max_model_len=max_seq_length,
            tensor_parallel_size=tensor_parallel_size,
            enable_chunked_prefill=False,
            trust_remote_code=trust_remote_code,
            swap_space=64,
        )
        if attn_type != "vllm":
            if MInference is not None:
                minference_patch = MInference(
                    attn_type,
                    model_name,
                    config_path=topk_dims_file_path,
                    starting_layer=starting_layer,
                    attn_kwargs=hyper_param,
                )
                llm = minference_patch(llm)
            else:
                print(f"Warning: minference is not installed. Skipping patch for {attn_type}")
        llm = GreedySearch_vLLM(llm, tok)
    else:
        if MInference is not None:
            minference_patch = MInference(
                attn_type.replace("_sink", ""),
                model_name,
                config_path=topk_dims_file_path,
                starting_layer=starting_layer,
                kv_type=kv_type,
                is_search=is_search,
                kv_cache_cpu=kv_cache_cpu,
                kv_cache_cpu_device=kv_cache_cpu_device,
                attn_kwargs=hyper_param,
            )
        else:
            minference_patch = None
            print(f"Warning: minference is not installed. Skipping patch for {attn_type}")

        if "mamba" in model_name.lower() or "recurrentgemma" in model_name.lower():
            llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                resume_download=None,
                trust_remote_code=trust_remote_code,
            )
            llm = GreedySearch_Mamba2(llm, tok)

            return llm, tok
        else:
            llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=trust_remote_code,
                attn_implementation="flash_attention_2",
            )
            if minference_patch is not None:
                llm = minference_patch(llm)

        if attn_type == "inf_llm":
            llm = GreedySearch_InfLLM(llm.model, tok)
            return llm, tok
        elif kv_type in ["retr_attn", "kivi"]:
            llm = GreedySearch_RetrAttn(
                llm,
                tok,
            )
            return llm, tok

        llm = GreedySearch(
            llm,
            tok,
        )

    print("Model and tokenizer loaded.")
    return llm, tok


if __name__ == "__main__":
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    # check_benchmark_availability(args.data_dir)
    model_name = args.model_name_or_path
    max_seq_length = args.max_seq_length
    real_model_name = model_name.split("/")[-1]
    data_name = args.task
    scdq_mode = args.same_context_different_query

    if "," in data_name:
        data_names = data_name.split(",")
    else:
        data_names = [data_name]

    if max_seq_length == -1:
        max_seq_length = 160_000

    result_dir, real_model_name_tag, use_scdq, use_llmlingua = _build_result_dir(args, real_model_name, scdq_mode)
    result_dir.mkdir(exist_ok=True, parents=True)

    if args.ws > 1:
        procs = []
        for rank in range(args.ws):
            p = mp.Process(
                target=_run_scbench_worker,
                args=(rank, args.ws, args, data_names, max_seq_length, scdq_mode),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        for p in procs:
            if p.exitcode != 0:
                raise RuntimeError(f"SCBench worker exited with code {p.exitcode}")

        results = {}
        for data_name in data_names:
            merged_path = result_dir / f"prediction_{data_name}{use_scdq}{use_llmlingua}.jsonl"
            shard_paths = [
                result_dir / f"prediction_{data_name}{use_scdq}{use_llmlingua}.rank{rank}.jsonl"
                for rank in range(args.ws)
            ]
            _merge_rank_jsonl(merged_path, shard_paths)
            score = compute_scores(
                merged_path,
                data_name,
                real_model_name_tag,
                max_seq_length=max_seq_length,
                scdq_mode=scdq_mode,
            )
            results[data_name] = score
    else:
        # Model
        model, tok = load_model(
            model_name,
            args.topk,
            args.starting_layer,
            args.topk_dims_file_path,
            args.use_sparq,
            attn_type=args.attn_type,
            max_seq_length=max_seq_length,
            is_search=args.is_search,
            kv_type=args.kv_type,
            trust_remote_code=args.trust_remote_code,
            kv_cache_cpu=args.kv_cache_cpu,
            kv_cache_cpu_device=args.kv_cache_cpu_device,
            tensor_parallel_size=args.tensor_parallel_size,
            hyper_param=args.hyper_param.copy(),
            copy_on_gpu=args.copy_on_gpu,
        )
        subset_indices = _load_subset_indices(args.subset_indices_file)

        results = {}
        for data_name in data_names:
            max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
            if isinstance(max_new_tokens, dict):
                assert (
                    max(max_new_tokens.values()) <= max_seq_length
                ), "max_new_tokens must be less than max_seq_length"
            elif max_new_tokens >= max_seq_length:
                max_new_tokens = 500

            output_path = result_dir / f"prediction_{data_name}{use_scdq}{use_llmlingua}.jsonl"
            examples = load_dataset("microsoft/SCBench", data_name, split="test")

            if args.use_llmlingua:
                compression_ratio = args.hyper_param.get("llmlingua_ratio", 3) if args.hyper_param else 3
                examples = get_compressed_examples(examples, data_name, args.data_dir, rate=1 / compression_ratio)

            examples = _filter_examples(
                examples,
                tok=tok,
                subset_indices=subset_indices,
                context_min_tokens=args.context_min_tokens,
                context_max_tokens=args.context_max_tokens,
            )
            max_turn_size = len(examples[0]["multi_turns"])
            if args.max_turns > 0 and args.max_turns < max_turn_size:
                examples = [{**eg, "multi_turns": eg["multi_turns"][: args.max_turns]} for eg in examples]
                max_turn_size = args.max_turns

            if args.num_eval_examples != -1:
                num_eval_examples = min(args.num_eval_examples, len(examples))
                if isinstance(examples, list):
                    examples = examples[:num_eval_examples]
                else:
                    examples = examples.select(range(num_eval_examples))

            preds = []
            print(f"==== Evaluation {data_name}====")
            print(f"# examples: {len(examples)}")
            print(f"Num eval examples: {args.num_eval_examples}")
            print(f"Verbose: {args.verbose}")
            print(f"Max new tokens: {max_new_tokens}")
            print(f"Num of turns: {max_turn_size}")

            for i in tqdm(range(len(examples))):
                if i < args.start_example_id:
                    continue

                eg = examples[i]

                if isinstance(eg, str):
                    try:
                        eg = json.loads(eg)
                    except:
                        pass

                if data_name in ["scbench_summary_with_needles", "scbench_repoqa_and_kv"]:
                    tokens_to_sum = sum(list(max_new_tokens.values())) if isinstance(max_new_tokens, dict) else max_new_tokens
                    max_input_length = max_seq_length - (tokens_to_sum * max_turn_size // 2)
                else:
                    max_input_length = max_seq_length - max_new_tokens * max_turn_size

                pred = get_pred(
                    model,
                    eg,
                    data_name,
                    max_new_tokens,
                    max_input_length=max_input_length,
                    attn_type=args.attn_type,
                    tok=tok,
                    use_chat_template=args.use_chat_template,
                    scdq_mode=scdq_mode,
                    disable_golden_context=args.disable_golden_context,
                )
                gts = get_ground_truth(eg, data_name)
                for turn_idx, (ans, gt, turn) in enumerate(zip(pred["answers"], gts, eg["multi_turns"])):
                    case = {
                        "id": i,
                        "turn_idx": turn_idx,
                        "prediction": ans,
                        "ground_truth": gt,
                    }
                    if "task" in pred:
                        case["task"] = pred["task"][turn_idx]
                    if data_name == "scbench_repoqa":
                        case["lang"] = eg["lang"]
                        case["repo"] = eg["repo"]
                        case["func_name"] = turn["name"]
                    if data_name == "scbench_repoqa_and_kv":
                        case["lang"] = eg["lang"]
                        case["repo"] = eg["repo"]
                        if turn["task"] == "scbench_repoqa":
                            case["func_name"] = turn["name"]
                    if data_name == "scbench_kv_compressible":
                        case["task"] = eg["task"]
                    preds.append(case)
                dump_jsonl(preds, output_path)
                torch.cuda.empty_cache()

            score = compute_scores(
                output_path,
                data_name,
                real_model_name_tag,
                max_seq_length=max_seq_length,
                scdq_mode=scdq_mode,
            )
            results[data_name] = score

    print("==== Results ====")
    print(json.dumps(results, indent=2))

    # 记录评测信息到日志文件
    os.makedirs(BASE_PATH, exist_ok=True)
    log_path = os.path.join(BASE_PATH, "scbench_eval.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: python {' '.join(sys.argv)}\n")
        f.write(f"Args: {json.dumps(vars(args), indent=2)}\n")
        f.write("-" * 80 + "\n")

    # 记录结果到日志
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Evaluation Results:\n")
        f.write(json.dumps(results, indent=4, ensure_ascii=False))
        f.write("\n" + "="*80 + "\n\n")

    try:
        lmcache_vllm.close_lmcache_engine()
    except:
        pass
