"""Reproducible GLM attention-layout trace with a fixed expert-parallel size.

Use identical checkpoint, seed, methods and prompts for both runs. Tiny random
runs validate runtime contracts only; use a real checkpoint for quality checks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import torch

from sparsevllm import LLM, SamplingParams


async def validate_dispatcher(llm, prompts):
    from sparsevllm.entrypoints.openai.dispatcher import AsyncEngineDispatcher

    dispatcher = AsyncEngineDispatcher(llm)
    try:
        params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
        handles = await asyncio.gather(
            *(
                dispatcher.submit_admitted(prompt, params, i)
                for i, prompt in enumerate(prompts[:2])
            )
        )

        async def consume(handle):
            events = []
            while True:
                event = await asyncio.wait_for(handle.output_queue.get(), timeout=60)
                assert event["type"] != "error", event
                events.append(event)
                if event["type"] == "final":
                    assert event["completion_tokens"] == 4
                    return events

        outputs = await asyncio.gather(*(consume(handle) for handle in handles))
        handle = await dispatcher.submit_admitted(
            prompts[0],
            SamplingParams(temperature=0, max_tokens=128, ignore_eos=True),
            2,
        )
        await dispatcher.discard(handle)
        assert handle.terminal.is_set()
        assert await dispatcher.control("is_finished")
        # A cancellation must not poison the other replica or the next request.
        resumed = await dispatcher.submit_admitted(prompts[1], params, 3)
        await consume(resumed)
        assert dispatcher.failure_message is None
        load = await dispatcher.control("worker_routing_load")
        return {"name": "async_dispatcher", "outputs": outputs, "final_load": load}
    finally:
        dispatcher.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--mode", choices=["ep", "dp"], default="dp")
    parser.add_argument(
        "--size",
        type=int,
        default=2,
        help="Expert-parallel size; also DP size in dp mode.",
    )
    parser.add_argument(
        "--attention-tp",
        type=int,
        default=1,
        help="Attention TP size: ep mode with TP=1 replicates attention; TP>1 shards it.",
    )
    parser.add_argument("--method", default="")
    parser.add_argument("--moe-communication", choices=["auto", "agrs", "all2all"], default="auto")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--dispatcher", action="store_true")
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--tiny-config")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--mlp-chunk-size", type=int, default=16384)
    parser.add_argument("--logits-trace", action="store_true")
    parser.add_argument("--logits-reference", type=Path)
    parser.add_argument("--teacher-tokens", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.logits_trace or args.logits_reference or args.teacher_tokens:
        os.environ["SPARSEVLLM_DEBUG_RUNTIME"] = "1"
    config = {
        "tensor_parallel_size": args.attention_tp,
        "expert_parallel_size": args.size,
        "data_parallel_size": args.size if args.mode == "dp" else 1,
        "moe_communication_backend": args.moe_communication,
        "sparse_method": args.method,
        "decode_graph": args.graph,
        "enable_prefix_caching": args.prefix_cache,
        "tiny_random": bool(args.tiny_config),
        "tiny_random_config": args.tiny_config,
        "max_model_len": 512,
        "max_num_batched_tokens": 512,
        "max_num_seqs_in_batch": 2,
        "max_decoding_seqs": 2,
        "decode_graph_capture_sizes": [1, 2],
        "gpu_memory_utilization": 0.75,
        "sink_keep_tokens": 8,
        "decode_keep_tokens": 32,
        "recent_keep_tokens": 8,
        "prefix_cache_block_size": 16 if args.prefix_cache else None,
        "throughput_log_interval_s": 0,
        "mlp_chunk_size": args.mlp_chunk_size,
    }
    if args.tiny_config:
        config.update(
            full_attention_layers=[0],
            h2o_decode_budget=32,
            h2o_prefill_budget=32,
            quest_skip_layers=0,
        )
    report = {
        "config": config,
        "model": args.model,
        "torch": torch.__version__,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "status": "running",
        "cases": [],
    }
    llm = None
    try:
        llm = LLM(args.model, **config)
        report["initial_state"] = llm.debug_sparse_state_summaries()
        prompts = [
            llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )
            for text in [
                "What is 2 + 3? Answer with the number only.",
                "中国的首都是哪里？只回答城市名。",
                "Complete the sequence: 2, 4, 6, 8,",
            ]
        ]
        params = [
            SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)
            for n in (args.max_tokens, max(1, args.max_tokens // 2), 3)
        ]
        started = perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        report["cases"].append(
            {
                "name": "ragged_and_idle",
                "prompts": prompts,
                "outputs": outputs,
                "seconds": perf_counter() - started,
            }
        )
        # Force an idle owner to meet an active graph, then inject a prefill
        # while another replica is decoding. This catches mismatched AG sizes
        # and collective order that a static balanced batch cannot exercise.
        llm.add_request(prompts[0], params[0])
        llm.step()
        llm.add_request(prompts[1], params[1])
        steps = []
        churn = []
        while not llm.is_finished():
            start = perf_counter()
            done, count = llm.step()
            steps.append({"seconds": perf_counter() - start, "signed_tokens": count})
            churn.extend(done)
        report["cases"].append(
            {"name": "mixed_prefill_decode", "outputs": churn, "raw_steps": steps}
        )
        # A very short prefill fits the prepared communication capacity. Its
        # owner and a decoding peer must still choose the same backend.
        llm.add_request(prompts[0], params[0])
        llm.step()
        llm.add_request([prompts[1][-1]], params[1])
        churn = []
        while not llm.is_finished():
            done, _ = llm.step()
            churn.extend(done)
        report["cases"].append(
            {"name": "one_token_prefill_with_decode", "outputs": churn}
        )
        short_long_outputs = llm.generate(
            [prompts[0], prompts[1] * 5],
            SamplingParams(temperature=0, max_tokens=4, ignore_eos=True),
            use_tqdm=False,
        )
        report["cases"].append(
            {"name": "different_sparse_paths", "outputs": short_long_outputs}
        )
        if args.logits_trace or args.logits_reference or args.teacher_tokens:
            teacher_path = args.teacher_tokens or args.logits_reference
            expected = (
                torch.load(teacher_path, weights_only=True) if teacher_path else None
            )
            trace_seq_id = llm.add_request(
                prompts[0],
                SamplingParams(temperature=0, max_tokens=16, ignore_eos=True),
            )
            logits_trace = []
            tokens = []
            while not llm.is_finished():
                llm.step()
                logits = llm.debug_last_logits()
                logits_trace.append(
                    (logits[0] if isinstance(logits, dict) else logits).clone()
                )
                tokens.extend(llm.last_step_token_outputs[0][1])
                if expected is not None and not llm.is_finished():
                    llm.debug_set_next_decode_token(
                        trace_seq_id, expected["tokens"][len(tokens) - 1]
                    )
            tensor = torch.cat(logits_trace).float()
            logits_path = args.output.with_suffix(".pt")
            torch.save(
                {
                    "logits": tensor,
                    "tokens": tokens,
                    "teacher_tokens": expected["tokens"]
                    if expected is not None
                    else tokens,
                },
                logits_path,
            )
            report["logits_trace"] = str(logits_path)
            if args.logits_reference:
                reference_logits = torch.load(args.logits_reference, weights_only=True)[
                    "logits"
                ]
                delta = tensor - reference_logits
                report["logits_error"] = {
                    "max_abs": float(delta.abs().max()),
                    "relative_rms": float(
                        delta.square().mean().sqrt()
                        / reference_logits.square().mean().sqrt()
                    ),
                }
                torch.testing.assert_close(
                    tensor, reference_logits, atol=0.25, rtol=0.02
                )
        if args.prefix_cache:
            prompt = (prompts[0] * 5)[:160]
            admission = llm.admit_request(
                prompt, SamplingParams(temperature=0, max_tokens=3, ignore_eos=True)
            )
            cold_tokens = []
            while not llm.is_finished():
                llm.step()
                for _, step_tokens in llm.last_step_token_outputs:
                    cold_tokens.extend(step_tokens)
            if admission.chain_id:
                resumed = llm.admit_request(
                    [20, 21, 22],
                    SamplingParams(temperature=0, max_tokens=3),
                    chain_id=admission.chain_id,
                    chain_append_only=True,
                )
                assert resumed.seq_id == admission.seq_id
                assert resumed.reused_tokens > 0
                while not llm.is_finished():
                    llm.step()
                llm.discard_chain(admission.chain_id, expected_seq_id=admission.seq_id)
                cache_result = {"mode": "chain", "reused_tokens": resumed.reused_tokens}
            else:
                matched = llm.prefix_cache_match(prompt)
                assert matched["matched_tokens"] > 0, matched
                cached = llm.generate(
                    [prompt],
                    SamplingParams(temperature=0, max_tokens=3, ignore_eos=True),
                    use_tqdm=False,
                )
                assert cached[0]["token_ids"] == cold_tokens
                cache_result = {
                    "mode": "radix", "match": matched,
                    "cold_tokens": cold_tokens, "cached_tokens": cached[0]["token_ids"],
                }
            report["cases"].append(dict(name="prefix_resume", **cache_result))
        report["final_state"] = llm.debug_sparse_state_summaries()
        report["bindings"] = llm.operator_runtime_stats()
        for initial, final in zip(report["initial_state"], report["final_state"]):
            assert final["decode_graph"]["recapture_count"] == 0
            assert (
                initial["decode_graph"]["capture_count"]
                == final["decode_graph"]["capture_count"]
            )
        if args.reference:
            reference = json.loads(args.reference.read_text())
            expected = reference["cases"][0]["outputs"]
            eos = set(llm.config.eos_token_ids)

            def before_eos(output):
                ids = output["token_ids"]
                end = next((i for i, token in enumerate(ids) if token in eos), len(ids))
                return ids[:end]

            report["forced_post_eos_exact_match"] = outputs == expected
            if [before_eos(o) for o in outputs] != [before_eos(o) for o in expected]:
                raise AssertionError(
                    "Greedy token outputs before EOS differ from reference."
                )
        if args.dispatcher:
            report["cases"].append(asyncio.run(validate_dispatcher(llm, prompts)))
        report["status"] = "success"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
