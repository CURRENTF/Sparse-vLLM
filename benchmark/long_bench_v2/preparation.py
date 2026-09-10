"""CPU-only chat preparation and bounded parallel full-dataset preprocessing."""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import partial
import multiprocessing
import os
from typing import Any

from transformers import AutoTokenizer
from tqdm import tqdm

from benchmark.long_bench.pred import build_chat
from benchmark.long_bench_v2.contracts import prepare_official_samples


def prepare_chat(tokenizer: Any, prompt: str, *, no_chat_template: bool) -> tuple[str, list[int]]:
    prompt = build_chat(
        tokenizer, prompt, "longbench_v2",
        no_chat_template=no_chat_template, thinking_mode="off",
    )
    add_special_tokens = bool(tokenizer.bos_token is not None and not prompt.startswith(tokenizer.bos_token))
    token_ids = [int(token) for token in tokenizer.encode(prompt, add_special_tokens=add_special_tokens)]
    if not token_ids:
        raise ValueError("LongBench v2 prompt tokenized to zero tokens.")
    return prompt, token_ids


_worker_options: dict[str, Any] = {}


def _initialize_worker(tokenizer_path, template, no_chat_template, truncate_max_tokens, max_prompt_tokens):
    # Parallelism is across samples; prevent nested tokenizer/Torch thread pools.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch

    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    _worker_options.update(
        template=template, tokenizer=tokenizer,
        prepare_chat=partial(prepare_chat, tokenizer, no_chat_template=no_chat_template),
        truncate_max_tokens=truncate_max_tokens, max_prompt_tokens=max_prompt_tokens,
    )


def _prepare_row(row):
    result = prepare_official_samples([row], **_worker_options)[0]
    del result["sample"]  # The parent already owns the potentially multi-megabyte source row.
    return result


def prepare_official_samples_parallel(
    rows: list[dict[str, Any]], *, tokenizer_path: str, template: str,
    no_chat_template: bool, truncate_max_tokens: int, max_prompt_tokens: int,
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 0:
        raise ValueError("preprocess workers must be positive.")
    if not rows:
        return []
    workers = min(workers, len(rows))
    selected = [None] * len(rows)
    indices = iter(range(len(rows)))
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(tokenizer_path, template, no_chat_template, truncate_max_tokens, max_prompt_tokens),
    ) as pool, tqdm(
        total=len(rows), desc="LongBench v2 preparation", unit="it",
        dynamic_ncols=True, disable=None,
    ) as progress:
        # Bound queued source texts and returned token arrays instead of submitting all 503 rows.
        pending = {}
        for _ in range(min(2 * workers, len(rows))):
            index = next(indices)
            pending[pool.submit(_prepare_row, rows[index])] = index
        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    result = future.result()
                    result.update(index=index, source_index=index, sample=rows[index])
                    selected[index] = result
                    progress.update(1)
                    next_index = next(indices, None)
                    if next_index is not None:
                        pending[pool.submit(_prepare_row, rows[next_index])] = next_index
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return selected
