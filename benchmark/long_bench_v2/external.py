"""Generation-only adapters; sample selection and scoring stay in pred.py."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen


def get_generate_api(*, engine, model_path, max_model_len, seed, engine_kwargs, server_url,
                     sparse_method="vanilla"):
    if engine == "vllm":
        scorer = engine_kwargs.get("compression_scorer")
        if not ((sparse_method == "vanilla" and scorer in (None, "none")) or
                (sparse_method == "snapkv" and scorer == "snapkv" and
                 engine_kwargs.get("compression_budget_tokens", 0) > 0)):
            raise ValueError("vLLM method label does not match its explicit compression configuration.")
        import vllm
        config = dict(model=model_path, max_model_len=max_model_len, seed=seed,
                      trust_remote_code=True, enable_prefix_caching=False)
        if set(config) & set(engine_kwargs):
            raise ValueError("Engine kwargs may not override paired model/context/seed/prefix settings.")
        config.update(engine_kwargs)
        llm = vllm.LLM(**config)

        def generate(prompts, **kwargs):
            params = vllm.SamplingParams(
                temperature=kwargs["temperature"], top_p=kwargs["top_p"],
                top_k=kwargs["top_k"], max_tokens=kwargs["max_new_tokens"],
                stop_token_ids=kwargs["eos_token_id"], seed=seed,
            )
            return [row.outputs[0].text for row in llm.generate(
                [{"prompt_token_ids": ids} for ids in prompts], params, use_tqdm=False)]

        return generate, {"version": vllm.__version__, "package_source": str(Path(vllm.__file__).resolve()),
                          "config": config}
    if engine != "sglang-http" or not server_url:
        raise ValueError("sglang-http requires an explicit server URL.")
    if engine_kwargs:
        raise ValueError("HTTP runtime parameters belong in the preserved server launch config.")
    base = server_url.rstrip("/")
    with urlopen(base + "/get_server_info", timeout=30) as response:
        info = json.load(response)
    args = info.get("server_args", info)
    actual_model = args.get("model_path")
    if not actual_model or Path(actual_model).resolve() != Path(model_path).resolve():
        raise ValueError(f"HTTP model mismatch: {actual_model!r} != {model_path!r}")
    if not args.get("disable_radix_cache", False):
        raise ValueError("Paired quality requires the server radix cache to be disabled.")

    def generate(prompts, **kwargs):
        payload = {"input_ids": prompts, "sampling_params": {
            "temperature": kwargs["temperature"], "top_p": kwargs["top_p"],
            "top_k": kwargs["top_k"], "max_new_tokens": kwargs["max_new_tokens"],
            "stop_token_ids": kwargs["eos_token_id"], "skip_special_tokens": True},
            "stream": False}
        request = Request(base + "/generate", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=1800) as response:
            rows = json.load(response)
        if isinstance(rows, dict):
            rows = [rows]
        return [row["text"] for row in rows]

    return generate, info
