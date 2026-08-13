from __future__ import annotations

__all__ = ["LLM", "MultiModalPrompt", "SamplingParams"]


def __getattr__(name: str):
    if name == "LLM":
        from sparsevllm.llm import LLM

        return LLM
    if name == "SamplingParams":
        from sparsevllm.sampling_params import SamplingParams

        return SamplingParams
    if name == "MultiModalPrompt":
        from sparsevllm.multimodal import MultiModalPrompt

        return MultiModalPrompt
    raise AttributeError(f"module 'sparsevllm' has no attribute {name!r}")
