import os
import sys
import torch
from typing import Union, List
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from deltakv.configs.model_config_cls import KVQwen2Config, KVQwen3Config, KVLlamaConfig
from deltakv.baseline_adapters import load_omnikv_model, load_kivi_model
from safetensors.torch import load_file


def load_compressor(compressor_path, device='cuda:0'):
    state_dict = load_file(os.path.join(compressor_path, 'model.safetensors'), device)
    return state_dict


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch, vocabulary size)
            if top_k > 0: keep only top k tokens with the highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the end
        From: https://gist.github.com/thomwolf/1a5a29f69620886c271b93575f97Self
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep - 1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


@torch.inference_mode()
def manual_generate(model, tokenizer, prompt: Union[str, List[str]], past_key_values=None, return_kv_cache=False, **kwargs):
    """
    手动实现的生成函数，支持KV Cache复用及采样。
    """
    if isinstance(prompt, str):
        prompts = [prompt]
        is_single = True
    else:
        prompts = prompt
        is_single = False

    max_new_tokens = kwargs.get('max_new_tokens', 128)
    do_sample = kwargs.get('do_sample', False)
    temperature = kwargs.get('temperature', 1.0)
    top_k = kwargs.get('top_k', 50)
    top_p = kwargs.get('top_p', 1.0)
    add_special_tokens = True
    if tokenizer.bos_token is None or prompts[0].startswith(tokenizer.bos_token):
        add_special_tokens = False
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=add_special_tokens).to(model.device)
    input_ids = inputs.input_ids

    batch_size = input_ids.shape[0]
    unfinished_sequences = input_ids.new_ones(batch_size, device=model.device)

    # 保存每一轮生成的 token
    generated_tokens = []

    cur_input_ids = input_ids
    # 支持分块 Prefill 以降低激活显存占用
    chunk_prefill_size = int(os.environ.get('MANUAL_GEN_CHUNK_PREFILL_SIZE', 0))
    if chunk_prefill_size > 0 and input_ids.shape[1] > chunk_prefill_size:
        seq_len = input_ids.shape[1]
        for j in range(0, seq_len - 1, chunk_prefill_size):
            end_idx = min(j + chunk_prefill_size, seq_len - 1)
            chunk = input_ids[:, j:end_idx]
            outputs = model(input_ids=chunk, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
        cur_input_ids = input_ids[:, -1:]

    eos_token_ids = kwargs.get('eos_token_id', [tokenizer.eos_token_id])
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    eos_token_ids_tensor = torch.tensor(eos_token_ids, device=model.device)

    for i in range(max_new_tokens):
        outputs = model(input_ids=cur_input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        if os.environ.get('BAN_EOS') and tokenizer.eos_token_id is not None:
            logits[:, tokenizer.eos_token_id] = -float('inf')

        if do_sample:
            if temperature != 1.0:
                logits = logits / temperature
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(logits, dim=-1)

        # 考虑 EOS (支持多个 EOS ID)
        is_eos = torch.isin(next_tokens, eos_token_ids_tensor)
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (1 - unfinished_sequences)
        generated_tokens.append(next_tokens)

        # 更新状态
        unfinished_sequences = unfinished_sequences.mul((~is_eos).long())
        if unfinished_sequences.max() == 0:
            break

        cur_input_ids = next_tokens.unsqueeze(-1)

    skip_special_tokens = True
    if os.environ.get('NOT_SKIP_SPECIAL_TOKENS', False):
        skip_special_tokens = False

    # 拼接并解码
    if not generated_tokens:
        results = ["" for _ in range(batch_size)]
    else:
        all_generated_ids = torch.stack(generated_tokens, dim=1)
        results = []
        for i in range(batch_size):
            text = tokenizer.decode(all_generated_ids[i], skip_special_tokens=skip_special_tokens)
            results.append(text)

    if return_kv_cache:
        return (results[0], past_key_values) if is_single else (results, past_key_values)
    return results[0] if is_single else results


def hf_gen(model, tokenizer, prompt: Union[str, List[str]], return_kv_cache, past_key_values=None, **kwargs):
    assert past_key_values is None
    if isinstance(prompt, str):
        prompts = [prompt]
        is_single = True
    else:
        prompts = prompt
        is_single = False

    inputs = tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)

    gen_config = {
        "max_new_tokens": kwargs.get('max_new_tokens', 128),
        "do_sample": kwargs.get('do_sample', False),
        "temperature": kwargs.get('temperature', 1.0),
        "top_k": kwargs.get('top_k', 50),
        "top_p": kwargs.get('top_p', 1.0),
        "eos_token_id": kwargs.get('eos_token_id', tokenizer.eos_token_id),
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }

    # 如果不采样，移除相关参数以避免警告
    if not gen_config["do_sample"]:
        gen_config.pop("temperature", None)
        gen_config.pop("top_k", None)
        gen_config.pop("top_p", None)

    output_ids = model.generate(**inputs, **gen_config)

    input_len = inputs.input_ids.shape[1]
    generated_ids = output_ids[:, input_len:]

    results = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    if return_kv_cache:
        return (results[0], None) if is_single else (results, None)
    return results[0] if is_single else results


def get_generate_api(model_path: str, infer_config: dict, compressor_path: str,
                     tokenizer_path: str = None, model_cls: str = 'deltakv',  use_cache: bool = True,
                     cuda_device: Union[int, str] = 0, backend: str = 'hf', return_kv_cache: bool = False,
                     return_model: bool = False):
    """
    Returns:
        function: 一个生成函数，输入prompt和生成参数，返回生成内容。
    """

    if backend == 'sparsevllm':
        from sparsevllm import LLM, SamplingParams
        # sparsevllm 内部管理 tokenizer 和设备
        # TODO 这边逻辑稍微不太统一，不通过 compressor_path 传 compressor
        
        llm = LLM(
            model_path, 
            **infer_config,
        )

        def generate(prompt: Union[str, List[str]], past_key_values=None, **kwargs):
            if isinstance(prompt, str):
                prompts = [prompt]
                is_single = True
            else:
                prompts = prompt
                is_single = False

            # 将 HF 风格参数映射到 SamplingParams
            max_tokens = kwargs.get('max_new_tokens', kwargs.get('max_tokens', 128))
            temperature = kwargs.get('temperature', 1.0)
            
            # greedy decoding should be exact argmax, not low-temperature sampling.
            if not kwargs.get('do_sample', True):
                temperature = 0.0
            elif temperature < 1e-5:
                temperature = 1e-5
            
            sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            
            results = [out['text'] for out in outputs]
            if return_kv_cache:
                return (results[0], None) if is_single else (results, None)
            return results[0] if is_single else results
        
        if return_model:
            raise ValueError('sparse vllm 不支持 return_model=True')
        return generate

    assert use_cache, '还要做padding才能用训练代码推理'
    if model_cls == 'deltakv':
        base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if base_config.model_type == 'qwen2':
            if use_cache:
                from deltakv.modeling.qwen2.qwen2_with_compress_inference import Qwen2KVCompress as KVModel
            else:
                from deltakv.modeling.qwen2.qwen2_e2e import Qwen2KVCompress as KVModel
            config_cls = KVQwen2Config
        elif base_config.model_type == 'qwen3':
            if use_cache:
                from deltakv.modeling.qwen3.qwen3_with_compress_inference import Qwen3KVCompress as KVModel
            else:
                from deltakv.modeling.qwen3.qwen3_e2e import Qwen3KVCompress as KVModel
            config_cls = KVQwen3Config
        elif base_config.model_type == 'llama':
            from deltakv.modeling.llama.llama_with_compress_inference import LlamaKVCompress as KVModel
            config_cls = KVLlamaConfig
        else:
            raise ValueError(f"Unsupported model type: {base_config.model_type}")

        if compressor_path is not None:
            config = config_cls.from_pretrained(compressor_path)
        else:
            config = config_cls.from_pretrained(model_path)
        config.set_infer_args(**infer_config)
        model = KVModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=cuda_device,
            attn_implementation="flash_attention_2",
        )
        if compressor_path is not None:
            load_device = f'cuda:{cuda_device}' if isinstance(cuda_device, int) else 'cpu'
            comp_state_dict = load_compressor(compressor_path, device=load_device)
            _, unexpected = model.load_state_dict(comp_state_dict, strict=False)
            assert len(unexpected) == 0, f'compressor 加载有问题: {unexpected}'
            del comp_state_dict

    elif model_cls == 'snapkv':
        base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if base_config.model_type == 'qwen2':
            from deltakv.modeling.qwen2.qwen2_snapkv import Qwen2SnapKVForCausalLM as KVModel
            config_cls = KVQwen2Config
        elif base_config.model_type == 'llama':
            from deltakv.modeling.llama.llama_snapkv import LlamaSnapKVForCausalLM as KVModel
            config_cls = KVLlamaConfig
        else:
            raise ValueError(f"Unsupported model type: {base_config.model_type}")

        print('💡💡💡 SnapKV')
        config = config_cls.from_pretrained(model_path)
        config.set_infer_args(**infer_config)
        print(f'[Config] {config}')
        model = KVModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=cuda_device,
            attn_implementation="flash_attention_2",
        )

    elif model_cls == 'pyramidkv':
        base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if base_config.model_type == 'qwen2':
            from deltakv.modeling.qwen2.qwen2_pyramidkv import Qwen2PyramidKVForCausalLM as KVModel
            config_cls = KVQwen2Config
        elif base_config.model_type == 'llama':
            from deltakv.modeling.llama.llama_pyramidkv import LlamaPyramidKVForCausalLM as KVModel
            config_cls = KVLlamaConfig
        else:
            raise ValueError(f"Unsupported model type: {base_config.model_type}")

        print('💡💡💡 PyramidKV')
        config = config_cls.from_pretrained(model_path)
        config.set_infer_args(**infer_config)
        print(f'[Config] {config}')
        model = KVModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=cuda_device,
            attn_implementation="flash_attention_2",
        )

    elif model_cls == 'omnikv':
        print('💡💡💡 OmniKV')
        model = load_omnikv_model(model_path, infer_config, cuda_device)

    elif model_cls == 'auto':
        print('💡💡💡 Auto')
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=cuda_device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        auto_chunk_prefill_size = infer_config.get('chunk_prefill_size', None)
        if auto_chunk_prefill_size is not None:
            from types import MethodType

            def chunked_forward(self, input_ids=None, past_key_values=None, **kwargs):
                if input_ids is not None and input_ids.shape[1] > auto_chunk_prefill_size:
                    seq_len = input_ids.shape[1]
                    outputs = None
                    for i in range(0, seq_len, auto_chunk_prefill_size):   # noqa
                        chunk = input_ids[:, i:i + auto_chunk_prefill_size]   # noqa
                        outputs = self.original_forward(input_ids=chunk, past_key_values=past_key_values, **kwargs)
                        past_key_values = outputs.past_key_values
                    return outputs
                return self.original_forward(input_ids=input_ids, past_key_values=past_key_values, **kwargs)

            print('monkey patch raw full attn')
            model.original_forward = model.forward
            model.forward = MethodType(chunked_forward, model)

    elif model_cls == 'quest':
        print('💡💡💡 Quest')
        # 加入 quest 的路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        quest_base_dir = os.path.abspath(os.path.join(current_dir, "../../baselines/quest"))
        if quest_base_dir not in sys.path:
            sys.path.insert(0, quest_base_dir)

        from baselines.quest.evaluation.llama import enable_tuple_kv_cache_for_llama
        from baselines.quest.evaluation.quest_attention import enable_quest_attention_eval

        # 启用 Quest 所需的补丁，改变 KV Cache 处理方式
        enable_tuple_kv_cache_for_llama()

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=cuda_device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        class QuestArgs:
            def __init__(self, token_budget, chunk_size):
                self.token_budget = token_budget
                self.chunk_size = chunk_size

        # 从 infer_config 中获取参数
        quest_args = QuestArgs(
            token_budget=infer_config['num_top_tokens'],
            chunk_size=infer_config.get('chunk_size', 16)
        )
        enable_quest_attention_eval(model, quest_args)

    elif model_cls == 'palu':
        print('💡💡💡 Palu')
        import transformers
        assert transformers.__version__ == '4.37.2'

        current_dir = os.path.dirname(os.path.abspath(__file__))
        palu_base_dir = os.path.abspath(os.path.join(current_dir, "../../baselines/palu"))
        if palu_base_dir not in sys.path:
            sys.path.insert(0, palu_base_dir)

        # 必须先导入 palu.model 以触发 AutoConfig/AutoModel 注册
        import palu.model  # noqa: F401
        from palu.quant_utils import configure_latent_quantizer

        # Palu 模型通常使用 float16，且依赖其自定义的 Triton kernel
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=cuda_device,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        # 如果配置了低比特量化（lt_bits < 16），则进行配置
        lt_bits = infer_config.get('lt_bits', 16)
        if lt_bits < 16:
            configure_latent_quantizer(
                model,
                n_bits=lt_bits,
                group_size=infer_config.get('lt_group_size', 0),
                sym=infer_config.get('lt_sym', True),
                clip_ratio=infer_config.get('lt_clip_ratio', 1.0),
                hadamard=infer_config.get('lt_hadamard', False)
            )

    elif model_cls == 'kivi':
        print('💡💡💡 KIVI')
        model = load_kivi_model(model_path, infer_config, cuda_device)
    elif model_cls == 'adakv':
        print('💡💡💡 AdaKV')
        os.environ['ENABLE_HF_GEN'] = '1'  # 生成hack流程依赖于generate函数
        current_dir = os.path.dirname(os.path.abspath(__file__))
        adakv_base_dir = os.path.abspath(os.path.join(current_dir, "../../baselines/adakv"))
        if adakv_base_dir not in sys.path:
            sys.path.insert(0, adakv_base_dir)

        from adaptive_snapkv.monkeypatch.monkeypatch import (
            replace_mistral_adaptive, replace_llama_adaptive,
            replace_mistral_fixed, replace_llama_fixed,
            config_compress
        )

        base_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        use_adaptive = infer_config.get('use_adaptive', True)

        if base_config.model_type == 'llama':
            if use_adaptive:
                replace_llama_adaptive()
            else:
                replace_llama_fixed()
        elif base_config.model_type == 'mistral':
            if use_adaptive:
                replace_mistral_adaptive()
            else:
                replace_mistral_fixed()
        else:
            raise ValueError(f"AdaKV does not support model type: {base_config.model_type}")

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=cuda_device,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        # config hyperparameters
        model = config_compress(
            model,
            window_size=infer_config.get('snapkv_window_size', 32),
            base_capacity=infer_config.get('num_top_tokens', 512),
            kernel_size=infer_config.get('kernel_size', 7),
            pooling=infer_config.get('pooling', 'maxpool'),
            floor_alpha=infer_config.get('floor_alpha', 0.2),
            pyram_mode=infer_config.get('pyram_mode', False),
            beta=infer_config.get('pyram_beta', 20),
            gqa_support=infer_config.get('gqa_support', False),
            gqa_func=infer_config.get('gqa_func', 'mean')
        )
    elif model_cls == 'kvzip':
        print('💡💡💡 KVzip')
        current_dir = os.path.dirname(os.path.abspath(__file__))
        kvzip_base_dir = os.path.abspath(os.path.join(current_dir, "../../baselines/kvzip"))
        if kvzip_base_dir not in sys.path:
            sys.path.insert(0, kvzip_base_dir)

        from model import ModelKVzip

        kvzip_model = ModelKVzip(
            model_path,
            kv_type=infer_config.get('kv_type', 'evict'),
        )
        base_gen_kwargs = kvzip_model.gen_kwargs.copy()
        default_prefill_chunk_size = int(
            infer_config.get('prefill_chunk_size', infer_config.get('chunk_prefill_size', 16000))
        )
        default_ratio = float(infer_config.get('ratio', 0.3))
        default_level = infer_config.get('level', 'pair')
        default_load_score = _as_bool(infer_config.get('load_score', False))
        default_do_score = _as_bool(infer_config.get('do_score', True))
        default_update_cache = _as_bool(infer_config.get('update_cache', False))
        model = kvzip_model.model
        tokenizer = kvzip_model.tokenizer

        def generate(prompt: Union[str, List[str]], past_key_values=None, **kwargs):
            if past_key_values is not None:
                raise ValueError('KVzip adapter does not support external past_key_values.')

            if isinstance(prompt, str):
                prompts = [prompt]
                is_single = True
            else:
                prompts = prompt
                is_single = False

            prefill_chunk_size = int(kwargs.pop('prefill_chunk_size', default_prefill_chunk_size))
            ratio = float(kwargs.pop('ratio', default_ratio))
            level = kwargs.pop('level', default_level)
            load_score = _as_bool(kwargs.pop('load_score', default_load_score))
            do_score = _as_bool(kwargs.pop('do_score', default_do_score))
            update_cache = _as_bool(kwargs.pop('update_cache', default_update_cache))

            gen_kwargs = base_gen_kwargs.copy()
            gen_kwargs.update({
                'max_new_tokens': kwargs.pop('max_new_tokens', base_gen_kwargs.get('max_new_tokens', 512)),
                'do_sample': kwargs.pop('do_sample', base_gen_kwargs.get('do_sample', False)),
                'temperature': kwargs.pop('temperature', base_gen_kwargs.get('temperature', 1.0)),
                'top_p': kwargs.pop('top_p', base_gen_kwargs.get('top_p', 1)),
                'top_k': kwargs.pop('top_k', base_gen_kwargs.get('top_k', None)),
            })
            num_beams = int(kwargs.pop('num_beams', 1))
            if num_beams != 1:
                raise ValueError(f'KVzip adapter only supports num_beams=1, got {num_beams}')

            eos_token_id = kwargs.pop('eos_token_id', None)
            if eos_token_id is not None:
                gen_kwargs['eos_token_id'] = eos_token_id

            if kwargs:
                raise ValueError(f'Unsupported KVzip generation kwargs: {sorted(kwargs.keys())}')

            results = []
            old_gen_kwargs = kvzip_model.gen_kwargs
            kvzip_model.gen_kwargs = gen_kwargs
            try:
                for single_prompt in prompts:
                    prompt_ids = kvzip_model.encode(single_prompt)
                    prefill_ids = prompt_ids[:, :-1]
                    query_ids = prompt_ids[:, -1:]
                    kv = kvzip_model.prefill(
                        prefill_ids,
                        prefill_chunk_size=prefill_chunk_size,
                        load_score=load_score,
                        do_score=do_score,
                    )
                    kv.prune(ratio=ratio, level=level)
                    results.append(kvzip_model.generate(query_ids, kv=kv, update_cache=update_cache))
            finally:
                kvzip_model.gen_kwargs = old_gen_kwargs

            if return_kv_cache:
                return (results[0], None) if is_single else (results, None)
            return results[0] if is_single else results

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        if return_model:
            return generate, model
        return generate
    else:
        raise ValueError(f"Unknown model_cls: {model_cls}")

    manual_chunk_prefill_size = int(os.environ.get('MANUAL_GEN_CHUNK_PREFILL_SIZE', 0))
    if manual_chunk_prefill_size > 0:
        assert model_cls == 'kivi' or model_cls == 'palu' or model_cls == 'quest' or model_cls == 'kvzip', '其他方法在代码内部实现了chunk prefill'

    model.eval()
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    def generate(prompt: Union[str, List[str]], past_key_values=None, **kwargs):
        if os.getenv('ENABLE_HF_GEN'):
            return hf_gen(model, tokenizer, prompt, return_kv_cache, past_key_values, **kwargs)
        else:
            return manual_generate(model, tokenizer, prompt, past_key_values, return_kv_cache, **kwargs)

    if return_model:
        return generate, model

    return generate
