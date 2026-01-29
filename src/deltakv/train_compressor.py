# -*- coding: utf-8 -*-
import fire
import torch
import os
import wandb
from datetime import datetime
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    set_seed
)
from datasets import load_from_disk, Dataset
from deltakv.data_prepare.data_collator import get_naive_collator
from deltakv.save_trainable_trainer import SaveTrainableParamsTrainer
from accelerate import Accelerator

# 设定torch线程数，优化CPU使用
torch.set_num_threads(8)


def main(
    # 模型与数据路径
    model_name_or_path: str = '/root/autodl-fs/models/Qwen2.5-7B-Instruct-1M',
    dataset_path: str = '/root/autodl-fs/datasets/fineweb-edu-tokenized',
    output_dir: str = '/root/autodl-fs/checkpoints/compressor',
    data_max_len: int = -1,

    # 压缩器相关配置
    model_type: str = 'parallel',  # 'parallel', 'sequential', 'e2e' or 'cluster_e2e'
    kv_compressed_size: int = 64,
    seq_chunk_size: int = 1,
    layer_chunk_size: int = 1,
    recon_mode: str = 'delta_in_latent',  # delta_in_origin or delta_in_latent
    ref_mode: str = 'last',  # last or avg or first
    use_nonlinear_compressor: bool = False,
    compressor_intermediate_size: int = -1,
    compressor_down_type: str = 'auto',  # auto|linear|mlp_gelu|mlp_swiglu
    compressor_up_type: str = 'auto',  # auto|linear|mlp_gelu|mlp_swiglu
    compressor_down_intermediate_size: int = -1,
    compressor_up_intermediate_size: int = -1,
    collect_kv_before_rope: bool = True,
    compressor_linear_bias: bool = True,
    cluster_metric: str = 'l2',
    cluster_on_kv: bool = True,
    cluster_temp: float = 10.0,
    cluster_soft_assignment: bool = True,
    cluster_ratio: float = 0.1,
    split_kv: bool = False,

    # 训练超参数
    num_epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.05,
    gradient_accumulation_steps: int = 1,
    log_steps: int = 10,
    save_steps: float = 0.2,
    save_total_limit: int = 5,
    max_steps: int = -1,
    max_grad_norm: float = 1.0,
    gradient_checkpointing: bool = False,
    use_qlora_style: bool = False,
    use_8bit: bool = False,

    # W&B相关
    project_name: str = "DeltaKV",
    run_name: str = None,
    wandb_group: str = None,
):
    """
    训练KV Cache压缩器的主要脚本。
    
    该脚本执行以下操作:
    1. 加载预训练的Qwen2模型和对应的tokenizer。
    2. 修改模型配置以集成KV Cache压缩层。
    3. 冻结原始模型参数，只训练新增的压缩层权重。
    4. 加载预处理（tokenized和packed）的数据集。
    5. 使用Hugging Face Trainer和Accelerate进行分布式训练。
    6. 训练结束后，仅保存压缩层的权重。
    """
    print("Params:\n", locals())
    set_seed(42)
    accelerator = Accelerator()

    if gradient_checkpointing:
        assert model_type == 'e2e' or model_type == 'cluster_e2e_big', '其他训练方法在中间activation上训练，不能开gradient checkpointing'

    # --- 1. 加载模型和Tokenizer ---
    from transformers import AutoConfig
    raw_config = AutoConfig.from_pretrained(model_name_or_path)
    is_llama = "llama" in raw_config.model_type.lower()
    is_qwen2 = "qwen2" in raw_config.model_type.lower()
    if os.getenv('FORCE_QWEN'):
        is_qwen2 = True
        is_llama = False

    if is_llama:
        from deltakv.configs.model_config_cls import KVLlamaConfig as KVConfig
        if model_type == 'e2e':
            from deltakv.modeling.llama.llama_e2e import LlamaKVCompress as KVCompressModel
        elif model_type == 'cluster_e2e':
            from deltakv.modeling.llama.llama_e2e_cluster import KVModelCompress as KVCompressModel
        else:
            raise ValueError(f"Unknown model_type for Llama: {model_type}")
    elif is_qwen2:
        from deltakv.configs.model_config_cls import KVQwen2Config as KVConfig
        if model_type == 'e2e':
            from deltakv.modeling.qwen2.qwen2_e2e import KVCompressModel
        elif model_type == 'cluster_e2e':
            from deltakv.modeling.qwen2.qwen2_e2e_cluster import Qwen2KVClusterCompress as KVCompressModel
        elif model_type == 'cluster_e2e_big':
            from deltakv.modeling.qwen2.qwen2_e2e_cluster_for_big_model import Qwen2KVClusterCompress as KVCompressModel
        else:
            raise ValueError(f"Unknown model_type for Qwen2: {model_type}")
    else:
        raise ValueError(f"Unsupported model architecture: {raw_config.model_type}")
        
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = KVConfig.from_pretrained(model_name_or_path)
    config.set_extra_args(
        kv_compressed_size=kv_compressed_size,
        seq_chunk_size=seq_chunk_size,
        layer_chunk_size=layer_chunk_size,
        recon_mode=recon_mode,
        ref_mode=ref_mode,
        use_nonlinear_compressor=use_nonlinear_compressor,
        compressor_intermediate_size=compressor_intermediate_size,
        compressor_down_type=compressor_down_type,
        compressor_up_type=compressor_up_type,
        compressor_down_intermediate_size=compressor_down_intermediate_size,
        compressor_up_intermediate_size=compressor_up_intermediate_size,
        collect_kv_before_rope=collect_kv_before_rope,
        compressor_linear_bias=compressor_linear_bias,
        cluster_metric=cluster_metric,
        cluster_on_kv=cluster_on_kv,
        cluster_temp=cluster_temp,
        cluster_soft_assignment=cluster_soft_assignment,
        cluster_ratio=cluster_ratio,
        split_kv=split_kv,
        use_compression=True,
    )
    print(f'[Config]\n{config=}')

    # --- 2. 加载模型 ---
    quantization_config = None
    if use_qlora_style or use_8bit:
        from transformers import BitsAndBytesConfig
        # 排除压缩层，防止它们被量化。
        skip_modules = ["compress_down", "compress_up", "k_compress_down", "k_compress_up", 
                        "v_compress_down", "v_compress_up", "cluster", "transform"]
        
        if use_8bit:
            print("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=skip_modules
            )
        else:
            print("Using QLoRA (4-bit quantization)")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=skip_modules
            )

    # 加载模型
    model = KVCompressModel.from_pretrained(
        model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map='auto' if (use_qlora_style or use_8bit) else 'cuda',
        quantization_config=quantization_config,
    )

    if use_qlora_style or use_8bit:
        from peft import prepare_model_for_kbit_training  # noqa
        # 注意：即便不开启梯度检查点，prepare_model_for_kbit_training 也会处理一些量化模型训练所需的 LayerNorm 和 input 梯度逻辑
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)

    if gradient_checkpointing or use_qlora_style or use_8bit:
        # 当基础模型被冻结且使用梯度检查点，或者进行量化训练时，必须开启此项以确保梯度流
        model.enable_input_require_grads()
    
    torch.cuda.empty_cache()

    # --- 3. 冻结非压缩层参数 ---
    # 我们只训练压缩相关的权重
    print("--- Trainable Parameters ---")
    for name, param in model.named_parameters():
        if 'compress' not in name and 'cluster' not in name and 'transform' not in name:
            param.requires_grad = False
        else:
            # 确保可训练参数具有梯度
            param.requires_grad = True
            # 在非量化模式下统一转为bf16，在量化模式下 prepare_model_for_kbit_training 会处理
            # 如果skip成功，理论上应该就是bf16吧，哦，会给转为 fp32
            # assert param.data.dtype == torch.bfloat16, f'{param.data.dtype=}'
            if use_qlora_style or use_8bit:
                param.data = param.data.to(torch.bfloat16)
            print(f"  - {name}")
    print("--------------------------")

    if use_qlora_style or use_8bit:
        # 绕过 Trainer 的安全检查：Trainer 不允许对非 PeftModel 的量化模型进行微调
        # 但我们手动添加了可训练的压缩层，所以这是安全的。
        model.is_quantized = False

    # --- 4. 加载数据集 ---
    tokenized_dataset = load_from_disk(dataset_path)
    if isinstance(tokenized_dataset, Dataset):
        train_dataset = tokenized_dataset
    elif 'train' in tokenized_dataset:
        train_dataset = tokenized_dataset['train']
    else:
        raise ValueError("无法在数据集中找到'train'分割，请确保数据集已正确准备。")

    if data_max_len > 0:
        print(f"截断数据集样本长度为: {data_max_len}")
        train_dataset = train_dataset.map(
            lambda x: {k: v[:data_max_len] if isinstance(v, list) else v for k, v in x.items()},
            batched=True,
            num_proc=16,
            desc=f"Truncating dataset to {data_max_len}"
        )

    # --- 5. 配置训练器 ---
    # 根据超参数创建唯一的输出目录和运行名称
    if model_type == 'e2e':
        run_name_suffix = f"e2e_ref{ref_mode}_cs{kv_compressed_size}_scs{seq_chunk_size}_lcs{layer_chunk_size}"
        if split_kv:
            run_name_suffix += "_split"
    elif model_type in ['cluster_e2e', 'cluster_e2e_big']:
        run_name_suffix = f"{model_type}_cs{kv_compressed_size}_bias{compressor_linear_bias}_{cluster_metric}_ratio{cluster_ratio}"
        if not cluster_on_kv:
            run_name_suffix += "_clusSoft"
        if not cluster_soft_assignment:
            run_name_suffix += "_clusMean"
    else:  # parallel or sequential
        raise ValueError

    if model_type in ['e2e', 'cluster_e2e']:
        if collect_kv_before_rope:
            run_name_suffix += "_before_rope"
        else:
            run_name_suffix += "_after_rope"

    if use_qlora_style:
        run_name_suffix += "_qlora"
    if use_8bit:
        run_name_suffix += "_8bit"

    run_name_suffix += f"_lr{learning_rate}"

    has_directional_arch = (
        compressor_down_type != 'auto'
        or compressor_up_type != 'auto'
        or compressor_down_intermediate_size > 0
        or compressor_up_intermediate_size > 0
    )
    if has_directional_arch:
        run_name_suffix += f"_cdown{compressor_down_type}"
        if compressor_down_intermediate_size > 0:
            run_name_suffix += f"d{compressor_down_intermediate_size}"
        run_name_suffix += f"_cup{compressor_up_type}"
        if compressor_up_intermediate_size > 0:
            run_name_suffix += f"u{compressor_up_intermediate_size}"
    elif use_nonlinear_compressor:
        run_name_suffix += f"_nonlinear_inter{compressor_intermediate_size}"

    # 加入时间戳，避免覆盖保存
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    final_output_dir = os.path.join(output_dir, f"{run_name_suffix}_{timestamp}")

    if accelerator.is_main_process:
        os.makedirs(final_output_dir, exist_ok=True)
        if run_name is None:
            run_name = f"{run_name_suffix}_{timestamp}"
        wandb.init(project=project_name, name=run_name, config=locals(), group=wandb_group if wandb_group else model_type)

    training_args = TrainingArguments(
        output_dir=final_output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        logging_steps=log_steps,
        save_steps=save_steps,
        max_steps=max_steps,
        save_total_limit=save_total_limit,
        bf16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        max_grad_norm=max_grad_norm,
        logging_dir=f"{final_output_dir}/logs",
        report_to="wandb" if accelerator.is_main_process else "none",
        remove_unused_columns=False, # 我们不需要label，所以设为False
        gradient_checkpointing=gradient_checkpointing,
    )

    trainer = SaveTrainableParamsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=get_naive_collator(tokenizer),
    )

    # --- 6. 开始训练 ---
    trainer.train()

    trainer.save_model()


if __name__ == '__main__':
    fire.Fire(main)
