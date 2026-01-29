# -*- coding: utf-8 -*-
import fire
import torch
import tqdm
import torch.nn.functional as F
from datasets import load_from_disk

from deltakv.modeling.qwen2.qwen2_e2e import Qwen2KVCompress, Qwen2AttnKVCompress
from deltakv.configs.model_config_cls import KVQwen2Config

def patch_model_for_chunk_avg(model):
    """
    将模型中的滑动窗口平均逻辑替换为推理时的 Chunk 平均逻辑。
    """
    def patched_single_tensor_comp_then_reconstruct(self, kv, prev_recon_kv, compress_down, compress_up, layer_transform, compress_alpha):
        bs, seq_len, dim = kv.shape
        kv_chunks = kv.view(bs, -1, self.config.seq_chunk_size, dim)

        use_seq_ref = self.config.seq_chunk_size > 1
        use_layer_ref = self.config.layer_chunk_size > 1 and prev_recon_kv is not None

        seq_ref = None
        if use_seq_ref:
            # 关键修改：使用 Chunk 内的平均值作为参考，而不是滑动窗口
            # kv_chunks shape: (bs, num_chunks, chunk_size, dim)
            seq_ref = kv_chunks.mean(dim=2, keepdim=True) 
            # 广播将处理 (bs, num_chunks, 1, dim) -> (bs, num_chunks, chunk_size, dim)

        if use_seq_ref and use_layer_ref:
            a = torch.sigmoid(compress_alpha)
            # transform(kv_i-1)
            layer_ref = layer_transform(prev_recon_kv).view(bs, -1, self.config.seq_chunk_size, dim)

            # Formula: comp_kv_i = down(kv_i) - a*down(seq_ref_i) - (1-a)*down(layer_ref)
            comp_kv = compress_down(kv_chunks) - a * compress_down(seq_ref) - (1 - a) * compress_down(layer_ref)

            # Recon: recon_kv = up(comp_kv) + a*seq_ref + (1-a)*layer_ref
            recon_kv = compress_up(comp_kv) + a * seq_ref + (1 - a) * layer_ref

        elif use_seq_ref:
            comp_kv = compress_down(kv_chunks) - compress_down(seq_ref)
            recon_kv = compress_up(comp_kv) + seq_ref

        elif use_layer_ref:
            layer_ref = layer_transform(prev_recon_kv).view(bs, -1, self.config.seq_chunk_size, dim)
            comp_kv = compress_down(kv_chunks) - compress_down(layer_ref)
            recon_kv = compress_up(comp_kv) + layer_ref
        else:
            # 纯自编码模式
            comp_kv = compress_down(kv_chunks)
            recon_kv = compress_up(comp_kv)

        return recon_kv.reshape(bs, seq_len, dim)

    # 遍历所有模块进行替换
    count = 0
    for m in model.modules():
        if isinstance(m, Qwen2AttnKVCompress):
            # 使用 __get__ 将函数绑定为实例方法
            m._single_tensor_comp_then_reconstruct = patched_single_tensor_comp_then_reconstruct.__get__(m, Qwen2AttnKVCompress)
            count += 1
    print(f"Patched {count} Qwen2AttnKVCompress layers with Chunk Average logic.")

def main(
    model_path: str,
    compressor_path: str,
    dataset_path: str,
    num_samples: int = 100,
    batch_size: int = 1,
    do_patch: bool = True,
):
    """
    测试脚本：评估基于滑动窗口训练的模型在 Chunk 平均模式下的性能。
    """
    import os
    from safetensors.torch import load_file
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading config from: {compressor_path}")
    config = KVQwen2Config.from_pretrained(compressor_path)
    print(f'{config=}')
    
    print(f"Loading base model from: {model_path}")
    model = Qwen2KVCompress.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    
    print(f"Loading compressor weights from: {compressor_path}")
    # 根据 safetensors 文件名加载权重
    sf_path = os.path.join(compressor_path, "model.safetensors")
    if not os.path.exists(sf_path):
        # 兼容某些可能保存为 pytorch_model.bin 的情况
        sf_path = os.path.join(compressor_path, "pytorch_model.bin")
        if os.path.exists(sf_path):
            comp_state_dict = torch.load(sf_path, map_location=device)
        else:
            raise FileNotFoundError(f"Could not find weights in {compressor_path}")
    else:
        comp_state_dict = load_file(sf_path, device=str(device))
    
    _, unexpected = model.load_state_dict(comp_state_dict, strict=False)
    if len(unexpected) > 0:
        print(f"Warning: Unexpected keys when loading compressor: {unexpected}")
    
    model.to(device)
    model.eval()

    # 应用 Monkey Patch
    if do_patch:
        patch_model_for_chunk_avg(model)
    else:
        print("Skipping Monkey Patch, using raw model logic (likely sliding window).")

    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    if 'train' in dataset:
        dataset = dataset['train']
    
    num_samples = min(num_samples, len(dataset))
    eval_dataset = dataset.select(range(num_samples))
    
    total_ntp_loss = 0.0
    total_mse_loss = 0.0
    total_steps = 0

    print(f"Starting evaluation on {num_samples} samples...")
    with torch.no_grad():
        for i in tqdm.trange(0, num_samples, batch_size):
            batch = eval_dataset[i : i + batch_size]
            input_ids = torch.tensor(batch['input_ids'], dtype=torch.long).to(device)
            
            # Qwen2KVCompress.forward 会自动计算 NTP loss + MSE loss
            # 我们需要拦截它以分别统计
            outputs = model(input_ids=input_ids, labels=input_ids)
            
            # 在 Qwen2KVCompress 中，loss = ntp_loss + mse_loss
            # 我们重新手动统计一次各个部分的损耗
            
            current_ntp_loss = 0
            current_mse_loss = 0
            
            # 手动计算 NTP Loss 以确保准确分离
            logits = outputs.logits
            labels = input_ids
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ntp_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # 统计 MSE Loss
            for n, mod in model.named_modules():
                if isinstance(mod, Qwen2AttnKVCompress):
                    current_mse_loss += model.mse(mod.buffer_recon_kv, mod.buffer_raw_kv).item()
            
            total_ntp_loss += ntp_loss.item()
            total_mse_loss += current_mse_loss
            total_steps += 1

    avg_ntp = total_ntp_loss / total_steps
    avg_mse = total_mse_loss / total_steps
    
    print("\n" + "="*30)
    mode_str = "Chunk Average Mode" if do_patch else "Raw Model Mode"
    print(f"Results ({mode_str}):")
    print(f"Average NTP Loss: {avg_ntp:.6f}")
    print(f"Average MSE Loss: {avg_mse:.6f}")
    print(f"Total Combined:   {avg_ntp + avg_mse:.6f}")
    print("="*30)

if __name__ == '__main__':
    fire.Fire(main)
