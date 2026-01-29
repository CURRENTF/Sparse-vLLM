# -*- coding: utf-8 -*-
import fire
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, DefaultDataCollator
from datasets import load_from_disk
from torch.utils.data import DataLoader


def main(
    model_path: str,
    dataset_path: str,
    num_samples: int = 1000,
    subset: str = None,
):
    """
    计算给定模型在数据集子集上的平均损失。

    Args:
        model_path (str): 预训练模型的路径。
        dataset_path (str): tokenized数据集的路径。
        num_samples (int, optional): 默认评估的样本数量（从0开始）。默认为 1000。
        subset (str, optional): 指定评估的子集范围，格式为 'start_idx:end_idx'。
    """
    print(f"Loading model from: {model_path}")
    print(f"Loading dataset from: {dataset_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    model.to(device)
    model.eval()

    print("Loading dataset...")
    try:
        full_dataset = load_from_disk(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        return

    if isinstance(full_dataset, dict) and 'train' in full_dataset:
        dataset_to_eval = full_dataset['train']
    else:
        dataset_to_eval = full_dataset

    # Handle subset selection
    if subset:
        try:
            start_idx, end_idx = map(int, subset.split(':'))
            start_idx = max(0, start_idx)
            end_idx = min(len(dataset_to_eval), end_idx)
            indices = range(start_idx, end_idx)
            print(f"Evaluating on subset range: {start_idx}:{end_idx} (Total: {len(indices)} samples)")
        except ValueError:
            print(f"Error: Invalid subset format '{subset}'. Expected 'start:end'.")
            return
    else:
        num_samples = min(num_samples, len(dataset_to_eval))
        indices = range(num_samples)
        print(f"Evaluating on first {num_samples} samples")

    eval_dataset = dataset_to_eval.select(indices)
    num_eval_samples = len(eval_dataset)

    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        progress_bar = tqdm.tqdm(eval_dataset, desc="Evaluating Loss", leave=False)
        for sample in progress_bar:
            # The data collator should automatically create 'labels' from 'input_ids'
            input_ids = torch.tensor(sample['input_ids'], dtype=torch.long).to(model.device)
            input_ids = input_ids[None]
            assert input_ids.ndim == 2, f"{input_ids.shape}"

            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            if loss is not None:
                total_loss += loss.item()
                total_steps += 1
                progress_bar.set_postfix({"loss": loss.item()})
            else:
                print("Warning: Model output does not contain loss. Skipping batch.")

    if total_steps > 0:
        avg_loss = total_loss / total_steps
        print(f"\nAverage loss over {num_eval_samples} samples: {avg_loss:.4f}")
    else:
        print("\nCould not calculate average loss. No steps were completed.")


if __name__ == '__main__':
    fire.Fire(main)
