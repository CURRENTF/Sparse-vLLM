import torch
from transformers import Qwen2Tokenizer

def get_naive_collator(tokenizer: Qwen2Tokenizer):
    def data_collator(batch_lis):
        res = {"input_ids": []}
        for sample in batch_lis:
            res['input_ids'].append(sample['input_ids'])

        res = tokenizer.pad(res, return_tensors="pt")
        assert res['input_ids'].ndim == 2, f"{res['input_ids'].shape}"
        res["labels"] = res["input_ids"]
        return res

    return data_collator