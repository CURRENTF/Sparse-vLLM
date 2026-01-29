# -*- coding: utf-8 -*-
import fire
from datasets import load_dataset, Dataset
from sympy import limit
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import logging
from multiprocessing import Pool
from typing import Iterator, List

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 多进程相关函数 ---

# 全局tokenizer，用于在每个工作进程中初始化
_tokenizer = None

def init_worker(model_name_or_path: str):
    """
    多进程Pool的初始化函数，在每个工作进程中加载tokenizer。
    """
    global _tokenizer
    logging.info(f"工作进程 {os.getpid()} 正在加载Tokenizer...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

def tokenize_batch(batch_of_texts: List[str]) -> List[List[int]]:
    """
    【优化点】实际执行tokenize的函数，由工作进程调用。
    此函数现在接收一个文本列表（batch），并使用tokenizer的批处理能力进行处理。
    """
    global _tokenizer
    if not batch_of_texts:
        return []
    
    # 为批次中的每段文本手动包裹BOS/EOS

    texts_with_special_tokens = [f"{_tokenizer.bos_token}{t}{_tokenizer.eos_token}" for t in batch_of_texts]  # noqa
    
    # 使用tokenizer的批处理调用（__call__），效率远高于循环调用encode
    return _tokenizer(texts_with_special_tokens, add_special_tokens=False)['input_ids']  # noqa

def batch_generator(dataset_iterator: Iterator, batch_size: int) -> Iterator[List[str]]:
    """
    一个生成器，从数据流中按批次大小(batch_size)产出文本列表。
    """
    batch = []
    for example in dataset_iterator:
        text = example.get('text')
        if text:
            batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# --- 数据处理主类 ---

class Processor:
    """
    数据处理类，用于对文本数据进行tokenize和packing。
    """

    def __init__(self, model_name_or_path: str = 'Qwen/Qwen2.5-7B-Instruct-1M'):
        self.model_name_or_path = model_name_or_path
        # 主进程的tokenizer主要用于获取配置信息，实际工作由子进程完成
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def process_and_save(self,
                         dataset_name: str = 'HuggingFaceFW/fineweb-edu',
                         # dataset_sub_name: str = 'sample-10BT',
                         output_dir: str = '/root/autodl-fs/datasets/fineweb-edu-tokenized',
                         max_seq_len: int = 2048,
                         num_proc: int = 8,
                         text_batch_size: int = 1000):
        """
        核心处理函数：流式加载、多进程tokenize、打包并保存。

        :param dataset_name: Hugging Face Hub上的数据集名称。
        # :param dataset_sub_name: 数据集的子集或配置名称。
        :param output_dir: 处理后数据的保存目录。
        :param max_seq_len: 打包后每条数据的固定序列长度。
        :param num_proc: 用于tokenize的工作进程数，默认为CPU核心数。
        :param text_batch_size: 每个进程一次处理的文本批次大小。
        """
        logging.info(f"开始处理数据集: {dataset_name}，使用 {num_proc} 个进程。")
        os.makedirs(output_dir, exist_ok=True)

        dataset = load_dataset(dataset_name, split='train[:20%]').shuffle(seed=42)
        num_examples = len(dataset)
        total_batches = (num_examples + text_batch_size - 1) // text_batch_size
        logging.info(f"数据集包含 {num_examples} 个样本, 将分为 {total_batches} 个批次进行处理。")

        token_buffer = []
        packed_samples = []

        with Pool(num_proc, initializer=init_worker, initargs=(self.model_name_or_path,)) as pool:
            # 从数据流创建文本批次生成器
            text_batches = batch_generator(dataset, text_batch_size)  # noqa

            # 【优化点】为tqdm提供total参数，以显示准确的进度和预估时间
            progress_bar = tqdm(
                pool.imap(tokenize_batch, text_batches),
                total=total_batches,
                desc="Tokenizing Batches"
            )
            for list_of_tokenized_texts in progress_bar:
                # list_of_tokenized_texts 是一个二维列表, e.g., [[...], [...], ...]
                for tokens in list_of_tokenized_texts:
                    token_buffer.extend(tokens)
                    
                    # 缓冲池打包逻辑保持不变
                    while len(token_buffer) >= max_seq_len:
                        packed_sequence = token_buffer[:max_seq_len]
                        packed_samples.append({'input_ids': packed_sequence})
                        token_buffer = token_buffer[max_seq_len:]

        logging.info("数据处理完毕。")
        logging.warning(f"缓冲池中剩余 {len(token_buffer)} 个token将被丢弃。")
        logging.info(f"总共打包了 {len(packed_samples)} 条数据。")

        if packed_samples:
            logging.info("正在将打包好的数据转换为Hugging Face Dataset对象...")
            final_dataset = Dataset.from_dict({'input_ids': [sample['input_ids'] for sample in packed_samples]})
            
            logging.info(f"开始将处理好的数据集保存到磁盘: {output_dir}")
            final_dataset.save_to_disk(output_dir)
            logging.info("数据集保存成功！")
        else:
            logging.warning("没有生成任何打包数据。请检查数据集是否为空或max_seq_len设置是否过大。")


if __name__ == '__main__':
    fire.Fire(Processor)
