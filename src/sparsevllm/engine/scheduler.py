from collections import deque

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence, SequenceStatus
from sparsevllm.engine.cache_manager import CacheManager
from sparsevllm.utils.log import logger


class Scheduler:
    """
    请求调度器，负责管理待处理 (waiting) 和正在运行 (running) 的序列。
    主要职责：
    1. 决定每一轮 (step) GPU 应该处理哪些序列。
    2. 实现分块 Prefill (Chunked Prefill) 以处理长序列。
    3. 管理逻辑显存额度，并在显存不足时触发抢占 (Preemption/Eviction)。
    """

    def __init__(self, config: Config, memory_oracle: CacheManager):
        self.config = config
        self.max_num_seqs_in_batch = config.max_num_seqs_in_batch
        self.max_num_batched_tokens = config.max_num_batched_tokens
        logger.debug(f'set max_num_batched_tokens = {config.max_num_batched_tokens} in Scheduler')
        self.max_decoding_seqs = config.max_decoding_seqs

        self.chunk_prefill_size = config.chunk_prefill_size
        self.eos = config.eos

        self.num_sink_tokens = config.num_sink_tokens
        self.num_recent_tokens = config.num_recent_tokens
        self.num_top_tokens = config.num_top_tokens
        
        # memory_oracle 引用 Rank 0 的 CacheManager，作为全局显存余量参考。
        # 对多层异构预算，采用更保守的可用空间估计。
        self.memory_oracle = memory_oracle
        
        self.waiting: deque[Sequence] = deque()
        self.decoding: deque[Sequence] = deque()

    def _long_text_threshold(self, is_prefill: bool) -> int:
        """Long-text boundary for batch partitioning.

        Prefill: based on prompt length + chunk prefill size.
        Decode: based on current total tokens (prompt + generated), without chunk size.
        """
        base = self.num_sink_tokens + self.num_top_tokens + self.num_recent_tokens
        return base + (self.chunk_prefill_size if is_prefill else 0)

    def _is_long_text(self, seq: Sequence, is_prefill: bool) -> bool:
        threshold = self._long_text_threshold(is_prefill)
        seq_len = seq.num_prompt_tokens if is_prefill else seq.num_tokens
        return int(seq_len) > int(threshold)

    def _pop_waiting_at(self, idx: int) -> Sequence:
        if idx == 0:
            return self.waiting.popleft()
        self.waiting.rotate(-idx)
        seq = self.waiting.popleft()
        self.waiting.rotate(idx)
        return seq

    def _pop_next_prefill_seq(self, target_is_long: bool) -> Sequence | None:
        if not self.waiting:
            return None
        for idx, seq in enumerate(self.waiting):
            if self._is_long_text(seq, is_prefill=True) == target_is_long:
                return self._pop_waiting_at(idx)
        return None

    def _pop_decoding_at(self, idx: int) -> Sequence:
        if idx == 0:
            return self.decoding.popleft()
        self.decoding.rotate(-idx)
        seq = self.decoding.popleft()
        self.decoding.rotate(idx)
        return seq

    def _pop_next_decoding_seq(self, target_is_long: bool) -> Sequence | None:
        if not self.decoding:
            return None
        for idx, seq in enumerate(self.decoding):
            if self._is_long_text(seq, is_prefill=False) == target_is_long:
                return self._pop_decoding_at(idx)
        return None

    def is_finished(self):
        """判断所有请求是否已处理完成"""
        return len(self.waiting) == 0 and len(self.decoding) == 0

    def add(self, seq: Sequence):
        """将新请求加入等待队列"""
        self.waiting.append(seq)

    def _reserved_prefill_tokens(self) -> int:
        return int(self.memory_oracle.reserved_prefill_slots(self.waiting, self.chunk_prefill_size))

    def schedule(self) -> tuple[list[Sequence], bool, list[Sequence]]:
        """
        核心调度逻辑。
        返回：(本次要运行的序列列表, 是否是 Prefill 阶段, 本次被抢占的序列列表)
        
        注意：目前为了简化算子实现，单次 step 不支持 Prefill 和 Decode 混合。
        """
        scheduled_seqs = []
        preempted_seqs = []
        num_batched_seqs = 0
        num_batched_tokens = 0
        
        # 逻辑可用空间计数器，用于在本轮调度中预估显存占用
        physical_free_count = self.memory_oracle.num_free_slots
        logical_free_count = max(0, physical_free_count - self._reserved_prefill_tokens())
        step_free_count = physical_free_count
        admission_budgets = dict(self.memory_oracle.prompt_admission_budgets())
        margin_batched_tokens = self.memory_oracle.prefill_batched_tokens_margin()

        # --- 阶段 1: Prefill 调度 ---
        # 只要 waiting 队列有活，就优先处理 Prefill，因为它是计算密集型的。
        target_is_long = self._is_long_text(self.waiting[0], is_prefill=True) if self.waiting else False

        while (self.waiting and step_free_count > 0 and
               num_batched_tokens <= self.max_num_batched_tokens - margin_batched_tokens and
               num_batched_seqs < self.max_num_seqs_in_batch and len(self.decoding) < self.max_decoding_seqs):

            seq = self._pop_next_prefill_seq(target_is_long)
            if seq is None:
                break
            remaining_prefill_tokens = self.memory_oracle.remaining_prefill_tokens(seq)
            
            # 异常处理：如果由于某种原因已经 prefill 完却还在 waiting 队列
            if remaining_prefill_tokens <= 0:
                raise ValueError('BUG：理论上不应该在 waiting 里')

            # 确定本次 Chunk 的大小
            can_prefill_tokens = min(
                remaining_prefill_tokens,
                self.chunk_prefill_size,
                self.max_num_batched_tokens - num_batched_tokens,
                step_free_count,
            )

            if can_prefill_tokens <= 0:
                logger.debug(f'{can_prefill_tokens=} 结束 schedule prefill 请求')
                self.waiting.appendleft(seq)
                break
            
            # 逻辑显存分配检查：如果是新序列的起始，检查是否能容纳完整的 Prompt 长度。
            # 采用保守策略：预先逻辑占位整个 Prompt，即使后续可能会有稀疏逐出。
            # 只要我想尽可能地持续生成某个序列，那就应该提前都申请出来
            if seq.num_prefilled_tokens == 0:
                costs = self.memory_oracle.prompt_admission_costs(seq)
                failed = None
                for name, need in costs.items():
                    free = int(admission_budgets.get(name, 0) or 0)
                    if free < int(need):
                        failed = (name, int(need), free)
                        break
                if failed is not None:
                    action = self.memory_oracle.prompt_admission_failure_action()
                    if action == "defer":
                        # Queue: stop admitting new prompts; let other work free space first.
                        self.waiting.appendleft(seq)
                        break
                    # Fail fast instead of returning an empty schedule (which would deadlock callers).
                    name, need, free = failed
                    raise RuntimeError(
                        "Insufficient KV cache slots to admit prompt. "
                        f"cache_manager={type(self.memory_oracle).__name__} prompt_len={seq.num_prompt_tokens} "
                        f"failed_budget={name} need={need} free={free} budgets={admission_budgets} "
                        f"free_slots={physical_free_count} reserved_prefill={self._reserved_prefill_tokens()} "
                        f"logical_free={logical_free_count}"
                    )
                for name, need in costs.items():
                    admission_budgets[name] = int(admission_budgets.get(name, 0) or 0) - int(need)
                self.memory_oracle.on_prompt_admitted(seq, costs)
                logical_need = self.memory_oracle.prompt_logical_reservation_cost(seq)
                if logical_free_count < logical_need:
                    # Fail fast: this cache manager requires the logical budget to hold the whole prompt.
                    raise RuntimeError(
                        "Insufficient logical KV cache slots to admit prompt. "
                        f"cache_manager={type(self.memory_oracle).__name__} prompt_len={seq.num_prompt_tokens} "
                        f"logical_need={logical_need} logical_free={logical_free_count} "
                        f"free_slots={physical_free_count} reserved_prefill={self._reserved_prefill_tokens()}"
                    )
                logical_free_count -= int(logical_need)

            # 设置当前 Chunk 属性并标记状态
            logger.debug(f'Add chunk prefill with {can_prefill_tokens} tokens.')
            seq.current_chunk_size = can_prefill_tokens
            num_batched_seqs += 1
            num_batched_tokens += can_prefill_tokens
            step_free_count -= can_prefill_tokens
            seq.status = SequenceStatus.RUNNING
            scheduled_seqs.append(seq)

        # 如果有 Prefill 请求被选中，直接返回，本次 step 只跑 Prefill。
        if scheduled_seqs:
            return scheduled_seqs, True, []

        # --- 阶段 2: Decode 调度 ---
        # 只有在没有 Prefill 任务时才处理增量生成任务。
        # Decode 优先短序列：如果当前 decoding 队列里存在 short，则本轮只调度 short；
        # 仅当全部都是 long 时，才调度 long。（避免 short 被 long 淹没）
        if self.decoding:
            decode_threshold = self._long_text_threshold(is_prefill=False)
            has_short_decode = any(int(seq.num_tokens) <= int(decode_threshold) for seq in self.decoding)
            target_is_long_decode = not has_short_decode
        else:
            target_is_long_decode = False
        while self.decoding and num_batched_seqs < self.max_num_seqs_in_batch:
            seq = self._pop_next_decoding_seq(target_is_long_decode)
            if seq is None:
                break

            # 检查逻辑空间是否够塞下一个新 Token (Decode 步进)
            if logical_free_count < 1:
                # 显存耗尽，触发驱逐/抢占逻辑
                # 策略：牺牲当前 seq，并立刻返回，让上层先释放槽位再进入下一轮调度。
                # 这样可以避免在一次 schedule() 调用中反复驱逐多个请求造成抖动。
                victim = seq
                victim.status = SequenceStatus.WAITING
                victim.num_prefilled_tokens = 0  # 重置进度，下次回来重新跑 Prefill
                self.waiting.appendleft(victim) # 插到队首，显存释放后优先处理
                preempted_seqs.append(victim)
                logger.warning(f'驱逐请求 id = {victim.seq_id} | slots={self.memory_oracle.free_slot_stats()}')
                return [], False, preempted_seqs
            else:
                # 逻辑预占 1 个位子
                logical_free_count -= 1
                num_batched_seqs += 1
                scheduled_seqs.append(seq)
                # logger.debug('Add a decode req.')
        
        if not scheduled_seqs:
            return [], False, preempted_seqs
            
        # 将被选中的 Decode 序列放回 running 队列以保持顺序
        self.decoding.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False, preempted_seqs

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        模型运行后的后处理工作。
        1. 更新 Token 序列。
        2. 更新 Prefill 进度。
        3. 处理序列完成状态 (EOS 或 Max Tokens)。
        """
        if is_prefill:
            for seq, token_id in zip(seqs, token_ids):
                seq.num_prefilled_tokens += seq.current_chunk_size
                # 检查 Chunked Prefill 是否完成
                if seq.num_prefilled_tokens < seq.num_prompt_tokens:
                    # 没跑完，塞回等待队列头部下次继续
                    seq.status = SequenceStatus.WAITING
                    self.waiting.appendleft(seq)
                else:
                    # Prefill 彻底结束，进入正常生成流程
                    seq.status = SequenceStatus.RUNNING
                    self.decoding.append(seq)
                    # 记录模型生成的第一个 Token
                    seq.append_token(token_id)
                    # 检查是否命中结束条件
                    if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                        seq.status = SequenceStatus.FINISHED
                        self.decoding.remove(seq)
            return

        # 处理 Decode 步骤
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                if seq in self.decoding:
                    self.decoding.remove(seq)
