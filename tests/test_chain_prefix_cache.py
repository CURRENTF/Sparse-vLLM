from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from sparsevllm.engine.cache_manager.h2o import H2OCacheManager
from sparsevllm.engine.cache_manager.snapkv import SnapKVCacheManager
from sparsevllm.engine.chain_cache import (
    ChainAdmissionPlan,
    ChainBusyError,
    ChainCacheIndex,
    ChainCapacityError,
    ChainFingerprintMismatchError,
    ChainGoneError,
    ChainNotFoundError,
    ChainOwnerMismatchError,
    ChainPrefixMismatchError,
    ChainState,
    build_chain_cache_fingerprint,
    normalize_prefix_cache_mode,
    stable_token_digest,
)
from sparsevllm.engine.chain_cache import ChainCacheCoordinator
from sparsevllm.engine.llm_engine import LLMEngine
from sparsevllm.engine.runtime_state import RuntimeState
from sparsevllm.engine.sequence import Sequence
from sparsevllm.sampling_params import SamplingParams


FINGERPRINT = b"chain-test-fingerprint"


def _create_and_finish(
    index: ChainCacheIndex,
    chain_id: str,
    seq_id: int,
    token_ids: list[int],
    *,
    physical_slots: tuple[int, ...] = (3, 3),
):
    plan = index.plan_admission(
        chain_id=chain_id,
        seq_id=seq_id,
        token_ids=token_ids,
        fingerprint=FINGERPRINT,
    )
    index.apply_admission(plan, fingerprint=FINGERPRINT)
    return index.finish(
        chain_id,
        token_ids=token_ids,
        processed_token_count=len(token_ids),
        physical_slots_by_layer=physical_slots,
    )


def test_chain_create_finish_resume_and_processed_digest():
    index = ChainCacheIndex()
    created = index.plan_admission(
        chain_id="chain-a",
        seq_id=7,
        token_ids=[1, 2, 3],
        fingerprint=FINGERPRINT,
    )
    assert created.status == "created"
    assert created.reused_tokens == 0
    record = index.apply_admission(created, fingerprint=FINGERPRINT)
    assert record.state is ChainState.ACTIVE

    with pytest.raises(ChainBusyError):
        index.plan_admission(
            chain_id="chain-a",
            seq_id=7,
            token_ids=[1, 2, 3, 4],
            fingerprint=FINGERPRINT,
        )

    index.finish(
        "chain-a",
        token_ids=[1, 2, 3],
        processed_token_count=3,
        physical_slots_by_layer=(2, 3),
    )
    resumed = index.plan_admission(
        chain_id="chain-a",
        seq_id=7,
        token_ids=[1, 2, 3, 4, 5],
        fingerprint=FINGERPRINT,
    )
    assert resumed.status == "resumed"
    assert resumed.reused_tokens == 3
    assert resumed.seq_id == 7


def test_chain_text_continuation_preserves_resident_token_identity():
    class MergeTokenizer:
        bos_token = None

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            values = {
                "ab": [3],
                "ab!": [3, 4],
                "!": [4],
            }
            return values[text]

        def decode(
            self,
            token_ids,
            *,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            del skip_special_tokens, clean_up_tokenization_spaces
            if list(token_ids) != [1, 2]:
                raise AssertionError(token_ids)
            return "ab"

    index = ChainCacheIndex()
    record = _create_and_finish(index, "chain-a", 1, [1, 2])
    engine = object.__new__(LLMEngine)
    engine.tokenizer = MergeTokenizer()

    assert engine._tokenize_prompt("ab!") == [3, 4]
    assert engine._tokenize_chain_continuation("ab!", record) == [1, 2, 4]


def test_chain_text_continuation_preserves_identity_after_automatic_bos():
    class MergeTokenizer:
        bos_token = "<bos>"

        def encode(self, text, add_special_tokens=False):
            values = {
                "ab!": [3, 4],
                "!": [4],
            }
            token_ids = values[text]
            return [0, *token_ids] if add_special_tokens else token_ids

        def decode(
            self,
            token_ids,
            *,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ):
            del skip_special_tokens, clean_up_tokenization_spaces
            if list(token_ids) != [1, 2]:
                raise AssertionError(token_ids)
            return "ab"

    index = ChainCacheIndex()
    record = _create_and_finish(index, "chain-a", 1, [0, 1, 2])
    engine = object.__new__(LLMEngine)
    engine.tokenizer = MergeTokenizer()

    assert engine._tokenize_prompt("ab!") == [0, 3, 4]
    assert engine._tokenize_chain_continuation("ab!", record) == [0, 1, 2, 4]


def test_discard_chain_targets_expected_resident_sequence():
    class Scheduler:
        def __init__(self):
            self.aborted = []

        def abort(self, seq_id):
            self.aborted.append(int(seq_id))

    class ModelRunner:
        def __init__(self, coordinator):
            self.runtime_state = SimpleNamespace(
                chain_cache_coordinator=coordinator,
            )
            self.calls = []

        def call(self, method, *args):
            self.calls.append((method, args))

    record = SimpleNamespace(seq_id=7)
    coordinator = SimpleNamespace(
        index=SimpleNamespace(records={"chain-a": record}),
    )
    engine = object.__new__(LLMEngine)
    engine.scheduler = Scheduler()
    engine.model_runner = ModelRunner(coordinator)
    engine._active_chain_sequences = {7: SimpleNamespace()}

    assert engine.discard_chain("missing", expected_seq_id=8) is False
    with pytest.raises(ChainOwnerMismatchError):
        engine.discard_chain("chain-a", expected_seq_id=8)
    assert engine.discard_chain("chain-a", expected_seq_id=7) is True
    assert engine.scheduler.aborted == [7]
    assert engine._active_chain_sequences == {}
    assert engine.model_runner.calls == [
        ("chain_invalidate", ("chain-a", 7)),
    ]


def test_chain_resume_rejects_digest_and_fingerprint_mismatch():
    index = ChainCacheIndex()
    _create_and_finish(index, "chain-a", 1, [10, 20, 30])

    with pytest.raises(ChainPrefixMismatchError):
        index.plan_admission(
            chain_id="chain-a",
            seq_id=1,
            token_ids=[10, 99, 30, 40],
            fingerprint=FINGERPRINT,
        )
    with pytest.raises(ChainPrefixMismatchError) as shorter:
        index.plan_admission(
            chain_id="chain-a",
            seq_id=1,
            token_ids=[10, 20],
            fingerprint=FINGERPRINT,
        )
    assert shorter.value.chain_id == "chain-a"
    with pytest.raises(ChainFingerprintMismatchError):
        index.plan_admission(
            chain_id="chain-a",
            seq_id=1,
            token_ids=[10, 20, 30, 40],
            fingerprint=b"different",
        )


def test_chain_compact_validation_matches_full_token_validation():
    index = ChainCacheIndex()
    _create_and_finish(index, "chain-a", 1, [10, 20, 30])

    compact = index.plan_admission_digest(
        chain_id="chain-a",
        seq_id=1,
        input_token_count=5,
        input_prefix_digest=stable_token_digest([10, 20, 30]),
        fingerprint=FINGERPRINT,
    )

    assert compact.status == "resumed"
    assert compact.reused_tokens == 3
    with pytest.raises(ChainPrefixMismatchError):
        index.plan_admission_digest(
            chain_id="chain-a",
            seq_id=1,
            input_token_count=2,
            input_prefix_digest=stable_token_digest([10, 20]),
            fingerprint=FINGERPRINT,
        )
    with pytest.raises(ChainPrefixMismatchError):
        index.plan_admission_digest(
            chain_id="chain-a",
            seq_id=1,
            input_token_count=4,
            input_prefix_digest=stable_token_digest([10, 99, 30]),
            fingerprint=FINGERPRINT,
        )


def test_chain_unknown_and_tombstone_are_distinct():
    index = ChainCacheIndex()
    with pytest.raises(ChainNotFoundError):
        index.lookup("unknown")
    _create_and_finish(index, "chain-a", 1, [1])
    index.evict("chain-a")
    with pytest.raises(ChainGoneError):
        index.lookup("chain-a")


def test_chain_lru_is_strict_idle_only_and_accumulates_layer_deficits():
    index = ChainCacheIndex()
    old = _create_and_finish(
        index, "old", 1, [1], physical_slots=(2, 1)
    )
    newer = _create_and_finish(
        index, "newer", 2, [2], physical_slots=(1, 4)
    )
    active_plan = index.plan_admission(
        chain_id="active",
        seq_id=3,
        token_ids=[3],
        fingerprint=FINGERPRINT,
    )
    index.apply_admission(active_plan, fingerprint=FINGERPRINT)
    assert old.last_access < newer.last_access

    plan = index.plan_admission(
        chain_id="incoming",
        seq_id=4,
        token_ids=[4],
        fingerprint=FINGERPRINT,
        required_slots_by_layer=(3, 4),
        row_deficit=2,
    )
    assert plan.victim_chain_ids == ("old", "newer")
    assert "active" not in plan.victim_chain_ids


def test_chain_lru_tie_break_is_chain_id_and_active_capacity_fails():
    index = ChainCacheIndex()
    first = _create_and_finish(index, "b", 1, [1], physical_slots=(1,))
    second = _create_and_finish(index, "a", 2, [2], physical_slots=(1,))
    first.last_access = second.last_access = 9
    plan = index.plan_admission(
        chain_id="incoming",
        seq_id=3,
        token_ids=[3],
        fingerprint=FINGERPRINT,
        required_slots_by_layer=(1,),
    )
    assert plan.victim_chain_ids == ("a",)

    index.apply_admission(plan, fingerprint=FINGERPRINT)
    for chain_id in list(index.records):
        record = index.records[chain_id]
        record.state = ChainState.ACTIVE
    with pytest.raises(ChainCapacityError):
        index.plan_admission(
            chain_id="another",
            seq_id=4,
            token_ids=[4],
            fingerprint=FINGERPRINT,
            required_slots_by_layer=(1,),
        )


def test_chain_tombstones_are_bounded():
    index = ChainCacheIndex(max_tombstones=2)
    for seq_id, chain_id in enumerate(("a", "b", "c"), start=1):
        _create_and_finish(index, chain_id, seq_id, [seq_id])
        index.evict(chain_id)
    assert tuple(index.tombstones) == ("b", "c")
    with pytest.raises(ChainNotFoundError):
        index.lookup("a")
    with pytest.raises(ChainGoneError):
        index.lookup("b")


def test_chain_token_history_is_compact_bounded_and_reclaimed():
    index = ChainCacheIndex(max_token_history_tokens=4)
    first = _create_and_finish(index, "first", 1, [1, 2, 3])

    assert first.processed_token_ids.typecode == "I"
    assert list(first.processed_token_ids) == [1, 2, 3]
    assert index.stats()["chain_cache_token_history_tokens"] == 3
    assert index.stats()["chain_cache_token_history_capacity"] == 4
    assert (
        index.stats()["chain_cache_token_history_bytes"]
        == 3 * first.processed_token_ids.itemsize
    )
    assert (
        index.stats()["chain_cache_token_history_byte_capacity"]
        == 4 * first.processed_token_ids.itemsize
    )

    second_plan = index.plan_admission(
        chain_id="second",
        seq_id=2,
        token_ids=[4, 5],
        fingerprint=FINGERPRINT,
    )
    index.apply_admission(second_plan, fingerprint=FINGERPRINT)
    with pytest.raises(ChainCapacityError, match="token history"):
        index.finish(
            "second",
            token_ids=[4, 5],
            processed_token_count=2,
            physical_slots_by_layer=(2,),
        )
    assert index.lookup("second").state is ChainState.ACTIVE

    evicted = index.evict("first")
    assert list(evicted.processed_token_ids) == []
    index.finish(
        "second",
        token_ids=[4, 5],
        processed_token_count=2,
        physical_slots_by_layer=(2,),
    )
    assert index.stats()["chain_cache_token_history_tokens"] == 2


def test_chain_apply_plan_rejects_duplicate_resident_seq_owner():
    index = ChainCacheIndex()
    _create_and_finish(index, "chain-a", 1, [1])
    plan = ChainAdmissionPlan(
        chain_id="chain-b",
        seq_id=1,
        status="created",
        reused_tokens=0,
    )
    with pytest.raises(Exception, match="already owns chain"):
        index.apply_admission(plan, fingerprint=FINGERPRINT)


@pytest.mark.parametrize(
    ("method", "requested", "expected"),
    [
        ("", "auto", "radix"),
        ("omnikv", "auto", "radix"),
        ("quest", "radix", "radix"),
        ("snapkv", "auto", "chain"),
        ("h2o", "auto", "chain"),
        ("pyramidkv", "chain", "chain"),
        ("rkv", "auto", "chain"),
        ("skipkv", "chain", "chain"),
    ],
)
def test_prefix_cache_mode_resolution(method, requested, expected):
    assert (
        normalize_prefix_cache_mode(
            requested, enabled=True, method=method
        )
        == expected
    )


def test_prefix_cache_mode_rejects_incompatible_mode():
    with pytest.raises(ValueError, match="incompatible"):
        normalize_prefix_cache_mode("radix", enabled=True, method="snapkv")
    with pytest.raises(ValueError, match="incompatible"):
        normalize_prefix_cache_mode("radix", enabled=True, method="h2o")
    with pytest.raises(ValueError, match="incompatible"):
        normalize_prefix_cache_mode("chain", enabled=True, method="quest")
    assert (
        normalize_prefix_cache_mode("chain", enabled=False, method="snapkv")
        == "disabled"
    )


def test_sequence_ipc_preserves_chain_admission_fields():
    seq = Sequence(
        [1, 2, 3],
        SamplingParams(max_tokens=2, temperature=0.0),
    )
    seq.current_chunk_size = 1
    seq.chain_id = "chain-a"
    seq.chain_status = "resumed"
    seq.chain_reused_tokens = 2
    restored = pickle.loads(pickle.dumps(seq))
    assert restored.chain_id == "chain-a"
    assert restored.chain_status == "resumed"
    assert restored.chain_reused_tokens == 2


def test_chain_routing_snapshot_is_read_only_state_copy():
    index = ChainCacheIndex()
    _create_and_finish(index, "idle", 1, [1])
    plan = index.plan_admission(
        chain_id="active",
        seq_id=2,
        token_ids=[2],
        fingerprint=FINGERPRINT,
    )
    index.apply_admission(plan, fingerprint=FINGERPRINT)
    snapshot = index.routing_snapshot()
    index.invalidate("active")
    assert snapshot.match("idle")["state"] == "idle"
    assert snapshot.match("active")["state"] == "active"
    assert snapshot.match("unknown") == {
        "enabled": True,
        "present": False,
        "state": None,
        "tombstone": False,
    }


def test_runtime_chain_lru_reclaims_payload_before_reusing_capacity():
    class CacheManager:
        def __init__(self):
            self.freed = []

        def free_seq(self, seq_id):
            self.freed.append(int(seq_id))

    config = SimpleNamespace(
        vllm_sparse_method="snapkv",
        model="/models/test",
        hf_config=SimpleNamespace(model_type="test", torch_dtype="float16"),
        tensor_parallel_size=1,
        max_model_len=128,
        prefix_cache_salt="",
        chain_cache_max_tombstones=8,
        full_attn_layers=[],
        num_sink_tokens=1,
        num_recent_tokens=1,
        decode_keep_tokens=4,
        snapkv_window_size=2,
        snapkv_num_full_layers=0,
        sparse_attn_score_dtype="float32",
        pool_kernel_size=1,
    )
    cache_manager = CacheManager()
    coordinator = ChainCacheCoordinator(config, cache_manager)
    runtime_state = RuntimeState(
        config,
        cache_manager,
        chain_cache_coordinator=coordinator,
    )
    for seq_id, chain_id in ((1, "old"), (2, "newer")):
        plan = coordinator.index.plan_admission(
            chain_id=chain_id,
            seq_id=seq_id,
            token_ids=[seq_id],
            fingerprint=coordinator.fingerprint,
        )
        coordinator.index.apply_admission(
            plan,
            fingerprint=coordinator.fingerprint,
        )
        coordinator.index.finish(
            chain_id,
            token_ids=[seq_id],
            processed_token_count=1,
            physical_slots_by_layer=(2,),
        )
        runtime_state._resident_seq_ids.add(seq_id)

    incoming = coordinator.index.plan_admission(
        chain_id="incoming",
        seq_id=3,
        token_ids=[3],
        fingerprint=coordinator.fingerprint,
        required_slots_by_layer=(2,),
        row_deficit=1,
    )
    result = runtime_state.chain_apply_admission(incoming)

    assert result["victim_chain_ids"] == ["old"]
    assert cache_manager.freed == [1]
    assert 1 not in runtime_state._resident_seq_ids
    with pytest.raises(ChainGoneError):
        coordinator.index.lookup("old")
    assert coordinator.index.lookup("newer").state is ChainState.IDLE


def test_snapkv_resumed_prefill_score_uses_physical_coordinates():
    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(
        vllm_sparse_method="snapkv",
        snapkv_num_full_layers=0,
        snapkv_window_size=8,
        num_sink_tokens=2,
        decode_keep_tokens=16,
        num_recent_tokens=4,
    )
    manager.kv_layer_index = lambda layer_idx: int(layer_idx)
    manager.chain_physical_kv_len = lambda layer_idx, seq_id: 30
    seq = Sequence(
        list(range(100)),
        SamplingParams(max_tokens=2, temperature=0.0),
    )
    seq.chain_id = "chain-a"
    seq.chain_status = "resumed"
    seq.chain_reused_tokens = 90
    seq.num_prefilled_tokens = 90
    seq.current_chunk_size = 10

    rows = manager._prefill_score_rows(0, [seq])

    assert rows == [(0, seq, 22, 30)]


def test_snapkv_resumed_prefill_skips_score_below_physical_budget():
    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(
        vllm_sparse_method="snapkv",
        snapkv_num_full_layers=0,
        snapkv_window_size=8,
        num_sink_tokens=2,
        decode_keep_tokens=16,
        num_recent_tokens=4,
    )
    manager.kv_layer_index = lambda layer_idx: int(layer_idx)
    manager.chain_physical_kv_len = lambda layer_idx, seq_id: 20
    seq = Sequence(
        list(range(100)),
        SamplingParams(max_tokens=2, temperature=0.0),
    )
    seq.chain_status = "resumed"
    seq.chain_reused_tokens = 90
    seq.num_prefilled_tokens = 90
    seq.current_chunk_size = 10

    assert manager._prefill_score_rows(0, [seq]) == []


def test_pyramid_chain_resume_disables_long_prefill_staging():
    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(vllm_sparse_method="pyramidkv")
    manager._pyramidkv_can_use_full_prefill_staging = lambda: True
    manager.pyramidkv_prefill_staging_kv_cache = object()
    seq = Sequence(
        list(range(100)),
        SamplingParams(max_tokens=2, temperature=0.0),
    )
    seq.chain_status = "resumed"
    seq.num_prefilled_tokens = 90
    seq.current_chunk_size = 10

    assert manager.requires_long_prefill_offload(seq) is False


def test_new_pyramid_chain_capacity_uses_materialized_layer_budgets():
    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(
        vllm_sparse_method="pyramidkv",
        num_sink_tokens=1,
        num_recent_tokens=1,
    )
    manager.kv_transformer_layer_indices = lambda: [0, 1]
    manager.kv_layer_index = lambda layer_idx: int(layer_idx)
    manager._num_free_slots = [2, 2]
    manager.free_rows = [[0], [0]]
    manager._pyramidkv_can_use_full_prefill_staging = lambda: True
    manager._pyramidkv_layer_budget = lambda layer_idx: (10, 5)[layer_idx]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=100,
            generation_tokens=1,
            needs_resident_row=True,
        )
    )

    assert required == (10, 5)
    assert required_rows == 1
    assert deficits == (8, 3)
    assert row_deficit == 0

    _, _, decode_deficits, _ = manager.chain_capacity_deficits(
        suffix_tokens=100,
        generation_tokens=5,
        needs_resident_row=True,
    )
    assert decode_deficits == (12, 6)


def test_resumed_snapkv_capacity_uses_physical_peak_not_token_sum():
    manager = object.__new__(SnapKVCacheManager)
    manager.config = SimpleNamespace(
        vllm_sparse_method="snapkv",
        snapkv_num_full_layers=0,
        num_sink_tokens=1,
        decode_keep_tokens=8,
        num_recent_tokens=1,
    )
    manager.kv_transformer_layer_indices = lambda: [0]
    manager.kv_layer_index = lambda layer_idx: int(layer_idx)
    manager._num_free_slots = [3]
    manager.free_rows = [[]]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=2,
            generation_tokens=100,
            existing_slots_by_layer=(10,),
            needs_resident_row=False,
        )
    )

    # Prefill peaks at 12 physical tokens, then compression returns to budget
    # 10; decode peaks at trigger 16. Only six new slots are ever simultaneous.
    assert required == (6,)
    assert required_rows == 0
    assert deficits == (3,)
    assert row_deficit == 0


def test_resumed_h2o_capacity_uses_chunked_physical_peak():
    manager = object.__new__(H2OCacheManager)
    manager.config = SimpleNamespace(
        vllm_sparse_method="h2o",
        h2o_decode_budget=4,
        h2o_prefill_budget=8,
        chunk_prefill_size=4,
    )
    manager.kv_transformer_layer_indices = lambda: [0]
    manager._num_free_slots = [3]
    manager.free_rows = [[]]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=100,
            generation_tokens=10,
            existing_slots_by_layer=(4,),
            needs_resident_row=False,
        )
    )

    # Intermediate prefill peaks at prefill_budget + one chunk (12), so an
    # already resident four-slot row needs eight more slots, not the 109
    # logical suffix/decode tokens.
    assert required == (8,)
    assert required_rows == 0
    assert deficits == (5,)
    assert row_deficit == 0


def test_new_h2o_chain_capacity_reserves_row_and_outstanding_slots():
    manager = object.__new__(H2OCacheManager)
    manager.config = SimpleNamespace(
        vllm_sparse_method="h2o",
        h2o_decode_budget=4,
        h2o_prefill_budget=8,
        chunk_prefill_size=4,
    )
    manager.kv_transformer_layer_indices = lambda: [0]
    manager._num_free_slots = [16]
    manager.free_rows = [[0, 1]]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=100,
            generation_tokens=10,
            outstanding_reserved_slots_by_layer=(8,),
            outstanding_reserved_rows=1,
            needs_resident_row=True,
        )
    )

    assert required == (12,)
    assert required_rows == 1
    assert deficits == (4,)
    assert row_deficit == 0


def test_h2o_chain_capacity_reserves_one_decode_ring_slot():
    manager = object.__new__(H2OCacheManager)
    manager.config = SimpleNamespace(
        vllm_sparse_method="h2o",
        h2o_decode_budget=4,
        h2o_prefill_budget=8,
        chunk_prefill_size=4,
    )
    manager.kv_transformer_layer_indices = lambda: [0]
    manager._num_free_slots = [1]
    manager.free_rows = [[]]

    required, required_rows, deficits, row_deficit = (
        manager.chain_capacity_deficits(
            suffix_tokens=0,
            generation_tokens=10,
            existing_slots_by_layer=(4,),
            needs_resident_row=False,
        )
    )

    assert required == (1,)
    assert required_rows == 0
    assert deficits == (0,)
    assert row_deficit == 0


def _h2o_fingerprint_config(**overrides):
    values = {
        "vllm_sparse_method": "h2o",
        "model": "/models/test",
        "hf_config": SimpleNamespace(
            model_type="qwen2",
            torch_dtype="float16",
        ),
        "tensor_parallel_size": 1,
        "max_model_len": 128,
        "full_attn_layers": [],
        "prefix_cache_salt": "",
        "h2o_decode_budget": 4,
        "h2o_prefill_budget": 8,
        "h2o_recent_ratio": 0.5,
        "h2o_prefill_score_window": 4,
        "sparse_attn_score_dtype": "float32",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    [
        ("h2o_decode_budget", 5),
        ("h2o_prefill_budget", 9),
        ("h2o_recent_ratio", 0.25),
        ("h2o_prefill_score_window", 8),
        ("sparse_attn_score_dtype", "float16"),
    ],
)
def test_h2o_chain_fingerprint_covers_physical_state_config(
    field_name,
    changed_value,
):
    baseline = build_chain_cache_fingerprint(_h2o_fingerprint_config())
    changed = build_chain_cache_fingerprint(
        _h2o_fingerprint_config(**{field_name: changed_value})
    )
    assert changed != baseline


def test_chain_admission_reserves_capacity_before_prefill_allocation():
    config = SimpleNamespace(
        vllm_sparse_method="snapkv",
        model="/models/test",
        hf_config=SimpleNamespace(model_type="test", torch_dtype="float16"),
        tensor_parallel_size=1,
        max_model_len=128,
        prefix_cache_salt="",
        chain_cache_max_tombstones=8,
        full_attn_layers=[],
        snapkv_num_full_layers=0,
        num_sink_tokens=1,
        decode_keep_tokens=8,
        num_recent_tokens=1,
    )
    manager = object.__new__(SnapKVCacheManager)
    manager.config = config
    manager.kv_transformer_layer_indices = lambda: [0]
    manager.kv_layer_index = lambda layer_idx: int(layer_idx)
    manager._num_free_slots = [5]
    manager.free_rows = [[0, 1]]
    manager.seq_id_to_row = [{}]
    coordinator = ChainCacheCoordinator(config, manager)

    first = coordinator.plan_admission(
        chain_id="first",
        seq_id=1,
        token_ids=[1, 2, 3, 4],
        generation_tokens=1,
    )
    coordinator.apply_admission(first)

    assert first.reserved_slots_by_layer == (4,)
    assert first.reserved_rows == 1
    with pytest.raises(ChainCapacityError):
        coordinator.plan_admission(
            chain_id="second",
            seq_id=2,
            token_ids=[5, 6, 7, 8],
            generation_tokens=1,
        )


def test_engine_chain_admission_reuses_resident_seq_and_logical_boundary():
    class CacheManager:
        def __init__(self):
            self.finished_turns = []

        def chain_capacity_deficits(self, **_kwargs):
            return (), 0, (), 0

        def chain_physical_residency(self, _seq_id):
            return (3, 4)

        def on_chain_turn_finished(self, seq_id, processed_token_count):
            self.finished_turns.append(
                (int(seq_id), int(processed_token_count))
            )

    config = SimpleNamespace(
        resolved_prefix_cache_mode="chain",
        max_model_len=128,
        vllm_sparse_method="snapkv",
        model="/models/test",
        hf_config=SimpleNamespace(model_type="test", torch_dtype="float16"),
        tensor_parallel_size=1,
        prefix_cache_salt="",
        chain_cache_max_tombstones=8,
        full_attn_layers=[],
        num_sink_tokens=1,
        num_recent_tokens=1,
        decode_keep_tokens=4,
        snapkv_window_size=2,
        snapkv_num_full_layers=0,
        sparse_attn_score_dtype="float32",
        pool_kernel_size=1,
    )
    cache_manager = CacheManager()
    coordinator = ChainCacheCoordinator(config, cache_manager)
    runtime_state = RuntimeState(
        config,
        cache_manager,
        chain_cache_coordinator=coordinator,
    )

    class ModelRunner:
        def __init__(self):
            self.runtime_state = runtime_state
            self.calls = []

        def call(self, method, *args):
            self.calls.append((method, args))
            if method == "chain_invalidate":
                return self.runtime_state.chain_invalidate(
                    args[0],
                    expected_seq_id=args[1],
                )
            return getattr(self.runtime_state, method)(*args)

    class Scheduler:
        def __init__(self):
            self.added = []

        def add(self, seq):
            self.added.append(seq)

        def abort(self, _seq_id):
            return False

    engine = object.__new__(LLMEngine)
    engine.config = config
    engine.model_runner = ModelRunner()
    engine.scheduler = Scheduler()
    engine._active_chain_sequences = {}
    params = SamplingParams(max_tokens=2, temperature=0.0)

    first = engine.admit_request([1, 2, 3], params)
    assert first.chain_status == "created"
    validation_args = next(
        args
        for method, args in engine.model_runner.calls
        if method == "chain_validate_admission_plan"
    )
    assert validation_args[1:] == (
        3,
        stable_token_digest([], count=0),
        2,
    )
    with pytest.raises(ChainBusyError):
        engine.admit_request([1, 2, 3, 4], params, chain_id=first.chain_id)

    runtime_state.chain_finish(
        first.chain_id,
        first.seq_id,
        stable_token_digest([1, 2, 3, 9], count=3),
        3,
    )
    assert cache_manager.finished_turns == [(first.seq_id, 3)]
    engine._active_chain_sequences.clear()
    resumed = engine.admit_request(
        [1, 2, 3, 9, 5],
        params,
        chain_id=first.chain_id,
    )

    assert resumed.seq_id == first.seq_id
    assert resumed.chain_status == "resumed"
    assert resumed.reused_tokens == 3
    assert resumed.prefilled_tokens == 2
    resumed_seq = engine.scheduler.added[-1]
    assert resumed_seq.num_prefilled_tokens == 3
    assert resumed_seq.token_ids[3:] == [9, 5]

    with pytest.raises(ChainNotFoundError):
        engine.admit_request([1, 2], params, chain_id="unknown")

    runtime_state.chain_finish(
        resumed.chain_id,
        resumed.seq_id,
        stable_token_digest([1, 2, 3, 9, 5], count=4),
        4,
    )
    assert cache_manager.finished_turns == [
        (first.seq_id, 3),
        (resumed.seq_id, 4),
    ]
    engine._active_chain_sequences.clear()
    engine.abort_request(resumed.seq_id)
    engine.abort_request(resumed.seq_id)
    with pytest.raises(ChainGoneError):
        coordinator.index.lookup(resumed.chain_id)


def test_engine_abort_rejects_unsafe_chain_retain_disposition():
    engine = object.__new__(LLMEngine)
    engine.scheduler = SimpleNamespace(abort=lambda _seq_id: False)
    engine._active_chain_sequences = {}

    with pytest.raises(ValueError, match="only supports 'invalidate'"):
        engine.abort_request(7, disposition="retain")
