from __future__ import annotations

import hashlib
import json
import secrets
import struct
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable


CHAIN_PREFIX_METHODS = frozenset({"snapkv", "pyramidkv", "rkv", "skipkv"})
RADIX_PREFIX_METHODS = frozenset({"", "omnikv", "quest"})
PREFIX_CACHE_MODES = frozenset({"auto", "radix", "chain"})


class ChainState(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"


class ChainCacheError(RuntimeError):
    status_code = 500
    error_code = "chain_cache_error"

    def __init__(self, message: str, *, chain_id: str | None = None):
        super().__init__(message)
        self.chain_id = chain_id


class ChainNotFoundError(ChainCacheError):
    status_code = 404
    error_code = "chain_not_found"


class ChainBusyError(ChainCacheError):
    status_code = 409
    error_code = "chain_busy"


class ChainPrefixMismatchError(ChainCacheError):
    status_code = 409
    error_code = "chain_prefix_mismatch"


class ChainFingerprintMismatchError(ChainCacheError):
    status_code = 409
    error_code = "chain_fingerprint_mismatch"


class ChainGoneError(ChainCacheError):
    status_code = 410
    error_code = "chain_gone"


class ChainCapacityError(ChainCacheError):
    status_code = 503
    error_code = "chain_capacity_unavailable"


class ChainOwnerMismatchError(ChainCacheError):
    status_code = 500
    error_code = "chain_owner_mismatch"


class ChainModeError(ChainCacheError):
    status_code = 400
    error_code = "chain_mode_disabled"


@dataclass(frozen=True, slots=True)
class RequestAdmission:
    seq_id: int
    chain_id: str | None
    chain_status: str
    reused_tokens: int
    prefilled_tokens: int = 0


@dataclass(slots=True)
class ChainRecord:
    chain_id: str
    seq_id: int
    fingerprint: bytes
    state: ChainState
    processed_token_count: int = 0
    processed_token_digest: bytes = b""
    processed_token_ids: tuple[int, ...] = ()
    last_access: int = 0
    physical_slots_by_layer: tuple[int, ...] = ()
    resident_rows: int = 1
    reserved_slots_by_layer: tuple[int, ...] = ()
    reserved_rows: int = 0


@dataclass(frozen=True, slots=True)
class ChainAdmissionPlan:
    chain_id: str
    seq_id: int
    status: str
    reused_tokens: int
    victim_chain_ids: tuple[str, ...] = ()
    reserved_slots_by_layer: tuple[int, ...] = ()
    reserved_rows: int = 0


@dataclass(frozen=True, slots=True)
class ChainRoutingSnapshot:
    enabled: bool
    active_chain_ids: frozenset[str] = frozenset()
    idle_chain_ids: frozenset[str] = frozenset()
    tombstone_chain_ids: frozenset[str] = frozenset()

    def match(self, chain_id: str) -> dict[str, object]:
        chain_id = str(chain_id)
        if chain_id in self.active_chain_ids:
            return {
                "enabled": self.enabled,
                "present": True,
                "state": ChainState.ACTIVE.value,
                "tombstone": False,
            }
        if chain_id in self.idle_chain_ids:
            return {
                "enabled": self.enabled,
                "present": True,
                "state": ChainState.IDLE.value,
                "tombstone": False,
            }
        return {
            "enabled": self.enabled,
            "present": False,
            "state": None,
            "tombstone": chain_id in self.tombstone_chain_ids,
        }


def normalize_prefix_cache_mode(
    requested: str | None,
    *,
    enabled: bool,
    method: str,
) -> str:
    mode = str(requested or "auto").strip().lower()
    if mode not in PREFIX_CACHE_MODES:
        supported = ", ".join(sorted(PREFIX_CACHE_MODES))
        raise ValueError(
            f"prefix_cache_mode must be one of {supported}; got {requested!r}."
        )
    if not bool(enabled):
        return "disabled"
    method = str(method or "")
    expected = (
        "radix"
        if method in RADIX_PREFIX_METHODS
        else "chain"
        if method in CHAIN_PREFIX_METHODS
        else None
    )
    if expected is None:
        raise ValueError(
            "prefix caching is not supported for "
            f"vllm_sparse_method={method!r}."
        )
    if mode == "auto":
        return expected
    if mode != expected:
        raise ValueError(
            f"prefix_cache_mode={mode!r} is incompatible with "
            f"vllm_sparse_method={method!r}; use {expected!r}."
        )
    return mode


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return str(value)


def build_chain_cache_fingerprint(config: Any) -> bytes:
    hf_config = getattr(config, "hf_config", None)
    method = str(getattr(config, "vllm_sparse_method", "") or "")
    method_fields = {
        "snapkv": (
            "num_sink_tokens",
            "num_recent_tokens",
            "decode_keep_tokens",
            "snapkv_window_size",
            "snapkv_num_full_layers",
            "sparse_attn_score_dtype",
            "pool_kernel_size",
        ),
        "pyramidkv": (
            "num_sink_tokens",
            "num_recent_tokens",
            "decode_keep_tokens",
            "snapkv_window_size",
            "pyramid_layer_ratios",
            "pyramidkv_start_layer",
            "pyramidkv_start_ratio",
            "pyramidkv_least_layer",
            "pyramidkv_least_ratio",
            "sparse_attn_score_dtype",
            "pool_kernel_size",
        ),
        "rkv": (
            "num_sink_tokens",
            "num_recent_tokens",
            "decode_keep_tokens",
            "rkv_compression_interval",
            "rkv_observation_tokens",
            "rkv_alpha",
            "rkv_similarity_threshold",
            "rkv_recent_similar_keep",
            "rkv_max_redundancy_tokens",
            "rkv_redundancy_window",
        ),
        "skipkv": (
            "num_sink_tokens",
            "num_recent_tokens",
            "decode_keep_tokens",
            "rkv_observation_tokens",
            "skipkv_compression_interval",
            "skipkv_alpha",
            "skipkv_similarity_threshold",
            "skipkv_segment_size",
            "skipkv_max_redundancy_tokens",
            "skipkv_redundancy_window",
            "skipkv_enable_sentence_scoring",
            "skipkv_sentence_score_weight",
            "skipkv_sentence_min_tokens",
            "skipkv_sentence_max_tokens",
            "skipkv_sentence_embedding_layer",
            "skipkv_max_tracked_sentences",
            "skipkv_enable_activation_steering",
            "skipkv_steering_vector_path",
            "skipkv_steering_layer",
            "skipkv_steering_alpha",
            "skipkv_steering_alpha_increment",
            "skipkv_steering_alpha_max",
        ),
    }
    payload: dict[str, Any] = {
        "schema": 1,
        "model": getattr(config, "model", None),
        "model_type": getattr(hf_config, "model_type", None),
        "dtype": str(getattr(hf_config, "torch_dtype", None)),
        "tp_size": int(getattr(config, "tensor_parallel_size", 1)),
        "max_model_len": int(getattr(config, "max_model_len", 0)),
        "full_attention_layers": _jsonable(
            getattr(config, "full_attn_layers", ())
        ),
        "method": method,
        "salt": str(getattr(config, "prefix_cache_salt", "") or ""),
    }
    for field_name in method_fields.get(method, ()):
        payload[field_name] = _jsonable(getattr(config, field_name, None))
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()


def stable_token_digest(
    token_ids: Iterable[int],
    *,
    count: int | None = None,
) -> bytes:
    hasher = hashlib.sha256()
    seen = 0
    for token_id in token_ids:
        if count is not None and seen >= int(count):
            break
        hasher.update(struct.pack("<q", int(token_id)))
        seen += 1
    if count is not None and seen != int(count):
        raise ChainPrefixMismatchError(
            "Input is shorter than the chain's processed token boundary: "
            f"input_tokens={seen}, processed_tokens={int(count)}."
        )
    return hasher.digest()


class ChainCacheIndex:
    """Logical lifecycle for non-branching, resident sparse-KV chains."""

    def __init__(self, *, max_tombstones: int = 1024):
        max_tombstones = int(max_tombstones)
        if max_tombstones <= 0:
            raise ValueError(
                f"chain_cache_max_tombstones must be > 0, got {max_tombstones}."
            )
        self.max_tombstones = max_tombstones
        self.records: dict[str, ChainRecord] = {}
        self.seq_id_to_chain_id: dict[int, str] = {}
        self.tombstones: OrderedDict[str, int] = OrderedDict()
        self._clock = 0
        self._stats = {
            "chain_cache_created": 0,
            "chain_cache_resumed": 0,
            "chain_cache_finished": 0,
            "chain_cache_evicted": 0,
            "chain_cache_invalidated": 0,
            "chain_cache_prefix_mismatch": 0,
            "chain_cache_busy": 0,
        }

    @staticmethod
    def new_chain_id() -> str:
        return f"chain_{secrets.token_urlsafe(24)}"

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def _add_tombstone(self, chain_id: str) -> None:
        self.tombstones.pop(chain_id, None)
        self.tombstones[chain_id] = self._tick()
        while len(self.tombstones) > self.max_tombstones:
            self.tombstones.popitem(last=False)

    def lookup(self, chain_id: str) -> ChainRecord:
        chain_id = str(chain_id)
        record = self.records.get(chain_id)
        if record is not None:
            return record
        if chain_id in self.tombstones:
            raise ChainGoneError(
                f"Chain {chain_id!r} was evicted or invalidated.",
                chain_id=chain_id,
            )
        raise ChainNotFoundError(
            f"Unknown chain_id {chain_id!r}.",
            chain_id=chain_id,
        )

    def plan_admission(
        self,
        *,
        chain_id: str,
        seq_id: int,
        token_ids: list[int],
        fingerprint: bytes,
        required_slots_by_layer: tuple[int, ...] = (),
        row_deficit: int = 0,
        reserved_slots_by_layer: tuple[int, ...] = (),
        reserved_rows: int = 0,
    ) -> ChainAdmissionPlan:
        return self._plan_admission(
            chain_id=chain_id,
            seq_id=seq_id,
            input_token_count=len(token_ids),
            input_prefix_digest=None,
            token_ids=token_ids,
            fingerprint=fingerprint,
            required_slots_by_layer=required_slots_by_layer,
            row_deficit=row_deficit,
            reserved_slots_by_layer=reserved_slots_by_layer,
            reserved_rows=reserved_rows,
        )

    def plan_admission_digest(
        self,
        *,
        chain_id: str,
        seq_id: int,
        input_token_count: int,
        input_prefix_digest: bytes,
        fingerprint: bytes,
        required_slots_by_layer: tuple[int, ...] = (),
        row_deficit: int = 0,
        reserved_slots_by_layer: tuple[int, ...] = (),
        reserved_rows: int = 0,
    ) -> ChainAdmissionPlan:
        """Validate a rank-local plan without broadcasting full token IDs."""
        return self._plan_admission(
            chain_id=chain_id,
            seq_id=seq_id,
            input_token_count=int(input_token_count),
            input_prefix_digest=bytes(input_prefix_digest),
            token_ids=None,
            fingerprint=fingerprint,
            required_slots_by_layer=required_slots_by_layer,
            row_deficit=row_deficit,
            reserved_slots_by_layer=reserved_slots_by_layer,
            reserved_rows=reserved_rows,
        )

    def _plan_admission(
        self,
        *,
        chain_id: str,
        seq_id: int,
        input_token_count: int,
        input_prefix_digest: bytes | None,
        token_ids: list[int] | None,
        fingerprint: bytes,
        required_slots_by_layer: tuple[int, ...],
        row_deficit: int,
        reserved_slots_by_layer: tuple[int, ...],
        reserved_rows: int,
    ) -> ChainAdmissionPlan:
        record = self.records.get(chain_id)
        if record is None:
            if chain_id in self.tombstones:
                self.lookup(chain_id)
            status = "created"
            reused_tokens = 0
        else:
            if record.state is ChainState.ACTIVE:
                self._stats["chain_cache_busy"] += 1
                raise ChainBusyError(
                    f"Chain {chain_id!r} already has an active writer.",
                    chain_id=chain_id,
                )
            if record.fingerprint != fingerprint:
                raise ChainFingerprintMismatchError(
                    f"Chain {chain_id!r} was created with a different method/config fingerprint.",
                    chain_id=chain_id,
                )
            if record.seq_id != int(seq_id):
                raise ChainOwnerMismatchError(
                    f"Chain owner mismatch for {chain_id!r}: "
                    f"resident_seq_id={record.seq_id}, requested_seq_id={int(seq_id)}.",
                    chain_id=chain_id,
                )
            try:
                if token_ids is not None:
                    digest = stable_token_digest(
                        token_ids,
                        count=record.processed_token_count,
                    )
                else:
                    if int(input_token_count) < record.processed_token_count:
                        raise ChainPrefixMismatchError(
                            "Input is shorter than the chain's processed token "
                            "boundary: "
                            f"input_tokens={int(input_token_count)}, "
                            f"processed_tokens={record.processed_token_count}.",
                            chain_id=chain_id,
                        )
                    digest = bytes(input_prefix_digest or b"")
            except ChainPrefixMismatchError as exc:
                self._stats["chain_cache_prefix_mismatch"] += 1
                if exc.chain_id is None:
                    exc.chain_id = chain_id
                raise
            if digest != record.processed_token_digest:
                self._stats["chain_cache_prefix_mismatch"] += 1
                raise ChainPrefixMismatchError(
                    f"Input prefix does not match chain {chain_id!r} at "
                    f"processed_token_count={record.processed_token_count}.",
                    chain_id=chain_id,
                )
            status = "resumed"
            reused_tokens = int(record.processed_token_count)

        slot_deficits = [max(0, int(value)) for value in required_slots_by_layer]
        remaining_row_deficit = max(0, int(row_deficit))
        victims: list[str] = []
        candidates = sorted(
            (
                candidate
                for candidate in self.records.values()
                if candidate.state is ChainState.IDLE
                and candidate.chain_id != chain_id
            ),
            key=lambda candidate: (candidate.last_access, candidate.chain_id),
        )
        for victim in candidates:
            if remaining_row_deficit <= 0 and not any(slot_deficits):
                break
            victims.append(victim.chain_id)
            remaining_row_deficit = max(
                0, remaining_row_deficit - int(victim.resident_rows)
            )
            for layer_idx in range(len(slot_deficits)):
                resident = (
                    int(victim.physical_slots_by_layer[layer_idx])
                    if layer_idx < len(victim.physical_slots_by_layer)
                    else 0
                )
                slot_deficits[layer_idx] = max(
                    0, slot_deficits[layer_idx] - resident
                )
        if remaining_row_deficit > 0 or any(slot_deficits):
            raise ChainCapacityError(
                "No capacity is available for the chain request after evicting "
                "all IDLE chains; ACTIVE chains remain pinned.",
                chain_id=chain_id,
            )
        return ChainAdmissionPlan(
            chain_id=chain_id,
            seq_id=int(seq_id),
            status=status,
            reused_tokens=reused_tokens,
            victim_chain_ids=tuple(victims),
            reserved_slots_by_layer=tuple(
                max(0, int(value)) for value in reserved_slots_by_layer
            ),
            reserved_rows=max(0, int(reserved_rows)),
        )

    def apply_admission(
        self,
        plan: ChainAdmissionPlan,
        *,
        fingerprint: bytes,
    ) -> ChainRecord:
        for victim_chain_id in plan.victim_chain_ids:
            self.evict(victim_chain_id)
        if plan.status == "created":
            if plan.chain_id in self.records:
                raise ChainOwnerMismatchError(
                    f"Chain {plan.chain_id!r} appeared between plan and admission.",
                    chain_id=plan.chain_id,
                )
            if plan.seq_id in self.seq_id_to_chain_id:
                raise ChainOwnerMismatchError(
                    f"resident_seq_id={plan.seq_id} already owns chain "
                    f"{self.seq_id_to_chain_id[plan.seq_id]!r}.",
                    chain_id=plan.chain_id,
                )
            record = ChainRecord(
                chain_id=plan.chain_id,
                seq_id=int(plan.seq_id),
                fingerprint=bytes(fingerprint),
                state=ChainState.ACTIVE,
                last_access=self._tick(),
            )
            self.records[plan.chain_id] = record
            self.seq_id_to_chain_id[int(plan.seq_id)] = plan.chain_id
            record.reserved_slots_by_layer = tuple(
                int(value) for value in plan.reserved_slots_by_layer
            )
            record.reserved_rows = int(plan.reserved_rows)
            self._stats["chain_cache_created"] += 1
            return record
        record = self.lookup(plan.chain_id)
        if record.state is not ChainState.IDLE:
            raise ChainBusyError(
                f"Chain {plan.chain_id!r} is not IDLE during admission.",
                chain_id=plan.chain_id,
            )
        record.state = ChainState.ACTIVE
        record.last_access = self._tick()
        record.reserved_slots_by_layer = tuple(
            int(value) for value in plan.reserved_slots_by_layer
        )
        record.reserved_rows = int(plan.reserved_rows)
        self._stats["chain_cache_resumed"] += 1
        return record

    def finish(
        self,
        chain_id: str,
        *,
        token_ids: list[int],
        processed_token_count: int,
        physical_slots_by_layer: tuple[int, ...],
        resident_rows: int = 1,
    ) -> ChainRecord:
        processed_token_count = int(processed_token_count)
        record = self.finish_digest(
            chain_id,
            processed_token_digest=stable_token_digest(
                token_ids,
                count=processed_token_count,
            ),
            processed_token_count=processed_token_count,
            physical_slots_by_layer=physical_slots_by_layer,
            resident_rows=resident_rows,
        )
        record.processed_token_ids = tuple(
            int(token_id)
            for token_id in token_ids[:processed_token_count]
        )
        return record

    def finish_digest(
        self,
        chain_id: str,
        *,
        processed_token_digest: bytes,
        processed_token_count: int,
        physical_slots_by_layer: tuple[int, ...],
        resident_rows: int = 1,
    ) -> ChainRecord:
        record = self.lookup(chain_id)
        if record.state is not ChainState.ACTIVE:
            raise ChainOwnerMismatchError(
                f"Cannot finish non-ACTIVE chain {chain_id!r}.",
                chain_id=chain_id,
            )
        processed_token_count = int(processed_token_count)
        if processed_token_count < record.processed_token_count:
            raise ChainOwnerMismatchError(
                "Chain processed boundary moved backwards: "
                f"previous={record.processed_token_count}, new={processed_token_count}.",
                chain_id=chain_id,
            )
        processed_token_digest = bytes(processed_token_digest)
        if len(processed_token_digest) != hashlib.sha256().digest_size:
            raise ValueError(
                "processed_token_digest must be a SHA-256 digest, got "
                f"{len(processed_token_digest)} bytes."
            )
        record.processed_token_digest = processed_token_digest
        record.processed_token_count = processed_token_count
        record.physical_slots_by_layer = tuple(
            int(value) for value in physical_slots_by_layer
        )
        record.resident_rows = int(resident_rows)
        record.reserved_slots_by_layer = ()
        record.reserved_rows = 0
        record.state = ChainState.IDLE
        record.last_access = self._tick()
        self._stats["chain_cache_finished"] += 1
        return record

    def invalidate(self, chain_id: str) -> ChainRecord:
        record = self.lookup(chain_id)
        self.records.pop(chain_id, None)
        self.seq_id_to_chain_id.pop(int(record.seq_id), None)
        self._add_tombstone(chain_id)
        self._stats["chain_cache_invalidated"] += 1
        return record

    def evict(self, chain_id: str) -> ChainRecord:
        record = self.lookup(chain_id)
        if record.state is ChainState.ACTIVE:
            raise ChainBusyError(
                f"Cannot evict ACTIVE chain {chain_id!r}.",
                chain_id=chain_id,
            )
        self.records.pop(chain_id, None)
        self.seq_id_to_chain_id.pop(int(record.seq_id), None)
        self._add_tombstone(chain_id)
        self._stats["chain_cache_evicted"] += 1
        return record

    def routing_match(self, chain_id: str) -> dict[str, object]:
        record = self.records.get(str(chain_id))
        if record is not None:
            return {
                "present": True,
                "state": record.state.value,
                "tombstone": False,
            }
        return {
            "present": False,
            "state": None,
            "tombstone": str(chain_id) in self.tombstones,
        }

    def routing_snapshot(self) -> ChainRoutingSnapshot:
        return ChainRoutingSnapshot(
            enabled=True,
            active_chain_ids=frozenset(
                record.chain_id
                for record in self.records.values()
                if record.state is ChainState.ACTIVE
            ),
            idle_chain_ids=frozenset(
                record.chain_id
                for record in self.records.values()
                if record.state is ChainState.IDLE
            ),
            tombstone_chain_ids=frozenset(self.tombstones),
        )

    def stats(self) -> dict[str, int]:
        return {
            **self._stats,
            "chain_cache_entries": len(self.records),
            "chain_cache_active": sum(
                record.state is ChainState.ACTIVE
                for record in self.records.values()
            ),
            "chain_cache_idle": sum(
                record.state is ChainState.IDLE
                for record in self.records.values()
            ),
            "chain_cache_tombstones": len(self.tombstones),
            "chain_cache_tombstone_capacity": self.max_tombstones,
        }

    def reset(self) -> None:
        self.records.clear()
        self.seq_id_to_chain_id.clear()
        self.tombstones.clear()
        self._clock = 0
        for name in self._stats:
            self._stats[name] = 0


class ChainCacheCoordinator:
    """Coordinates chain metadata; cache managers continue to own all payload."""

    def __init__(self, config: Any, cache_manager: Any):
        self.config = config
        self.cache_manager = cache_manager
        self.fingerprint = build_chain_cache_fingerprint(config)
        self.index = ChainCacheIndex(
            max_tombstones=int(
                getattr(config, "chain_cache_max_tombstones", 1024)
            )
        )

    def owner_seq_id(self, chain_id: str) -> int:
        return int(self.index.lookup(chain_id).seq_id)

    def admission_requirements(
        self,
        *,
        chain_id: str,
        token_count: int,
        generation_tokens: int = 0,
    ) -> tuple[tuple[int, ...], int, tuple[int, ...], int]:
        record = self.index.records.get(chain_id)
        reused = 0 if record is None else int(record.processed_token_count)
        existing_slots = (
            ()
            if record is None
            else tuple(int(value) for value in record.physical_slots_by_layer)
        )
        suffix_tokens = max(0, int(token_count) - reused)
        hook = getattr(self.cache_manager, "chain_capacity_deficits", None)
        if not callable(hook):
            return (), 0, (), 0
        outstanding_slots, outstanding_rows = (
            self._outstanding_active_reservations()
        )
        required_slots, required_rows, slot_deficits, row_deficit = hook(
            suffix_tokens=suffix_tokens,
            generation_tokens=max(0, int(generation_tokens)),
            existing_slots_by_layer=existing_slots,
            outstanding_reserved_slots_by_layer=outstanding_slots,
            outstanding_reserved_rows=outstanding_rows,
            needs_resident_row=record is None,
        )
        return (
            tuple(int(value) for value in required_slots),
            int(required_rows),
            tuple(int(value) for value in slot_deficits),
            int(row_deficit),
        )

    def _outstanding_active_reservations(
        self,
    ) -> tuple[tuple[int, ...], int]:
        has_residency = getattr(
            self.cache_manager, "chain_has_residency", None
        )
        physical_residency = getattr(
            self.cache_manager, "chain_physical_residency", None
        )
        outstanding: list[int] = []
        outstanding_rows = 0
        for record in self.index.records.values():
            if record.state is not ChainState.ACTIVE:
                continue
            reserved = tuple(
                int(value) for value in record.reserved_slots_by_layer
            )
            if len(outstanding) < len(reserved):
                outstanding.extend([0] * (len(reserved) - len(outstanding)))
            resident = (
                bool(has_residency(int(record.seq_id)))
                if callable(has_residency)
                else False
            )
            current = (
                tuple(
                    int(value)
                    for value in physical_residency(int(record.seq_id))
                )
                if resident and callable(physical_residency)
                else tuple(
                    int(value)
                    for value in record.physical_slots_by_layer
                )
            )
            baseline = tuple(
                int(value) for value in record.physical_slots_by_layer
            )
            for layer_idx, reserved_slots in enumerate(reserved):
                current_slots = (
                    current[layer_idx] if layer_idx < len(current) else 0
                )
                baseline_slots = (
                    baseline[layer_idx] if layer_idx < len(baseline) else 0
                )
                realized = max(0, current_slots - baseline_slots)
                outstanding[layer_idx] += max(
                    0, reserved_slots - realized
                )
            outstanding_rows += max(
                0,
                int(record.reserved_rows) - (1 if resident else 0),
            )
        return tuple(outstanding), outstanding_rows

    def plan_admission(
        self,
        *,
        chain_id: str,
        seq_id: int,
        token_ids: list[int],
        generation_tokens: int = 0,
    ) -> ChainAdmissionPlan:
        required_slots, required_rows, slots, rows = (
            self.admission_requirements(
                chain_id=chain_id,
                token_count=len(token_ids),
                generation_tokens=generation_tokens,
            )
        )
        return self.index.plan_admission(
            chain_id=chain_id,
            seq_id=seq_id,
            token_ids=token_ids,
            fingerprint=self.fingerprint,
            required_slots_by_layer=slots,
            row_deficit=rows,
            reserved_slots_by_layer=required_slots,
            reserved_rows=required_rows,
        )

    def validate_admission_plan(
        self,
        expected: ChainAdmissionPlan,
        *,
        input_token_count: int,
        input_prefix_digest: bytes,
        generation_tokens: int = 0,
    ) -> ChainAdmissionPlan:
        required_slots, required_rows, slots, rows = (
            self.admission_requirements(
                chain_id=expected.chain_id,
                token_count=int(input_token_count),
                generation_tokens=generation_tokens,
            )
        )
        local = self.index.plan_admission_digest(
            chain_id=expected.chain_id,
            seq_id=expected.seq_id,
            input_token_count=int(input_token_count),
            input_prefix_digest=bytes(input_prefix_digest),
            fingerprint=self.fingerprint,
            required_slots_by_layer=slots,
            row_deficit=rows,
            reserved_slots_by_layer=required_slots,
            reserved_rows=required_rows,
        )
        if local != expected:
            raise RuntimeError(
                "Chain-cache admission plan diverged from the rank-0 plan: "
                f"rank0={expected!r}, local={local!r}."
            )
        return local

    def apply_admission(self, plan: ChainAdmissionPlan) -> ChainRecord:
        return self.index.apply_admission(plan, fingerprint=self.fingerprint)

    def finish(self, seq: Any) -> ChainRecord:
        chain_id = str(getattr(seq, "chain_id", "") or "")
        if not chain_id:
            raise ChainOwnerMismatchError(
                f"Sequence {getattr(seq, 'seq_id', None)} has no chain_id."
            )
        processed_token_count = max(0, int(seq.num_tokens) - 1)
        residency_hook = getattr(
            self.cache_manager, "chain_physical_residency", None
        )
        physical_slots = (
            tuple(int(value) for value in residency_hook(int(seq.seq_id)))
            if callable(residency_hook)
            else ()
        )
        return self.index.finish(
            chain_id,
            token_ids=list(seq.token_ids),
            processed_token_count=processed_token_count,
            physical_slots_by_layer=physical_slots,
        )

    def finish_values(
        self,
        *,
        chain_id: str,
        seq_id: int,
        processed_token_digest: bytes,
        processed_token_count: int,
    ) -> ChainRecord:
        record = self.index.lookup(chain_id)
        if int(record.seq_id) != int(seq_id):
            raise ChainOwnerMismatchError(
                f"Chain owner mismatch for {chain_id!r}: "
                f"resident_seq_id={record.seq_id}, finished_seq_id={int(seq_id)}.",
                chain_id=chain_id,
            )
        residency_hook = getattr(
            self.cache_manager, "chain_physical_residency", None
        )
        physical_slots = (
            tuple(int(value) for value in residency_hook(int(seq_id)))
            if callable(residency_hook)
            else ()
        )
        return self.index.finish_digest(
            chain_id,
            processed_token_digest=bytes(processed_token_digest),
            processed_token_count=int(processed_token_count),
            physical_slots_by_layer=physical_slots,
        )

    def remember_processed_tokens(
        self,
        *,
        chain_id: str,
        seq_id: int,
        token_ids: list[int],
        processed_token_count: int,
    ) -> ChainRecord:
        record = self.index.lookup(str(chain_id))
        processed_token_count = int(processed_token_count)
        if int(record.seq_id) != int(seq_id):
            raise ChainOwnerMismatchError(
                f"Chain owner mismatch for {chain_id!r}: "
                f"resident_seq_id={record.seq_id}, finished_seq_id={int(seq_id)}.",
                chain_id=str(chain_id),
            )
        if record.state is not ChainState.IDLE:
            raise ChainOwnerMismatchError(
                f"Cannot remember tokens for non-IDLE chain {chain_id!r}.",
                chain_id=str(chain_id),
            )
        logical_tokens = tuple(
            int(token_id)
            for token_id in token_ids[:processed_token_count]
        )
        if len(logical_tokens) != processed_token_count:
            raise ChainOwnerMismatchError(
                "Finished token sequence is shorter than the processed boundary: "
                f"tokens={len(logical_tokens)}, processed={processed_token_count}.",
                chain_id=str(chain_id),
            )
        digest = stable_token_digest(logical_tokens)
        if (
            processed_token_count != int(record.processed_token_count)
            or digest != record.processed_token_digest
        ):
            raise ChainOwnerMismatchError(
                "Finished logical tokens do not match the recorded chain boundary.",
                chain_id=str(chain_id),
            )
        record.processed_token_ids = logical_tokens
        return record

    def invalidate(self, chain_id: str) -> ChainRecord:
        return self.index.invalidate(chain_id)

    def routing_match(self, chain_id: str) -> dict[str, object]:
        return self.index.routing_match(chain_id)

    def stats(self) -> dict[str, int]:
        return self.index.stats()

    def reset(self) -> None:
        self.index.reset()
