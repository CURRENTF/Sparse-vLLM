import ast
import inspect
import textwrap
from dataclasses import fields

from sparsevllm.config import Config as PublicConfig
from sparsevllm.configs import (
    Config,
    DecodeCudaGraphConfig,
    DeltaKVConfig,
    ObservabilityConfig,
    PrefixCacheConfig,
    SparseMethodConfig,
)


CONFIG_GROUPS = (
    PrefixCacheConfig,
    DecodeCudaGraphConfig,
    SparseMethodConfig,
    DeltaKVConfig,
    ObservabilityConfig,
)


def test_public_config_import_remains_compatible():
    assert PublicConfig is Config


def test_config_composes_disjoint_responsibility_groups():
    config_fields = {item.name for item in fields(Config)}
    claimed_fields: set[str] = set()

    for group in CONFIG_GROUPS:
        group_fields = {item.name for item in fields(group)}
        assert group_fields
        assert group_fields <= config_fields
        assert claimed_fields.isdisjoint(group_fields)
        claimed_fields.update(group_fields)


def test_config_groups_are_keyword_only():
    for group in CONFIG_GROUPS:
        assert all(item.kw_only for item in fields(group) if item.init)


def test_runtime_invariant_validation_is_opt_in_and_profiler_independent():
    assert ObservabilityConfig().validate_runtime_invariants is False
    assert (
        ObservabilityConfig(enable_profiler=True).validate_runtime_invariants
        is False
    )
    assert (
        ObservabilityConfig(validate_runtime_invariants=True)
        .validate_runtime_invariants
        is True
    )


def test_runtime_constructor_only_orchestrates_config_stages():
    tree = ast.parse(textwrap.dedent(inspect.getsource(Config.__post_init__)))
    forbidden = (ast.For, ast.If, ast.Raise, ast.Try, ast.While)

    assert not any(isinstance(node, forbidden) for node in ast.walk(tree))

    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "normalize_bootstrap",
        "normalize_sparse_method_name",
        "normalize_prefix_cache",
        "normalize_scheduling",
        "normalize_deltakv_storage",
        "normalize_platform",
        "normalize_decode_cuda_graph",
        "load_and_validate_model",
        "normalize_sparse_methods",
        "finalize_prefix_cache",
        "validate_deltakv_runtime",
        "finalize_sparse_layout",
    } <= calls
