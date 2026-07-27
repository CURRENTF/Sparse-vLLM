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
