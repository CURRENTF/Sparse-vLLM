from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sparsevllm.configs.full_attention_profiles import (
    _parse_profile_catalog,
    load_full_attention_layer_profiles,
    resolve_auto_full_attention_layers,
    resolve_full_attention_layer_profile,
)


def _profiles(*entries):
    return _parse_profile_catalog({"schema_version": 1, "profiles": list(entries)})


def _entry(profile_id="test", model_names=None, layers=None):
    return {
        "id": profile_id,
        "model_names": model_names or ["Model-X"],
        "sparse_methods": ["omnikv"],
        "full_attention_layers": [0, 3] if layers is None else layers,
    }


@pytest.mark.parametrize(
    "model_name",
    [
        "Model-X",
        "org/Model-X",
        "/models/Model-X/",
        "/cache/models--org--Model-X/snapshots/revision",
        "model-x",
    ],
)
def test_profile_resolution_uses_exact_model_suffixes(model_name):
    profile = resolve_full_attention_layer_profile(
        model_name,
        "omnikv",
        profiles=_profiles(_entry()),
    )

    assert profile.profile_id == "test"


def test_profile_resolution_does_not_fuzzy_match_related_models():
    with pytest.raises(ValueError, match="No automatic full_attention_layers profile"):
        resolve_full_attention_layer_profile(
            "org/Model-X-Chat",
            "omnikv",
            profiles=_profiles(_entry()),
        )


def test_profile_catalog_rejects_ambiguous_aliases():
    with pytest.raises(ValueError, match="is ambiguous"):
        _profiles(
            _entry("first"),
            _entry("second", model_names=["model-x"]),
        )


@pytest.mark.parametrize("layers", [[], [True], [-1], [3, 0], [0, 0]])
def test_profile_catalog_rejects_invalid_layer_contracts(layers):
    entry = _entry()
    entry["full_attention_layers"] = layers
    with pytest.raises(ValueError, match="sorted list of unique non-negative integers"):
        _profiles(entry)


def test_non_profile_methods_resolve_auto_to_no_full_layers():
    config = SimpleNamespace(
        model="unregistered-model",
        sparse_method="quest",
        full_attention_layers="auto",
    )

    resolve_auto_full_attention_layers(config)

    assert config.full_attention_layers == []


def test_packaged_profile_catalog_satisfies_schema_contract():
    profiles = load_full_attention_layer_profiles()

    assert profiles
    assert all(profile.full_attention_layers for profile in profiles)


@pytest.mark.parametrize("sparse_method", ["omnikv", "deltakv"])
def test_config_auto_resolution_consumes_packaged_profile(tmp_path, sparse_method):
    from sparsevllm.config import Config

    profile = load_full_attention_layer_profiles()[0]
    model_dir = tmp_path / profile.model_names[0]
    model_dir.mkdir()
    hf_config = SimpleNamespace(
        model_type="qwen2",
        torch_dtype="float16",
        max_position_embeddings=32768,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=max(profile.full_attention_layers) + 1,
    )
    with patch(
        "sparsevllm.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        config = Config(
            model=str(model_dir),
            sparse_method=sparse_method,
            allow_missing_deltakv_path=sparse_method == "deltakv",
        )

    assert config.full_attention_layers == list(profile.full_attention_layers)
    assert config.resolved_full_attention_profile == profile.profile_id
