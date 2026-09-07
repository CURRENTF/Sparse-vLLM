"""Regression: serialized FP8 checkpoints may omit optional generation metadata."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from benchmark.long_bench.pred import load_model_and_tokenizer


@pytest.mark.parametrize('malformed', [False, True])
def test_longbench_missing_generation_metadata_preserves_eos_but_bad_file_fails(tmp_path, malformed):
    (tmp_path / 'config.json').write_text(json.dumps({'model_type': 'llama', 'eos_token_id': [2, 3]}))
    if malformed:
        (tmp_path / 'generation_config.json').write_text('{bad json')
    args = SimpleNamespace(model_path=str(tmp_path), tokenizer_path=str(tmp_path),
                           deltakv_checkpoint_path=None, sparse_method='vanilla', max_model_len=1024)
    tokenizer = SimpleNamespace(eos_token_id=3, eot_token_id=4)
    with patch('benchmark.long_bench.pred.get_sparsevllm_generate_api'), patch(
        'benchmark.long_bench.pred.AutoTokenizer.from_pretrained', return_value=tokenizer
    ):
        if malformed:
            with pytest.raises(OSError):
                load_model_and_tokenizer(0, args, {})
        else:
            assert load_model_and_tokenizer(0, args, {})[3] == [2, 3, 4]
