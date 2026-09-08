"""Protect portable experiment serialization and reject ambiguous output identities."""
import copy
import hashlib
import json

import pytest

from scripts.official_experiments.sparsevllm_vs_vortex.prepare_runs import (
    build_outputs, check_vortex_sources,
)


def build(cases, root, **variables):
    return build_outputs({'cases': cases}, {'OUTPUT_ROOT': str(root), **variables},
                         {'qwen'}, {'qwen': 0}, {'qwen': (1234, 1235)})


def native_case():
    return {'id': 'example', 'model_key': 'qwen', 'directory': 'qwen/example',
            'hyper_params': {'budget': 12},
            'job': {'command': ['python', '--config', '${HYPER_PARAMS_FILE}']}}


def test_nested_vortex_json_keeps_paths_as_single_arguments(tmp_path):
    # Paths containing quotes/spaces must survive both JSON serialization layers.
    case = {'id': 'example', 'model_key': 'qwen', 'directory': 'qwen/example',
            'vortex_config': {'module_path': '${MODEL_PATH}', 'budget': 12},
            'server_config': {'argv': ['--config', '${VORTEX_CONFIG_JSON}']},
            'job': {'command': ['python', '${SERVER_CONFIG_FILE}']}}
    model_path = str(tmp_path / 'a "quoted" model' / 'weights')
    result = build([case], tmp_path, MODEL_PATH=model_path)
    roundtrip = json.loads(json.dumps(result))
    argv = roundtrip['configs/example.server.json']['argv']
    assert len(argv) == 2
    assert json.loads(argv[1]) == {'module_path': model_path, 'budget': 12}
    assert roundtrip['qwen/jobs/example.json']['command'][1] == str(
        tmp_path / 'configs/example.server.json')


def test_missing_template_variable_cannot_create_a_partial_plan(tmp_path):
    case = native_case()
    case['job']['command'].append('${MISSING_MODEL}')
    with pytest.raises(KeyError, match='MISSING_MODEL'):
        build([case], tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('field,value', [('directory', '../escape'),
    ('directory', '/absolute/escape'), ('directory', 'glm/wrong_model'),
    ('id', '../escape')])
def test_output_identity_cannot_escape_its_case_directory(tmp_path, field, value):
    case = native_case()
    case[field] = value
    with pytest.raises(ValueError, match='Invalid case'):
        build([case], tmp_path)


@pytest.mark.parametrize('same_id', [True, False])
def test_duplicate_configs_or_job_targets_are_not_silently_overwritten(tmp_path, same_id):
    first = native_case()
    second = copy.deepcopy(first)
    if same_id:
        second['directory'] = 'qwen/different_job'
    else:
        second['id'] = 'different_config'
    with pytest.raises(ValueError, match='Duplicate case'):
        build([first, second], tmp_path)


def test_baseline_source_drift_or_missing_file_fails_explicitly(tmp_path):
    source = tmp_path / 'method.py'
    source.write_bytes(b'recorded implementation')
    provenance = {'engines': {'vortex': {'source_sha256': {
        'method.py': hashlib.sha256(source.read_bytes()).hexdigest()}}}}
    check_vortex_sources(tmp_path, provenance)
    source.write_bytes(b'changed implementation')
    with pytest.raises(ValueError, match='baseline mismatch'):
        check_vortex_sources(tmp_path, provenance)
    source.unlink()
    with pytest.raises(ValueError, match='baseline mismatch'):
        check_vortex_sources(tmp_path, provenance)
