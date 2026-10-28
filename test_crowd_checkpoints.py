"""Restart each algorithm and compare it with uninterrupted computation."""
from functools import partial
import subprocess
import sys
import pytest
from common import ROOT, np, read_config
from crowd import _CLASSES, create_crowd_game, decode_flow, layout_from_config
from crowd_checkpoints import read_checkpoint, restore_checkpoint, save_checkpoint
from crowd_methods import METHOD_SETTINGS, resolve_settings
from crowd_results import load_case, save_case
from crowd_value_correction import NormalizedBoltzmannPolicyIteration, NormalizedMirrorDescent, normalized_crowd_values
from open_spiel.python.mfg.algorithms import nash_conv
import json
from importlib.metadata import version

@pytest.mark.parametrize('method', list(METHOD_SETTINGS))
@pytest.mark.parametrize('noise', [0.0, 0.5])
def test_restart_matches_uninterrupted(method, noise, tmp_path):
    config = read_config(ROOT / 'configs/crowd_maze_smoke.json')
    (game, _) = create_crowd_game(layout_from_config(config), 3, noise, 1)
    settings = resolve_settings([method])[method]
    classes = dict(_CLASSES, mirror_descent=NormalizedMirrorDescent, boltzmann_policy_iteration=NormalizedBoltzmannPolicyIteration)
    factory = partial(classes[settings['class']], **settings['parameters'])
    config['method_settings'] = {method: settings}
    params = dict(game.get_parameters())
    with normalized_crowd_values():
        full = factory(game)
        reference = [nash_conv.NashConv(game, full.get_policy()).nash_conv()]
        for _ in range(5):
            full.iteration()
            reference.append(nash_conv.NashConv(game, full.get_policy()).nash_conv())
        first = factory(game)
        prefix = [nash_conv.NashConv(game, first.get_policy()).nash_conv()]
        for _ in range(2):
            first.iteration()
            prefix.append(nash_conv.NashConv(game, first.get_policy()).nash_conv())
        path = tmp_path / 'checkpoint.npz'
        save_checkpoint(path, first, method, 2, prefix, config, params, {})
        (metadata, arrays) = read_checkpoint(path)
        resumed = factory(game)
        history = restore_checkpoint(resumed, game, metadata, arrays, settings, params)
        assert history == pytest.approx(reference[:3], abs=1e-12)
        for k in range(3, 6):
            resumed.iteration()
            assert nash_conv.NashConv(game, resumed.get_policy()).nash_conv() == pytest.approx(reference[k], abs=1e-11)
        assert decode_flow(game, resumed.get_policy()) == pytest.approx(decode_flow(game, full.get_policy()), abs=1e-12)

def test_result_archive_is_self_contained(tmp_path):
    flow = np.zeros((3, 3, 3))
    flow[:, 1, 1] = 1
    layout = {'grid': ['###', '#o#', '###'], 'targets': [[1, 1]]}
    save_case(tmp_path, {'fp': [2.0, 1.0]}, {'fp': flow}, layout, {'horizon': 3}, METHOD_SETTINGS, experiment={'seed': 19, 'config': {'iterations': 1}})
    (tmp_path / 'case_metadata.json').unlink()
    (metadata, arrays, _) = load_case(tmp_path)
    assert metadata['experiment']['seed'] == 19
    assert metadata['final_update']['fp'] == 1
    assert arrays['updates_fp'].tolist() == [0, 1]

def test_cli_restart(tmp_path):
    (first, second) = (tmp_path / 'first', tmp_path / 'second')
    subprocess.run([sys.executable, '-B', str(ROOT / 'crowd.py'), '--config', str(ROOT / 'configs/crowd_maze_smoke.json'), '--iterations', '2', '--noise', '0', '--methods', 'omd_0p1', '--reward-mode', 'markov_action_cost_v1', '--value-backend', 'normalized', '--save-checkpoints', '--checkpoint-interval', '1', '--output-root', str(first)], check=True, capture_output=True)
    saved = next(first.rglob('update_000002.npz'))
    subprocess.run([sys.executable, '-B', str(ROOT / 'crowd.py'), '--resume', str(saved), '--iterations', '4', '--no-save-checkpoints', '--output-root', str(second)], check=True, capture_output=True)
    directory = next(second.rglob('case_metadata.json')).parent
    (metadata, arrays, _) = load_case(directory)
    assert metadata['final_update']['omd_0p1'] == 4
    assert metadata['experiment']['config']['resumed_from'] == str(saved)
    assert len(arrays['exploitability_omd_0p1']) == 5
    assert not list(second.rglob('update_*.npz'))


@pytest.mark.parametrize('field,bad_value,message', [
    ('schema_version', 99, 'Unsupported checkpoint schema'),
    ('kind', 'not-a-checkpoint', 'Unsupported checkpoint schema'),
    ('open_spiel', 'incompatible', 'recorded OpenSpiel/NumPy'),
    ('numpy', 'incompatible', 'recorded OpenSpiel/NumPy'),
    ('update', 3, 'history length'),
])
def test_read_rejects_incompatible_metadata(field, bad_value, message, tmp_path):
    metadata = {'schema_version': 1, 'kind': 'crowd-solver-checkpoint',
                'open_spiel': version('open-spiel'), 'numpy': np.__version__, 'update': 0}
    metadata[field] = bad_value
    path = tmp_path / 'invalid.npz'
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata)),
                        exploitability=np.asarray([1.0]))
    with pytest.raises(ValueError, match=message):
        read_checkpoint(path)


def test_restore_rejects_changed_game_or_hyperparameters():
    metadata = {'method_settings': {'class': 'fixed_point'}, 'parameters': {'horizon': 3}}
    with pytest.raises(ValueError, match='game or hyperparameters differ'):
        restore_checkpoint(None, None, metadata, {}, {'class': 'fictitious_play'},
                           {'horizon': 3})
    with pytest.raises(ValueError, match='game or hyperparameters differ'):
        restore_checkpoint(None, None, metadata, {}, {'class': 'fixed_point'},
                           {'horizon': 4})


def test_cli_rejects_source_changes_and_model_overrides(tmp_path):
    command = [sys.executable, '-B', str(ROOT / 'crowd.py')]
    subprocess.run(command + ['--config', str(ROOT / 'configs/crowd_maze_smoke.json'),
                   '--iterations', '1', '--horizon', '3', '--noise', '0', '--methods', 'fp',
                   '--save-checkpoints', '--output-root', str(tmp_path / 'original')],
                   check=True, capture_output=True)
    saved = next((tmp_path / 'original').rglob('update_000001.npz'))
    result = subprocess.run(command + ['--resume', str(saved), '--iterations', '2',
                            '--horizon', '4'], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'retains the checkpoint' in result.stderr
    metadata, arrays = read_checkpoint(saved)
    metadata['provenance']['source_sha256']['crowd.py'] = '0' * 64
    arrays['metadata_json'] = np.asarray(json.dumps(metadata))
    altered = tmp_path / 'altered.npz'
    np.savez_compressed(altered, **arrays)
    result = subprocess.run(command + ['--resume', str(altered), '--iterations', '2'],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Checkpoint implementation differs: crowd.py' in result.stderr
