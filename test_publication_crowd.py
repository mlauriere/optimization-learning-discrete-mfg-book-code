"""Publication sampling and labels follow the saved data, not solver defaults."""

import pytest

from common import np
from crowd_methods import METHOD_SETTINGS, resolve_settings
from crowd_results import save_case
import publication_crowd as figures


def test_common_display_indices_keep_full_initial_segment_and_endpoint():
    expected = np.r_[np.arange(21), np.arange(25, 1001, 5)]
    assert len(expected) == 217
    assert np.array_equal(figures.exploitability_indices(1001, 5), expected)


@pytest.mark.parametrize('stride', [0, -1, 2, 4])
def test_display_stride_rejects_nonpositive_or_even_values(stride):
    with pytest.raises(ValueError, match='positive odd stride'):
        figures.exploitability_indices(1001, stride)


def test_paired_plot_reads_saved_budget_and_hyperparameters(tmp_path):
    settings = resolve_settings(list(METHOD_SETTINGS), {'omd_0p1': {'lr': 0.27}})
    flow = np.zeros((3, 3, 3))
    flow[:, 1, 1] = 1
    layout = {'grid': ['###', '#o#', '###'], 'targets': [[1, 1]]}
    for case, noise in (('maze_c1_p0', 0.0), ('maze_c1_p0p5', 0.5)):
        directory = tmp_path / case
        directory.mkdir()
        save_case(directory, {m: np.linspace(3, 1, 8) for m in settings},
                  {m: flow for m in settings}, layout,
                  {'horizon': 3, 'noise_intensity': noise}, settings)
        (directory / 'case_metadata.json').unlink()
    with figures.plt.rc_context({'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 8,
                                'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5}):
        details = figures.combined_exploitability(tmp_path, tmp_path, maze=True,
                                                  legend_columns=3, plot_stride=5)
    assert details['updates'] == [0, 7]
    assert details['legend_columns'] == 3
    assert not details['fixed_point_markers'] and not details['smoothing']
    assert any('0.27' in label for label in details['legend_order'])
    assert all(v['retained_updates'] == list(range(8))
               for case in details['sampling'].values() for v in case.values())
