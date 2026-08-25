import numpy as np

from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage1 import (
    donor_maps,
    factor_schedules,
)


def test_factor_schedules_are_balanced_and_changing():
    persistent, changing = factor_schedules()
    assert all(len(set(row.tolist())) == 1 for row in persistent)
    assert np.array_equal(persistent[:, 0], changing[:, 0])
    assert all(changing[:, e].tolist().count(f) == 8 for e in range(4) for f in range(4))
    assert np.all(changing[:, 1:] != changing[:, :-1])


def test_donors_are_cross_factor_nonself_and_shuffled_unique():
    wrong, shuffled = donor_maps()
    ids = np.arange(32)
    assert np.all(wrong != ids)
    assert np.all(wrong % 4 != ids % 4)
    assert all(len(set(row.tolist())) == 3 for row in shuffled)
    assert np.all(shuffled != ids[:, None])
    assert np.all(shuffled % 4 != (ids % 4)[:, None])
