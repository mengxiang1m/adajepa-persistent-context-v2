import numpy as np

from research.persistent_context_v2.pushobj_cog_stage0 import FACTORS, SEGMENT_INDICES, bootstrap_ci, scenarios


def _segments():
    result=[]
    for _ in range(500):
        states=np.zeros((11,7),dtype=np.float32);states[10,2]=12
        result.append({"states":states,"actions":np.zeros((10,2)),"shape":"T"})
    return result


def test_frozen_cog_scenarios_are_unique_balanced_and_early():
    rows=scenarios(_segments())
    assert len(rows)==32 and len(set(SEGMENT_INDICES))==32
    assert [row["factor_cog_x"] for row in rows[::8]]==list(FACTORS)
    assert all(row["nominal_block_displacement_at_10"]==12 for row in rows)


def test_bootstrap_is_deterministic():
    values=np.asarray([-1.,0.,1.,2.])
    assert bootstrap_ci(values)==bootstrap_ci(values)
