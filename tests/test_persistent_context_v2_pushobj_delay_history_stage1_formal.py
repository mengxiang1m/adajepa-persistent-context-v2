import json
from pathlib import Path

import numpy as np

from research.persistent_context_v2.pushobj_delay_history_stage1_formal import (
    FACTORS,
    bootstrap,
    factor_for,
    history_payload,
    split_pairs,
)
from research.persistent_context_v2.pushobj_delay_stage0 import apply_discrete_delay
from research.persistent_context_v2.pushobj_rotation_stage1 import pd_coefficients


ROOT = Path(__file__).resolve().parents[1]


def load_design_selection():
    design = json.loads((ROOT / "docs/research/persistent_context_v2_pushobj_delay_history_stage1_formal_design.json").read_text(encoding="utf-8"))
    selection = json.loads((ROOT / "docs/research/persistent_context_v2_pushobj_delay_history_stage1_formal_selection.json").read_text(encoding="utf-8"))
    return design, selection


def states_from_effective(effective):
    effective = np.asarray(effective, dtype=np.float64)
    pc, vc, tc = pd_coefficients()
    states = np.zeros((len(effective) + 1, 7), dtype=np.float64)
    states[0, :2] = [1.5, -0.75]
    for index, action in enumerate(effective):
        p0, v0 = states[index, :2], states[index, 5:7]
        states[index + 1, :2] = pc * p0 + vc * v0 + tc * (p0 + 100.0 * action)
    return states


def evidence(n):
    rows = {}
    for sequence_id in range(n):
        delay = FACTORS[sequence_id % 4]
        commands = np.stack([np.linspace(-0.5, 0.4, 10), np.linspace(0.25, -0.45, 10)], axis=1) + sequence_id * 1e-3
        states = states_from_effective(apply_discrete_delay(commands, delay))
        rows[sequence_id] = {"commands": commands, "states": states, "factor_steps": delay, "evidence_sha256": str(sequence_id)}
    return rows


def test_frozen_selection_sizes_and_global_uniqueness():
    _, selection = load_design_selection()
    smoke, formal = split_pairs(selection, "smoke"), split_pairs(selection, "formal")
    assert len(smoke) == 4 and len(formal) == 32
    selected = [value for pairs in (smoke, formal) for pair in pairs for value in pair] + selection["reserve_segment_indices"]
    assert len(selected) == len(set(selected))


def test_conditions_are_factor_balanced_and_wrong_offset_avoids_both_currents():
    design, selection = load_design_selection()
    for split in ("smoke", "formal"):
        n = len(split_pairs(selection, split))
        for condition in ("persistent", "no_persistence"):
            values = [factor_for(condition, sequence_id, 1) for sequence_id in range(n)]
            assert {factor: values.count(factor) for factor in FACTORS} == {factor: n // 4 for factor in FACTORS}
        for sequence_id in range(n):
            donor = (sequence_id + design["wrong_donor_sequence_offset"]) % n
            donor_factor = factor_for("persistent", donor, 0)
            assert donor_factor != factor_for("persistent", sequence_id, 1)
            assert donor_factor != factor_for("no_persistence", sequence_id, 1)


def test_wrong_history_payload_uses_frozen_collision_free_donor():
    design, _ = load_design_selection()
    bank = evidence(32)
    for sequence_id in range(32):
        payload = history_payload(bank, "wrong_sequence_history", sequence_id, design)
        donor = (sequence_id + 2) % 32
        assert payload["donors"][0]["sequence_id"] == donor
        assert payload["context_delay_steps"] == FACTORS[donor % 4]
        assert payload["current_episode_evidence_count"] == 0


def test_bootstrap_is_sequence_level_and_deterministic():
    design, _ = load_design_selection()
    values = np.arange(32, dtype=np.float64)
    first = bootstrap(values, design, 123)
    second = bootstrap(values, design, 123)
    other = bootstrap(values, design, 124)
    assert first == second
    assert first != other
