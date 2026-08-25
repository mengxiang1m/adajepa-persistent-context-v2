import json
from pathlib import Path

import numpy as np

from research.persistent_context_v2.matrix_f0_soft_policy_formal import (
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_DESIGN_SHA256,
    EXPECTED_SELECTION_SHA256,
    load_frozen,
    model_prediction,
    scenario,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import sha256


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "docs/research/persistent_context_v2_matrix_f0_soft_policy_design.json"
CONTRACT = ROOT / "docs/research/persistent_context_v2_matrix_f0_soft_policy_contract_zh.md"
SELECTION = ROOT / "docs/research/persistent_context_v2_matrix_f0_soft_policy_selection.json"


def test_frozen_hashes_and_split_counts():
    assert sha256(DESIGN) == EXPECTED_DESIGN_SHA256
    assert sha256(CONTRACT) == EXPECTED_CONTRACT_SHA256
    assert sha256(SELECTION) == EXPECTED_SELECTION_SHA256
    design, selection = load_frozen(DESIGN, CONTRACT, SELECTION)
    assert [len(selection[name]) for name in ("train", "dev", "formal", "reserve")] == [128, 64, 192, 29]
    assert len(set(selection["train"] + selection["dev"] + selection["formal"])) == 384
    assert all(len(selection[name]) == 2 * design["splits"][name]["sequences"] for name in ("train", "dev", "formal"))


def test_scenario_is_balanced_and_uses_two_segments():
    design, selection = load_frozen(DESIGN, CONTRACT, SELECTION)
    first = scenario(design, selection, "formal", 0, 0)
    second = scenario(design, selection, "formal", 0, 1)
    assert first["factor_index"] == 0
    assert first["segment_index"] != second["segment_index"]
    assert first["env_seed"] + 1 == second["env_seed"]
    assert scenario(design, selection, "formal", 8, 0)["factor_index"] == 0


def test_model_prediction_zero_alpha_is_zero():
    model = {"mean": [0] * 6, "scale": [1] * 6, "beta": list(range(12))}
    prediction = model_prediction(model, np.ones(6), [0.0, 0.5, 1.0])
    assert prediction[0] == 0.0
    assert np.all(np.isfinite(prediction))
