import json
from pathlib import Path

import numpy as np

from research.persistent_context_v2.cross_shape_matrix_history_formal import (
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_DESIGN_SHA256,
    EXPECTED_SELECTION_SHA256,
    feature,
    load_frozen,
    predict_f0,
)
from research.persistent_context_v2.pushobj_rotation_stage0 import sha256


ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "docs/research/persistent_context_v2_cross_shape_matrix_history_design.json"
CONTRACT = ROOT / "docs/research/persistent_context_v2_cross_shape_matrix_history_contract_zh.md"
SELECTION = ROOT / "docs/research/persistent_context_v2_cross_shape_matrix_history_selection.json"


def test_frozen_hashes_and_balance():
    assert sha256(DESIGN) == EXPECTED_DESIGN_SHA256
    assert sha256(CONTRACT) == EXPECTED_CONTRACT_SHA256
    assert sha256(SELECTION) == EXPECTED_SELECTION_SHA256
    design, selection = load_frozen(DESIGN, CONTRACT, SELECTION)
    assert len(selection["smoke"]) == 6
    assert len(selection["formal"]) == 96
    assert [sum(row["shape_pair_index"] == index for row in selection["formal"]) for index in range(6)] == [16] * 6
    assert [sum(row["factor_index"] == index for row in selection["formal"]) for index in range(8)] == [12] * 8
    assert all(sum(row["shape_pair_index"] == pair and row["factor_index"] == factor for row in selection["formal"]) == 2
               for pair in range(6) for factor in range(8))
    assert design["formal"]["sequences"] == 96


def test_selection_identifiers_are_globally_unique():
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    items = [row[episode] for split in ("smoke", "formal") for row in selection[split] for episode in ("e1", "e2")]
    items += [item for rows in selection["reserve"].values() for item in rows]
    assert len(items) == 228
    assert len({item["segment_sha256"] for item in items}) == 228
    assert len({item["provenance_key"] for item in items}) == 228


def test_external_f0_prediction_uses_ascending_tie_rule():
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    posterior = {"mean_z": [design["gain_center"], 0.0]}
    x = feature(posterior, design)
    assert np.allclose(x, [1, 0, 0, 0, 0, 0])
    model = {"alphas": [0, .25, .5, .75, 1], "mean": [0] * 6, "scale": [1] * 6, "beta": [0] * 12}
    prediction = predict_f0(model, x)
    assert np.argmax(prediction) == 0
    assert np.all(prediction == 0)
