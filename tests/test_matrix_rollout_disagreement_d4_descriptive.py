import json

import numpy as np

from scripts.summarize_matrix_rollout_disagreement_d4_descriptive import bootstrap_mean_ci, summarize_group


def row(sequence_id, population, context):
    return {
        "sequence_id": sequence_id,
        "e2": {"policies": {
            "population": {"metrics": {"pose_auc10": population}},
            "context": {"metrics": {"pose_auc10": context}},
        }},
    }


def test_group_summary_uses_sequence_as_unit_and_reports_all_signs():
    rows = [row(0, 3, 2), row(1, 3, 4), row(2, 3, 3)]
    result = summarize_group(rows, [0, 1, 2], 0)
    assert result["n"] == 3
    assert result["mean_benefit"] == 0.0
    assert result["positive_count"] == result["negative_count"] == result["tie_count"] == 1
    assert result["sequence_ids"] == [0, 1, 2]
    assert result["unit_benefit"] == [1.0, -1.0, 0.0]


def test_bootstrap_is_deterministic():
    first = bootstrap_mean_ci(np.asarray([-1.0, 0.0, 2.0]), 123)
    second = bootstrap_mean_ci(np.asarray([-1.0, 0.0, 2.0]), 123)
    assert first == second
