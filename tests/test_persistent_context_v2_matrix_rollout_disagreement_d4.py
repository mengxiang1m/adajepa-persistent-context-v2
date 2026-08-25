import json
import sys
import types

import numpy as np
import torch

from research.persistent_context_v2.matrix_rollout_disagreement_d4 import (
    flatten_latent,
    nested_leave_pair_out,
    preoutcome_traces,
    quartile_summary,
    rankdata,
    rms_pair,
    roc_auc,
    rollout_features,
)
from scripts.create_matrix_rollout_disagreement_d4_selection import prior_exclusions


def latent(value):
    return {"b": np.full((1, 2, 1), value, dtype=np.float32),
            "a": np.full((1, 2, 2), value, dtype=np.float32)}


def test_flatten_and_frozen_rollout_feature_order():
    traces = {
        "prior_population": latent(0),
        "prior_context": latent(2),
        "context_population": latent(1),
        "context_context": latent(4),
    }
    assert flatten_latent(traces["prior_population"]).shape == (2, 3)
    assert rms_pair(traces["prior_population"], traces["context_population"]) == (1.0, 1.0)
    np.testing.assert_allclose(rollout_features(traces), [1, 1, 2, 2, 2, 2, 3, 3])


def test_rank_and_auc_handle_ties_and_single_class():
    np.testing.assert_allclose(rankdata([3, 1, 1, 2]), [4, 1.5, 1.5, 3])
    assert roc_auc([False, False, True, True], [0, 1, 2, 3]) == 1.0
    assert roc_auc([False, True], [1, 1]) == 0.5
    assert roc_auc([False, False], [0, 1]) is None


def test_quartiles_preserve_all_independent_units():
    result = quartile_summary(np.arange(10), np.arange(10) - 5, 1e-12)
    assert [row["n"] for row in result] == [3, 3, 2, 2]
    assert sorted(index for row in result for index in row["sequence_indices"]) == list(range(10))


def test_nested_pair_prediction_is_out_of_group():
    rows = []
    for pair in range(3):
        for value in (-2.0, -1.0, 1.0, 2.0):
            rows.append({"x": np.asarray([value, value**2]), "benefit": value, "shape_pair_index": pair})
    prediction, chosen = nested_leave_pair_out(rows, [0.01, 1.0])
    assert prediction.shape == (12,)
    assert set(chosen) == {"0", "1", "2"}
    assert np.corrcoef(prediction, [row["benefit"] for row in rows])[0, 1] > 0.99


def test_prior_exclusions_include_all_frozen_old_splits():
    item = lambda index: {"segment_sha256": f"h{index}", "provenance_key": f"p{index}"}
    selection = {
        "smoke": [{"e1": item(1), "e2": item(2)}],
        "formal": [{"e1": item(3), "e2": item(4)}],
        "reserve": {"I": [item(5)]},
    }
    hashes, provenance = prior_exclusions(selection)
    assert hashes == {"h1", "h2", "h3", "h4", "h5"}
    assert provenance == {"p1", "p2", "p3", "p4", "p5"}


class _Preprocessor:
    def transform_obs(self, obs):
        return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in obs.items()}

    def normalize_actions(self, actions):
        return actions


class _WorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.matrix = np.eye(2)

    def set_context(self, matrix):
        self.matrix = np.asarray(matrix)

    def encode_obs(self, obs):
        return obs

    def rollout(self, obs, actions):
        matrix = actions.new_tensor(self.matrix)
        physical = actions.reshape(actions.shape[0], actions.shape[1], 5, 2) @ matrix.T
        value = physical.mean(dim=(2, 3), keepdim=False)[..., None]
        return {"visual": value, "proprio": value * 2}, None


def test_preoutcome_probe_has_frozen_score_names_and_does_not_consume_rng(monkeypatch):
    monkeypatch.setitem(sys.modules, "utils", types.SimpleNamespace(move_to_device=lambda value, device: value))
    wm = _WorldModel()
    obs_0 = {"visual": np.asarray([0.0]), "proprio": np.asarray([0.0])}
    obs_g = {"visual": np.asarray([1.0]), "proprio": np.asarray([2.0])}
    population = np.zeros((10, 2), dtype=np.float32)
    context = np.ones((10, 2), dtype=np.float32)
    traces, goal, scores, latent_features, objective_features, before, after = preoutcome_traces(
        wm, _Preprocessor(), obs_0, obs_g, population, context, np.asarray([[1.1, 0.0], [0.0, 0.9]])
    )
    assert set(traces) == {"prior_population", "prior_context", "context_population", "context_context"}
    assert set(scores) == {"J_prior_a_prior", "J_prior_a_context", "J_context_a_prior", "J_context_a_context"}
    assert set(goal) == {"visual", "proprio"}
    assert latent_features.shape == (8,)
    assert objective_features.shape == (4,)
    assert before == after
