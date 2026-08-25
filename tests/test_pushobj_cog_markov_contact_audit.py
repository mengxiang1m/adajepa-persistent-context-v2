import numpy as np

from research.persistent_context_v2.pushobj_cog_markov_contact_audit import (
    ACTION_COUNT,
    CONTACT_FEATURES,
    CONTACT_KINDS,
    bootstrap_delta,
    choose_alpha,
    feature_map,
    representation_inputs,
    standardizer,
    trajectory_errors,
)


def test_representation_dimensions_are_nested():
    commands = np.zeros((ACTION_COUNT, 2), dtype=np.float32)
    states = np.zeros((ACTION_COUNT + 1, 10), dtype=np.float32)
    contacts = np.zeros((ACTION_COUNT, len(CONTACT_KINDS), len(CONTACT_FEATURES)), dtype=np.float32)
    values = representation_inputs(commands, states, contacts)
    assert values["R0_legacy"].shape == (108,)
    assert values["R1_markov"].shape == (141,)
    assert values["R2_nominal_agent_block_contact"].shape == (141 + ACTION_COUNT * len(CONTACT_FEATURES),)
    np.testing.assert_array_equal(values["R1_markov"][:108], values["R0_legacy"])
    np.testing.assert_array_equal(values["R2_nominal_agent_block_contact"][:141], values["R1_markov"])


def test_context_feature_map_has_exact_zero_identity():
    x = np.arange(24, dtype=np.float64).reshape(6, 4)
    mean, scale = standardizer(x)
    mapped = feature_map(x, np.zeros(6), mean, scale)
    assert mapped.shape == (6, 10)
    assert np.max(np.abs(mapped)) == 0.0


def test_trajectory_pose_error_uses_position_norm_and_angle():
    target = np.zeros((2, ACTION_COUNT * 3), dtype=np.float32)
    prediction = target.copy().reshape(2, ACTION_COUNT, 3)
    prediction[0, :, 0] = 3.0
    prediction[0, :, 1] = 4.0
    prediction[0, :, 2] = 2.0
    errors = trajectory_errors(prediction.reshape(2, -1), target)
    np.testing.assert_allclose(errors, [7.0, 0.0], atol=1e-6)


def test_bootstrap_counts_segments_not_trajectories():
    segment_ids = np.repeat([10, 20, 30], 4)
    first = np.repeat([3.0, 2.0, 1.0], 4)
    second = np.repeat([2.0, 2.0, 2.0], 4)
    result = bootstrap_delta(first, second, segment_ids, seed=7, resamples=1000)
    assert result["n_segments"] == 3
    assert result["positive"] == 1
    assert result["tie"] == 1
    assert result["negative"] == 1
    assert abs(result["mean_delta"]) < 1e-12


def test_group_cv_reduces_fold_count_for_smoke_without_empty_folds():
    x = np.arange(32, dtype=np.float64).reshape(8, 4)
    target = np.zeros((8, ACTION_COUNT * 3), dtype=np.float64)
    context = np.tile([-1.0, 1.0], 4)
    segment_ids = np.repeat([10, 20], 4)
    selected, rows = choose_alpha(x, target, context, segment_ids, [0.1, 1.0], folds=4)
    assert selected in (0.1, 1.0)
    assert all(row["effective_folds"] == 2 for row in rows)
    assert all(len(row["fold_scores"]) == 2 for row in rows)
    assert all(np.isfinite(row["mean_segment_pose_error"]) for row in rows)
