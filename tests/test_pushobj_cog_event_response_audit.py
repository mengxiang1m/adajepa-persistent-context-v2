import numpy as np

from research.persistent_context_v2.pushobj_cog_event_response_audit import (
    AGENT_BLOCK,
    BLOCK_WALL,
    CONTACT_FEATURES,
    CONTACT_KINDS,
    aggregate_contact_substeps,
    choose_alpha,
    eligible_event_mask,
    feature_map,
    grouped_means,
    representation_rows,
    response_target,
    standardizer,
)


def _index(name):
    return CONTACT_FEATURES.index(name)


def _trace():
    count = 100
    contacts = np.zeros((count, len(CONTACT_KINDS), len(CONTACT_FEATURES)), dtype=np.float32)
    contacts[3, AGENT_BLOCK, _index("point_count")] = 1
    contacts[3, AGENT_BLOCK, _index("first_contact_count")] = 1
    contacts[3, AGENT_BLOCK, _index("normal_x")] = 1
    contacts[3, AGENT_BLOCK, _index("contact_rel_x")] = 4
    contacts[3, AGENT_BLOCK, _index("contact_radius")] = 4
    contacts[3, AGENT_BLOCK, _index("min_distance")] = -0.2
    contacts[3, AGENT_BLOCK, _index("impulse_x")] = 2
    contacts[3, AGENT_BLOCK, _index("impulse_norm_sum")] = 2
    contacts[3, AGENT_BLOCK, _index("impulse_norm_max")] = 2
    contacts[3, AGENT_BLOCK, _index("total_ke_sum")] = 5
    contacts[3, AGENT_BLOCK, _index("total_ke_max")] = 5
    pre = np.zeros((count, 10), dtype=np.float32)
    post = pre.copy()
    boundary = np.zeros((11, 10), dtype=np.float32)
    commands = np.zeros((count, 2), dtype=np.float32)
    targets = np.ones((count, 2), dtype=np.float32)
    return {
        "boundary_states": boundary,
        "pre_states": pre,
        "post_states": post,
        "contacts": contacts,
        "commands": commands,
        "targets": targets,
        "control_ids": np.repeat(np.arange(10), 10).astype(np.int16),
        "substep_ids": np.tile(np.arange(10), 10).astype(np.int16),
    }


def test_event_rule_uses_nominal_agent_block_and_excludes_wall():
    trace = _trace()
    mask = eligible_event_mask(trace)
    assert np.flatnonzero(mask).tolist() == [3]
    trace["contacts"][3, BLOCK_WALL, _index("point_count")] = 1
    assert not eligible_event_mask(trace).any()


def test_representation_dimensions_and_nesting():
    nominal = _trace()
    true = _trace()
    mask = eligible_event_mask(nominal)
    rows = representation_rows(nominal, true, mask)
    assert rows["C10_aggregate"].shape == (1, 29)
    assert rows["S100_state"].shape == (1, 15)
    assert rows["S100_state_geometry"].shape == (1, 23)
    assert rows["S100_state_geometry_impulse"].shape == (1, 29)
    assert rows["P100_true_contact"].shape == (1, 43)
    np.testing.assert_array_equal(rows["S100_state_geometry"][:, :15], rows["S100_state"])
    np.testing.assert_array_equal(rows["S100_state_geometry_impulse"][:, :23], rows["S100_state_geometry"])


def test_response_target_is_true_minus_nominal_generalized_velocity_change():
    nominal = _trace()
    true = _trace()
    mask = eligible_event_mask(nominal)
    nominal["post_states"][3, 7:10] = [1, 2, 3]
    true["post_states"][3, 7:10] = [4, 8, 12]
    np.testing.assert_allclose(response_target(nominal, true, mask), [[3, 6, 9]])


def test_contact_aggregation_sums_impulse_and_weights_geometry():
    contacts = np.zeros((2, len(CONTACT_KINDS), len(CONTACT_FEATURES)), dtype=np.float32)
    contacts[:, AGENT_BLOCK, _index("point_count")] = [1, 3]
    contacts[:, AGENT_BLOCK, _index("impulse_norm_sum")] = [1, 3]
    contacts[:, AGENT_BLOCK, _index("impulse_x")] = [1, 3]
    contacts[:, AGENT_BLOCK, _index("normal_x")] = [1, -1]
    contacts[:, AGENT_BLOCK, _index("contact_rel_x")] = [2, 6]
    contacts[:, AGENT_BLOCK, _index("min_distance")] = [-0.1, -0.4]
    aggregate = aggregate_contact_substeps(contacts, np.zeros(2, dtype=np.int16), action_count=1)[0, AGENT_BLOCK]
    assert aggregate[_index("point_count")] == 4
    assert aggregate[_index("impulse_x")] == 4
    np.testing.assert_allclose(aggregate[_index("normal_x")], -0.5)
    np.testing.assert_allclose(aggregate[_index("contact_rel_x")], 5.0)
    np.testing.assert_allclose(aggregate[_index("min_distance")], -0.4)


def test_context_feature_map_is_exactly_zero_at_zero_cog():
    values = np.arange(24, dtype=np.float64).reshape(6, 4)
    mean, scale = standardizer(values)
    mapped = feature_map(values, np.zeros(6), mean, scale)
    assert mapped.shape == (6, 10)
    assert np.max(np.abs(mapped)) == 0


def test_grouped_cv_and_grouped_metric_use_segments():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(48, 5))
    contexts = np.tile([-30.0, -15.0, 15.0, 30.0], 12)
    segment_ids = np.repeat([10, 20, 30, 40], 12)
    targets = np.stack([contexts / 30.0, np.zeros(48), np.zeros(48)], axis=1)
    design = {"group_cv_folds": 4, "ridge_alphas": [0.01, 1.0]}
    selected, rows = choose_alpha(values, targets, contexts, segment_ids, design)
    assert selected in (0.01, 1.0)
    assert all(row["effective_folds"] == 4 for row in rows)
    unique, means = grouped_means(np.arange(48), segment_ids)
    assert unique.tolist() == [10, 20, 30, 40]
    assert means.shape == (4,)
