#!/usr/bin/env python3
"""Descriptive heterogeneity analysis from frozen Stage-1 raw artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.persistent_context_v2.pushobj_rotation_stage0 import read_jsonl
from research.persistent_context_v2.pushobj_rotation_stage1 import (
    CONDITIONS,
    POLICIES,
    wrapped_degrees_error,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("repro_outputs/persistent_context_v2_pushobj_rotation_stage1"),
    )
    args = parser.parse_args()
    summary = json.loads((args.output_dir / "runner_summary.json").read_text(encoding="utf-8"))
    result = {
        "schema": "persistent-context-v2-pushobj-rotation-stage1-descriptive-v1",
        "note": "Post-formal descriptive analysis; no automatic decision threshold.",
        "conditions": {},
    }
    for condition in CONDITIONS:
        raw = read_jsonl(args.output_dir / f"{condition}_raw.jsonl")
        evidence = [row for row in raw if row.get("record_type") == "evidence_episode"]
        evaluations = [row for row in raw if row.get("record_type") == "evaluation_episode"]
        evidence_by_key = {
            (int(row["sequence_id"]), int(row["episode_index"])): row for row in evidence
        }
        later = [row for row in evaluations if int(row["episode_index"]) > 0]
        policy_context = {}
        for policy in ("correct_history", "shuffled_history", "wrong_sequence_history"):
            errors = np.asarray(
                [
                    abs(
                        wrapped_degrees_error(
                            row["policies"][policy]["context_degrees"], row["factor_deg"]
                        )
                    )
                    for row in later
                ],
                dtype=np.float64,
            )
            changes = [
                row["policies"][policy]["command_sha256"]
                != row["policies"]["current_only"]["command_sha256"]
                for row in later
            ]
            donor_factor_matches = []
            for row in later:
                for donor in row["policies"][policy]["donors"]:
                    source = evidence_by_key[
                        (int(donor["donor_sequence_id"]), int(donor["history_episode"]) - 1)
                    ]
                    donor_factor_matches.append(float(source["factor_deg"]) == float(row["factor_deg"]))
            policy_context[policy] = {
                "angle_mae_degrees": float(errors.mean()),
                "angle_median_absolute_error_degrees": float(np.median(errors)),
                "action_changed_fraction": float(np.mean(changes)),
                "donor_transition_episode_factor_match_fraction": float(np.mean(donor_factor_matches)),
            }
        sequence_delta = np.asarray(
            summary[f"{condition}_correct_history"]["sequence_deltas"], dtype=np.float64
        )
        contact_fractions = []
        for sequence_id in range(len(sequence_delta)):
            contacts = []
            for episode_index in range(3):
                contacts.extend(evidence_by_key[(sequence_id, episode_index)]["contacts"])
            contact_fractions.append(float(np.mean(np.asarray(contacts) > 0)))
        contact_fractions = np.asarray(contact_fractions, dtype=np.float64)
        median = float(np.median(contact_fractions))
        low = contact_fractions <= median
        high = contact_fractions > median
        correlation = float(np.corrcoef(contact_fractions, sequence_delta)[0, 1])
        result["conditions"][condition] = {
            "policy_context": policy_context,
            "evidence_contact_fraction": {
                "mean": float(contact_fractions.mean()),
                "median": median,
                "min": float(contact_fractions.min()),
                "max": float(contact_fractions.max()),
                "pearson_with_sequence_behavior_delta": correlation,
                "low_or_equal_median_n": int(low.sum()),
                "low_or_equal_median_mean_delta": float(sequence_delta[low].mean()),
                "above_median_n": int(high.sum()),
                "above_median_mean_delta": float(sequence_delta[high].mean()),
            },
        }
    target = args.output_dir / "descriptive_analysis.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
