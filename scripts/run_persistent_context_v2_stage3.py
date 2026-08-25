#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.persistent_context_v2.stage3 import Stage3Config, run_stage3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--train-steps", type=int, default=2_000)
    parser.add_argument("--formal-sequences", type=int, default=384)
    parser.add_argument("--formal-bootstrap-resamples", type=int, default=20_000)
    args = parser.parse_args()
    config = Stage3Config(
        train_steps=args.train_steps,
        formal_sequences=args.formal_sequences,
        formal_bootstrap_resamples=args.formal_bootstrap_resamples,
    )
    command = " ".join(shlex.quote(item) for item in sys.argv)
    summary = run_stage3(args.output_dir, config=config, command=command, device_name=args.device)
    print(json.dumps({
        "development_passed": summary["development_gate"]["passed"],
        "formal_outcomes_generated": summary["formal_outcomes_generated"],
        "decision": summary["decision"],
    }, indent=2, sort_keys=True))
    return 2 if summary["decision"]["verdict"] == "INVALID_EXECUTION" else 0


if __name__ == "__main__":
    raise SystemExit(main())
