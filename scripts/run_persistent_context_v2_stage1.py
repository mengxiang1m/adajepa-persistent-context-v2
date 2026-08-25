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

from research.persistent_context_v2.stage1 import Stage1Config, run_stage1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-sequences", type=int, default=384)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    args = parser.parse_args()
    config = Stage1Config(n_sequences=args.n_sequences, bootstrap_resamples=args.bootstrap_resamples)
    command = " ".join(shlex.quote(item) for item in sys.argv)
    summary = run_stage1(args.output_dir, config=config, command=command)
    print(json.dumps({"decision": summary["decision"], "criteria": summary["criteria"]}, indent=2, sort_keys=True))
    return 2 if summary["decision"]["verdict"] == "INVALID_EXECUTION" else 0


if __name__ == "__main__":
    raise SystemExit(main())
