#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, shlex, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))
from research.persistent_context_v2.stage2 import Stage2Config, run_stage2
def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--n-sequences", type=int, default=384); parser.add_argument("--bootstrap-resamples", type=int, default=20_000); args = parser.parse_args()
    config = Stage2Config(n_sequences=args.n_sequences, bootstrap_resamples=args.bootstrap_resamples)
    summary = run_stage2(args.output_dir, config, " ".join(shlex.quote(item) for item in sys.argv))
    print(json.dumps({"decision": summary["decision"], "criteria": summary["criteria"]}, indent=2, sort_keys=True)); return 2 if summary["decision"]["verdict"] == "INVALID_EXECUTION" else 0
if __name__ == "__main__": raise SystemExit(main())
