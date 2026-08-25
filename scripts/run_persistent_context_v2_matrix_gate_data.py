#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.persistent_context_v2.pushobj_matrix_gate_data import main


if __name__ == "__main__":
    main()
