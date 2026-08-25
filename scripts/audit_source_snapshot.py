#!/usr/bin/env python3
"""Independently verify a source snapshot without importing its creator."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    failures = []
    repo = Path(snapshot["repo_root"]).resolve()
    patch = Path(snapshot["tracked_binary_patch"]["path"]).resolve()
    if not patch.is_file() or file_sha256(patch) != snapshot["tracked_binary_patch"]["sha256"]:
        failures.append("patch_hash")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != snapshot["head_commit"]:
        failures.append("head_commit")
    bad_files = []
    rows = snapshot["source_files"]
    for row in rows:
        path = (repo / row["path"]).resolve()
        try:
            path.relative_to(repo)
        except ValueError:
            bad_files.append(row["path"])
            continue
        if not path.is_file() or file_sha256(path) != row["sha256"] or path.stat().st_size != row["size_bytes"]:
            bad_files.append(row["path"])
    if bad_files:
        failures.append("source_files")
    source_tree = hashlib.sha256(
        "".join(f"{row['path']}\0{row['sha256']}\n" for row in rows).encode("utf-8")
    ).hexdigest()
    if source_tree != snapshot["source_tree_sha256"]:
        failures.append("source_tree_hash")
    audit = {
        "valid": not failures,
        "failures": failures,
        "snapshot": str(args.snapshot.resolve()),
        "snapshot_sha256": file_sha256(args.snapshot),
        "source_file_count": len(rows),
        "bad_source_files": bad_files,
        "source_tree_sha256": source_tree,
        "patch_sha256": file_sha256(patch) if patch.is_file() else None,
        "head_commit": head,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
