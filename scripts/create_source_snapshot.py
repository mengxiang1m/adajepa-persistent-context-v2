#!/usr/bin/env python3
"""Create an auditable source snapshot for a dirty experiment worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_INCLUDES = (
    "conf",
    "datasets",
    "env",
    "metrics",
    "models",
    "planning",
    "research",
    "scripts",
    "tests",
    "任务说明",
    "AGENTS.md",
    "custom_resolvers.py",
    "plan.py",
    "preprocessor.py",
    "train.py",
    "utils.py",
)
SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".sh", ".toml", ".txt"}
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", "repro_outputs"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str, text: bool = True):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=text,
    ).stdout


def resolve_inside(repo: Path, value: str) -> Path:
    candidate = (repo / value).resolve()
    try:
        candidate.relative_to(repo)
    except ValueError as exc:
        raise ValueError(f"include path escapes repository: {value}") from exc
    return candidate


def source_files(repo: Path, includes: list[str]) -> list[Path]:
    files: set[Path] = set()
    for value in includes:
        candidate = resolve_inside(repo, value)
        if not candidate.exists():
            continue
        candidates = [candidate] if candidate.is_file() else candidate.rglob("*")
        for path in candidates:
            if not path.is_file():
                continue
            relative = path.relative_to(repo)
            if any(part in EXCLUDED_PARTS for part in relative.parts):
                continue
            if path.suffix.lower() in SOURCE_SUFFIXES:
                files.add(path)
    return sorted(files, key=lambda path: path.relative_to(repo).as_posix())


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "torch", "hydra", "omegaconf"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = None
    return versions


def gpu_inventory() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",", 4)]
        if len(fields) == 5:
            rows.append(dict(zip(("index", "uuid", "name", "driver_version", "memory_total_mib"), fields)))
    return rows


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def build_snapshot(repo: Path, includes: list[str], patch_path: Path) -> dict:
    repo = repo.resolve()
    if git(repo, "rev-parse", "--is-inside-work-tree").strip() != "true":
        raise RuntimeError(f"not a git worktree: {repo}")
    patch = git(repo, "diff", "--binary", "--no-ext-diff", text=False)
    atomic_write(patch_path, patch)
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all").splitlines()
    files = source_files(repo, includes)
    file_rows = []
    for path in files:
        relative = path.relative_to(repo).as_posix()
        file_rows.append({
            "path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "tracked": bool(git(repo, "ls-files", "--error-unmatch", "--", relative).strip())
            if relative in set(git(repo, "ls-files").splitlines())
            else False,
        })
    return {
        "schema_version": 1,
        "created_unix": time.time(),
        "repo_root": str(repo),
        "head_commit": git(repo, "rev-parse", "HEAD").strip(),
        "head_tree": git(repo, "rev-parse", "HEAD^{tree}").strip(),
        "dirty": bool(status),
        "git_status_porcelain": status,
        "tracked_binary_patch": {
            "path": str(patch_path),
            "sha256": sha256_bytes(patch),
            "size_bytes": len(patch),
        },
        "include_roots": includes,
        "source_files": file_rows,
        "source_tree_sha256": sha256_bytes(
            "".join(f"{row['path']}\0{row['sha256']}\n" for row in file_rows).encode("utf-8")
        ),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "gpus": gpu_inventory(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-output", type=Path)
    parser.add_argument("--include", action="append", default=[])
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    patch_output = (args.patch_output or output.with_suffix(".patch")).resolve()
    includes = args.include or list(DEFAULT_INCLUDES)
    snapshot = build_snapshot(repo, includes, patch_output)
    atomic_write(output, (json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({
        "output": str(output),
        "patch_output": str(patch_output),
        "dirty": snapshot["dirty"],
        "source_file_count": len(snapshot["source_files"]),
        "source_tree_sha256": snapshot["source_tree_sha256"],
        "patch_sha256": snapshot["tracked_binary_patch"]["sha256"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
