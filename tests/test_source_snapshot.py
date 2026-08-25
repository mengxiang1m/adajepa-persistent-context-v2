import json
import subprocess
from pathlib import Path

from scripts.create_source_snapshot import build_snapshot, sha256_file


def run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_snapshot_captures_tracked_patch_and_untracked_source(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init")
    run_git(repo, "config", "user.email", "test@example.com")
    run_git(repo, "config", "user.name", "Test")
    tracked = repo / "tracked.py"
    tracked.write_text("VALUE = 1\n", encoding="utf-8")
    run_git(repo, "add", "tracked.py")
    run_git(repo, "commit", "-m", "initial")
    tracked.write_text("VALUE = 2\n", encoding="utf-8")
    untracked = repo / "new.yaml"
    untracked.write_text("enabled: true\n", encoding="utf-8")

    patch_path = repo / "snapshot.patch"
    snapshot = build_snapshot(repo, ["tracked.py", "new.yaml"], patch_path)

    assert snapshot["dirty"] is True
    assert snapshot["tracked_binary_patch"]["size_bytes"] > 0
    assert snapshot["tracked_binary_patch"]["sha256"] == sha256_file(patch_path)
    rows = {row["path"]: row for row in snapshot["source_files"]}
    assert rows["tracked.py"]["tracked"] is True
    assert rows["new.yaml"]["tracked"] is False
    json.dumps(snapshot)


def test_snapshot_rejects_include_outside_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init")
    run_git(repo, "config", "user.email", "test@example.com")
    run_git(repo, "config", "user.name", "Test")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    run_git(repo, "add", "a.py")
    run_git(repo, "commit", "-m", "initial")
    try:
        build_snapshot(repo, ["../outside.py"], repo / "snapshot.patch")
    except ValueError as exc:
        assert "escapes repository" in str(exc)
    else:
        raise AssertionError("outside include was accepted")


def test_source_tree_hash_is_path_and_content_bound(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init")
    run_git(repo, "config", "user.email", "test@example.com")
    run_git(repo, "config", "user.name", "Test")
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    run_git(repo, "add", "a.py")
    run_git(repo, "commit", "-m", "initial")
    snapshot = build_snapshot(repo, ["a.py"], repo / "snapshot.patch")
    original = snapshot["source_tree_sha256"]
    snapshot["source_files"][0]["path"] = "renamed.py"
    import hashlib

    changed = hashlib.sha256(
        "".join(f"{row['path']}\0{row['sha256']}\n" for row in snapshot["source_files"]).encode("utf-8")
    ).hexdigest()
    assert changed != original
