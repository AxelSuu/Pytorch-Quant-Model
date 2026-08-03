"""Tests for provenance.py -- version/git-sha capture for reproducibility."""

import subprocess
from unittest.mock import patch

from pyquant import provenance


def test_git_sha_is_memoized_after_the_first_call(tmp_path, monkeypatch):
    """PYQ-279: git_sha() shells out to resolve a working tree's sha on the
    first call, but a tree's sha can't change mid-process, so a second call
    for the same working tree must not shell out again."""
    repo = tmp_path / "repo"
    (repo / "pyquant").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=a@b",
            "-c",
            "user.name=t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "init",
        ],
        cwd=repo,
        check=True,
    )
    fake_module = repo / "pyquant" / "provenance.py"
    fake_module.write_text("")
    monkeypatch.setattr(provenance, "__file__", str(fake_module))

    with patch("subprocess.run", wraps=subprocess.run) as spy:
        first = provenance.git_sha()
        calls_after_first_call = spy.call_count
        second = provenance.git_sha()

    assert first is not None
    assert first == second
    assert spy.call_count == calls_after_first_call
