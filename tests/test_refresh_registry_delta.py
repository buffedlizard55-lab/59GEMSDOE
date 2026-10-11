import json
from types import SimpleNamespace

from scripts import refresh_registry_delta as registry_delta


def test_current_open_pr_heads_pins_owner_repository_main_heads(monkeypatch):
    rows = [
        {
            "number": 16,
            "baseRefName": "main",
            "headRefName": "arena/session-a",
            "headRefOid": "a" * 40,
            "headRepository": {"name": "57GEMSDOE"},
            "headRepositoryOwner": {"login": registry_delta.OWNER},
        },
        {
            "number": 17,
            "baseRefName": "main",
            "headRefName": "arena/session-b",
            "headRefOid": "b" * 40,
            "headRepository": {"name": "57GEMSDOE"},
            "headRepositoryOwner": {"login": registry_delta.OWNER},
        },
        {
            "number": 18,
            "baseRefName": "release",
            "headRefName": "arena/not-main",
            "headRefOid": "c" * 40,
            "headRepository": {"name": "57GEMSDOE"},
            "headRepositoryOwner": {"login": registry_delta.OWNER},
        },
    ]
    monkeypatch.setattr(
        registry_delta.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(rows), stderr=""
        ),
    )

    heads, errors = registry_delta.current_open_pr_heads()

    assert errors == []
    assert heads == [
        {
            "pr": 16,
            "repo": "57GEMSDOE",
            "branch": "arena/session-a",
            "commit": "a" * 40,
        },
        {
            "pr": 17,
            "repo": "57GEMSDOE",
            "branch": "arena/session-b",
            "commit": "b" * 40,
        },
    ]


def test_current_open_pr_heads_fails_closed_on_external_head(monkeypatch):
    rows = [
        {
            "number": 19,
            "baseRefName": "main",
            "headRefName": "contrib/branch",
            "headRefOid": "d" * 40,
            "headRepository": {"name": "57GEMSDOE-fork"},
            "headRepositoryOwner": {"login": "external-owner"},
        }
    ]
    monkeypatch.setattr(
        registry_delta.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(rows), stderr=""
        ),
    )

    heads, errors = registry_delta.current_open_pr_heads()

    assert heads == []
    assert errors == [
        {
            "pr": 19,
            "error": "open main-target PR head is not in the indexed owner repository",
        }
    ]


def test_open_pr_tree_scan_uses_the_pinned_head_sha(monkeypatch):
    seen = []

    def fake_gh_json(endpoint):
        seen.append(endpoint)
        return {"truncated": False, "tree": [{"path": "candidate.tif"}]}

    monkeypatch.setattr(registry_delta, "gh_json", fake_gh_json)
    heads = [
        {
            "pr": 16,
            "repo": "57GEMSDOE",
            "branch": "arena/session-a",
            "commit": "a" * 40,
        }
    ]

    trees, errors = registry_delta.open_pr_trees(heads, workers=1)

    assert errors == []
    assert trees == {16: [{"path": "candidate.tif"}]}
    assert seen == [
        f"repos/{registry_delta.OWNER}/57GEMSDOE/git/trees/{'a' * 40}?recursive=1"
    ]
