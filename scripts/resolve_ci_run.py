from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen


class ResolveCiRunError(RuntimeError):
    pass


def resolve_ci_run(
    *,
    repository: str,
    commit: str,
    token: str,
) -> int:
    url = (
        "https://api.github.com/repos/"
        f"{quote(repository, safe='/')}/actions/workflows/ci.yml/runs"
        "?branch=main&event=push&status=success&per_page=100"
    )
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urlopen(request, timeout=30) as response:
        value: Any = json.loads(response.read())
    runs = value.get("workflow_runs") if isinstance(value, dict) else None
    if not isinstance(runs, list):
        raise ResolveCiRunError("GitHub workflow run response is invalid")
    matches = [
        item
        for item in runs
        if isinstance(item, dict)
        and item.get("head_sha") == commit
        and item.get("conclusion") == "success"
        and item.get("event") == "push"
    ]
    if not matches:
        raise ResolveCiRunError(f"no successful main CI run found for commit {commit}")
    return int(max(matches, key=lambda item: int(item["id"]))["id"])


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=os.getenv("GITHUB_REPOSITORY"))
    parser.add_argument("--commit", default=os.getenv("GITHUB_SHA"))
    parser.add_argument("--token", default=os.getenv("GITHUB_TOKEN"))
    options = parser.parse_args(arguments)
    if not all((options.repository, options.commit, options.token)):
        print(
            "resolve-ci-run: repository, commit and token are required",
            file=sys.stderr,
        )
        return 2
    try:
        run_id = resolve_ci_run(
            repository=options.repository,
            commit=options.commit,
            token=options.token,
        )
    except (OSError, ResolveCiRunError, ValueError, json.JSONDecodeError) as exc:
        print(f"resolve-ci-run: {exc}", file=sys.stderr)
        return 2
    print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
