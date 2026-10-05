#!/usr/bin/env bash
# Rebase the current branch onto upstream main and run the tests.
#
#   scripts/sync-upstream.sh
#
# Stops at a conflict, for you to resolve, then `git rebase --continue`.
# Never pushes: the rebase rewrites history, so that stays a decision.
set -euo pipefail

cd "$(dirname "$0")/.."
git fetch origin --tags --force   # edge moves
latest=$(git tag --sort=-v:refname | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' | head -1)
behind=$(git rev-list --count HEAD..origin/main)
echo "upstream $latest, $behind commits behind origin/main"
[[ "$behind" -gt 0 ]] || exit 0
git diff --quiet HEAD || { echo "!! uncommitted changes, commit or stash first" >&2; exit 1; }
git rebase origin/main
.venv/bin/python -m pytest tests -q
echo "synced. Push with: git push --force-with-lease personal $(git branch --show-current)"
