# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RECAST Constraint Evaluation Framework for STAT 453. Evaluates zero-shot constraint-following performance of small LLMs (1B-9B params) on the RECAST-30K dataset across 4 difficulty levels (L1-L4).

## Repository Info

- **Remote:** https://github.com/natarajanWisc/stat-453.git
- **Main branch:** main
- **Git LFS:** enabled

## Shell Notes

- **NEVER chain commands with `&&`** — this gets blocked by settings.json permission rules. Always run commands as separate Bash invocations.
- Flags with dashes can also cause issues; prefer separate invocations over complex one-liners.
- Use `python3` not `python` (macOS).

## Permissions & Settings

- Claude has permission to modify `.claude/settings.local.json` to grant itself tool and command permissions needed for the current task.
- Destructive git commands (`push --force`, `reset --hard`, `clean -f`, `branch -D`, `rm -rf`) are explicitly denied — never attempt these without user confirmation and a settings change.
- If a Bash command gets blocked, check if it needs to be added to the allow list in `.claude/settings.local.json` before retrying.

## Guardrails

- **Never add yourself as a contributor or co-author.** No `Co-Authored-By` lines, no attribution to Claude in commits.
- **Never commit to `main`.** All work happens on feature branches.
- **Never push without being asked.** The user controls when code goes to remote.
- **Never merge PRs** without explicit user request.
- **Never delete files, branches, or data** without explicit user confirmation.
- **Never run destructive git operations** (`--force`, `--hard`, `-D`, `clean -f`).
- **Never commit secrets** (`.env`, tokens, credentials). Warn if the user asks.
- **Never modify code you haven't read first.** Always `Read` before `Edit`.
- **Test before committing.** Run `python3 -m pytest` on any modified Python module before staging.
- **Scope your changes.** Only touch what the user asked for — no drive-by refactors, extra docstrings, or speculative features.
- **Separate Bash calls.** Never chain with `&&`. One command per Bash invocation.

## Git Workflow

- Never commit directly to `main`. Always work on a feature branch.
- Create a new branch for each feature/task: `git checkout -b feature/description`
- Commit frequently — small, logical commits as you go.
- Push once at the end of every user prompt (after all commits for that prompt are done).
- When the feature is complete, open a PR to `main` via `gh pr create`.
- Do not merge PRs without the user asking.

## Architecture

- `baseline_testing/constraint_checker.py` — Standalone module with `ConstraintChecker` class. Dispatch-dict maps 18 constraint types to checker methods. `check_all()` returns per-constraint and hard CSR.
- `baseline_testing/viz_utils.py` — Three plotting functions: CSR degradation curve, per-type bar chart, constraint distribution histogram. Uses matplotlib/seaborn.
- `baseline_testing/judge.py` — LLM judge (Gemma-2-9B-IT) for qualitative constraints. Not yet committed; pending.
- `baseline_testing/recast_eval.ipynb` — Main evaluation notebook (runs on Google Colab T4). Sections 0-8 cover setup, model loading, dataset parsing, inference, constraint evaluation, metrics, visualization, team merge, and optional LLM judge.
- `baseline_testing/test_constraint_checker.py` — 47 pytest tests covering all 18 constraint types + integration.

## Testing

- Run tests: `python3 -m pytest baseline_testing/test_constraint_checker.py -v`
- Tests are local-only (no GPU, no model downloads).
- All constraint checker methods must have test coverage before committing changes.

## Key Metrics

- **Per-constraint CSR:** fraction of checkable constraints passed per instance, averaged
- **Hard CSR:** fraction of instances where all checkable constraints passed
- Both metrics computed per difficulty level (L1-L4) and overall

## Skills

Downloaded skill libraries live in `.claude-skills/` (gitignored). Two collections are available:
- `everything-claude-code/` — Agents, skills, commands, hooks, rules, MCP configs
- `awesome-claude-skills/` — Composio-based skills for document processing, app automation, etc.

Reference these when you need patterns, slash commands, or agent definitions beyond what's built in.

## Self-Improvement Protocol

After every user prompt, before finishing your response, do the following:

1. **Reflect:** Consider what you learned during this interaction — new codebase facts, gotchas, conventions, architectural decisions, or workflow patterns.
2. **Filter:** Ask yourself: *"Would every coding agent working in this codebase need this info?"* Only proceed if yes.
3. **Update CLAUDE.md:** Append or edit the relevant section of this file with the new knowledge. Keep entries concise (one bullet or short paragraph). Prefer updating existing sections over creating new ones.
4. **What qualifies:** Build/run commands, environment quirks, non-obvious conventions, architectural invariants, data pipeline details, dependency notes, common failure modes and fixes.
5. **What does NOT qualify:** User-specific preferences (those go in auto-memory), transient debugging details, things already obvious from reading the code, or one-off task context.

## Learnings

- `_check_all_caps_count` previously had a bug where it ignored the `relation` field and hardcoded `<=`. Fixed to use `_compare()`. Any new constraint checker method must use `_compare()` for relation handling.
- Files in `baseline_testing/` are split into deterministic (committed) and GPU-dependent (uncommitted: `judge.py`, `recast_eval.ipynb`). Only deterministic code gets pushed without explicit user approval.
- The notebook's fallback `parse_constraints_from_prompt()` sets `type: "parsed_from_prompt"` which is not in the dispatch dict — these constraints are effectively skipped. Needs fixing if fallback parsing is ever relied upon.
