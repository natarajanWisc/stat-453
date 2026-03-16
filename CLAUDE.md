# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RECAST Constraint Evaluation Framework for STAT 453. Evaluates zero-shot constraint-following performance of small LLMs (1B-9B params) on the RECAST-30K dataset across 4 difficulty levels (L1-L4).

## Repository Info

- **Remote:** https://github.com/natarajanWisc/stat-453.git
- **Main branch:** main
- **Git LFS:** enabled

## Shell Notes

When running bash commands, avoid chaining with `&&` — run commands separately instead. Flags with dashes can also cause issues; prefer separate invocations over complex one-liners.

## Git Workflow

- Never commit directly to `main`. Always work on a feature branch.
- Create a new branch for each feature/task: `git checkout -b feature/description`
- Commit frequently — small, logical commits as you go.
- Push once at the end of every user prompt (after all commits for that prompt are done).
- When the feature is complete, open a PR to `main` via `gh pr create`.
- Do not merge PRs without the user asking.

## Architecture

- `recast_eval.ipynb` — Main evaluation notebook (runs on Google Colab T4). Sections 0-7 cover setup, model loading, dataset parsing, inference, constraint evaluation, metrics, visualization, and team merge.
- `constraint_checker.py` — Standalone module with `ConstraintChecker` class. Dispatch-dict maps 18 constraint types to checker methods. `check_all()` returns per-constraint and hard CSR.
- `viz_utils.py` — Three plotting functions: CSR degradation curve, per-type bar chart, constraint distribution histogram. Uses matplotlib/seaborn.

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
