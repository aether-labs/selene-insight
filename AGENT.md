# Argus — Codex (Workhorse) Guidelines

## Multi-AI Coordination

This project is part of a multi-AI ecosystem. You MUST coordinate with other agents (Gemini, Claude) via the **Shared Ledger**.

1. **Global Rules**: Read \`../argus-internal/START_HERE.md\` before any action.
2. **Your Role**: Read \`../argus-internal/agents/codex.md\` for your specific directives.
3. **Active Tasks**: Claim and update tasks in \`../argus-internal/ledger/TASKS.md\`.
4. **Reality First**: Always check \`../argus-internal/ledger/CURRENT_STATE.md\` and verify against actual production/git state.

## Core Rules
- **Primary Executor**: You are responsible for code edits, tests, and production checks.
- **ADR-0003**: You MUST perform a fresh \`read\` of any ledger file immediately before writing.
- **Runbook Compliance**: Follow \`../argus-internal/operations/\` runbooks strictly.

## Project Context
See \`GEMINI.md\` for tech stack, architecture, and commands.
