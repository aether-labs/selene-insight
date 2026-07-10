# T-119-B Task 3 Report

## Implemented

- Added `scenarios_to_jsonable(scenarios: Sequence[ArgusDemoScenario]) -> list[dict[str, object]]` in `services/demo_scenarios.py`.
- Added `render_scenarios_markdown(scenarios: Sequence[ArgusDemoScenario]) -> str` in `services/demo_scenarios.py`.
- Both functions call `validate_demo_scenarios(scenarios)` before exporting data.
- JSON conversion now uses `dataclasses.asdict()` to produce stable nested dictionaries for `ArgusDemoScenario`, `ScenarioMetric`, and `ScenarioDecisionStep`.
- Markdown rendering now emits the review packet structure from the task brief with scenario headers, summaries, IDs, hypotheses, metrics, decision steps, expected posture, and expert prompts.

## Tests

- `python3 -m pytest tests/test_demo_scenarios.py -q` failed because the environment Python did not have `pytest` installed.
- `/Users/yong/projects/substratum/argus/.venv/bin/python -m pytest tests/test_demo_scenarios.py -q` initially failed one markdown assertion on the `Hypotheses:` label.
- After adjusting the label to the test contract, `/Users/yong/projects/substratum/argus/.venv/bin/python -m pytest tests/test_demo_scenarios.py -q` passed: `6 passed, 1 warning`.

## TDD Evidence

- Read `tests/test_demo_scenarios.py` before editing.
- Implemented the two export/render functions.
- Ran the focused test file and used the failure output to refine the markdown formatting.
- Re-ran the same focused test file until it passed.

## Files Changed

- `services/demo_scenarios.py`
- `.superpowers/sdd/task-3-report.md`

## Self-Review

- The implementation is narrowly scoped to the two required functions.
- JSON output is deterministic because it preserves dataclass field order and returns plain nested Python structures.
- Markdown output is stable and newline-terminated.

## Concerns

- The focused test run emitted a pytest cache write warning because the sandbox could not write `.pytest_cache`.
- I did not change any unrelated files.

## Fix Report: Hypotheses Label

- Updated `services/demo_scenarios.py` so Scenario 1 now renders the hypotheses line with the bold label required by the brief: `**Hypotheses:** 1, 2, 5, 10, 16, 19`.
- Tightened `tests/test_demo_scenarios.py` to assert the exact bold markdown label instead of the looser unbolded substring.
- Files changed for this fix: `services/demo_scenarios.py`, `tests/test_demo_scenarios.py`, `.superpowers/sdd/task-3-report.md`.
