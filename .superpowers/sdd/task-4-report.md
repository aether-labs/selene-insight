# Task 4 Report: Artifact Generator Script

## Status

DONE_WITH_CONCERNS

## Implemented

- Added `test_generator_writes_json_and_markdown` to `tests/test_demo_scenarios.py`.
- Created `scripts/generate_demo_scenarios.py`.
- The generator imports the canonical T-119-B scenario builders/serializers, validates the scenario packet, writes:
  - `data/demo_scenarios/t119b_argus_demo_scenarios.json`
  - `data/demo_scenarios/t119b_argus_demo_scenarios.md`
- `main() -> tuple[Path, Path]` returns the JSON and Markdown output paths and prints both paths when run as a script.

## TDD Evidence

1. RED test added first.
2. Initial command from the brief could not run because default `python3` lacks pytest:

   ```bash
   python3 -m pytest tests/test_demo_scenarios.py::test_generator_writes_json_and_markdown -q
   ```

   Result:

   ```text
   /opt/homebrew/opt/python@3.14/bin/python3.14: No module named pytest
   ```

3. Re-ran with the project venv as instructed:

   ```bash
   /Users/yong/projects/substratum/argus/.venv/bin/python -m pytest tests/test_demo_scenarios.py::test_generator_writes_json_and_markdown -q
   ```

   Result: failed for the expected reason:

   ```text
   ImportError: cannot import name 'generate_demo_scenarios' from 'scripts' (unknown location)
   ```

4. Implemented `scripts/generate_demo_scenarios.py`.
5. Verified generator test:

   ```bash
   /Users/yong/projects/substratum/argus/.venv/bin/python -m pytest -p no:cacheprovider tests/test_demo_scenarios.py::test_generator_writes_json_and_markdown -q
   ```

   Result:

   ```text
   1 passed in 0.01s
   ```

6. Verified full focused test file:

   ```bash
   /Users/yong/projects/substratum/argus/.venv/bin/python -m pytest -p no:cacheprovider tests/test_demo_scenarios.py -q
   ```

   Result:

   ```text
   7 passed in 0.01s
   ```

## Files Changed

- `tests/test_demo_scenarios.py`
- `scripts/generate_demo_scenarios.py`
- `.superpowers/sdd/task-4-report.md`

## Self-Review

- Scope stayed limited to the requested generator test, generator script, and this report.
- Did not touch the pre-existing unrelated dirty file `scripts/backup-vps.sh`.
- Generator uses the exact output directory, filenames, imports, JSON indentation, UTF-8 writes, validation call, print behavior, and return shape from the task brief.
- Test uses `tmp_path` plus `monkeypatch` so it verifies artifact writing without creating repository data artifacts.

## Concerns

- Default `python3` cannot run pytest in this checkout, so verification used `/Users/yong/projects/substratum/argus/.venv/bin/python`.
- Running pytest without `-p no:cacheprovider` in this sandbox emits `.pytest_cache` permission warnings. The passing verification commands disabled the cache provider to keep output clean.

---

## Review Fix: Direct Script Import Path

### Status

DONE_WITH_CONCERNS

### Reviewer Finding Addressed

- Updated `scripts/generate_demo_scenarios.py` to match the existing Argus direct-script import convention before importing `services.demo_scenarios`.
- Compared against local examples:
  - `scripts/generate_weekly_report.py`
  - `scripts/test_node_model.py`

### Files Changed

- `scripts/generate_demo_scenarios.py`
- `.superpowers/sdd/task-4-report.md`

### Verification Evidence

```bash
/Users/yong/projects/substratum/argus/.venv/bin/python -m pytest -p no:cacheprovider tests/test_demo_scenarios.py -q
```

Result:

```text
.......                                                                  [100%]
7 passed in 0.01s
```

```bash
/Users/yong/projects/substratum/argus/.venv/bin/python scripts/generate_demo_scenarios.py
```

Result:

```text
PermissionError: [Errno 1] Operation not permitted: '/Users/yong/projects/substratum/argus/data/demo_scenarios'
```

The direct script invocation now reaches artifact directory creation; the remaining direct-run failure is the controller sandbox write restriction on `data/demo_scenarios`.
