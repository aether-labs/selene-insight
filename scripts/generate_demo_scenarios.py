#!/usr/bin/env python3
"""Generate deterministic T-119-B Argus demo scenario artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from services.demo_scenarios import (
    build_demo_scenarios,
    render_scenarios_markdown,
    scenarios_to_jsonable,
    validate_demo_scenarios,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "demo_scenarios"
JSON_NAME = "t119b_argus_demo_scenarios.json"
MARKDOWN_NAME = "t119b_argus_demo_scenarios.md"


def main() -> tuple[Path, Path]:
    scenarios = build_demo_scenarios()
    validate_demo_scenarios(scenarios)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / JSON_NAME
    markdown_path = OUTPUT_DIR / MARKDOWN_NAME

    json_path.write_text(
        json.dumps(scenarios_to_jsonable(scenarios), indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_scenarios_markdown(scenarios),
        encoding="utf-8",
    )

    print(json_path)
    print(markdown_path)
    return json_path, markdown_path


if __name__ == "__main__":
    main()
