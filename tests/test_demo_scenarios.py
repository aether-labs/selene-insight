import json

import pytest

from services.demo_scenarios import (
    REQUIRED_HYPOTHESIS_IDS,
    build_demo_scenarios,
    render_scenarios_markdown,
    scenarios_to_jsonable,
    validate_demo_scenarios,
)


def test_builds_exactly_three_named_demo_scenarios():
    scenarios = build_demo_scenarios()

    assert len(scenarios) == 3
    assert {scenario.scenario_id for scenario in scenarios} == {
        "high_pc_stale_data",
        "pc_dilution_small_debris",
        "active_active_secondary_screening",
    }


def test_each_scenario_has_expert_review_surface():
    scenarios = build_demo_scenarios()

    for scenario in scenarios:
        assert scenario.title
        assert scenario.summary
        assert scenario.hypotheses
        assert scenario.metrics
        assert scenario.decision_steps
        assert scenario.expert_prompts
        assert scenario.expected_posture


def test_required_t119a_hypotheses_are_covered():
    scenarios = build_demo_scenarios()
    covered = {
        hypothesis_id
        for scenario in scenarios
        for hypothesis_id in scenario.hypotheses
    }

    assert REQUIRED_HYPOTHESIS_IDS <= covered


def test_validation_rejects_duplicate_ids():
    scenarios = build_demo_scenarios()
    scenarios[1].scenario_id = scenarios[0].scenario_id

    with pytest.raises(ValueError, match="duplicate scenario_id"):
        validate_demo_scenarios(scenarios)


def test_json_contract_is_serializable_and_stable():
    scenarios = build_demo_scenarios()
    payload = scenarios_to_jsonable(scenarios)

    encoded = json.dumps(payload, indent=2, sort_keys=True)

    assert '"scenario_id": "high_pc_stale_data"' in encoded
    assert '"hypotheses": [' in encoded
    assert payload[0]["metrics"][0]["name"] == "probability_of_collision"


def test_markdown_packet_contains_titles_prompts_and_hypotheses():
    markdown = render_scenarios_markdown(build_demo_scenarios())

    assert "# T-119-B Argus Demo Scenarios" in markdown
    assert "High Pc With Stale And Unstable Data" in markdown
    assert "Low Miss Distance With Pc Dilution And Small Debris" in markdown
    assert "Active-Active CAM With Secondary Screening Constraints" in markdown
    assert "## Expert Correction Prompts" in markdown
    assert "**Hypotheses:** 1, 2, 5, 10, 16, 19" in markdown


def test_generator_writes_json_and_markdown(tmp_path, monkeypatch):
    from scripts import generate_demo_scenarios

    monkeypatch.setattr(generate_demo_scenarios, "OUTPUT_DIR", tmp_path)

    json_path, markdown_path = generate_demo_scenarios.main()

    assert json_path == tmp_path / "t119b_argus_demo_scenarios.json"
    assert markdown_path == tmp_path / "t119b_argus_demo_scenarios.md"
    assert json_path.exists()
    assert markdown_path.exists()
    assert "high_pc_stale_data" in json_path.read_text(encoding="utf-8")
    assert "Expert Correction Prompts" in markdown_path.read_text(encoding="utf-8")
