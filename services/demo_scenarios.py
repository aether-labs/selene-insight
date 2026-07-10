"""T-119-B deterministic Argus demo scenario definitions.

The scenarios are expert-review probes derived from public CDM/SSA workflow
hypotheses. They are not production automation rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


REQUIRED_HYPOTHESIS_IDS = {
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    23,
    24,
    27,
    29,
}


@dataclass
class ScenarioMetric:
    name: str
    value: str
    unit: str | None = None


@dataclass
class ScenarioDecisionStep:
    stage: str
    action: str
    rationale: str


@dataclass
class ArgusDemoScenario:
    scenario_id: str
    title: str
    summary: str
    hypotheses: list[int]
    metrics: list[ScenarioMetric]
    decision_steps: list[ScenarioDecisionStep]
    expert_prompts: list[str]
    expected_posture: str


def build_demo_scenarios() -> list[ArgusDemoScenario]:
    """Return the canonical T-119-B demo scenario packet."""
    return [
        ArgusDemoScenario(
            scenario_id="high_pc_stale_data",
            title="High Pc With Stale And Unstable Data",
            summary=(
                "A high-Pc conjunction approaches TCA with stale observations, "
                "unstable CDM timing, and a narrowing command upload window."
            ),
            hypotheses=[1, 2, 5, 10, 16, 19],
            metrics=[
                ScenarioMetric("probability_of_collision", "2.4e-4"),
                ScenarioMetric("time_since_last_observation_primary", "82", "hours"),
                ScenarioMetric("time_since_last_observation_secondary", "91", "hours"),
                ScenarioMetric("tca_variation_last_three_cdms", "1.8", "seconds"),
                ScenarioMetric("time_to_tca", "31", "hours"),
                ScenarioMetric("next_command_window_before_tca", "5.5", "hours"),
            ],
            decision_steps=[
                ScenarioDecisionStep(
                    "Intake",
                    "Escalate the CDM to analyst review.",
                    "Pc exceeds the public high-risk threshold.",
                ),
                ScenarioDecisionStep(
                    "Tracking",
                    "Request fresh tracking before final burn commitment.",
                    "Both objects have stale observations and unstable TCA updates.",
                ),
                ScenarioDecisionStep(
                    "Planning",
                    "Start preliminary CAM planning while waiting for updated OD.",
                    "The event is inside the 48-hour planning window.",
                ),
                ScenarioDecisionStep(
                    "Go/No-Go",
                    "Lock a decision before the final viable command pass.",
                    "The upload window may close before the next CDM resolves uncertainty.",
                ),
            ],
            expert_prompts=[
                "Would your team wait for another CDM here, or lock the command-window decision earlier?",
                "Which data source would you request first when both TSLO and TCA variation are bad?",
                "What threshold would make this transition from monitoring to maneuver execution?",
            ],
            expected_posture=(
                "Escalate, request fresh tracking, prepare a CAM, and defer final "
                "execution until tracking improves or the command window forces a choice."
            ),
        ),
        ArgusDemoScenario(
            scenario_id="pc_dilution_small_debris",
            title="Low Miss Distance With Pc Dilution And Small Debris",
            summary=(
                "A low computed Pc masks dangerous geometry because the secondary "
                "object is small debris with a large, possibly non-realistic covariance."
            ),
            hypotheses=[3, 4, 8, 13, 14, 17, 29],
            metrics=[
                ScenarioMetric("probability_of_collision", "7.5e-6"),
                ScenarioMetric("total_miss_distance", "142", "m"),
                ScenarioMetric("combined_hard_body_radius", "18", "m"),
                ScenarioMetric("relative_velocity", "14.2", "km/s"),
                ScenarioMetric("secondary_rcs", "0.04", "m^2"),
                ScenarioMetric("secondary_position_sigma_major_axis", "3.8", "km"),
                ScenarioMetric("fragmentation_event_age", "5", "days"),
            ],
            decision_steps=[
                ScenarioDecisionStep(
                    "Intake",
                    "Generate a pre-alert despite Pc being below the action threshold.",
                    "The object is inside the screening volume with hypervelocity geometry.",
                ),
                ScenarioDecisionStep(
                    "Covariance Review",
                    "Flag covariance dilution and inspect Mahalanobis proximity.",
                    "Large uncertainty can reduce Pc while preserving dangerous geometry.",
                ),
                ScenarioDecisionStep(
                    "Debris Handling",
                    "Increase safety margin and request additional tracking.",
                    "Small, recent-fragment debris has higher covariance and drag uncertainty.",
                ),
                ScenarioDecisionStep(
                    "Operator Review",
                    "Ask the analyst whether geometry should override low Pc.",
                    "The scenario tests whether Pc-only automation would suppress a real concern.",
                ),
            ],
            expert_prompts=[
                "Would a low Pc suppress this alert in your current workflow?",
                "Do you use Mahalanobis distance or another geometry check for dilution cases?",
                "How do you alter thresholds for small debris or fresh fragmentation events?",
            ],
            expected_posture=(
                "Do not dismiss the event solely because Pc is low; escalate based on "
                "geometry, debris uncertainty, and covariance realism concerns."
            ),
        ),
        ArgusDemoScenario(
            scenario_id="active_active_secondary_screening",
            title="Active-Active CAM With Secondary Screening Constraints",
            summary=(
                "Two active spacecraft can maneuver, but one proposed CAM creates "
                "a secondary conjunction and another violates thermal attitude constraints."
            ),
            hypotheses=[6, 7, 9, 18, 20, 23, 24, 27],
            metrics=[
                ScenarioMetric("probability_of_collision", "1.6e-4"),
                ScenarioMetric("time_to_tca", "22", "hours"),
                ScenarioMetric("primary_delta_v_budget_remaining", "11.4", "m/s"),
                ScenarioMetric("secondary_delta_v_budget_remaining", "42.0", "m/s"),
                ScenarioMetric("best_safe_cam_delta_v", "0.18", "m/s"),
                ScenarioMetric("candidate_secondary_conjunction_pc", "3.1e-5"),
                ScenarioMetric("post_maneuver_slot_recovery", "2.6", "days"),
            ],
            decision_steps=[
                ScenarioDecisionStep(
                    "Coordination",
                    "Contact the secondary operator before committing to a CAM.",
                    "Both objects are active and the secondary has more maneuver margin.",
                ),
                ScenarioDecisionStep(
                    "Option Screening",
                    "Reject CAM candidates that create secondary conjunctions.",
                    "Post-maneuver ephemerides must be screened before upload.",
                ),
                ScenarioDecisionStep(
                    "Constraint Check",
                    "Reject the thermal-unsafe thrust vector and search alternatives.",
                    "A maneuver that points star trackers near the sun is not operationally viable.",
                ),
                ScenarioDecisionStep(
                    "Execution",
                    "Commit only after safe option selection and final command-window review.",
                    "The event is within the 24-hour go/no-go window.",
                ),
                ScenarioDecisionStep(
                    "Recovery",
                    "Request post-maneuver OD and rescreen achieved ephemeris.",
                    "Delta-V execution error can introduce new hazards.",
                ),
            ],
            expert_prompts=[
                "Who would maneuver in this active-active case, and how is that negotiated?",
                "Would secondary conjunction screening block the proposed safe-looking CAM?",
                "Which spacecraft constraints are checked before operators accept a burn plan?",
            ],
            expected_posture=(
                "Coordinate with the secondary operator, reject unsafe or secondary-risk "
                "CAMs, select the lowest-risk feasible option, and verify the achieved orbit."
            ),
        ),
    ]


def validate_demo_scenarios(scenarios: Sequence[ArgusDemoScenario]) -> None:
    """Validate the canonical scenario packet before export."""
    if len(scenarios) != 3:
        raise ValueError(f"expected exactly 3 scenarios, got {len(scenarios)}")

    seen_ids: set[str] = set()
    covered_hypotheses: set[int] = set()
    for scenario in scenarios:
        if scenario.scenario_id in seen_ids:
            raise ValueError(f"duplicate scenario_id: {scenario.scenario_id}")
        seen_ids.add(scenario.scenario_id)

        if not scenario.title:
            raise ValueError(f"{scenario.scenario_id} missing title")
        if not scenario.summary:
            raise ValueError(f"{scenario.scenario_id} missing summary")
        if not scenario.hypotheses:
            raise ValueError(f"{scenario.scenario_id} missing hypotheses")
        if not scenario.metrics:
            raise ValueError(f"{scenario.scenario_id} missing metrics")
        if not scenario.decision_steps:
            raise ValueError(f"{scenario.scenario_id} missing decision_steps")
        if not scenario.expert_prompts:
            raise ValueError(f"{scenario.scenario_id} missing expert_prompts")
        if not scenario.expected_posture:
            raise ValueError(f"{scenario.scenario_id} missing expected_posture")

        covered_hypotheses.update(scenario.hypotheses)

    missing = REQUIRED_HYPOTHESIS_IDS - covered_hypotheses
    if missing:
        raise ValueError(f"missing required hypothesis coverage: {sorted(missing)}")


def scenarios_to_jsonable(scenarios):
    return []


def render_scenarios_markdown(scenarios):
    return ""
