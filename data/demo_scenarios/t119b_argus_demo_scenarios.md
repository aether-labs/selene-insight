# T-119-B Argus Demo Scenarios

These scenarios compress public CDM/SSA workflow hypotheses into concrete review surfaces.
They are probes for expert correction, not production automation rules.

## Scenario 1: High Pc With Stale And Unstable Data

A high-Pc conjunction approaches TCA with stale observations, unstable CDM timing, and a narrowing command upload window.

**Scenario ID:** `high_pc_stale_data`
**Hypotheses:** 1, 2, 5, 10, 16, 19

### Metrics
- `probability_of_collision`: 2.4e-4
- `time_since_last_observation_primary`: 82 hours
- `time_since_last_observation_secondary`: 91 hours
- `tca_variation_last_three_cdms`: 1.8 seconds
- `time_to_tca`: 31 hours
- `next_command_window_before_tca`: 5.5 hours

### Decision Steps
- **Intake:** Escalate the CDM to analyst review. Rationale: Pc exceeds the public high-risk threshold.
- **Tracking:** Request fresh tracking before final burn commitment. Rationale: Both objects have stale observations and unstable TCA updates.
- **Planning:** Start preliminary CAM planning while waiting for updated OD. Rationale: The event is inside the 48-hour planning window.
- **Go/No-Go:** Lock a decision before the final viable command pass. Rationale: The upload window may close before the next CDM resolves uncertainty.

### Expected Posture
Escalate, request fresh tracking, prepare a CAM, and defer final execution until tracking improves or the command window forces a choice.

## Expert Correction Prompts
- Would your team wait for another CDM here, or lock the command-window decision earlier?
- Which data source would you request first when both TSLO and TCA variation are bad?
- What threshold would make this transition from monitoring to maneuver execution?

## Scenario 2: Low Miss Distance With Pc Dilution And Small Debris

A low computed Pc masks dangerous geometry because the secondary object is small debris with a large, possibly non-realistic covariance.

**Scenario ID:** `pc_dilution_small_debris`
**Hypotheses:** 3, 4, 8, 13, 14, 17, 29

### Metrics
- `probability_of_collision`: 7.5e-6
- `total_miss_distance`: 142 m
- `combined_hard_body_radius`: 18 m
- `relative_velocity`: 14.2 km/s
- `secondary_rcs`: 0.04 m^2
- `secondary_position_sigma_major_axis`: 3.8 km
- `fragmentation_event_age`: 5 days

### Decision Steps
- **Intake:** Generate a pre-alert despite Pc being below the action threshold. Rationale: The object is inside the screening volume with hypervelocity geometry.
- **Covariance Review:** Flag covariance dilution and inspect Mahalanobis proximity. Rationale: Large uncertainty can reduce Pc while preserving dangerous geometry.
- **Debris Handling:** Increase safety margin and request additional tracking. Rationale: Small, recent-fragment debris has higher covariance and drag uncertainty.
- **Operator Review:** Ask the analyst whether geometry should override low Pc. Rationale: The scenario tests whether Pc-only automation would suppress a real concern.

### Expected Posture
Do not dismiss the event solely because Pc is low; escalate based on geometry, debris uncertainty, and covariance realism concerns.

## Expert Correction Prompts
- Would a low Pc suppress this alert in your current workflow?
- Do you use Mahalanobis distance or another geometry check for dilution cases?
- How do you alter thresholds for small debris or fresh fragmentation events?

## Scenario 3: Active-Active CAM With Secondary Screening Constraints

Two active spacecraft can maneuver, but one proposed CAM creates a secondary conjunction and another violates thermal attitude constraints.

**Scenario ID:** `active_active_secondary_screening`
**Hypotheses:** 6, 7, 9, 18, 20, 23, 24, 27

### Metrics
- `probability_of_collision`: 1.6e-4
- `time_to_tca`: 22 hours
- `primary_delta_v_budget_remaining`: 11.4 m/s
- `secondary_delta_v_budget_remaining`: 42.0 m/s
- `best_safe_cam_delta_v`: 0.18 m/s
- `candidate_secondary_conjunction_pc`: 3.1e-5
- `post_maneuver_slot_recovery`: 2.6 days

### Decision Steps
- **Coordination:** Contact the secondary operator before committing to a CAM. Rationale: Both objects are active and the secondary has more maneuver margin.
- **Option Screening:** Reject CAM candidates that create secondary conjunctions. Rationale: Post-maneuver ephemerides must be screened before upload.
- **Constraint Check:** Reject the thermal-unsafe thrust vector and search alternatives. Rationale: A maneuver that points star trackers near the sun is not operationally viable.
- **Execution:** Commit only after safe option selection and final command-window review. Rationale: The event is within the 24-hour go/no-go window.
- **Recovery:** Request post-maneuver OD and rescreen achieved ephemeris. Rationale: Delta-V execution error can introduce new hazards.

### Expected Posture
Coordinate with the secondary operator, reject unsafe or secondary-risk CAMs, select the lowest-risk feasible option, and verify the achieved orbit.

## Expert Correction Prompts
- Who would maneuver in this active-active case, and how is that negotiated?
- Would secondary conjunction screening block the proposed safe-looking CAM?
- Which spacecraft constraints are checked before operators accept a burn plan?
