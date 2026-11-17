# `tests/test_planners/test_mbd_planner.py`

**Source file:** [tests/test_planners/test_mbd_planner.py](../../../tests/test_planners/test_mbd_planner.py)

## Overview
Defines the `TestMBDPlanner(BaseMBDTest)` unittest test suite covering kinematic (`tt2d`) and acceleration (`acc_tt2d`) parking scenarios (with and without demonstration trajectories) plus fixture utility tests.

---

## Class: `TestMBDPlanner(BaseMBDTest)` (lines 38–220)

### `run_scenario_test(self, scenario_name: str, enable_demo: Optional[bool] = None, visualize: Optional[bool] = None)` (lines 41–75)
Loads a `TestConfig` via `get_test_config(scenario_name)` (or `create_demo_variant` if `enable_demo` is explicitly passed), optionally overrides `visualize` / `render` / `show_animation`, runs `self.run_mbd_test(config)`, and returns `(distance, actions, states, timing)`.

### Kinematic Default Tests (lines 78–96)
- `test_parking_basic_forward`
- `test_parking_basic_backward`
- `test_parking_no_preference`
- `test_parking_enforce_forward`
- `test_parking_enforce_backward`

### Kinematic Demo Tests (lines 99–117)
- `test_parking_basic_forward_demo`
- `test_parking_basic_backward_demo`
- `test_parking_no_preference_demo`
- `test_parking_enforce_forward_demo`
- `test_parking_enforce_backward_demo`

### Acceleration Default Tests (lines 120–138)
- `test_acc_parking_basic_forward`
- `test_acc_parking_basic_backward`
- `test_acc_parking_no_preference`
- `test_acc_parking_enforce_forward` (note: commented out in `ACC_TEST_SCENARIOS`, so calling it raises `KeyError` unless uncommented)
- `test_acc_parking_enforce_backward` (same as above)

### Acceleration Demo Tests (lines 141–159)
- `test_acc_parking_basic_forward_demo`
- `test_acc_parking_basic_backward_demo`
- `test_acc_parking_no_preference_demo`
- `test_acc_parking_enforce_forward_demo` (see note above)
- `test_acc_parking_enforce_backward_demo` (see note above)

### Utility Tests (lines 162–219)
- `test_demo_flag_override(self, visualize=None)`: Verifies `enable_demo=True` and `enable_demo=False` overrides both succeed and yield non-negative distances.
- `test_scenario_list_functions(self, visualize=None)`: Verifies `list_default_scenarios`, `list_demo_scenarios`, `list_kinematic_scenarios`, `list_acceleration_scenarios`, and `get_scenario_pairs` return expected counts and naming patterns (note: asserts `len(default_scenarios) == 10`, though 2 `acc_parking_enforce_*` scenarios are currently commented out in `test_configs.py`).

## Extension Notes
- Add new `test_<new_vehicle>_...` methods here after defining corresponding scenarios in `fixtures/test_configs.py`.
