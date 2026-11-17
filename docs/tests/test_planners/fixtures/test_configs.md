# `tests/test_planners/fixtures/test_configs.py`

**Source file:** [tests/test_planners/fixtures/test_configs.py](../../../../tests/test_planners/fixtures/test_configs.py)

## Overview
Defines `TestConfig` (extending `MBDConfig`) and predefined scenario dictionaries (`TEST_SCENARIOS`, `ACC_TEST_SCENARIOS`, `DEMO_TEST_SCENARIOS`, `ACC_DEMO_TEST_SCENARIOS`, `ALL_TEST_SCENARIOS`) plus lookup and helper constructors.

---

## Class: `@dataclass class TestConfig(MBDConfig)` (lines 24–70)
Extends `MBDConfig` with test metadata (`test_name`, `description`, `expected_distance_min`, `expected_distance_max`, `timeout_seconds`, `visualize`, `min_final_distance_to_goal`), obstacle/parking overrides (`custom_circular_obstacles`, `custom_rectangular_obstacles`, `custom_parking_config`), and geometric initial/goal pose fields (`init_dx`, `init_dy`, `init_theta1`, `init_theta2`, `goal_theta1`, `goal_theta2`).

### `__post_init__(self)` (lines 62–70)
Enables `render` and `show_animation` when `self.visualize` is True, and configures Python `logging` level (`DEBUG` if `self.verbose` else `INFO`).

---

## Module-Level Scenario Dictionaries & Functions

### `_create_demo_scenarios()` (lines 244–268)
Populates `DEMO_TEST_SCENARIOS` and `ACC_DEMO_TEST_SCENARIOS` by deep-copying each entry in `TEST_SCENARIOS` and `ACC_TEST_SCENARIOS`, appending `"_demo"`, and setting `enable_demo = True`. Combined into `ALL_TEST_SCENARIOS`.

### `get_test_config(scenario_name: str) -> TestConfig` (lines 277–294)
Returns a deep copy of `ALL_TEST_SCENARIOS[scenario_name]`; raises `KeyError` if not found.

### `create_demo_variant(scenario_name: str, enable_demo: bool = False) -> TestConfig` (lines 297–325)
Looks up base scenario in `TEST_SCENARIOS` (stripping `_demo`) and returns a deep copy with `enable_demo` applied.

### `create_custom_test_config(base_scenario: str, **overrides) -> TestConfig` (lines 328–348)
Loads `get_test_config(base_scenario)` and sets attributes for each `key, value` in `overrides` (raises `ValueError` for unknown attributes).

### `list_available_scenarios() -> List[str]` (lines 351–353)
Returns all keys in `ALL_TEST_SCENARIOS`.

### `list_default_scenarios() -> List[str]` (lines 356–358)
Returns keys in `ALL_TEST_SCENARIOS` not ending with `"_demo"`.

### `list_demo_scenarios() -> List[str]` (lines 361–363)
Returns keys in `ALL_TEST_SCENARIOS` ending with `"_demo"`.

### `list_kinematic_scenarios() -> List[str]` (lines 366–368)
Returns keys starting with `"parking_"`.

### `list_acceleration_scenarios() -> List[str]` (lines 371–373)
Returns keys starting with `"acc_parking_"`.

### `get_scenario_pairs() -> List[Tuple[str, str]]` (lines 376–396)
Returns `(base_name, f"{base_name}_demo")` pairs for all entries in `TEST_SCENARIOS` and `ACC_TEST_SCENARIOS`.

## Extension Notes
- **New vehicle class**: Add a new scenario dictionary (e.g., `DYNAMIC_TEST_SCENARIOS`), auto-generate demo variants in `_create_demo_scenarios()`, merge it into `ALL_TEST_SCENARIOS`, and add a `list_<new>_scenarios()` helper if desired.
- **Perturbed tractor-trailer tests**: Use `create_custom_test_config(...)` or add perturbation fields to `MBDConfig`/`TestConfig` to run regression tests with disturbances.
