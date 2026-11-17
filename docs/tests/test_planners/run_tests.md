# `tests/test_planners/run_tests.py`

**Source file:** [tests/test_planners/run_tests.py](../../../tests/test_planners/run_tests.py)

## Overview
CLI runner for the `TestMBDPlanner` unittest suite supporting filtering by dynamics type (`--kinematic`, `--acc`), demonstration mode (`--demo`), single test (`--single`), comparison (`--compare`), full suite (`--all`), test listing (`--list`), and visualization (`--visualize`).

## Functions

### `setup_visualization()` (lines 32–42)
When global `ENABLE_VISUALIZATION` is True, mutates every `TestConfig` in `ALL_TEST_SCENARIOS` to set `visualize = True`, `render = True`, `show_animation = True`.

### `run_single_test(test_name: str, visualize: bool = False, enable_demo: Optional[bool] = None)` (lines 45–76)
Instantiates `TestMBDPlanner(test_name)`, optionally wraps `run_scenario_test` to force `visualize=True`, and runs it via `unittest.TextTestRunner(verbosity=2)`. Returns `bool` success.

### `run_default_tests(visualize: bool = False)` (lines 79–95)
Runs all non-demo scenarios from `list_default_scenarios()` (`test_{scenario}`).

### `run_demo_tests(visualize: bool = False)` (lines 98–113)
Runs all demo scenarios from `list_demo_scenarios()`.

### `run_acceleration_tests(visualize: bool = False)` (lines 116–132)
Runs all non-demo `acc_tt2d` scenarios from `list_acceleration_scenarios()`.

### `run_kinematic_tests(visualize: bool = False)` (lines 135–151)
Runs all non-demo `tt2d` scenarios from `list_kinematic_scenarios()`.

### `run_scenario_comparison(scenario_base: str, visualize: bool = False)` (lines 154–178)
Runs both `test_{scenario_base}` and `test_{scenario_base}_demo` back-to-back.

### `run_all_tests(visualize: bool = False)` (lines 181–192)
Loads and runs all test methods on `TestMBDPlanner`.

### `list_tests()` (lines 195–236)
Categorizes and logs all `test_*` methods on `TestMBDPlanner` into kinematic (default/demo), acceleration (default/demo), and utility tests.

### `main()` (lines 239–312)
Parses CLI arguments (`--all`, `--demo`, `--acc`, `--kinematic`, `--single`, `--compare`, `--list`, `--visualize`, `--timeout`), triggers `setup_visualization()`, dispatches the requested runner (defaulting to `run_kinematic_tests`), and exits with code `0` on pass or `1` on failure/error.

## Extension Notes
- **New vehicle class tests**: Add scenario configs to `fixtures/test_configs.py`, test methods to `test_mbd_planner.py`, and optionally a new CLI flag/category (e.g., `--dynamic`) here.
