# `tests/test_perturbations.py`

**Source file:** [tests/test_perturbations.py](../../tests/test_perturbations.py)

## Overview
Standalone testing script that loads an already-generated nominal action sequence (`final_actions.npy`) from any results directory (e.g. `results/acc_tt2d`, `results/tt2d`, `results/kinematic_bicycle2d`, `results/n_trailer2d`) and rolls it out under any perturbation mode (Levels 0, 1, 2, 3) **without re-running the diffusion planner**.

Saves both `nominal_trajectory_states.npy` and `perturbed_trajectory_states.npy` (plus `perturbed_actions.npy` and `env_metadata.json`), prints step-noise statistics, parameter uncertainty values, trajectory deviation, and collision/jackknife diagnostics, and optionally launches the standalone visualizer (`render.visualize_results`).

---

## Functions

### `check_trajectory_safety(env, states_np, use_perturbed_geom=False)`
Iterates through `states_np` and checks `env.check_obstacle_collision(..., use_perturbed=use_perturbed_geom)` and `env.check_hitch_violation(...)`, returning a dict with `collision`, `collision_steps`, `hitch_violation`, and `hitch_steps`.

### `perturb_existing_trajectory(...)`
1. Loads `final_actions.npy` (and `nominal_trajectory_states.npy` / `trajectory_states.npy`) from `results_dir`.
2. Reconstructs the environment and obstacle configuration via `load_env_from_results(results_path)`.
3. Configures the requested perturbation mode and noise parameters via `env.set_perturbation(...)`.
4. Rolls out `nominal_actions` through `jax.jit(env.step_perturbed)` using `rollout_perturbed_us`.
5. Saves `nominal_trajectory_states.npy`, `perturbed_trajectory_states.npy`, `trajectory_states.npy`, `perturbed_actions.npy`, and `env_metadata.json`.
6. Prints a diagnostic summary and optionally calls `render.visualize_results(results_path, animate=True)`.

### `main()`
Provides default perturbation parameters at the top of `main()` and CLI flags (`--level`, `--seed`, `--control-noise-std`, `--param-uncertainty-std`, `--dist`, `--df`, `--state-disturbance-scale`, `--perturbed-l1`, `--perturbed-l2`, `--perturbed-lh`, `--no-visualize`).
