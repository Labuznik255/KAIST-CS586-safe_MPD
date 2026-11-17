# `render.py`

**Source file:** [render.py](../render.py)

## Overview
Standalone visualizer that can be run on **any** already-generated trajectory directory (`results/acc_tt2d`, `results/tt2d`, `results/kinematic_bicycle2d`, `results/n_trailer2d`) or `.npz` test result archive. Reconstructs the environment (obstacles, parking layout, start/goal poses, vehicle geometry) via `load_env_from_results` and displays:
1. A static two-panel figure showing the full environment map with **Nominal** and **Perturbed** trajectories + heading quivers (left) and **Nominal** vs. **Perturbed** control inputs over the horizon (right), saving `rollout.png` and `rollout.svg`.
2. Full rigid-body vehicle animations (`create_animation`) for the nominal trajectory and/or perturbed trajectory (with the nominal trajectory overlaid as reference).

---

## Functions

### `_load_trajectories_from_target(target_path: Path)` (lines 28–60)
Loads `(results_dir, nominal_states, perturbed_states, actions, perturbed_actions, mu_0ts)` from either:
- A `.npz` archive (`nominal_states` / `states`, `perturbed_states`, `actions`, `perturbed_actions`, `mu_0ts`), or
- A results directory containing `nominal_trajectory_states.npy` (falling back to `trajectory_states.npy`), `perturbed_trajectory_states.npy`, `final_actions.npy`, `perturbed_actions.npy`, and `mu_0ts.npy`.

### `visualize_results(target_path="results/acc_tt2d", mode="both", animate=True, save_animation=False, show_static=True)` (lines 63–225)
- Calls `_load_trajectories_from_target(target_path)` and `load_env_from_results(results_dir)`.
- When `show_static=True`: creates a 1×2 figure, renders the environment and trajectories (`mode` in `{"both", "nominal", "perturbed"}`) with start/goal markers and heading quiver arrows on `axes[0]`, plots `actions` and `perturbed_actions` on `axes[1]`, saves `rollout.png` and `rollout.svg`, and displays the plot.
- When `animate=True` or `save_animation=True`: runs `create_animation` for the nominal trajectory (`nominal_trajectory_animation.mp4`) and/or the perturbed trajectory (`perturbed_trajectory_animation.mp4`) with the nominal path overlaid.

### `main()` (lines 228–253)
Parses CLI arguments:
- `target` (positional, default `"results/acc_tt2d"` or `"."`)
- `--mode {both,nominal,perturbed}` (default `"both"`)
- `--no-animate` (skip rigid-body animation)
- `--save-animation` (export MP4 video)
- `--no-static` (skip static figure and go straight to animation)
