# `mbd/robust/rmps_runner.py`

**Source file:** [mbd/robust/rmps_runner.py](../../../mbd/robust/rmps_runner.py)

## Overview
Implements **Stage 4: Online Closed-Loop RMPS Execution & Replanning Runner** (RMPS Algorithm 1) for `AccTractorTrailer2d` under Perturbation Levels 0–3, together with standard `results/` export (compatible with [render.py](../../../render.py)) and offscreen `FigureCanvasAgg` diagnostic plotting (`rmps_diagnostics.png`).

---

## Functions

### `compute_min_obstacle_clearance_np(env, state_np, use_perturbed_geom=False) -> float` (lines 28–92)
Samples points along the tractor and trailer rectangle edges and computes the minimum signed Euclidean distance (in meters, negative = penetration) to all active circular and oriented rectangular obstacles.

### `run_rmps_episode(...) -> dict[str, Any]` (lines 128–399)
Executes a closed-loop episode against `jax.jit(env.step_perturbed)`:
1. **Initial Plan**: Plans `(x_plan, u_plan)` via `planner.plan(x0, x_goal)` (or reuses a paired `initial_plan` for controlled benchmarks).
2. **RMPS Switching (`_decide_rmps_control`)**:
   - When `enable_nmpc_shield=False`: executes `u_plan[plan_step]` open-loop (and brakes afterward).
   - When `enable_nmpc_shield=True`: checks whether the one-step nominal prediction `x_after_mpd = f_nom(x_eval, u_mpd)` satisfies a close-tracking clearance fast-path or is certified recoverable by `nmpc_backup.is_recoverable(x_after_mpd, r_tube, b_hitch)`. If not recoverable, triggers/continues `nmpc_backup` (`initialize_backup` + `step`).
   - Supports **1-step delay compensation** (`use_delay_compensation=True`), evaluating `x_eval = f_nom(x_prev, u_apply)` with 1-step shifted tubes (`_shift_tube_for_delay`).
3. **Stopped-Equilibrium Replanning**: When `enable_replanning=True`, an active backup stops the vehicle (`|v| <= v_replan_tol`), and `replan_count < max_replans`, triggers `planner.plan(x_real, x_goal)` from the stopped state and resets backup state.
4. **Metrics & Return Dict**: Returns `states`, `actions`, `nominal_plans`, `shield_interventions`, `replan_steps`, `collision`, `hitch_violation`, `goal_reached`, `final_pos_error`, `min_clearance`, `clearances`, `computation_times`, `initial_plan_time`, `intervention_rate`, `num_replans`, and `perturbation_level`. Optionally calls `save_rmps_results` when `save_dir` is set.

### `save_rmps_results(env, config, result, initial_u_plan, tube_data, save_dir, save_plot=True) -> Path` (lines 402–430)
Exports `nominal_trajectory_states.npy`, `trajectory_states.npy`, `perturbed_trajectory_states.npy`, `final_actions.npy`, `perturbed_actions.npy`, `env_metadata.json`, and `rmps_diagnostics.png` (without altering Matplotlib's global GUI backend).

### `plot_rmps_diagnostics(env, result, tube_data, save_path) -> Path` (lines 433–543)
Renders a multi-panel diagnostic figure offscreen (`Figure` + `FigureCanvasAgg`) showing the obstacle map, nominal plan(s), color-coded Safe-MPD vs. NMPC Shield segments, replanning markers, velocity $v(t)$, hitch angle with nominal & tube-tightened limits, and obstacle clearance over time.
