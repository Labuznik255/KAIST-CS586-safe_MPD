# `mbd/robust/rmps_runner.py`

**Source file:** [mbd/robust/rmps_runner.py](../../../mbd/robust/rmps_runner.py)

## Overview
Implements **Stage 4: Online Closed-Loop RMPS Execution & Replanning Runner** (RMPS Algorithm 1) for `AccTractorTrailer2d` under Perturbation Levels 0–3, together with closed-loop TV-LQR trajectory tracking, buffer-escape margin adaptation, commitment hysteresis, standard `results/` export (compatible with [render.py](../../../render.py)), and offscreen `FigureCanvasAgg` diagnostic plotting (`rmps_diagnostics.png`).

---

## Functions

### `compute_min_obstacle_clearance_np(env, state_np, use_perturbed_geom=False, obs_c=None, obs_r=None) -> float`
Samples points along the tractor and trailer rectangle edges and computes the minimum signed Euclidean distance (in meters, negative = penetration) to all active circular and oriented rectangular obstacles. Accepts pre-converted NumPy `(obs_c, obs_r)` arrays to avoid per-step JAX-to-NumPy device syncs.

### `_prepare_online_tubes(env, tube_data, T_bak, pert_level, use_delay_compensation) -> tuple[np.ndarray, np.ndarray]`
Builds length-`(T_bak + 1)` online NMPC tube schedules `(r_eval, b_eval)`:
- Keeps `k = 0` at `0.0` (the current state `x_real` is already measured).
- Optionally shifts future margins (`k >= 1`) by 1 step for delay compensation (`_shift_tube_for_delay`).
- Adds short-horizon safety floors for Level 2 (geometric link-length cushion $\ge 0.12\,\text{m}$) and Level 3 (heavy-tailed impulse floor $\ge 0.08\,\text{m}$), and caps hitch tightening at 30% of `env.phi_max`.

### `_compute_tvlqr_gains_np(env, x_ref, u_ref) -> np.ndarray`
Fast pure-NumPy finite-horizon TV-LQR gain synthesizer (`< 0.5 ms`, zero JAX compilation) using exact analytic Jacobians of `AccTractorTrailer2d`'s Euler dynamics and the same `Q/R/Q_terminal` weights used in offline tube calibration ([mbd/robust/tube_calibration.py](tube_calibration.md)).

### `run_rmps_episode(...) -> dict[str, Any]`
Executes one closed-loop episode against `jax.jit(env.step_perturbed)`. Key arguments: `max_steps=150`, `max_replans=5`, `enable_replanning=True`, `enable_nmpc_shield=True`, `enable_tube_tracking=True`, `use_delay_compensation=True`, `replan_cooldown_steps=8`, `pos_tol=1.5`, `ang_tol=0.75`, `v_tol=0.5`, `episode_seed`, `initial_plan`, `save_dir`, `save_plot`.

Execution flow:
1. **Initial Plan & TV-LQR Gains**: Plans `(x_plan, u_plan)` via `planner.plan(x0, x_goal)` (or reuses a paired `initial_plan`) and synthesizes `K_plan = _compute_tvlqr_gains_np(env, x_plan, u_plan)` when `enable_nmpc_shield` and `enable_tube_tracking` are enabled. (Unshielded `baseline` and `tube_only` remain strictly open-loop.)
2. **RMPS Switching (`_decide_rmps_control`, always evaluated from true measured `x_real`)**:
   - **Commitment Hysteresis**: Once `active_backup` is engaged, commits to bringing the vehicle safely to a stopped equilibrium inside $G_e$ via `nmpc_backup.step(x_real)` rather than aborting mid-brake and oscillating on the safety boundary.
   - **Closed-Loop Nominal Proposal**: Proposes `u_mpd = clip(u_plan[plan_step] + K_plan[plan_step] @ wrap(x_real - x_plan[plan_step]), -1, 1)` (when deviation is bounded), testing recoverability via `_try_accept_mpd_action` (with fallback check on open-loop `u_plan[plan_step]` before intervening).
   - **Buffer-Escape Margin Adaptation (`_adapt_margins_for_buffer_escape`)**: If a real-world perturbation has already pushed `x_real` inside the nominal safety buffer (`clr_curr < r_eval[1]`), allows candidate escape actions **only** when they maintain/improve safety (`clr_after >= max(0.015, clr_curr - 0.005)`) and remain recoverable under locally capped margins, preventing $v=0$ replan deadlocks.
   - **Intervention**: If unrecoverable (or plan horizon exhausted), invokes `nmpc_backup.initialize_backup(x_real, r_tube, b_hitch)` and sets `active_backup = True`.
3. **Stopped-Equilibrium Replanning (`_plan_from_stopped_state`)**:
   - When `enable_replanning=True`, `active_backup=True`, and `|v| <= v_replan_tol`, enforces `max_replans`, `replan_cooldown_steps`, and anti-chatter stall detection (`stalled_replans >= 2` within `0.35 m`), tracking skipped triggers in `replan_skip_counts`.
   - Only relaxes `env` planning margins if `x_real` strictly violates step-0 tightened margins (`clr_0 < r0 + 0.005`), avoiding unnecessary XLA JIT cache flushes; calls `planner.plan(x_real, x_goal)` (retrying once with a fresh seed if displacement `< 0.25 m`), recomputes `K_plan`, and resets `active_backup = False`.
4. **Metrics & Return Dict**: Returns `states`, `actions`, `nominal_plans`, `shield_interventions`, `replan_steps`, `collision`, `hitch_violation`, `goal_reached`, `termination_reason` (`"goal"`, `"timeout"`, `"max_replans"`, `"stopped_no_replan"`), `final_pos_error`, `min_clearance`, `clearances`, `computation_times`, `initial_plan_time`, `intervention_rate`, `num_replans`, `replan_skip_counts`, and `perturbation_level`.

### `save_rmps_results(...) -> Path` & `plot_rmps_diagnostics(...) -> Path`
Exports `nominal_trajectory_states.npy`, `trajectory_states.npy`, `perturbed_trajectory_states.npy`, `final_actions.npy`, `perturbed_actions.npy`, `env_metadata.json`, and offscreen `rmps_diagnostics.png` (obstacle map, nominal plans, color-coded Safe-MPD vs. NMPC segments, replan stars, $v(t)$, hitch angle with limits, and clearance over time).
