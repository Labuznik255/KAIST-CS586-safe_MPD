# `mbd/robust/nmpc_backup.py`

**Source file:** [mbd/robust/nmpc_backup.py](../../../mbd/robust/nmpc_backup.py)

## Overview
Implements **Stage 3 (Online CPU Shield): `RobustNMPCBackup`**, a Tube-Based Recovery and Tracking Nonlinear Model Predictive Control backup controller (RMPS Algorithms 2 & 3) for `AccTractorTrailer2d`. Built purely in Python / NumPy + CasADi (`casadi.Opti()` + `ipopt`) on CPU, strictly isolated from JAX tracing.

Key features:
- **Lazy-compiled reusable parametric CasADi Opti bundles** (`_recovery_bundle`, `_tracking_bundle`, `expand=True`): built only if IPOPT is genuinely needed after fast-path braking checks.
- **Cached NumPy obstacle arrays** (`_get_active_obstacles`, `refresh_obstacles()`): avoids repeated JAX-to-NumPy device syncs during per-step SAT checks.
- **Kinematic stopping-time tube freezing** (`_effective_stopping_schedules`): freezes tube growth past $k_{\text{stop}}(v_0)$ so slow parking maneuvers are not rejected by multi-second cruise tubes while stationary in $G_e$.
- **Exact smooth vehicle-rectangle vs. obstacle distance constraints** + exact CPU SAT & hitch verification on every candidate trajectory.
- **Sub-millisecond braking fast-path + LRU solution caching**: tests `("straighten", 0.0, 0.55, -0.55)` braking rollouts before IPOPT, caches `(feasible, x_sol, u_sol)` across `is_recoverable` and `initialize_backup`, and uses a close-tracking fast-path in `step(x_curr)`.

---

## Utilities & Class

### `_wrap_angle_np(angle)` & `_format_tube_schedule(val, length)`
Wraps angles onto $(-\pi, \pi]$ and normalizes scalar or 1D array tube margins into non-negative length-`length` NumPy schedules.

### `class RobustNMPCBackup`
- **`__init__(self, env, T_bak=12, v_eps=0.05, max_iter=30, disc_safety_pad=0.01, hitch_safety_pad=0.01, cache_size=128)`**: Extracts nominal `AccTractorTrailer2d` dimensions/limits from `env`, initializes backup state (`x_ref`, `u_ref`, `r_tube_active`, `b_hitch_active`, `t_step`, `active`), LRU `_solution_cache`, obstacle caches, and lazy solver bundles.
- **`step_nominal_np(self, x, u_norm) -> np.ndarray`**: Exact CPU NumPy Forward-Euler nominal step matching `AccTractorTrailer2d.tractor_trailer_dynamics`.
- **`refresh_obstacles(self)` & `_get_active_obstacles(self)`**: Returns cached NumPy `(obs_c, obs_r)` arrays, auto-invalidating whenever `env.obs_circles` or `env.obs_rectangles` object identity changes.
- **`check_state_safety_np(self, x, r_margin=0.0, b_margin=0.0) -> tuple[bool, bool]`**: Exact CPU SAT collision & hitch checker with optional tightened margins.
- **`_build_opti_problem(self, obs_c, obs_r, tracking_mode: bool)`**:
  - **`tracking_mode=False` (Recovery NLP, RMPS Algorithm 2)**: Finds `u[0..T_bak-1]` driving `x` safely to $|v[T_{\text{bak}}]| \le v_\varepsilon$ while strongly penalizing speed increases (`120.0 * smax0(v^2 - v0^2)`) and hitch articulation (`12.0 * (1 - cos(phi))`) under hard tightened obstacle and hitch constraints.
  - **`tracking_mode=True` (Tracking NMPC, RMPS Algorithm 3)**: Tracks `(x_ref[t_step:], u_ref[t_step:])` with hitch-straightening weights and soft slack `S_obs >= 0` so perturbed states inside the tube stay feasible.
- **`_rollout_braking_guess(self, x0, steer_bias=0.0)`**: Generates deadbeat braking trajectories with `"straighten"` (hitch-stabilizing steering in reverse, centered steering forward), straight (`0.0`), or biased steering.
- **`_effective_stopping_schedules(self, x0, r_sched, b_sched)`**: Caps tube growth at $k_{\text{stop}}(v_0) = \text{clip}(\lceil |v_0| / (a_{\max} \Delta t) \rceil + 1, 2, T_{\text{bak}})$ with `k = 0` kept at `0.0`.
- **`is_recoverable(self, x_cand, r_tube=0.0, b_hitch=0.0, return_solution=False)`**: Checks $k=0$ physical safety -> 4 NumPy braking fast-paths -> lazy-compiled IPOPT (`("straighten", 0.65, -0.65)` warm-starts, retrying `pad_extra=0.14` only if a converged IPOPT solution clipped an exact SAT corner) and caches the verified trajectory.
- **`initialize_backup(self, x_init, r_tube=0.0, b_hitch=0.0) -> tuple[bool, np.ndarray, np.ndarray]`**: Initializes `(x_ref, u_ref, t_step=0, active=True)` with progressive margin fallback (`1.0 -> 0.5 -> 0.0`) so emergency stops remain feasible even from tight perturbed states.
- **`stabilizing_brake_policy(x_curr)`**, **`in_equilibrium_set(x_curr)`**, **`reset()`**: Brake-hold policy $\pi_{\text{stable}}$ inside $G_e$, equilibrium check, and state reset.
- **`step(self, x_curr) -> np.ndarray`**: Applies `u_ref[t_step]` directly via a zero-overhead fast-path when close to `x_ref[t_step]` (`pos_err <= 0.03 m`, `ang_err <= 0.03 rad`, `vel_err <= 0.06 m/s`); otherwise solves Tracking NMPC while moving (`|v| > v_eps`, `t_step < T_bak`), falling back to `stabilizing_brake_policy` inside $G_e$.
