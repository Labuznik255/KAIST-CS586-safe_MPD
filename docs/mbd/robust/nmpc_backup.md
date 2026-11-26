# `mbd/robust/nmpc_backup.py`

**Source file:** [mbd/robust/nmpc_backup.py](../../../mbd/robust/nmpc_backup.py)

## Overview
Implements **Stage 3 (Online CPU Shield): `RobustNMPCBackup`**, a Tube-Based Recovery and Tracking Nonlinear Model Predictive Control backup controller (RMPS Algorithms 2 & 3) for `AccTractorTrailer2d`. Built purely in Python / NumPy + CasADi (`casadi.Opti()` + `ipopt`) on CPU, isolated from JAX tracing.

Key features:
- **Reusable parametric CasADi Opti bundles** (`_recovery_bundle`, `_tracking_bundle`) with `expand=True` and obstacle signature caching so IPOPT solves do not rebuild symbolic graphs online.
- **Exact vehicle-rectangle vs. obstacle collision formulation**: smooth distance constraints between the tractor and trailer rectangles and circular/oriented rectangular obstacles inflated by `r_tube` + `disc_safety_pad`.
- **Fast-path verification + symmetry-breaking warm-starts + LRU solution caching**: `is_recoverable(x_cand)` tests fast braking warm-starts (~0.1 ms) first, then IPOPT with `(0.0, +0.65, -0.65)` steering biases, and caches `(feasible, x_sol, u_sol)` so a subsequent `initialize_backup(x_cand)` reuses the solution with zero extra IPOPT calls.

---

## Utilities & Class

### `_wrap_angle_np(angle)` & `_format_tube_schedule(val, length)` (lines 20–39)
Wraps angles onto $(-\pi, \pi]$ and normalizes scalar or 1D array tube margins into non-negative length-`length` NumPy schedules.

### `class RobustNMPCBackup` (lines 42–769)

- **`__init__(self, env, T_bak=12, v_eps=0.05, max_iter=40, disc_safety_pad=0.01, hitch_safety_pad=0.01, cache_size=128)`** (lines 51–104): Extracts nominal `AccTractorTrailer2d` dimensions/limits from `env`, initializes backup state (`x_ref`, `u_ref`, `r_tube_active`, `b_hitch_active`, `t_step`, `active`), LRU `_solution_cache`, and lazy compiled solver bundles.
- **`step_nominal_np(self, x, u_norm) -> np.ndarray`** (lines 108–139): Exact CPU NumPy Forward-Euler nominal step matching `AccTractorTrailer2d.tractor_trailer_dynamics`.
- **`check_state_safety_np(self, x, r_margin=0.0, b_margin=0.0) -> tuple[bool, bool]`** (lines 217–267): Exact CPU SAT collision & hitch checker with optional tightened margins.
- **`_build_opti_problem(self, obs_c, obs_r, tracking_mode: bool)`** (lines 352–510):
  - **`tracking_mode=False` (Recovery NLP, RMPS Algorithm 2)**: Finds a control sequence `u[0..T_bak-1]` driving `x` safely to a stopped equilibrium (`|v[T_bak]| <= v_eps`) subject to hard tightened obstacle (`r_tube + P_disc_pad`) and hitch (`phi_max - b_hitch - hitch_safety_pad`) constraints.
  - **`tracking_mode=True` (Tracking NMPC, RMPS Algorithm 3)**: Minimizes tracking error to `(x_ref[t_step:], u_ref[t_step:])` + terminal velocity using softened slack `S_obs >= 0` on tightened constraints so perturbed states within the tube stay feasible.
- **`is_recoverable(self, x_cand, r_tube=0.0, b_hitch=0.0, return_solution=False)`** (lines 629–648): Checks whether `x_cand` admits a verified recovery trajectory to $G_e$ (`|v| <= v_eps`), caching the result in `_solution_cache`.
- **`initialize_backup(self, x_init, r_tube=0.0, b_hitch=0.0) -> tuple[bool, np.ndarray, np.ndarray]`** (lines 665–686): Stores active `(x_ref, u_ref, t_step=0, active=True)` for online tracking.
- **`stabilizing_brake_policy(x_curr)`**, **`in_equilibrium_set(x_curr)`**, **`reset()`** (lines 688–703): Closed-loop brake-hold policy $\pi_{\text{stable}}$ inside $G_e$, equilibrium check, and state reset.
- **`step(self, x_curr) -> np.ndarray`** (lines 705–769): Executes one Tracking NMPC step along `(x_ref[t_step:], u_ref[t_step:])` while moving (`|v| > v_eps` and `t_step < T_bak`), falling back to `stabilizing_brake_policy` inside $G_e$.
