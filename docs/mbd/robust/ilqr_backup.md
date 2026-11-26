# `mbd/robust/ilqr_backup.py`

**Source file:** [mbd/robust/ilqr_backup.py](../../../mbd/robust/ilqr_backup.py)

## Overview
Implements **Stage 2: Pure-JAX, Branchless, Vectorized Constrained iLQR Backup Solver** for `AccTractorTrailer2d` (`6D` state, `2D` normalized control).

Used as Stage 2 of the two-stage GPU backup shield inside `AccTractorTrailer2d.shielded_rollout` whenever opted in via `enable_ilqr_backup=True`. When Stage 1's fast straight braking rollout reports an obstacle collision or hitch-angle violation, `solve_ilqr_backup` searches for an evasive braking + steering recovery maneuver using fixed iterations (`jax.lax.scan`, default `n_iters=4`), smooth obstacle/hitch barriers inflated by the tube margins `(r_margins, b_margins)`, closed-form 2×2 Levenberg-Marquardt regularization, and parallel line search over `[1.0, 0.5, 0.1]`.

---

## Data Structures & Functions

### `class ILQRBackupParams(NamedTuple)` (lines 21–46)
PyTree-compatible container holding physical dimensions/limits (`dt`, `l1`, `l2`, `lh`, `lf1`, `lr`, `lf2`, `lr2`, `tractor_width`, `trailer_width`, `v_max`, `delta_max`, `a_max`, `omega_max`, `phi_max`) and Gauss-Newton barrier cost weights (`w_u=0.25`, `w_v_stage=0.4`, `w_obs=120.0`, `w_hitch=100.0`, `w_v_term=200.0`, `w_hitch_term=15.0`, `lm_reg=1e-2`).

### `_wrap_angle(angle)` & `_splus(z, eps=1e-2)` (lines 49–56)
Smooth branchless angle wrap and $C^\infty$ soft-plus hinge $\frac{1}{2}(z + \sqrt{z^2 + \varepsilon^2})$.

### `acc_tt2d_step(x, u_norm, params) -> jax.Array` (lines 59–92)
Exact nominal Forward-Euler integration step for 6D `AccTractorTrailer2d`.

### `acc_tt2d_jacobians(x, u_norm, params) -> tuple[jax.Array, jax.Array]` (lines 95–152)
Closed-form discrete-time Jacobians `(F_x, F_u)` of shapes `(6, 6)` and `(6, 2)` matching `acc_tt2d_step`.

### `_vehicle_body_discs(x, params)` & `_state_barrier_residuals(...)` (lines 155–254)
Covers the tractor and trailer rectangles with 6 body discs (`3` tractor + `3` trailer) and constructs Gauss-Newton residual vectors for tightened hitch-angle limits (`phi_max - b_margin - 0.02`), inflated circular obstacles (`r + r_margin`), and inflated rotated rectangular obstacles (`hw + r_margin`, `hh + r_margin`).

### `_stage_state_residuals`, `_terminal_state_residuals`, `_trajectory_cost` (lines 257–311)
Builds stage velocity/barrier residuals, terminal stopping (`v -> 0`) + hitch alignment (`theta1 - theta2 -> 0`) residuals, and `0.5 * sum(residuals^2)` trajectory cost.

### `generate_braking_warmstart(x0, params, horizon=8) -> tuple[jax.Array, jax.Array]` (lines 314–336)
Rolls out a smooth deadbeat-braking + steering-centering warm-start `(xs_init, us_init)`.

### `_regularize_quu_2x2(Q_uu, lm_reg)` (lines 339–348)
Closed-form analytical 2×2 symmetric eigenvalue floor guaranteeing $Q_{uu} \succeq \text{lm\_reg} \cdot I_2$ without Python branching.

### `solve_ilqr_backup(x0, obs_circles, obs_rectangles, r_margins, b_margins, params, n_iters=4, horizon=8)` (lines 351–471)
JIT-compiled (`static_argnames=("n_iters", "horizon")`) fixed-iteration constrained iLQR solver:
1. Initializes from `generate_braking_warmstart`.
2. In each iteration via `jax.lax.scan`, evaluates vectorized Gauss-Newton stage/terminal quadratizations and dynamics Jacobians (`F_x`, `F_u`).
3. Runs a reverse backward pass (`jax.lax.scan(..., reverse=True)`) to compute feedforward `k_ff` and feedback gains `K_fb`.
4. Evaluates candidate forward rollouts in parallel (`jax.vmap`) over `alphas = [1.0, 0.5, 0.1]`, selects the lowest-cost candidate via `jnp.argmin`, and accepts it via `jnp.where` if cost improves.
- **Returns**: `(xs_opt, us_opt, cost_opt)` with static shapes `(horizon + 1, 6)` and `(horizon, 2)`.
