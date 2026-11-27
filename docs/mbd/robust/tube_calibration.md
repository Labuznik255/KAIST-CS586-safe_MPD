# `mbd/robust/tube_calibration.py`

**Source file:** [mbd/robust/tube_calibration.py](../../../mbd/robust/tube_calibration.py)

## Overview
Implements **Stage 1: Offline Reachable Disturbance Tube Calibration** (RMPS Algorithm 4 + Section IV-E) for `AccTractorTrailer2d` across Perturbation Levels 0–3.

The calibrator constructs a curved reversing nominal maneuver, synthesizes closed-loop Time-Varying LQR (TV-LQR) feedback gains around it using JAX autodiff linearization, executes $N_{\text{mc}}$ closed-loop Monte Carlo rollouts in parallel (`jax.vmap`) under the active perturbation model, bounds empirical state errors $e_i(t) = x_i(t) - x_{\text{ref}}(t)$ with axis-aligned boxes $B(t)$, and exports scalar constraint tightening margins `r_tube`, `b_hitch`, and `b_v` to `.npz` archives under `results/tubes/`.

---

## Functions

### `_wrap_angle_jnp(angle: jax.Array) -> jax.Array` (lines 22–24)
Pure, branchless JAX angle wrapping onto $(-\pi, \pi]$ via `jnp.arctan2(jnp.sin(angle), jnp.cos(angle))`.

### `generate_nominal_maneuver(env, T_horizon: int = 25) -> tuple[np.ndarray, np.ndarray]`
Generates a representative curved reversing maneuver `(x_ref, u_ref)` of shapes `(T_horizon + 1, 6)` and `(T_horizon, 2)` from zero initial state using `jax.lax.scan`, `env.tractor_trailer_dynamics`, and `env.input_scaler`. Uses hitch-stabilizing steering feedback (`d_des = clip(1.6 * phi + d_exc, -0.65 * delta_max, 0.65 * delta_max)`) around sinusoidal excitation so arbitrary horizons (`T_horizon = 25, 100, ...`) stay well below `env.phi_max` (40 deg) without jackknifing.

### `compute_tvlqr_gains(env, x_ref, u_ref, Q=None, R=None, Q_terminal=None) -> np.ndarray` (lines 70–128)
Computes finite-horizon time-varying discrete LQR gains `K_fb` of shape `(T, 2, 6)` using `jax.jacfwd` linearization `(A_t, B_t)` along `(x_ref[:-1], u_ref)` and backward Riccati recursion (`Q = diag([2, 2, 5, 8, 2, 1])`, `R = diag([0.5, 0.5])`, `Q_terminal = 5 * Q`).

### `_rollout_mc_stochastic(env, x_ref, u_ref, K_fb, N_mc, seed) -> np.ndarray` (lines 131–168)
Simulates `N_mc` closed-loop rollouts via `jax.jit(jax.vmap(...))` calling `env.step_perturbed(st, u_fb)` with `u_fb = clip(u_ref[t] + K_fb[t] @ wrap(x - x_ref[t]), -1, 1)` for stochastic Levels 0, 1, and 3.

### `_rollout_mc_level2_params(env, x_ref, u_ref, K_fb, N_mc, seed, param_uncertainty_std) -> np.ndarray` (lines 171–230)
Simulates `N_mc` closed-loop rollouts for Level 2 time-invariant parameter uncertainty, sampling an independent `(l1_p, l2_p, lh_p)` tuple per Monte Carlo trajectory matching `TractorTrailer2d._init_perturbation_params` (`N(l_nom, (param_uncertainty_std * l_nom)^2)`, clipped to `[0.2 * l_nom, 5.0 * l_nom]`).

### `default_tube_path(level: int, model_name: str = "acc_tt2d", base_dir: str | Path = "results/tubes") -> Path` (lines 233–239)
Returns `Path(base_dir) / f"{model_name}_level{level}.npz"`.

### `save_tubes(tube_data: dict, path: str | Path) -> Path` & `load_tubes(path: str | Path) -> dict` (lines 242–277)
Save and reload calibrated tube dictionaries cleanly using `.npz` with scalar metadata type restoration (`level`, `N_mc`, `T_horizon`, `seed`, `dt`, `quantile`, `param_uncertainty_std`, `non_gaussian_dist`, `non_gaussian_df`, `state_disturbance_scale`).

### `calibrate_tubes(env, level, N_mc=1000, T_horizon=25, seed=42, quantile=0.99, x_ref=None, u_ref=None, K_fb=None, save_path=None, **perturbation_kwargs) -> dict` (lines 280–417)
Temporarily configures `env.set_perturbation(level=level, seed=seed, **perturbation_kwargs)` (restoring the caller's perturbation config in a `finally` block), runs the $N_{\text{mc}}$ closed-loop rollouts, computes `box_halfwidths`, `box_min`, `box_max`, and derives:
- `r_tube[t] = sqrt(box_halfwidths[t, 0]^2 + box_halfwidths[t, 1]^2)` [m]
- `b_hitch[t] = box_halfwidths[t, 2] + box_halfwidths[t, 3]` [rad]
- `b_v[t] = box_halfwidths[t, 4]` [m/s]

Saves to `save_path` if provided and returns the dictionary.

---

## CLI Entry Point (`main()`, lines 420–478)
Calibrates and verifies tubes for Levels 0, 1, 2, and 3 on `acc_tt2d`. Supports:
- `--N-mc` (default `1000`)
- `--horizon` (default `25`)
- `--dt` (default `0.25`)
- `--seed` (default `42`)
- `--quantile` (default `0.99`; use `1.0` for empirical min/max worst case)
- `--out-dir` (default `"results/tubes"`)
