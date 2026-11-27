# `tests/test_robust_rmps.py`

**Source file:** [tests/test_robust_rmps.py](../../tests/test_robust_rmps.py)

## Overview
Implements **Stage 5: 4-Variant Comparative Evaluation Benchmark** for Robust Safe-MPD on `AccTractorTrailer2d` across Perturbation Levels 0, 1, 2, and 3 over multiple seeds:
1. **`baseline`**: Original Safe-MPD (nominal open-loop plan executed under perturbations).
2. **`tube_only`**: Tube-Safe MPD (offline calibrated tube margins `r_tube`, `b_hitch` applied in `AccTractorTrailer2d` planning rollout, no online NMPC).
3. **`rmps_no_replan`**: Tube-Safe MPD + Online CPU NMPC Shield (`enable_nmpc_shield=True`, stops safely at equilibrium on intervention without replanning).
4. **`rmps_replan`**: Full RMPS (Tube-Safe MPD + Online NMPC Shield + Safe-MPD replanning from stopped states).

Uses paired initial plans per seed per planning family: `baseline_plans[s_idx]` (`enable_robust_shielding=False`) for `baseline`, and dedicated `tube_plans[(lv, s_idx)]` (`enable_robust_shielding=True`, including at Level 0 so robust variants always use accurate `dt` braking rollout and $t=0$ weight masking) shared across `tube_only`, `rmps_no_replan`, and `rmps_replan`.

---

## Functions

### `build_benchmark_env(perturbation_level, perturbation_seed, dt=0.25, H=50, enable_robust_shielding=False, r_tube=0.0, b_hitch=0.0)` (lines 52–105)
Builds the same `"navigation"` scenario as `tests/test_mbd.py` (`32x32` workspace, 5 rectangular obstacles, 4 circular obstacles, `x0=(-10, 5, 0, 0)`, `xg=(5, -5, -pi/2, -pi/2)`), forwarding `enable_robust_shielding`, `r_tube`, `b_hitch`, and perturbation defaults (`control_noise_std=(0.05, 0.05)`, `param_uncertainty_std=0.10`, `non_gaussian_dist="student_t"`, `non_gaussian_df=2.5`, `state_disturbance_scale=0.05`).

### `ensure_tubes_loaded(levels, dt, recalibrate=False) -> dict[int, dict]` (lines 108–120)
Loads `results/tubes/acc_tt2d_level{L}.npz` for each level, automatically calling `calibrate_tubes(...)` if a tube file is missing or `--recalibrate` is passed.

### `cvar_upper(values, alpha=0.01)` & `cvar_lower(values, alpha=0.01)` (lines 123–134)
Computes tail $\alpha$-CVaR metrics (upper tail for position error, lower tail for minimum obstacle clearance).

### `plot_benchmark_summary(rows, levels, out_path)` (lines 137–176)
Saves `benchmark_comparison.png` comparing collision rate, safe success rate, 1%-CVaR min clearance, and average online step runtime across variants and levels.

### `classify_episode_outcome(res)` & `plot_all_trajectories_overview(...)`
- Classifies every episode into a mutually exclusive primary outcome (`safe_goal`, `collision`, `jackknife`, `collision+jackknife`, `timeout`, `max_replans`, `stopped_no_replan`).
- Saves `all_trajectories_overview.png` as a `(len(levels) x len(variants))` grid overlaying every seed's trajectory on the obstacle map, color-coded by outcome.

### `main()`
Parses CLI options:
- `--seeds` (default `5`) and `--seed-list` (optional explicit seed list, e.g. `--seed-list 9123 42 43 44 45`)
- `--levels` (default `0 1 2 3`)
- `--variants` (default `baseline tube_only rmps_no_replan rmps_replan`; pass e.g. `--variants rmps_replan` to run a subset and skip unused baseline/tube precomputation)
- `--Nsample` (default `800`), `--Ndiffuse` (default `25`), `--horizon` (default `50`), `--dt` (default `0.25`)
- `--max-steps` (default `150`), `--max-replans` (default `5`)
- `--out-dir` (default `"results/rmps_benchmark"`)
- `--save-episodes {sample,per_level,all}` (default `"all"`; saves `.npy` trajectories + `env_metadata.json` + optional `rmps_diagnostics.png` under `episodes/level<L>/<variant>/seed_<S>/` so `render.py` can visualize any level/seed)
- `--no-episode-plots` (skip per-episode `rmps_diagnostics.png` to speed up large sweeps)
- `--recalibrate` (force fresh tube calibration)

Prints a full metrics table + per-row failure type breakdown (`Collision`, `Jackknife`, `Collision+Jackknife`, `Safe Timeout`, `Safe Max-Replans`, `Safe Stopped`), and writes `summary.json`, `summary.csv`, `benchmark_comparison.png`, `all_trajectories_overview.png`, and saved episode folders.
