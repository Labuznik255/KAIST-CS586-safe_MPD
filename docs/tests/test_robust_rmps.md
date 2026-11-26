# `tests/test_robust_rmps.py`

**Source file:** [tests/test_robust_rmps.py](../../tests/test_robust_rmps.py)

## Overview
Implements **Stage 5: 4-Variant Comparative Evaluation Benchmark** for Robust Safe-MPD on `AccTractorTrailer2d` across Perturbation Levels 0, 1, 2, and 3 over multiple seeds:
1. **`baseline`**: Original Safe-MPD (nominal open-loop plan executed under perturbations).
2. **`tube_only`**: Tube-Safe MPD (offline calibrated tube margins `r_tube`, `b_hitch` applied in `AccTractorTrailer2d` planning rollout, no online NMPC).
3. **`rmps_no_replan`**: Tube-Safe MPD + Online CPU NMPC Shield (`enable_nmpc_shield=True`, stops safely at equilibrium on intervention without replanning).
4. **`rmps_replan`**: Full RMPS (Tube-Safe MPD + Online NMPC Shield + Safe-MPD replanning from stopped states).

Uses paired initial plans per seed (`baseline_plans[s_idx]` shared at Level 0, `tube_plans[(lv, s_idx)]` shared across `tube_only`, `rmps_no_replan`, and `rmps_replan`) so differences reflect pure shielding & replanning effects.

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

### `main()` (lines 179–388)
Parses CLI options:
- `--seeds` (default `5`)
- `--levels` (default `0 1 2 3`)
- `--Nsample` (default `800`), `--Ndiffuse` (default `25`), `--horizon` (default `50`), `--dt` (default `0.25`)
- `--max-steps` (default `120`), `--max-replans` (default `2`)
- `--out-dir` (default `"results/rmps_benchmark"`)
- `--recalibrate` (force fresh tube calibration)

Writes `summary.json`, `summary.csv`, `benchmark_comparison.png`, and representative Level-`max(levels)` / Seed-`0` episode folders `sample_{variant}_level{L}/` (containing `.npy` trajectories, `env_metadata.json`, and `rmps_diagnostics.png`).
