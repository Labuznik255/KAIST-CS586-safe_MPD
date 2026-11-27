# `tests/sweep_perturbation_magnitude.py`

**Source file:** [tests/sweep_perturbation_magnitude.py](../../tests/sweep_perturbation_magnitude.py)

## Overview
Standalone perturbation-magnitude sweep evaluator and publication-quality plotting suite comparing `baseline` (unshielded Safe-MPD) vs. `rmps_replan` (Tube-Safe MPD + closed-loop TV-LQR tracking + online NMPC shield + replanning) — and optionally `tube_only` and `rmps_no_replan` — across dimensionless perturbation scale factors $\alpha$ on `AccTractorTrailer2d` for Perturbation Levels 1, 2, and 3.

---

## Perturbation Scaling (`scaled_perturbation_params`)
- **$\alpha = 0.0$**: Maps to nominal Level 0 (`effective_level = 0`, zero noise/uncertainty, zero tube margins).
- **Level 1**: `control_noise_std = (0.05 * alpha, 0.05 * alpha)`.
- **Level 2**: `param_uncertainty_std = 0.10 * alpha`.
- **Level 3**: `control_noise_std = (0.05 * alpha, 0.05 * alpha)` and `state_disturbance_scale = 0.05 * alpha` (`non_gaussian_dist="student_t"`, `non_gaussian_df=2.5`).

---

## Key Functions

- **`build_scaled_benchmark_env(level, scale, perturbation_seed, dt=0.25, H=50, enable_robust_shielding=False, r_tube=0.0, b_hitch=0.0)`**: Builds the standard 32×32 m custom navigation obstacle environment with scaled perturbation parameters.
- **`ensure_scaled_tubes(levels, scales, dt, tube_dir, tube_mode="calibrate", n_mc=1000, recalibrate=False)`**:
  - `tube_mode="calibrate"` (default): Runs `calibrate_tubes(...)` per `(level, alpha)` and caches `.npz` archives in `results/tubes/acc_tt2d_level{L}_scale{tag}.npz` (reusing `level0` at $\alpha=0$ and `level{L}` at $\alpha=1$).
  - `tube_mode="scale"`: Linearly scales the $\alpha=1.0$ tube margins by $\alpha$.
- **`plot_perturbation_sweep_main(summary_rows, levels, scales, variants, out_dir)`**: Generates `perturbation_sweep_main.{png,pdf,svg}` (300 DPI PNG + vector PDF/SVG) as a $3 \times |\text{levels}|$ grid:
  - **Row 1**: Safe Success Rate (%) vs. $\alpha$ with $\pm 1\,\text{SE}$ shaded bands.
  - **Row 2**: Collision Rate (%) (solid) and Hitch Violation Rate (%) (dashed) vs. $\alpha$.
  - **Row 3**: Mean Minimum Clearance [m] (left axis, with `0.0 m` collision threshold) & NMPC Intervention Rate (%) (right axis bars).
- **`plot_safety_vs_success_tradeoff(summary_rows, levels, scales, variants, out_dir)`**: Generates `safety_vs_success_tradeoff.{png,pdf,svg}` plotting Collision Rate (%) vs. Safe Success Rate (%) parameterized by $\alpha$ with directional arrows, coinciding-point label merging, and the ideal `(0%, 100%)` target marker.
- **`generate_all_plots(payload, out_dir)`**: Generates both figures directly from a computed or loaded `sweep_results.json` payload.

---

## Plan Caching & Fairness
- `baseline` plans (`enable_robust_shielding=False`) are precomputed once per seed across all levels and scales.
- Tube/RMPS variants (`enable_robust_shielding=True`) always run dedicated robust diffusion plans (including at $\alpha=0.0$, cached once as `robust_zero_plans` across Levels 1–3) so they never inherit legacy unshielded `baseline` plans.

---

## CLI Options (`main()`)
- `--scales` (default `0.0 0.2 0.4 0.6 0.8 1.0`)
- `--levels` (default `1 2 3`)
- `--variants` (default `baseline rmps_replan`)
- `--num-seeds` (default `5`) / `--seed-list` (explicit seed override)
- `--Nsample` (default `800`), `--Ndiffuse` (default `25`), `--horizon` (default `50`), `--dt` (default `0.25`)
- `--max-steps` (default `150`), `--max-replans` (default `5`)
- `--tube-mode {calibrate,scale}` (default `calibrate`), `--tube-N-mc` (default `1000`), `--recalibrate`
- `--out-dir` (default `results/perturbation_sweep`)
- `--plot-only <path/to/sweep_results.json>`: Regenerates all 6 figure files (`.png`, `.pdf`, `.svg`) from saved results without re-running simulations.

Outputs saved to `--out-dir`: `sweep_results.json`, `sweep_summary.csv`, `perturbation_sweep_main.{png,pdf,svg}`, and `safety_vs_success_tradeoff.{png,pdf,svg}`.
