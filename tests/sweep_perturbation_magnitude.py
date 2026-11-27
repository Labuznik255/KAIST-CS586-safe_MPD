"""
Perturbation-Magnitude Sweep Evaluator & Publication-Quality Visualization Suite.

Systematically evaluates how perturbation magnitude (dimensionless scale factor `alpha`)
affects Safe Success Rate (%), Collision Rate (%), Hitch Violation Rate (%), Minimum
Obstacle Clearance [m], and NMPC Intervention Rate (%) for `baseline` (unshielded
Safe-MPD) vs. `rmps_replan` (Tube-Safe MPD + Online NMPC Backup + Replanning) on
`AccTractorTrailer2d` across Perturbation Levels 1, 2, and 3:

  - alpha = 0.0 : Nominal Level 0 (zero perturbation, zero tube margins)
  - Level 1     : Gaussian control noise, `control_noise_std = (0.05 * alpha, 0.05 * alpha)`
  - Level 2     : Model parameter uncertainty, `param_uncertainty_std = 0.10 * alpha`
  - Level 3     : Non-Gaussian Student-t (df=2.5) disturbances,
                  `control_noise_std = (0.05 * alpha, 0.05 * alpha)` and
                  `state_disturbance_scale = 0.05 * alpha`

Usage examples:
  # Fast CPU smoke test (3 scales, 3 seeds, small diffusion budget):
  JAX_PLATFORMS=cpu MPLBACKEND=Agg ./.venv/bin/python tests/sweep_perturbation_magnitude.py \\
      --scales 0.0 0.5 1.0 --levels 1 2 3 --num-seeds 3 --Nsample 800 --Ndiffuse 25 \\
      --out-dir results/perturbation_sweep_smoke

  # Full GPU sweep on virtualized NVIDIA GRID A100D-20C (VMM disabled):
  XLA_FLAGS="--xla_gpu_experimental_vmm_disabled" MPLBACKEND=Agg ./.venv/bin/python \\
      tests/sweep_perturbation_magnitude.py \\
      --scales 0.0 0.2 0.4 0.6 0.8 1.0 --levels 1 2 3 --num-seeds 10 \\
      --Nsample 20000 --Ndiffuse 100 --out-dir results/perturbation_sweep_gpu

  # Regenerate figures from saved sweep results without re-running simulations:
  MPLBACKEND=Agg ./.venv/bin/python tests/sweep_perturbation_magnitude.py \\
      --plot-only results/perturbation_sweep_gpu/sweep_results.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mbd  # noqa: E402
from mbd.envs.env import Env  # noqa: E402
from mbd.planners.mbd_planner import MBDConfig, clear_jit_cache  # noqa: E402
from mbd.robust import (  # noqa: E402
    RobustNMPCBackup,
    SafeMPDPlanner,
    calibrate_tubes,
    default_tube_path,
    load_tubes,
    run_rmps_episode,
)
from tests.test_robust_rmps import (  # noqa: E402
    VARIANTS,
    classify_episode_outcome,
    cvar_lower,
)

# Default base perturbation magnitudes at alpha = 1.0
BASE_CONTROL_NOISE_STD = 0.05
BASE_PARAM_UNCERTAINTY_STD = 0.10
BASE_STATE_DISTURBANCE_SCALE = 0.05
BASE_NON_GAUSSIAN_DIST = "student_t"
BASE_NON_GAUSSIAN_DF = 2.5

LEVEL_METADATA: dict[int, dict[str, str]] = {
    1: {
        "title": "Level 1: Gaussian Control Noise",
        "subtitle": r"$\sigma_u = 0.05\,\alpha$",
        "short": "L1 (Control Noise)",
    },
    2: {
        "title": "Level 2: Model Parameter Uncertainty",
        "subtitle": r"$\sigma_{\mathrm{param}} = 0.10\,\alpha$",
        "short": "L2 (Param Uncert.)",
    },
    3: {
        "title": "Level 3: Non-Gaussian (Student-t) Noise",
        "subtitle": r"$\sigma_u = 0.05\,\alpha,\; s_x = 0.05\,\alpha$",
        "short": "L3 (Student-t)",
    },
}

VARIANT_STYLE: dict[str, dict[str, Any]] = {
    "baseline": {
        "label": "Baseline (Safe-MPD)",
        "color": "#D55E00",
        "marker": "o",
        "linestyle": "-",
        "zorder": 3,
    },
    "tube_only": {
        "label": "Tube-Only MPD",
        "color": "#E69F00",
        "marker": "D",
        "linestyle": "-",
        "zorder": 3,
    },
    "rmps_no_replan": {
        "label": "RMPS (No Replan)",
        "color": "#0072B2",
        "marker": "^",
        "linestyle": "-",
        "zorder": 4,
    },
    "rmps_replan": {
        "label": "RMPS + Replan",
        "color": "#009E73",
        "marker": "s",
        "linestyle": "-",
        "zorder": 5,
    },
}


# ---------------------------------------------------------------------------
# Perturbation Scaling, Environment Construction & Tube Calibration Helpers
# ---------------------------------------------------------------------------
def scaled_perturbation_params(level: int, scale: float) -> dict[str, Any]:
    """Compute exact perturbation kwargs for `(level, scale)` with `scale=0.0 -> Level 0`."""
    alpha = float(scale)
    if alpha < 0.0:
        raise ValueError(f"Perturbation scale alpha must be >= 0.0; got {alpha}")
    if alpha <= 1e-12:
        return {
            "effective_level": 0,
            "control_noise_std": (0.0, 0.0),
            "param_uncertainty_std": 0.0,
            "non_gaussian_dist": BASE_NON_GAUSSIAN_DIST,
            "non_gaussian_df": BASE_NON_GAUSSIAN_DF,
            "state_disturbance_scale": 0.0,
        }

    if level == 1:
        s_u = BASE_CONTROL_NOISE_STD * alpha
        return {
            "effective_level": 1,
            "control_noise_std": (s_u, s_u),
            "param_uncertainty_std": BASE_PARAM_UNCERTAINTY_STD,
            "non_gaussian_dist": BASE_NON_GAUSSIAN_DIST,
            "non_gaussian_df": BASE_NON_GAUSSIAN_DF,
            "state_disturbance_scale": BASE_STATE_DISTURBANCE_SCALE,
        }
    if level == 2:
        return {
            "effective_level": 2,
            "control_noise_std": (BASE_CONTROL_NOISE_STD, BASE_CONTROL_NOISE_STD),
            "param_uncertainty_std": BASE_PARAM_UNCERTAINTY_STD * alpha,
            "non_gaussian_dist": BASE_NON_GAUSSIAN_DIST,
            "non_gaussian_df": BASE_NON_GAUSSIAN_DF,
            "state_disturbance_scale": BASE_STATE_DISTURBANCE_SCALE,
        }
    if level == 3:
        s_u = BASE_CONTROL_NOISE_STD * alpha
        s_x = BASE_STATE_DISTURBANCE_SCALE * alpha
        return {
            "effective_level": 3,
            "control_noise_std": (s_u, s_u),
            "param_uncertainty_std": BASE_PARAM_UNCERTAINTY_STD,
            "non_gaussian_dist": BASE_NON_GAUSSIAN_DIST,
            "non_gaussian_df": BASE_NON_GAUSSIAN_DF,
            "state_disturbance_scale": s_x,
        }
    raise ValueError(f"Sweep level must be 1, 2, or 3; got {level}")


def build_scaled_benchmark_env(
    level: int,
    scale: float,
    perturbation_seed: int,
    dt: float = 0.25,
    H: int = 50,
    enable_robust_shielding: bool = False,
    r_tube: float | np.ndarray = 0.0,
    b_hitch: float | np.ndarray = 0.0,
) -> Any:
    """Build the standard `AccTractorTrailer2d` navigation environment with scaled perturbations."""
    p = scaled_perturbation_params(level, scale)
    env_config = Env(width=32.0, height=32.0, case="navigation")
    env = mbd.envs.get_env(
        "acc_tt2d",
        case="navigation",
        env_config=env_config,
        dt=dt,
        H=H,
        perturbation_level=p["effective_level"],
        perturbation_seed=perturbation_seed,
        control_noise_std=p["control_noise_std"],
        param_uncertainty_std=p["param_uncertainty_std"],
        non_gaussian_dist=p["non_gaussian_dist"],
        non_gaussian_df=p["non_gaussian_df"],
        state_disturbance_scale=p["state_disturbance_scale"],
        enable_robust_shielding=enable_robust_shielding,
        tube_obstacle_margin=r_tube,
        tube_hitch_margin=b_hitch,
    )

    parking_space_width = 2.5
    parking_space_length = 5.0
    parking_x = 5.0
    parking_y = -5.0

    rectangular_obstacles = [
        [parking_x - parking_space_width - 1.0, parking_y, 1.0, parking_space_length * 2, 0.0],
        [parking_x, parking_y - parking_space_length - 1.0, parking_space_width * 2 + 4.0, 1.0, 0.0],
        [parking_x - parking_space_width - 6.0, parking_y, 1.0, parking_space_length * 2, -jnp.pi / 4],
        [parking_x - parking_space_width - 6.0, parking_y + 18.0, 1.2, parking_space_length * 2, jnp.pi / 4],
        [parking_x - parking_space_width - 12.0, parking_y + 18.0, 0.8, parking_space_length, -jnp.pi / 6],
    ]
    env.set_rectangle_obs(rectangular_obstacles, coordinate_mode="center", padding=0.0)

    circular_obstacles = [
        [parking_x + parking_space_width + 0.0, parking_y + 3.0, 1.0],
        [parking_x + parking_space_width + 0.0, parking_y - 3.0, 2.0],
        [parking_x + parking_space_width + 1.0, parking_y, 1.5],
        [parking_x + parking_space_width + 1.0, parking_y + 12.0, 3.5],
    ]
    env.set_circle_obs(circular_obstacles, padding=0.0)

    env.set_init_pos(x=-10.0, y=5.0, theta1=0.0, theta2=0.0)
    env.set_goal_pos(x=parking_x, y=parking_y, theta1=-jnp.pi / 2, theta2=-jnp.pi / 2)
    return env


def _scale_tag(scale: float) -> str:
    return f"{float(scale):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def _tube_cache_path(level: int, scale: float, tube_dir: Path) -> Path:
    if scale <= 1e-12:
        return default_tube_path(level=0, model_name="acc_tt2d", base_dir=tube_dir)
    if abs(scale - 1.0) <= 1e-9:
        return default_tube_path(level=level, model_name="acc_tt2d", base_dir=tube_dir)
    return tube_dir / f"acc_tt2d_level{int(level)}_scale{_scale_tag(scale)}.npz"


def ensure_scaled_tubes(
    levels: list[int],
    scales: list[float],
    dt: float,
    tube_dir: Path,
    tube_mode: str = "calibrate",
    n_mc: int = 1000,
    recalibrate: bool = False,
) -> dict[tuple[int, float], dict[str, Any]]:
    """
    Load or calibrate reachable disturbance tubes for every `(level, scale)` pair.

    - `tube_mode="calibrate"`: Runs `calibrate_tubes` at the exact scaled perturbation
      parameters and caches each `.npz` file in `tube_dir`.
    - `tube_mode="scale"`: Loads/calibrates the `alpha = 1.0` tube per level and
      linearly scales `r_tube`, `b_hitch`, `b_v`, and `box_halfwidths` by `alpha`.
    """
    tube_dir.mkdir(parents=True, exist_ok=True)
    tubes: dict[tuple[int, float], dict[str, Any]] = {}
    env_cal = None

    def _get_env_cal() -> Any:
        nonlocal env_cal
        if env_cal is None:
            env_cal = mbd.envs.get_env("acc_tt2d", case="navigation", dt=dt, H=50)
        return env_cal

    if tube_mode == "scale":
        base_tubes: dict[int, dict[str, Any]] = {}
        for lv in levels:
            base_path = default_tube_path(level=lv, model_name="acc_tt2d", base_dir=tube_dir)
            if recalibrate or not base_path.exists():
                base_tubes[lv] = calibrate_tubes(
                    _get_env_cal(),
                    level=lv,
                    N_mc=n_mc,
                    T_horizon=25,
                    seed=42,
                    save_path=base_path,
                )
            else:
                base_tubes[lv] = load_tubes(base_path)

        for lv in levels:
            bt = base_tubes[lv]
            for sc in scales:
                t_scaled = dict(bt)
                t_scaled["r_tube"] = np.asarray(bt["r_tube"], dtype=np.float32) * float(sc)
                t_scaled["b_hitch"] = np.asarray(bt["b_hitch"], dtype=np.float32) * float(sc)
                if "b_v" in bt:
                    t_scaled["b_v"] = np.asarray(bt["b_v"], dtype=np.float32) * float(sc)
                if "box_halfwidths" in bt:
                    t_scaled["box_halfwidths"] = np.asarray(bt["box_halfwidths"], dtype=np.float32) * float(sc)
                tubes[(lv, sc)] = t_scaled
        return tubes

    # Exact per-(level, scale) empirical calibration with disk caching
    zero_tube: dict[str, Any] | None = None
    for lv in levels:
        for sc in scales:
            if sc <= 1e-12:
                if zero_tube is None:
                    z_path = _tube_cache_path(0, 0.0, tube_dir)
                    if recalibrate or not z_path.exists():
                        zero_tube = calibrate_tubes(
                            _get_env_cal(),
                            level=0,
                            N_mc=n_mc,
                            T_horizon=25,
                            seed=42,
                            save_path=z_path,
                        )
                    else:
                        zero_tube = load_tubes(z_path)
                tubes[(lv, sc)] = zero_tube
                continue

            c_path = _tube_cache_path(lv, sc, tube_dir)
            if recalibrate or not c_path.exists():
                p = scaled_perturbation_params(lv, sc)
                tubes[(lv, sc)] = calibrate_tubes(
                    _get_env_cal(),
                    level=p["effective_level"],
                    N_mc=n_mc,
                    T_horizon=25,
                    seed=42,
                    save_path=c_path,
                    control_noise_std=p["control_noise_std"],
                    param_uncertainty_std=p["param_uncertainty_std"],
                    non_gaussian_dist=p["non_gaussian_dist"],
                    non_gaussian_df=p["non_gaussian_df"],
                    state_disturbance_scale=p["state_disturbance_scale"],
                )
            else:
                tubes[(lv, sc)] = load_tubes(c_path)

    return tubes


# ---------------------------------------------------------------------------
# Aggregation Helpers
# ---------------------------------------------------------------------------
def _bernoulli_se_pct(flags: list[bool]) -> float:
    n = len(flags)
    if n <= 1:
        return 0.0
    arr = np.asarray(flags, dtype=np.float64)
    return float(100.0 * np.std(arr, ddof=1) / np.sqrt(n))


def _mean_std_se(values: list[float]) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean_v = float(np.mean(arr))
    if n == 1:
        return mean_v, 0.0, 0.0
    std_v = float(np.std(arr, ddof=1))
    return mean_v, std_v, float(std_v / np.sqrt(n))


# ---------------------------------------------------------------------------
# Publication-Quality Plotting
# ---------------------------------------------------------------------------
def _configure_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.8,
            "figure.titlesize": 13,
            "axes.linewidth": 0.9,
            "lines.linewidth": 2.1,
            "lines.markersize": 6.0,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_multi_format(fig: plt.Figure, out_stem: Path) -> list[Path]:
    saved: list[Path] = []
    for ext, kwargs in (
        (".png", {"dpi": 300}),
        (".pdf", {}),
        (".svg", {}),
    ):
        p = out_stem.with_suffix(ext)
        fig.savefig(p, bbox_inches="tight", **kwargs)
        saved.append(p)
    return saved


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)


def plot_perturbation_sweep_main(
    summary_rows: list[dict[str, Any]],
    levels: list[int],
    scales: list[float],
    variants: list[str],
    out_dir: Path,
) -> list[Path]:
    """
    Generate `perturbation_sweep_main.{png,pdf,svg}`: a 3 x len(levels) multi-panel grid.
      Row 1: Safe Success Rate (%) vs. alpha (with +/- 1 SE shaded bands)
      Row 2: Safety Violations (%) vs. alpha (Collision solid, Hitch Violation dashed)
      Row 3: Minimum Clearance [m] (left axis, 0 m threshold) & NMPC Intervention Rate (%) (right axis)
    """
    _configure_pub_style()
    n_cols = max(1, len(levels))
    fig, axes = plt.subplots(
        3,
        n_cols,
        figsize=(5.0 * n_cols, 10.6),
        sharex="col",
        squeeze=False,
    )

    by_key = {(int(r["level"]), round(float(r["scale"]), 6), str(r["variant"])): r for r in summary_rows}
    scales_sorted = sorted(float(s) for s in scales)
    x_arr = np.asarray(scales_sorted, dtype=np.float64)
    dx = float(np.min(np.diff(x_arr))) if x_arr.size > 1 else 0.25
    bar_w = min(0.11, 0.36 * dx)

    for col_idx, lv in enumerate(levels):
        meta = LEVEL_METADATA.get(lv, {"title": f"Level {lv}", "subtitle": ""})
        ax_succ = axes[0, col_idx]
        ax_viol = axes[1, col_idx]
        ax_clr = axes[2, col_idx]
        ax_int = ax_clr.twinx()

        # ---------------- Row 1: Safe Success Rate (%) ----------------
        _style_axis(ax_succ)
        for var in variants:
            st = VARIANT_STYLE.get(var, {"label": var, "color": "gray", "marker": "o", "zorder": 3})
            ys, ses = [], []
            for sc in scales_sorted:
                row = by_key.get((lv, round(sc, 6), var))
                ys.append(float(row["safe_success_rate_pct"]) if row else np.nan)
                ses.append(float(row.get("safe_success_se_pct", 0.0)) if row else 0.0)
            y_np = np.asarray(ys, dtype=np.float64)
            se_np = np.asarray(ses, dtype=np.float64)
            ax_succ.plot(
                x_arr,
                y_np,
                color=st["color"],
                marker=st["marker"],
                linestyle="-",
                label=st["label"],
                zorder=st["zorder"],
            )
            ax_succ.fill_between(
                x_arr,
                np.clip(y_np - se_np, 0.0, 100.0),
                np.clip(y_np + se_np, 0.0, 100.0),
                color=st["color"],
                alpha=0.16,
                linewidth=0.0,
                zorder=st["zorder"] - 1,
            )

        ax_succ.set_ylim(-5.0, 107.0)
        ax_succ.set_yticks([0, 20, 40, 60, 80, 100])
        title_txt = f"{meta['title']}\n({meta['subtitle']})" if meta["subtitle"] else meta["title"]
        ax_succ.set_title(title_txt, fontweight="bold", pad=8)
        if col_idx == 0:
            ax_succ.set_ylabel("Safe Success Rate (%)")

        # ---------------- Row 2: Collision & Hitch Violations (%) ----------------
        _style_axis(ax_viol)
        for var in variants:
            st = VARIANT_STYLE.get(var, {"label": var, "color": "gray", "marker": "o", "zorder": 3})
            col_ys, col_ses, hitch_ys = [], [], []
            for sc in scales_sorted:
                row = by_key.get((lv, round(sc, 6), var))
                col_ys.append(float(row["collision_rate_pct"]) if row else np.nan)
                col_ses.append(float(row.get("collision_se_pct", 0.0)) if row else 0.0)
                hitch_ys.append(float(row["hitch_violation_rate_pct"]) if row else np.nan)
            c_np = np.asarray(col_ys, dtype=np.float64)
            cse_np = np.asarray(col_ses, dtype=np.float64)
            h_np = np.asarray(hitch_ys, dtype=np.float64)

            ax_viol.plot(
                x_arr,
                c_np,
                color=st["color"],
                marker=st["marker"],
                linestyle="-",
                zorder=st["zorder"],
            )
            ax_viol.fill_between(
                x_arr,
                np.clip(c_np - cse_np, 0.0, 100.0),
                np.clip(c_np + cse_np, 0.0, 100.0),
                color=st["color"],
                alpha=0.14,
                linewidth=0.0,
                zorder=st["zorder"] - 1,
            )
            ax_viol.plot(
                x_arr,
                h_np,
                color=st["color"],
                marker="^",
                markerfacecolor="white",
                markeredgewidth=1.4,
                linestyle="--",
                linewidth=1.7,
                alpha=0.9,
                zorder=st["zorder"] + 1,
            )

        ax_viol.set_ylim(-5.0, 107.0)
        ax_viol.set_yticks([0, 20, 40, 60, 80, 100])
        if col_idx == 0:
            ax_viol.set_ylabel("Violation Rate (%)")

        # ---------------- Row 3: Min Clearance [m] + NMPC Intervention (%) ----------------
        ax_clr.spines["top"].set_visible(False)
        ax_clr.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        ax_int.spines["top"].set_visible(False)

        all_clr_vals: list[float] = []
        for var in variants:
            st = VARIANT_STYLE.get(var, {"label": var, "color": "gray", "marker": "o", "zorder": 3})
            clr_ys, clr_ses = [], []
            for sc in scales_sorted:
                row = by_key.get((lv, round(sc, 6), var))
                clr_ys.append(float(row["mean_min_clearance_m"]) if row else np.nan)
                clr_ses.append(float(row.get("se_min_clearance_m", 0.0)) if row else 0.0)
            y_np = np.asarray(clr_ys, dtype=np.float64)
            se_np = np.asarray(clr_ses, dtype=np.float64)
            all_clr_vals.extend((y_np - se_np).tolist())
            all_clr_vals.extend((y_np + se_np).tolist())

            ax_clr.plot(
                x_arr,
                y_np,
                color=st["color"],
                marker=st["marker"],
                linestyle="-",
                zorder=st["zorder"] + 2,
            )
            ax_clr.fill_between(
                x_arr,
                y_np - se_np,
                y_np + se_np,
                color=st["color"],
                alpha=0.14,
                linewidth=0.0,
                zorder=st["zorder"] + 1,
            )

        # NMPC Intervention Rate (%) bars on secondary y-axis for shielded variants
        shield_vars = [v for v in variants if v in ("rmps_replan", "rmps_no_replan")]
        for idx_sv, sv in enumerate(shield_vars):
            int_ys = []
            for sc in scales_sorted:
                row = by_key.get((lv, round(sc, 6), sv))
                int_ys.append(float(row["intervention_rate_pct"]) if row else 0.0)
            offset = (idx_sv - 0.5 * (len(shield_vars) - 1)) * bar_w
            bar_color = "#5E3C99" if sv == "rmps_replan" else "#0072B2"
            ax_int.bar(
                x_arr + offset,
                int_ys,
                width=bar_w,
                color=bar_color,
                alpha=0.25,
                edgecolor=bar_color,
                linewidth=0.9,
                zorder=1,
            )

        # Collision threshold line (0 m clearance) and light danger shading
        valid_clr = [v for v in all_clr_vals if np.isfinite(v)]
        y_lo = min(-0.25, min(valid_clr) - 0.15) if valid_clr else -1.5
        y_hi = max(0.35, max(valid_clr) + 0.15) if valid_clr else 0.5
        ax_clr.set_ylim(y_lo, y_hi)
        ax_clr.axhline(0.0, color="#B2182B", linestyle="--", linewidth=1.2, zorder=2)
        ax_clr.axhspan(y_lo, 0.0, color="#B2182B", alpha=0.06, zorder=0)

        ax_int.set_ylim(0.0, 105.0)
        ax_int.set_yticks([0, 25, 50, 75, 100])
        ax_int.tick_params(axis="y", colors="#5E3C99")
        if col_idx == n_cols - 1:
            ax_int.set_ylabel("NMPC Intervention Rate (%)", color="#5E3C99")
        else:
            ax_int.set_yticklabels([])

        if col_idx == 0:
            ax_clr.set_ylabel("Mean Min Clearance [m]")

        ax_clr.set_xlabel(r"Perturbation Magnitude Scale $\alpha$")
        ax_clr.set_xticks(x_arr)
        ax_clr.set_xticklabels([f"{s:g}" for s in scales_sorted])

    # Shared clean legend at bottom
    legend_handles: list[Any] = []
    for var in variants:
        st = VARIANT_STYLE.get(var, {"label": var, "color": "gray", "marker": "o"})
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=st["color"],
                marker=st["marker"],
                linewidth=2.2,
                markersize=6.5,
                label=st["label"],
            )
        )
    legend_handles.extend(
        [
            Line2D([0], [0], color="#333333", linestyle="-", linewidth=2.0, label="Row 2: Collision (%)"),
            Line2D(
                [0],
                [0],
                color="#333333",
                linestyle="--",
                marker="^",
                markerfacecolor="white",
                linewidth=1.7,
                label="Row 2: Hitch Violation (%)",
            ),
            Line2D([0], [0], color="#B2182B", linestyle="--", linewidth=1.3, label="Row 3: 0 m Collision Threshold"),
            Patch(facecolor="#5E3C99", edgecolor="#5E3C99", alpha=0.28, label="Row 3: NMPC Intervention (%)"),
        ]
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(4, len(legend_handles)),
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Perturbation-Magnitude Robustness Sweep — AccTractorTrailer2d (Baseline vs. RMPS)",
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.965))

    saved = _save_multi_format(fig, out_dir / "perturbation_sweep_main")
    plt.close(fig)
    return saved


def plot_safety_vs_success_tradeoff(
    summary_rows: list[dict[str, Any]],
    levels: list[int],
    scales: list[float],
    variants: list[str],
    out_dir: Path,
) -> list[Path]:
    """
    Generate `safety_vs_success_tradeoff.{png,pdf,svg}`:
    Collision Rate (%) vs. Safe Success Rate (%) curves parameterized by perturbation
    scale `alpha` across Levels 1-3, highlighting each method's robustness breakdown.
    """
    _configure_pub_style()
    n_cols = max(1, len(levels))
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(5.1 * n_cols, 4.7),
        sharey=True,
        squeeze=False,
    )

    by_key = {(int(r["level"]), round(float(r["scale"]), 6), str(r["variant"])): r for r in summary_rows}
    scales_sorted = sorted(float(s) for s in scales)

    for col_idx, lv in enumerate(levels):
        ax = axes[0, col_idx]
        _style_axis(ax)
        meta = LEVEL_METADATA.get(lv, {"title": f"Level {lv}", "subtitle": ""})

        # Highlight ideal low-collision, high-success region
        ax.axvspan(-4.0, 5.0, color="#009E73", alpha=0.07, zorder=0)
        ax.scatter(
            [0.0],
            [100.0],
            marker="*",
            s=190,
            color="#F0E442",
            edgecolor="#333333",
            linewidth=0.9,
            zorder=6,
            label="Ideal (0% Col, 100% Succ)" if col_idx == 0 else None,
        )

        for v_idx, var in enumerate(variants):
            st = VARIANT_STYLE.get(var, {"label": var, "color": "gray", "marker": "o", "zorder": 3})
            xs, ys, valid_scales = [], [], []
            for sc in scales_sorted:
                row = by_key.get((lv, round(sc, 6), var))
                if row is not None:
                    xs.append(float(row["collision_rate_pct"]))
                    ys.append(float(row["safe_success_rate_pct"]))
                    valid_scales.append(sc)

            if not xs:
                continue

            xs_np = np.asarray(xs, dtype=np.float64)
            ys_np = np.asarray(ys, dtype=np.float64)
            sizes = np.linspace(42.0, 96.0, len(xs_np))

            ax.plot(
                xs_np,
                ys_np,
                color=st["color"],
                linewidth=2.0,
                alpha=0.85,
                zorder=st["zorder"],
            )
            ax.scatter(
                xs_np,
                ys_np,
                s=sizes,
                color=st["color"],
                marker=st["marker"],
                edgecolor="white",
                linewidth=0.8,
                label=st["label"] if col_idx == 0 else None,
                zorder=st["zorder"] + 1,
            )

            # Directional arrows indicating increasing perturbation scale alpha
            for k in range(len(xs_np) - 1):
                dx = xs_np[k + 1] - xs_np[k]
                dy = ys_np[k + 1] - ys_np[k]
                if np.hypot(dx, dy) >= 6.0:
                    ax.annotate(
                        "",
                        xy=(xs_np[k] + 0.62 * dx, ys_np[k] + 0.62 * dy),
                        xytext=(xs_np[k] + 0.35 * dx, ys_np[k] + 0.35 * dy),
                        arrowprops={
                            "arrowstyle": "->",
                            "color": st["color"],
                            "lw": 1.6,
                            "shrinkA": 0,
                            "shrinkB": 0,
                        },
                        zorder=st["zorder"] + 2,
                    )

            # Label key scale points cleanly, merging coinciding (x, y) points to prevent text overlap
            label_indices = (
                list(range(len(valid_scales)))
                if len(valid_scales) <= 4
                else sorted({0, len(valid_scales) // 2, len(valid_scales) - 1})
            )
            grouped_pts: dict[tuple[float, float], list[float]] = {}
            for k in label_indices:
                pt_key = (round(float(xs_np[k]), 1), round(float(ys_np[k]), 1))
                grouped_pts.setdefault(pt_key, []).append(valid_scales[k])

            y_off = 9 if v_idx % 2 == 0 else -13
            x_off = 7 if v_idx % 2 == 0 else 7
            for (px_val, py_val), sc_list in grouped_pts.items():
                lbl_vals = ", ".join(f"{s:g}" for s in sc_list)
                lbl_txt = rf"$\alpha={lbl_vals}$"
                dx_pt = x_off if px_val < 75 else (-38 - 10 * (len(sc_list) - 1))
                dy_pt = y_off if py_val < 88 else -14
                if py_val <= 2.0 and dy_pt < 0:
                    dy_pt = 9 if v_idx % 2 == 0 else -12
                ax.annotate(
                    lbl_txt,
                    xy=(px_val, py_val),
                    xytext=(dx_pt, dy_pt),
                    textcoords="offset points",
                    fontsize=7.8,
                    color=st["color"],
                    fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.78},
                    zorder=st["zorder"] + 3,
                )

        ax.set_xlim(-5.0, 106.0)
        ax.set_ylim(-5.0, 108.0)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel("Collision Rate (%) [Lower is better]")
        if col_idx == 0:
            ax.set_ylabel("Safe Success Rate (%) [Higher is better]")
        title_txt = f"{meta['title']}\n({meta['subtitle']})" if meta["subtitle"] else meta["title"]
        ax.set_title(title_txt, fontweight="bold", pad=8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(handles)),
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        r"Safety vs. Task Success Tradeoff Across Perturbation Scale $\alpha$ (Marker Size Increases with $\alpha$)",
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))

    saved = _save_multi_format(fig, out_dir / "safety_vs_success_tradeoff")
    plt.close(fig)
    return saved


def generate_all_plots(payload: dict[str, Any], out_dir: Path) -> list[Path]:
    """Generate all publication-quality sweep figures from a loaded/computed sweep payload."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = payload["summary"]
    meta = payload.get("metadata", {})

    levels = [int(lv) for lv in meta.get("levels", sorted({int(r["level"]) for r in summary_rows}))]
    scales = [float(sc) for sc in meta.get("scales", sorted({float(r["scale"]) for r in summary_rows}))]
    variants = [str(v) for v in meta.get("variants", list(dict.fromkeys(r["variant"] for r in summary_rows)))]

    saved_files: list[Path] = []
    saved_files.extend(plot_perturbation_sweep_main(summary_rows, levels, scales, variants, out_dir))
    saved_files.extend(plot_safety_vs_success_tradeoff(summary_rows, levels, scales, variants, out_dir))
    return saved_files


# ---------------------------------------------------------------------------
# Main Sweep Runner
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perturbation-magnitude sweep benchmark & visualization suite for AccTractorTrailer2d."
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="Dimensionless perturbation magnitude scaling factors alpha (default: 0.0 0.2 0.4 0.6 0.8 1.0)",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="Perturbation levels to sweep (default: 1 2 3)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        choices=list(VARIANTS),
        default=["baseline", "rmps_replan"],
        help="Controller variants to compare (default: baseline rmps_replan)",
    )
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds per (level, scale) point")
    parser.add_argument(
        "--seed-list",
        type=int,
        nargs="+",
        default=None,
        help="Explicit planner/episode seed list (overrides --num-seeds)",
    )
    parser.add_argument("--Nsample", type=int, default=800, help="Safe-MPD diffusion sample count")
    parser.add_argument("--Ndiffuse", type=int, default=25, help="Safe-MPD diffusion steps")
    parser.add_argument("--horizon", type=int, default=50, help="Safe-MPD planning horizon H")
    parser.add_argument("--dt", type=float, default=0.25, help="Simulation timestep [s]")
    parser.add_argument("--max-steps", type=int, default=150, help="Max closed-loop steps per episode")
    parser.add_argument("--max-replans", type=int, default=5, help="Max online Safe-MPD replans per episode")
    parser.add_argument(
        "--tube-mode",
        type=str,
        choices=["calibrate", "scale"],
        default="calibrate",
        help="Tube margin strategy: 'calibrate' per (level, alpha) with disk caching, or linearly 'scale' alpha=1.0 tubes",
    )
    parser.add_argument("--tube-N-mc", type=int, default=1000, help="Monte Carlo rollouts for offline tube calibration")
    parser.add_argument("--recalibrate", action="store_true", help="Force recalibration of cached tubes")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/perturbation_sweep",
        help="Output directory for JSON, CSV, and publication figures",
    )
    parser.add_argument(
        "--plot-only",
        type=str,
        default=None,
        help="Path to an existing sweep_results.json to regenerate plots without running simulations",
    )
    args = parser.parse_args()

    # Plot-only fast regeneration mode
    if args.plot_only is not None:
        json_path = Path(args.plot_only)
        if not json_path.is_absolute():
            json_path = (REPO_ROOT / json_path).resolve()
        with open(json_path, "r") as f:
            payload = json.load(f)
        target_out_dir = Path(args.out_dir)
        if not target_out_dir.is_absolute():
            # Default to the JSON file's parent directory if --out-dir wasn't customized
            target_out_dir = json_path.parent if args.out_dir == "results/perturbation_sweep" else (REPO_ROOT / target_out_dir)
        saved = generate_all_plots(payload, target_out_dir)
        print(f"Regenerated {len(saved)} figure files from {json_path}:")
        for p in saved:
            print(f"  - {p}")
        return

    levels = list(dict.fromkeys(int(lv) for lv in args.levels))
    scales = sorted(dict.fromkeys(float(sc) for sc in args.scales))
    variants = list(dict.fromkeys(args.variants))
    need_baseline = "baseline" in variants
    need_tube = any(v != "baseline" for v in variants)

    plan_seeds = list(args.seed_list) if args.seed_list is not None else [42 + i for i in range(args.num_seeds)]
    n_seeds = len(plan_seeds)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tube_dir = REPO_ROOT / "results" / "tubes"

    print("=" * 158)
    print(
        f"Perturbation-Magnitude Sweep | levels={levels} | scales={scales} | variants={variants} | "
        f"seeds={plan_seeds} | Nsample={args.Nsample}, Ndiffuse={args.Ndiffuse}, tube_mode={args.tube_mode}"
    )
    print("=" * 158)

    # 1. Calibrate / load scaled disturbance tubes
    tubes: dict[tuple[int, float], dict[str, Any]] = {}
    tube_log: list[dict[str, Any]] = []
    if need_tube:
        tubes = ensure_scaled_tubes(
            levels=levels,
            scales=scales,
            dt=args.dt,
            tube_dir=tube_dir,
            tube_mode=args.tube_mode,
            n_mc=args.tube_N_mc,
            recalibrate=args.recalibrate,
        )
        print("Calibrated / Loaded Tube Margins:")
        print(f"  {'Level':<6} | {'Scale':>6} | {'max r_tube [m]':>14} | {'max b_hitch [deg]':>17}")
        print("  " + "-" * 52)
        for lv in levels:
            for sc in scales:
                t_info = tubes[(lv, sc)]
                max_r = float(np.max(t_info["r_tube"]))
                max_b_deg = float(np.rad2deg(float(np.max(t_info["b_hitch"]))))
                tube_log.append(
                    {
                        "level": lv,
                        "scale": sc,
                        "max_r_tube_m": round(max_r, 5),
                        "max_b_hitch_deg": round(max_b_deg, 3),
                    }
                )
                print(f"  L{lv:<5} | {sc:>6.2f} | {max_r:>14.4f} | {max_b_deg:>17.2f}")
        print("-" * 158, flush=True)

    # 2. Precompute unshielded baseline nominal plans ONCE per seed (shared across all levels & scales)
    baseline_plans: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    if need_baseline:
        clear_jit_cache()
        for s_idx, p_seed in enumerate(plan_seeds):
            env_b = build_scaled_benchmark_env(
                level=1,
                scale=0.0,
                perturbation_seed=p_seed,
                dt=args.dt,
                H=args.horizon,
                enable_robust_shielding=False,
            )
            cfg_b = MBDConfig(
                env_name="acc_tt2d",
                case="navigation",
                seed=p_seed,
                Nsample=args.Nsample,
                Hsample=args.horizon,
                Ndiffuse=args.Ndiffuse,
                dt=args.dt,
                enable_robust_shielding=False,
            )
            baseline_plans[s_idx] = SafeMPDPlanner(cfg_b, env_b).plan(np.asarray(env_b.x0), np.asarray(env_b.xg))

    # Cache nominal (scale=0.0, enable_robust_shielding=True) robust plans once across Levels 1-3
    robust_zero_plans: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # 3. Sweep across (level, scale, variant, seed)
    header = (
        f"{'Lv':<3} | {'Alpha':>5} | {'Variant':<14} | {'r_max':>6} | {'b_deg':>6} | "
        f"{'Safe%':>6} | {'Goal%':>6} | {'Col%':>6} | {'Hitch%':>6} | "
        f"{'MeanClr':>8} | {'CVaRClr':>8} | {'PosErr':>7} | {'Intv%':>6} | {'Rpln':>5} | {'Step[ms]':>8}"
    )
    print(header)
    print("-" * 158, flush=True)

    summary_rows: list[dict[str, Any]] = []
    episode_records: list[dict[str, Any]] = []

    for lv in levels:
        for sc in scales:
            p_info = scaled_perturbation_params(lv, sc)
            eff_lv = int(p_info["effective_level"])
            tube = tubes.get((lv, sc))
            r_tube_arr = np.asarray(tube["r_tube"], dtype=np.float32) if tube is not None else np.zeros(1, dtype=np.float32)
            b_hitch_arr = np.asarray(tube["b_hitch"], dtype=np.float32) if tube is not None else np.zeros(1, dtype=np.float32)
            r_tuple = tuple(float(v) for v in r_tube_arr.reshape(-1))
            b_tuple = tuple(float(v) for v in b_hitch_arr.reshape(-1))
            max_r_m = float(np.max(r_tube_arr))
            max_b_deg = float(np.rad2deg(float(np.max(b_hitch_arr))))

            # Precompute tube-tightened nominal plans once per (level, scale) across all tube-based variants
            # (never reuse baseline plans at scale=0.0 because `enable_robust_shielding=True` uses exact dt rollout)
            tube_plans: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            if need_tube:
                if eff_lv == 0 and max_r_m <= 1e-6 and max_b_deg <= 1e-6 and robust_zero_plans:
                    for s_idx in range(n_seeds):
                        tube_plans[s_idx] = robust_zero_plans[s_idx]
                else:
                    clear_jit_cache()
                    for s_idx, p_seed in enumerate(plan_seeds):
                        ep_seed = p_seed if args.seed_list is not None else (100 + 17 * s_idx + 1000 * lv)
                        env_t = build_scaled_benchmark_env(
                            level=lv,
                            scale=sc,
                            perturbation_seed=ep_seed,
                            dt=args.dt,
                            H=args.horizon,
                            enable_robust_shielding=True,
                            r_tube=r_tube_arr,
                            b_hitch=b_hitch_arr,
                        )
                        cfg_t = MBDConfig(
                            env_name="acc_tt2d",
                            case="navigation",
                            seed=p_seed,
                            Nsample=args.Nsample,
                            Hsample=args.horizon,
                            Ndiffuse=args.Ndiffuse,
                            dt=args.dt,
                            enable_robust_shielding=True,
                            tube_obstacle_margin=r_tuple,
                            tube_hitch_margin=b_tuple,
                        )
                        plan_t = SafeMPDPlanner(cfg_t, env_t).plan(
                            np.asarray(env_t.x0), np.asarray(env_t.xg)
                        )
                        tube_plans[s_idx] = plan_t
                        if eff_lv == 0 and max_r_m <= 1e-6 and max_b_deg <= 1e-6:
                            robust_zero_plans[s_idx] = plan_t

            for variant in variants:
                use_tube = variant != "baseline"
                use_nmpc = variant in ("rmps_no_replan", "rmps_replan")
                use_replan = variant == "rmps_replan"

                r_arg = r_tube_arr if use_tube else 0.0
                b_arg = b_hitch_arr if use_tube else 0.0

                clear_jit_cache()
                ep_results: list[dict[str, Any]] = []

                for s_idx, p_seed in enumerate(plan_seeds):
                    ep_seed = p_seed if args.seed_list is not None else (100 + 17 * s_idx + 1000 * lv)
                    env = build_scaled_benchmark_env(
                        level=lv,
                        scale=sc,
                        perturbation_seed=ep_seed,
                        dt=args.dt,
                        H=args.horizon,
                        enable_robust_shielding=use_tube,
                        r_tube=r_arg,
                        b_hitch=b_arg,
                    )
                    cfg = MBDConfig(
                        env_name="acc_tt2d",
                        case="navigation",
                        seed=p_seed,
                        Nsample=args.Nsample,
                        Hsample=args.horizon,
                        Ndiffuse=args.Ndiffuse,
                        dt=args.dt,
                        render=False,
                        show_animation=False,
                        save_animation=False,
                        perturbation_level=eff_lv,
                        perturbation_seed=ep_seed,
                        control_noise_std=p_info["control_noise_std"],
                        param_uncertainty_std=p_info["param_uncertainty_std"],
                        non_gaussian_dist=p_info["non_gaussian_dist"],
                        non_gaussian_df=p_info["non_gaussian_df"],
                        state_disturbance_scale=p_info["state_disturbance_scale"],
                        enable_robust_shielding=use_tube,
                        tube_obstacle_margin=r_tuple if use_tube else 0.0,
                        tube_hitch_margin=b_tuple if use_tube else 0.0,
                    )
                    planner = SafeMPDPlanner(cfg, env)
                    nmpc_backup = RobustNMPCBackup(env, T_bak=12, v_eps=0.05)
                    init_p = tube_plans[s_idx] if use_tube else baseline_plans[s_idx]

                    res = run_rmps_episode(
                        env=env,
                        planner=planner,
                        nmpc_backup=nmpc_backup,
                        tube_data=tube if use_tube else None,
                        max_steps=args.max_steps,
                        max_replans=args.max_replans,
                        enable_replanning=use_replan,
                        enable_nmpc_shield=use_nmpc,
                        use_delay_compensation=use_nmpc,
                        episode_seed=ep_seed,
                        initial_plan=init_p,
                        save_dir=None,
                        save_plot=False,
                    )
                    outcome = classify_episode_outcome(res)
                    safe_ok = bool(outcome == "safe_goal")
                    step_ms = (
                        float(np.mean(res["computation_times"]) * 1000.0)
                        if len(res["computation_times"]) > 0
                        else 0.0
                    )
                    res["outcome"] = outcome
                    res["safe_success"] = safe_ok
                    ep_results.append(res)

                    episode_records.append(
                        {
                            "level": lv,
                            "effective_level": eff_lv,
                            "scale": sc,
                            "variant": variant,
                            "plan_seed": p_seed,
                            "episode_seed": ep_seed,
                            "outcome": outcome,
                            "safe_success": safe_ok,
                            "goal_reached": bool(res["goal_reached"]),
                            "collision": bool(res["collision"]),
                            "hitch_violation": bool(res["hitch_violation"]),
                            "termination_reason": str(res.get("termination_reason", "timeout")),
                            "min_clearance_m": float(res["min_clearance"]),
                            "final_pos_error_m": float(res["final_pos_error"]),
                            "intervention_rate_pct": float(100.0 * res["intervention_rate"]),
                            "num_replans": int(res["num_replans"]),
                            "avg_step_runtime_ms": round(step_ms, 3),
                            "max_r_tube_m": round(max_r_m if use_tube else 0.0, 5),
                            "max_b_hitch_deg": round(max_b_deg if use_tube else 0.0, 3),
                        }
                    )

                safe_flags = [bool(r["safe_success"]) for r in ep_results]
                goal_flags = [bool(r["goal_reached"]) for r in ep_results]
                col_flags = [bool(r["collision"]) for r in ep_results]
                hitch_flags = [bool(r["hitch_violation"]) for r in ep_results]
                clr_vals = [float(r["min_clearance"]) for r in ep_results]
                err_vals = [float(r["final_pos_error"]) for r in ep_results]
                intv_vals = [float(100.0 * r["intervention_rate"]) for r in ep_results]
                rpln_vals = [float(r["num_replans"]) for r in ep_results]
                all_step_ms = [float(t) * 1000.0 for r in ep_results for t in r["computation_times"]]

                safe_pct = 100.0 * float(np.mean(safe_flags))
                goal_pct = 100.0 * float(np.mean(goal_flags))
                col_pct = 100.0 * float(np.mean(col_flags))
                hitch_pct = 100.0 * float(np.mean(hitch_flags))
                mean_clr, std_clr, se_clr = _mean_std_se(clr_vals)
                cvar_clr = cvar_lower(clr_vals, alpha=0.01)
                mean_err, std_err, _ = _mean_std_se(err_vals)
                mean_intv, std_intv, se_intv = _mean_std_se(intv_vals)
                avg_rpln = float(np.mean(rpln_vals))
                avg_ms = float(np.mean(all_step_ms)) if all_step_ms else 0.0

                row = {
                    "level": lv,
                    "effective_level": eff_lv,
                    "scale": sc,
                    "variant": variant,
                    "num_seeds": n_seeds,
                    "max_r_tube_m": round(max_r_m if use_tube else 0.0, 5),
                    "max_b_hitch_deg": round(max_b_deg if use_tube else 0.0, 3),
                    "safe_success_rate_pct": round(safe_pct, 2),
                    "safe_success_se_pct": round(_bernoulli_se_pct(safe_flags), 2),
                    "goal_reached_pct": round(goal_pct, 2),
                    "goal_reached_se_pct": round(_bernoulli_se_pct(goal_flags), 2),
                    "collision_rate_pct": round(col_pct, 2),
                    "collision_se_pct": round(_bernoulli_se_pct(col_flags), 2),
                    "hitch_violation_rate_pct": round(hitch_pct, 2),
                    "hitch_violation_se_pct": round(_bernoulli_se_pct(hitch_flags), 2),
                    "mean_min_clearance_m": round(mean_clr, 4),
                    "std_min_clearance_m": round(std_clr, 4),
                    "se_min_clearance_m": round(se_clr, 4),
                    "cvar_min_clearance_m": round(cvar_clr, 4),
                    "mean_final_pos_error_m": round(mean_err, 4),
                    "std_final_pos_error_m": round(std_err, 4),
                    "intervention_rate_pct": round(mean_intv, 2),
                    "intervention_std_pct": round(std_intv, 2),
                    "intervention_se_pct": round(se_intv, 2),
                    "avg_replans": round(avg_rpln, 2),
                    "avg_step_runtime_ms": round(avg_ms, 2),
                    "outcome_safe_goal": sum(1 for r in ep_results if r["outcome"] == "safe_goal"),
                    "fail_collision": sum(1 for r in ep_results if r["outcome"] == "collision"),
                    "fail_jackknife": sum(1 for r in ep_results if r["outcome"] == "jackknife"),
                    "fail_collision_and_jackknife": sum(1 for r in ep_results if r["outcome"] == "collision+jackknife"),
                    "fail_timeout": sum(1 for r in ep_results if r["outcome"] == "timeout"),
                    "fail_max_replans": sum(1 for r in ep_results if r["outcome"] == "max_replans"),
                    "fail_stopped_no_replan": sum(1 for r in ep_results if r["outcome"] == "stopped_no_replan"),
                }
                summary_rows.append(row)

                print(
                    f"L{lv:<2} | {sc:>5.2f} | {variant:<14} | {row['max_r_tube_m']:>6.3f} | {row['max_b_hitch_deg']:>6.2f} | "
                    f"{safe_pct:>6.1f} | {goal_pct:>6.1f} | {col_pct:>6.1f} | {hitch_pct:>6.1f} | "
                    f"{mean_clr:>8.3f} | {cvar_clr:>8.3f} | {mean_err:>7.2f} | {mean_intv:>6.1f} | {avg_rpln:>5.2f} | {avg_ms:>8.1f}",
                    flush=True,
                )

    print("=" * 158)

    # 4. Save structured JSON and CSV results
    payload: dict[str, Any] = {
        "metadata": {
            "levels": levels,
            "scales": scales,
            "variants": variants,
            "plan_seeds": plan_seeds,
            "num_seeds": n_seeds,
            "Nsample": args.Nsample,
            "Ndiffuse": args.Ndiffuse,
            "horizon": args.horizon,
            "dt": args.dt,
            "max_steps": args.max_steps,
            "max_replans": args.max_replans,
            "tube_mode": args.tube_mode,
            "tube_N_mc": args.tube_N_mc,
            "base_control_noise_std": BASE_CONTROL_NOISE_STD,
            "base_param_uncertainty_std": BASE_PARAM_UNCERTAINTY_STD,
            "base_state_disturbance_scale": BASE_STATE_DISTURBANCE_SCALE,
            "non_gaussian_dist": BASE_NON_GAUSSIAN_DIST,
            "non_gaussian_df": BASE_NON_GAUSSIAN_DF,
        },
        "tubes": tube_log,
        "summary": summary_rows,
        "episodes": episode_records,
    }

    json_out = out_dir / "sweep_results.json"
    with open(json_out, "w") as f:
        json.dump(payload, f, indent=2)

    csv_out = out_dir / "sweep_summary.csv"
    if summary_rows:
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    # 5. Generate publication-quality figures (.png @ 300 DPI, .pdf, .svg)
    saved_figs = generate_all_plots(payload, out_dir)
    print(f"Saved perturbation sweep results to {out_dir}")
    print(f"  JSON results : {json_out}")
    print(f"  CSV summary  : {csv_out}")
    for p in saved_figs:
        print(f"  Figure       : {p}")


if __name__ == "__main__":
    main()
