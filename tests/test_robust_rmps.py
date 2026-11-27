"""
Comparative Evaluation Benchmark for Robust Safe-MPD (Phase 5).

Compares 4 variants across Perturbation Levels 0, 1, 2, and 3 over multiple random seeds:
  1. `baseline`:       Original Safe-MPD (open-loop perturbed execution)
  2. `tube_only`:      Tube-Safe MPD (tightened constraints, no online NMPC)
  3. `rmps_no_replan`: Tube-Safe MPD + Online NMPC Shield (stops safely on intervention)
  4. `rmps_replan`:    Tube-Safe MPD + Online NMPC Shield + Replanning from stopped state

Usage examples:
    ./.venv/bin/python tests/test_robust_rmps.py --seeds 5
    ./.venv/bin/python tests/test_robust_rmps.py --seeds 10 --Nsample 20000 --Ndiffuse 100
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
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

# Ensure repo root is on sys.path
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


VARIANTS = ("baseline", "tube_only", "rmps_no_replan", "rmps_replan")


def build_benchmark_env(
    perturbation_level: int,
    perturbation_seed: int,
    dt: float = 0.25,
    H: int = 50,
    enable_robust_shielding: bool = False,
    r_tube: float | np.ndarray = 0.0,
    b_hitch: float | np.ndarray = 0.0,
) -> Any:
    """Build the custom navigation environment from tests/test_mbd.py for AccTractorTrailer2d."""
    env_config = Env(width=32.0, height=32.0, case="navigation")
    env = mbd.envs.get_env(
        "acc_tt2d",
        case="navigation",
        env_config=env_config,
        dt=dt,
        H=H,
        perturbation_level=perturbation_level,
        perturbation_seed=perturbation_seed,
        control_noise_std=(0.05, 0.05),
        param_uncertainty_std=0.10,
        non_gaussian_dist="student_t",
        non_gaussian_df=2.5,
        state_disturbance_scale=0.05,
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


def ensure_tubes_loaded(levels: list[int], dt: float, recalibrate: bool = False) -> dict[int, dict[str, Any]]:
    """Load precomputed tubes for requested levels, calibrating automatically if missing."""
    tubes: dict[int, dict[str, Any]] = {}
    env_cal = None
    for lv in levels:
        t_path = default_tube_path(level=lv, model_name="acc_tt2d", base_dir=REPO_ROOT / "results/tubes")
        if recalibrate or not t_path.exists():
            if env_cal is None:
                env_cal = mbd.envs.get_env("acc_tt2d", case="navigation", dt=dt, H=50)
            tubes[lv] = calibrate_tubes(env_cal, level=lv, N_mc=1000, T_horizon=25, seed=42, save_path=t_path)
        else:
            tubes[lv] = load_tubes(t_path)
    return tubes


def cvar_upper(values: list[float], alpha: float = 0.01) -> float:
    """Upper-tail alpha-CVaR (worst-case mean of top ceil(alpha*N) values; for position error)."""
    arr = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    k = max(1, int(np.ceil(alpha * arr.size)))
    return float(np.mean(arr[:k]))


def cvar_lower(values: list[float], alpha: float = 0.01) -> float:
    """Lower-tail alpha-CVaR (worst-case mean of bottom ceil(alpha*N) values; for min clearance)."""
    arr = np.sort(np.asarray(values, dtype=np.float64))
    k = max(1, int(np.ceil(alpha * arr.size)))
    return float(np.mean(arr[:k]))


def classify_episode_outcome(res: dict[str, Any]) -> str:
    """Classify an episode into a single primary outcome / failure type."""
    col = bool(res["collision"])
    hitch = bool(res["hitch_violation"])
    if col and hitch:
        return "collision+jackknife"
    if col:
        return "collision"
    if hitch:
        return "jackknife"
    if bool(res["goal_reached"]):
        return "safe_goal"
    term = str(res.get("termination_reason", "timeout"))
    if term == "max_replans":
        return "max_replans"
    if term == "stopped_no_replan":
        return "stopped_no_replan"
    return "timeout"


def plot_all_trajectories_overview(
    env_ref: Any,
    trajectories_by_key: dict[tuple[int, str], list[dict[str, Any]]],
    levels: list[int],
    variants: tuple[str, ...],
    out_path: Path,
) -> None:
    """
    Plot a (len(levels) x len(variants)) grid overlaying all seeds' trajectories
    on the obstacle map, color-coded by episode outcome / failure type.
    """
    n_rows = max(1, len(levels))
    n_cols = max(1, len(variants))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.8 * n_rows), squeeze=False)

    outcome_style = {
        "safe_goal": ("#2ca02c", "o", "Safe Goal"),
        "collision": ("#d62728", "X", "Collision"),
        "jackknife": ("#9467bd", "^", "Jackknife"),
        "collision+jackknife": ("#8c1d18", "X", "Col + Jackknife"),
        "max_replans": ("#ff7f0e", "s", "Safe Max-Replans"),
        "stopped_no_replan": ("#17becf", "D", "Safe Stopped (No Replan)"),
        "timeout": ("#1f77b4", "s", "Safe Timeout"),
    }

    obs_c = np.asarray(env_ref.obs_circles, dtype=np.float64) if hasattr(env_ref, "obs_circles") else np.zeros((0, 3))
    obs_r = np.asarray(env_ref.obs_rectangles, dtype=np.float64) if hasattr(env_ref, "obs_rectangles") else np.zeros((0, 5))
    xr, yr = env_ref.env.get_plot_limits() if hasattr(env_ref, "env") else ((-16.0, 16.0), (-16.0, 16.0))
    x0 = np.asarray(env_ref.x0, dtype=np.float64)
    xg = np.asarray(env_ref.xg, dtype=np.float64)

    for r_idx, lv in enumerate(levels):
        for c_idx, var in enumerate(variants):
            ax = axes[r_idx, c_idx]
            for c in obs_c:
                if c[2] > 1e-6:
                    ax.add_patch(Circle((c[0], c[1]), c[2], color="dimgray", alpha=0.55))
            for r in obs_r:
                rx, ry, rw, rh, rang = r
                if rw > 1e-6 and rh > 1e-6:
                    rect = Rectangle((-rw / 2, -rh / 2), rw, rh, facecolor="gray", edgecolor="black", alpha=0.6)
                    rect.set_transform(Affine2D().rotate(rang).translate(rx, ry) + ax.transData)
                    ax.add_patch(rect)

            eps = trajectories_by_key.get((lv, var), [])
            n_tot = max(1, len(eps))
            n_succ = sum(1 for e in eps if e["outcome"] == "safe_goal")
            n_col = sum(1 for e in eps if e["collision"])
            n_hitch = sum(1 for e in eps if e["hitch_violation"])

            for e in eps:
                xs = e["states"]
                out_k = e["outcome"]
                color, marker, _ = outcome_style.get(out_k, ("gray", "o", out_k))
                ax.plot(xs[:, 0], xs[:, 1], "-", color=color, linewidth=1.7, alpha=0.78)
                ax.scatter([xs[-1, 0]], [xs[-1, 1]], color=color, marker=marker, s=42, edgecolor="black", linewidth=0.5, zorder=5)

            ax.scatter([x0[0]], [x0[1]], color="blue", s=65, edgecolor="black", zorder=6)
            ax.scatter([xg[0]], [xg[1]], color="gold", marker="*", s=130, edgecolor="black", zorder=6)
            ax.set_xlim(xr)
            ax.set_ylim(yr)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.25)
            ax.set_title(
                f"L{lv} | {var} (Succ {100.0 * n_succ / n_tot:.0f}%, Col {100.0 * n_col / n_tot:.0f}%, Hitch {100.0 * n_hitch / n_tot:.0f}%)",
                fontsize=9.5,
                fontweight="bold",
            )
            if r_idx == n_rows - 1:
                ax.set_xlabel("x [m]")
            if c_idx == 0:
                ax.set_ylabel("y [m]")

    legend_handles = [
        Line2D([0], [0], color=col, marker=mk, markersize=6, linewidth=2.0, label=lbl)
        for col, mk, lbl in outcome_style.values()
    ] + [
        Line2D([0], [0], color="w", marker="o", markerfacecolor="blue", markeredgecolor="black", markersize=7, label="Start"),
        Line2D([0], [0], color="w", marker="*", markerfacecolor="gold", markeredgecolor="black", markersize=10, label="Goal"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("All-Seeds & All-Levels Trajectory Overview — AccTractorTrailer2d", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.97))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_benchmark_summary(
    rows: list[dict[str, Any]],
    levels: list[int],
    out_path: Path,
    variants: tuple[str, ...] = VARIANTS,
) -> None:
    """Save grouped bar charts comparing selected variants across Perturbation Levels 0-3."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    metrics = [
        ("collision_rate_pct", "Collision Rate (%) [Lower is better]", (0, 105)),
        ("success_rate_pct", "Task Success Rate (%) [Higher is better]", (0, 105)),
        ("cvar_min_clearance_m", "1%-CVaR / Worst Min Clearance [m]", None),
        ("avg_step_runtime_ms", "Avg Online Step Runtime [ms]", None),
    ]
    colors = {
        "baseline": "#d62728",
        "tube_only": "#ff7f0e",
        "rmps_no_replan": "#1f77b4",
        "rmps_replan": "#2ca02c",
    }

    x = np.arange(len(levels))
    n_var = max(1, len(variants))
    width = min(0.28, 0.76 / n_var)

    for ax, (m_key, title, ylim) in zip(axes.flat, metrics):
        for i, var in enumerate(variants):
            vals = []
            for lv in levels:
                match = [r for r in rows if r["level"] == lv and r["variant"] == var]
                vals.append(match[0][m_key] if match else 0.0)
            offset = (i - 0.5 * (n_var - 1)) * width
            ax.bar(x + offset, vals, width=width, label=var, color=colors[var], alpha=0.88)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Level {lv}" for lv in levels])
        if ylim is not None:
            ax.set_ylim(*ylim)
        if m_key == "cvar_min_clearance_m":
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Robust Safe-MPD (RMPS) Comparative Benchmark — AccTractorTrailer2d", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparative RMPS benchmark across Levels 0-3.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds per level (ignored if --seed-list is set)")
    parser.add_argument("--seed-list", type=int, nargs="+", default=None, help="Explicit planner/episode seeds (e.g. --seed-list 9123 42 43 44 45)")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3], help="Perturbation levels to evaluate")
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        choices=list(VARIANTS),
        default=list(VARIANTS),
        help="Variants to evaluate (default: all 4)",
    )
    parser.add_argument("--Nsample", type=int, default=800, help="Diffusion sample count (increase on GPU)")
    parser.add_argument("--Ndiffuse", type=int, default=25, help="Diffusion steps (increase on GPU)")
    parser.add_argument("--horizon", type=int, default=50, help="Safe-MPD planning horizon")
    parser.add_argument("--dt", type=float, default=0.25, help="Simulation timestep [s]")
    parser.add_argument("--max-steps", type=int, default=150, help="Max closed-loop steps per episode")
    parser.add_argument("--max-replans", type=int, default=5, help="Max replans per episode")
    parser.add_argument("--out-dir", type=str, default="results/rmps_benchmark", help="Output directory")
    parser.add_argument(
        "--save-episodes",
        type=str,
        choices=["sample", "per_level", "all"],
        default="all",
        help="Which episode trajectories to save for render.py: sample (seed 0 of max level), per_level (seed 0 of each level), or all (every level/variant/seed)",
    )
    parser.add_argument(
        "--no-episode-plots",
        action="store_true",
        help="Skip per-episode rmps_diagnostics.png generation (still saves .npy files & overview PNGs)",
    )
    parser.add_argument("--recalibrate", action="store_true", help="Force offline tube recalibration")
    args = parser.parse_args()

    selected_variants = tuple(dict.fromkeys(args.variants))
    need_baseline = "baseline" in selected_variants
    need_tube = any(v != "baseline" for v in selected_variants)
    plan_seeds = list(args.seed_list) if args.seed_list is not None else [42 + i for i in range(args.seeds)]
    n_seeds = len(plan_seeds)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tubes = ensure_tubes_loaded(args.levels, dt=args.dt, recalibrate=args.recalibrate)

    summary_rows: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    trajectories_by_key: dict[tuple[int, str], list[dict[str, Any]]] = {}
    env_ref_for_plot = build_benchmark_env(0, plan_seeds[0], dt=args.dt, H=args.horizon, enable_robust_shielding=False)

    print("=" * 162)
    print(
        f"Robust Safe-MPD Benchmark | variants={list(selected_variants)}, seeds={plan_seeds}, "
        f"levels={args.levels}, Nsample={args.Nsample}, Ndiffuse={args.Ndiffuse}, save_episodes={args.save_episodes}"
    )
    print("=" * 162)
    header = (
        f"{'Level':<6} | {'Variant':<15} | {'Succ%':>6} | {'Col%':>6} | {'Hitch%':>6} | "
        f"{'CVaR Err[m]':>11} | {'CVaR Clr[m]':>11} | {'Interv%':>7} | {'Replans':>7} | {'Step[ms]':>8} | "
        f"{'Fail(C/H/CH/TO/MR/ST)':>21}"
    )
    print(header)
    print("-" * 162)

    # Precompute nominal plans only for requested variant families
    baseline_plans: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    tube_plans: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

    if need_baseline:
        for s_idx, p_seed in enumerate(plan_seeds):
            env_b = build_benchmark_env(0, p_seed, dt=args.dt, H=args.horizon, enable_robust_shielding=False)
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

    for lv in args.levels:
        tube = tubes[lv]
        r_tube_arr = tube["r_tube"]
        b_hitch_arr = tube["b_hitch"]
        r_tuple = tuple(float(v) for v in np.asarray(r_tube_arr).reshape(-1))
        b_tuple = tuple(float(v) for v in np.asarray(b_hitch_arr).reshape(-1))

        if need_tube:
            # Always plan tube-based variants with `enable_robust_shielding=True` (even at Level 0,
            # since robust shielded rollout uses accurate `dt` braking & t=0 weight masking).
            for s_idx, p_seed in enumerate(plan_seeds):
                ep_seed = p_seed if args.seed_list is not None else (100 + 17 * s_idx + 1000 * lv)
                env_t = build_benchmark_env(
                    lv,
                    ep_seed,
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
                tube_plans[(lv, s_idx)] = SafeMPDPlanner(cfg_t, env_t).plan(np.asarray(env_t.x0), np.asarray(env_t.xg))

        for variant in selected_variants:
            use_tube = variant != "baseline"
            use_nmpc = variant in ("rmps_no_replan", "rmps_replan")
            use_replan = variant == "rmps_replan"

            r_arg = r_tube_arr if use_tube else 0.0
            b_arg = b_hitch_arr if use_tube else 0.0

            clear_jit_cache()
            ep_results: list[dict[str, Any]] = []

            for s_idx, p_seed in enumerate(plan_seeds):
                seed = p_seed if args.seed_list is not None else (100 + 17 * s_idx + 1000 * lv)
                env = build_benchmark_env(
                    perturbation_level=lv,
                    perturbation_seed=seed,
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
                    perturbation_level=lv,
                    perturbation_seed=seed,
                    enable_robust_shielding=use_tube,
                    tube_obstacle_margin=r_tuple if use_tube else 0.0,
                    tube_hitch_margin=b_tuple if use_tube else 0.0,
                )
                planner = SafeMPDPlanner(cfg, env)
                nmpc_backup = RobustNMPCBackup(env, T_bak=12, v_eps=0.05, max_iter=40)
                init_p = tube_plans[(lv, s_idx)] if use_tube else baseline_plans[s_idx]

                ep_save_dir: Path | None = None
                if args.save_episodes == "all":
                    ep_save_dir = out_dir / "episodes" / f"level{lv}" / variant / f"seed_{seed}"
                elif args.save_episodes == "per_level" and s_idx == 0:
                    ep_save_dir = out_dir / "episodes" / f"level{lv}" / variant / f"seed_{seed}"
                elif args.save_episodes == "sample" and s_idx == 0 and lv == max(args.levels):
                    ep_save_dir = out_dir / f"sample_{variant}_level{lv}"

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
                    episode_seed=seed,
                    initial_plan=init_p,
                    save_dir=ep_save_dir,
                    save_plot=(ep_save_dir is not None and not args.no_episode_plots),
                )
                outcome = classify_episode_outcome(res)
                res["outcome"] = outcome
                ep_results.append(res)
                trajectories_by_key.setdefault((lv, variant), []).append(
                    {
                        "seed": seed,
                        "states": res["states"].copy(),
                        "outcome": outcome,
                        "collision": bool(res["collision"]),
                        "hitch_violation": bool(res["hitch_violation"]),
                    }
                )
                raw_records.append(
                    {
                        "level": lv,
                        "variant": variant,
                        "seed": seed,
                        "outcome": outcome,
                        "goal_reached": res["goal_reached"],
                        "termination_reason": res.get("termination_reason", "timeout"),
                        "collision": res["collision"],
                        "hitch_violation": res["hitch_violation"],
                        "safe_success": bool(outcome == "safe_goal"),
                        "final_pos_error": res["final_pos_error"],
                        "min_clearance": res["min_clearance"],
                        "intervention_rate": res["intervention_rate"],
                        "num_replans": res["num_replans"],
                        "avg_step_ms": float(np.mean(res["computation_times"]) * 1000.0) if len(res["computation_times"]) else 0.0,
                        "saved_dir": str(ep_save_dir.relative_to(REPO_ROOT)) if (ep_save_dir is not None and ep_save_dir.is_relative_to(REPO_ROOT)) else (str(ep_save_dir) if ep_save_dir is not None else None),
                    }
                )

            succ_rate = 100.0 * float(np.mean([r["outcome"] == "safe_goal" for r in ep_results]))
            col_rate = 100.0 * float(np.mean([r["collision"] for r in ep_results]))
            hitch_rate = 100.0 * float(np.mean([r["hitch_violation"] for r in ep_results]))
            cvar_err = cvar_upper([r["final_pos_error"] for r in ep_results], alpha=0.01)
            cvar_clr = cvar_lower([r["min_clearance"] for r in ep_results], alpha=0.01)
            interv_pct = 100.0 * float(np.mean([r["intervention_rate"] for r in ep_results]))
            avg_replans = float(np.mean([r["num_replans"] for r in ep_results]))
            all_step_ms = [float(t) * 1000.0 for r in ep_results for t in r["computation_times"]]
            avg_step_ms = float(np.mean(all_step_ms)) if all_step_ms else 0.0

            n_goal = sum(1 for r in ep_results if r["outcome"] == "safe_goal")
            n_col_only = sum(1 for r in ep_results if r["outcome"] == "collision")
            n_hitch_only = sum(1 for r in ep_results if r["outcome"] == "jackknife")
            n_col_hitch = sum(1 for r in ep_results if r["outcome"] == "collision+jackknife")
            n_timeout = sum(1 for r in ep_results if r["outcome"] == "timeout")
            n_maxrp = sum(1 for r in ep_results if r["outcome"] == "max_replans")
            n_stopped = sum(1 for r in ep_results if r["outcome"] == "stopped_no_replan")

            row = {
                "level": lv,
                "variant": variant,
                "seeds": n_seeds,
                "success_rate_pct": round(succ_rate, 2),
                "collision_rate_pct": round(col_rate, 2),
                "hitch_violation_rate_pct": round(hitch_rate, 2),
                "cvar_pos_error_m": round(cvar_err, 4),
                "cvar_min_clearance_m": round(cvar_clr, 4),
                "intervention_rate_pct": round(interv_pct, 2),
                "avg_replans": round(avg_replans, 2),
                "avg_step_runtime_ms": round(avg_step_ms, 2),
                "outcome_safe_goal": n_goal,
                "fail_collision": n_col_only,
                "fail_jackknife": n_hitch_only,
                "fail_collision_and_jackknife": n_col_hitch,
                "fail_timeout": n_timeout,
                "fail_max_replans": n_maxrp,
                "fail_stopped_no_replan": n_stopped,
            }
            summary_rows.append(row)
            fail_str = f"{n_col_only}/{n_hitch_only}/{n_col_hitch}/{n_timeout}/{n_maxrp}/{n_stopped}"
            print(
                f"L{lv:<5} | {variant:<15} | {succ_rate:>6.1f} | {col_rate:>6.1f} | {hitch_rate:>6.1f} | "
                f"{cvar_err:>11.3f} | {cvar_clr:>11.3f} | {interv_pct:>7.1f} | {avg_replans:>7.2f} | {avg_step_ms:>8.2f} | {fail_str:>21}",
                flush=True,
            )

    print("=" * 162)
    print("Failure Type Breakdown (C=Collision, H=Jackknife, CH=Collision+Jackknife, TO=Safe Timeout, MR=Safe Max-Replans, ST=Safe Stopped No-Replan):")
    for r in summary_rows:
        print(
            f"  L{r['level']} {r['variant']:<15} | Safe Goal: {r['outcome_safe_goal']}/{r['seeds']} | "
            f"Col: {r['fail_collision']} | Jackknife: {r['fail_jackknife']} | Col+Jackknife: {r['fail_collision_and_jackknife']} | "
            f"Safe Timeout: {r['fail_timeout']} | Safe Max-Replans: {r['fail_max_replans']} | Safe Stopped: {r['fail_stopped_no_replan']}"
        )
    print("=" * 162)

    with open(out_dir / "summary.json", "w") as f:
        json.dump({"summary": summary_rows, "episodes": raw_records}, f, indent=2)

    if summary_rows:
        with open(out_dir / "summary.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    plot_benchmark_summary(
        summary_rows,
        args.levels,
        out_dir / "benchmark_comparison.png",
        variants=selected_variants,
    )
    plot_all_trajectories_overview(
        env_ref=env_ref_for_plot,
        trajectories_by_key=trajectories_by_key,
        levels=args.levels,
        variants=selected_variants,
        out_path=out_dir / "all_trajectories_overview.png",
    )
    print(f"Saved benchmark results & plots to {out_dir}")
    print(f"  Metrics comparison plot : {out_dir / 'benchmark_comparison.png'}")
    print(f"  All-seeds trajectory map: {out_dir / 'all_trajectories_overview.png'}")
    if args.save_episodes in ("all", "per_level"):
        print(f"  Individual episodes     : {out_dir / 'episodes'}/level<L>/<variant>/seed_<S>/ (visualize any with `render.py <folder> --mode both --no-static`)")


if __name__ == "__main__":
    main()
