"""
Online Closed-Loop RMPS Execution & Replanning Runner (RMPS Algorithm 1).

Runs a closed-loop episode against `env.step_perturbed` (Perturbation Levels 0-3)
combining:
- `SafeMPDPlanner` (JAX GPU planning / replanning from stopped equilibrium states)
- `RobustNMPCBackup` (CPU CasADi Tracking NMPC & recoverability shield)
- Optional 1-step delay compensation
- Standard `results/` output export (compatible with `render.py`) and diagnostic plots.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

from mbd.utils import save_env_metadata
from .nmpc_backup import RobustNMPCBackup, _format_tube_schedule, _wrap_angle_np  # noqa: E402
from .planner_wrapper import SafeMPDPlanner  # noqa: E402


def compute_min_obstacle_clearance_np(
    env: Any,
    state_np: np.ndarray,
    use_perturbed_geom: bool = False,
    obs_c: np.ndarray | None = None,
    obs_r: np.ndarray | None = None,
) -> float:
    """
    Compute exact signed Euclidean clearance [m] from the tractor & trailer rectangles
    to all active circular and oriented rectangular obstacles (negative = penetration).
    """
    px, py, theta1, theta2 = [float(v) for v in state_np[:4]]
    l1 = float(env.l1_perturbed if use_perturbed_geom else env.l1)
    l2 = float(env.l2_perturbed if use_perturbed_geom else env.l2)
    lh = float(env.lh_perturbed if use_perturbed_geom else env.lh)

    c1, s1 = np.cos(theta1), np.sin(theta1)
    trac_cx = px + 0.5 * (l1 + float(env.lf1) - float(env.lr)) * c1
    trac_cy = py + 0.5 * (l1 + float(env.lf1) - float(env.lr)) * s1
    trac_corners = RobustNMPCBackup._rect_corners_np(
        trac_cx, trac_cy, 0.5 * (l1 + float(env.lf1) + float(env.lr)), 0.5 * float(env.tractor_width), theta1
    )

    hx = px - lh * c1
    hy = py - lh * s1
    c2, s2 = np.cos(theta2), np.sin(theta2)
    trail_cx = hx + (0.5 * (float(env.lf2) - float(env.lr2)) - l2) * c2
    trail_cy = hy + (0.5 * (float(env.lf2) - float(env.lr2)) - l2) * s2
    trail_corners = RobustNMPCBackup._rect_corners_np(
        trail_cx, trail_cy, 0.5 * (float(env.lf2) + float(env.lr2)), 0.5 * float(env.trailer_width), theta2
    )

    if obs_c is None:
        obs_c = np.asarray(env.obs_circles, dtype=np.float64) if hasattr(env, "obs_circles") else np.zeros((0, 3))
    if obs_r is None:
        obs_r = np.asarray(env.obs_rectangles, dtype=np.float64) if hasattr(env, "obs_rectangles") else np.zeros((0, 5))

    min_d = float("inf")
    ts = np.linspace(0.0, 1.0, 17)[:, None]

    for corners in (trac_corners, trail_corners):
        edge_pts = []
        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            edge_pts.append(p1[None, :] * (1.0 - ts) + p2[None, :] * ts)
        pts = np.vstack(edge_pts)

        for c in obs_c:
            if c[2] <= 1e-6:
                continue
            d_c = float(np.min(np.linalg.norm(pts - c[:2][None, :], axis=1) - c[2]))
            min_d = min(min_d, d_c)

        for r in obs_r:
            rx, ry, rw, rh, rang = [float(v) for v in r]
            if rw <= 1e-6 or rh <= 1e-6:
                continue
            rel = pts - np.array([[rx, ry]])
            ca, sa = np.cos(-rang), np.sin(-rang)
            lx = rel[:, 0] * ca - rel[:, 1] * sa
            ly = rel[:, 0] * sa + rel[:, 1] * ca
            dx = np.maximum(np.abs(lx) - 0.5 * rw, 0.0)
            dy = np.maximum(np.abs(ly) - 0.5 * rh, 0.0)
            inside = (np.abs(lx) <= 0.5 * rw) & (np.abs(ly) <= 0.5 * rh)
            d_out = np.hypot(dx, dy)
            d_in = -np.minimum(0.5 * rw - np.abs(lx), 0.5 * rh - np.abs(ly))
            d_signed = np.where(inside, d_in, d_out)
            min_d = min(min_d, float(np.min(d_signed)))

    return min_d


def _check_goal_reached(
    x: np.ndarray,
    x_goal: np.ndarray,
    pos_tol: float,
    ang_tol: float,
    v_tol: float,
) -> tuple[bool, float, float]:
    pos_err = float(np.linalg.norm(x[:2] - x_goal[:2]))
    e_th1 = abs(float(_wrap_angle_np(x[2] - x_goal[2])))
    e_th2 = abs(float(_wrap_angle_np(x[3] - x_goal[3])))
    ang_err = max(e_th1, e_th2)
    reached = (pos_err <= pos_tol) and (ang_err <= ang_tol) and (abs(float(x[4])) <= v_tol)
    return reached, pos_err, ang_err


def _extract_tube_arrays(tube_data: dict[str, Any] | None, length: int) -> tuple[np.ndarray, np.ndarray]:
    if tube_data is None:
        return np.zeros(length, dtype=np.float64), np.zeros(length, dtype=np.float64)
    r_raw = tube_data.get("r_tube", 0.0)
    b_raw = tube_data.get("b_hitch", 0.0)
    return _format_tube_schedule(r_raw, length), _format_tube_schedule(b_raw, length)


def _shift_tube_for_delay(arr: np.ndarray) -> np.ndarray:
    """Expand future tube schedule (k >= 1) by 1 step to account for 1-step computation delay."""
    if arr.size <= 1:
        return arr.copy()
    shifted = np.empty_like(arr)
    shifted[0] = 0.0  # k = 0 is the already-measured current state
    shifted[1:-1] = arr[2:] if arr.size > 2 else arr[1:]
    shifted[-1] = arr[-1]
    return shifted


def _prepare_online_tubes(
    env: Any,
    tube_data: dict[str, Any] | None,
    T_bak: int,
    pert_level: int,
    use_delay_compensation: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build online NMPC tube schedules `(r_eval, b_eval)` of length `T_bak + 1`.

    - `k = 0` is kept at 0.0 because `x_real` is already measured.
    - For `pert_level > 0` and `k >= 1`, applies a non-zero short-horizon floor so
      calibration maneuvers starting from `v = 0` do not leave `k = 1..4` unguarded
      at cruise speeds.
    - For Level 2 (time-invariant link-length uncertainty), adds a geometric length
      cushion so perturbed tractor/trailer dimensions stay collision-free.
    """
    r_base, b_base = _extract_tube_arrays(tube_data, T_bak + 1)
    r_eval = _shift_tube_for_delay(r_base) if use_delay_compensation else r_base.copy()
    b_eval = _shift_tube_for_delay(b_base) if use_delay_compensation else b_base.copy()
    r_eval[0] = 0.0
    b_eval[0] = 0.0

    if pert_level > 0 and r_eval.size > 1:
        if pert_level == 2:
            # Cover time-invariant perturbed geometry (l1_perturbed, l2_perturbed, lh_perturbed)
            p_std = float(getattr(env, "param_uncertainty_std", 0.10))
            geom_pad = max(0.12, 1.2 * p_std)
            r_eval[1:] = np.maximum(r_eval[1:], geom_pad)
            b_eval[1:] = np.maximum(b_eval[1:], 0.05)
        elif pert_level == 3:
            # Small safety floor for heavy-tailed 1-step process noise even at low speeds
            r_eval[1:] = np.maximum(r_eval[1:], 0.08)
            b_eval[1:] = np.maximum(b_eval[1:], 0.04)

    # Cap hitch tightening so NMPC always retains >= 70% of physical phi_max
    phi_cap = 0.30 * float(getattr(env, "phi_max", np.deg2rad(40.0)))
    b_eval[1:] = np.minimum(b_eval[1:], phi_cap)
    return r_eval, b_eval


def _compute_tvlqr_gains_np(
    env: Any,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
) -> np.ndarray:
    """
    Fast pure-NumPy finite-horizon TV-LQR feedback gains `K_fb[t]` (shape `(T, 2, 6)`)
    matching the exact closed-loop tracking law used in offline tube calibration (`tube_calibration.py`).
    Runs in < 0.5 ms with zero JAX compilation overhead.
    """
    T = int(u_ref.shape[0])
    if T <= 0:
        return np.zeros((0, 2, 6), dtype=np.float64)

    dt = float(env.dt)
    l1 = float(getattr(env, "l1_nominal", env.l1))
    l2 = float(getattr(env, "l2_nominal", env.l2))
    lh = float(getattr(env, "lh_nominal", env.lh))
    v_max = float(env.v_max)
    delta_max = float(env.delta_max)
    a_max = float(getattr(env, "a_max", 2.0))
    omega_max = float(getattr(env, "omega_max", 1.0))

    Q = np.diag([2.0, 2.0, 5.0, 8.0, 2.0, 1.0])
    R = np.diag([0.5, 0.5])
    P = 5.0 * Q

    B = np.zeros((6, 2), dtype=np.float64)
    B[4, 0] = dt * a_max
    B[5, 1] = dt * omega_max

    K_fb = np.zeros((T, 2, 6), dtype=np.float64)
    I6 = np.eye(6, dtype=np.float64)

    for t in range(T - 1, -1, -1):
        _, _, th1, th2, v, delta = [float(val) for val in x_ref[t, :6]]
        v_c = float(np.clip(v, -v_max, v_max))
        dv_c = 1.0 if abs(v) < v_max else 0.0
        d_c = float(np.clip(delta, -delta_max, delta_max))
        dd_c = 1.0 if abs(delta) < delta_max else 0.0

        c1, s1 = np.cos(th1), np.sin(th1)
        cd = np.cos(d_c)
        td = np.tan(d_c)
        sec2 = 1.0 / max(cd * cd, 1e-6)
        phi = th1 - th2
        cp, sp = np.cos(phi), np.sin(phi)

        A = I6.copy()
        A[0, 2] = -dt * v_c * s1
        A[0, 4] = dt * c1 * dv_c
        A[1, 2] = dt * v_c * c1
        A[1, 4] = dt * s1 * dv_c
        A[2, 4] = dt * (td / l1) * dv_c
        A[2, 5] = dt * (v_c / l1) * sec2 * dd_c

        dth2_dth1 = (v_c / l2) * (cp + (lh / l1) * sp * td)
        A[3, 2] = dt * dth2_dth1
        A[3, 3] = 1.0 - dt * dth2_dth1
        A[3, 4] = dt * (1.0 / l2) * (sp - (lh / l1) * cp * td) * dv_c
        A[3, 5] = -dt * (v_c * lh / (l1 * l2)) * cp * sec2 * dd_c

        BtP = B.T @ P
        G = R + BtP @ B
        K_t = -np.linalg.solve(G, BtP @ A)
        K_fb[t] = K_t
        P = Q + A.T @ P @ (A + B @ K_t)

    return K_fb


def run_rmps_episode(
    env: Any,
    planner: SafeMPDPlanner,
    nmpc_backup: RobustNMPCBackup,
    tube_data: dict[str, Any] | None = None,
    max_steps: int = 150,
    max_replans: int = 5,
    enable_replanning: bool = True,
    enable_nmpc_shield: bool = True,
    enable_tube_tracking: bool = True,
    use_delay_compensation: bool = True,
    replan_cooldown_steps: int = 8,
    pos_tol: float = 1.5,
    ang_tol: float = 0.75,
    v_tol: float = 0.5,
    episode_seed: int | None = None,
    initial_plan: tuple[np.ndarray, np.ndarray] | None = None,
    save_dir: str | Path | None = None,
    save_plot: bool = True,
) -> dict[str, Any]:
    """
    Execute one closed-loop RMPS episode (RMPS Algorithm 1) against `env.step_perturbed`.

    Args:
        env: `AccTractorTrailer2d` environment (with perturbation level already configured).
        planner: `SafeMPDPlanner` instance wrapping Safe-MPD.
        nmpc_backup: `RobustNMPCBackup` instance for recoverability checking & tracking NMPC.
        tube_data: Optional calibrated tube dict containing `r_tube` and `b_hitch`.
        max_steps: Maximum simulation steps in the episode.
        max_replans: Maximum number of online Safe-MPD replans from stopped states.
        enable_replanning: Whether to replan from stopped equilibrium states after intervention.
        enable_nmpc_shield: Whether to run the online RMPS NMPC recoverability shield.
        use_delay_compensation: Whether to shift future tube margins by 1 step for delay compensation.
        replan_cooldown_steps: Minimum steps between consecutive replans to prevent chattering.
        pos_tol: Goal position tolerance [m].
        ang_tol: Goal heading tolerance [rad].
        v_tol: Goal terminal velocity tolerance [m/s].
        episode_seed: Optional RNG seed for `env.step_perturbed` rollout.
        initial_plan: Optional precomputed `(x_plan, u_plan)` to share across paired benchmarks.
        save_dir: Optional directory to save standard `results/` files + diagnostic plot.
        save_plot: Whether to save `rmps_diagnostics.png` when `save_dir` is provided.

    Returns:
        Dictionary containing states, actions, plans, shield interventions, replans,
        safety flags, clearance metrics, goal error, termination_reason, and step times.
    """
    pert_level = int(getattr(env, "perturbation_level", 0))
    use_pert_geom = pert_level == 2
    rollout_seed = int(getattr(env, "perturbation_seed", 123)) if episode_seed is None else int(episode_seed)

    x0 = np.asarray(env.x0, dtype=np.float64).copy()
    x_goal = np.asarray(env.xg, dtype=np.float64).copy()

    r_tube_eval, b_hitch_eval = _prepare_online_tubes(
        env=env,
        tube_data=tube_data,
        T_bak=nmpc_backup.T_bak,
        pert_level=pert_level,
        use_delay_compensation=use_delay_compensation,
    )

    # Initial Safe-MPD plan from x0 (or reuse paired plan)
    if initial_plan is not None:
        x_plan = np.asarray(initial_plan[0], dtype=np.float64).copy()
        u_plan = np.asarray(initial_plan[1], dtype=np.float64).copy()
        initial_plan_time = 0.0
    else:
        plan_t0 = time.perf_counter()
        x_plan, u_plan = planner.plan(x0, x_goal)
        initial_plan_time = time.perf_counter() - plan_t0

    nominal_plans: list[np.ndarray] = [x_plan.copy()]
    initial_u_plan = u_plan.copy()
    K_plan = _compute_tvlqr_gains_np(env, x_plan, u_plan) if (enable_nmpc_shield and enable_tube_tracking) else None
    plan_step = 0
    replan_count = 0
    replan_steps: list[int] = []
    last_replan_step = -replan_cooldown_steps
    last_replan_pos: np.ndarray | None = None
    stalled_replans = 0
    replan_skip_counts: dict[str, int] = {"cooldown": 0, "stalled": 0, "max_replans": 0}

    nmpc_backup.reset()
    active_backup = False

    # Pre-convert active obstacle arrays once to avoid per-step JAX -> NumPy syncs
    obs_c_np = np.asarray(env.obs_circles, dtype=np.float64) if hasattr(env, "obs_circles") else np.zeros((0, 3))
    obs_r_np = np.asarray(env.obs_rectangles, dtype=np.float64) if hasattr(env, "obs_rectangles") else np.zeros((0, 5))

    # Initialize JAX perturbed environment state
    step_perturbed_jit = jax.jit(env.step_perturbed)
    st = env.reset(jax.random.PRNGKey(rollout_seed)).replace(
        pipeline_state=jnp.asarray(x0, dtype=jnp.float32),
        obs=jnp.asarray(x0, dtype=jnp.float32),
    )

    states_log: list[np.ndarray] = [x0.copy()]
    actions_log: list[np.ndarray] = []
    shield_interventions: list[bool] = []
    computation_times: list[float] = []
    clearances_log: list[float] = [
        compute_min_obstacle_clearance_np(env, x0, use_perturbed_geom=use_pert_geom, obs_c=obs_c_np, obs_r=obs_r_np)
    ]

    any_collision = False
    any_hitch_violation = False
    goal_reached = False
    termination_reason = "timeout"

    # Slightly relaxed stop threshold under heavy-tailed Level 3 process noise so replanning can trigger
    v_replan_tol = max(nmpc_backup.v_eps, 0.14 if pert_level == 3 else nmpc_backup.v_eps + 0.03)
    phi_max_f = float(env.phi_max)

    def _plan_from_stopped_state(x_curr: np.ndarray, attempt_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Replan from a stopped equilibrium state. Only relaxes `env` planning margins if
        `x_curr` strictly violates the current step-0 tightened margin, avoiding unnecessary
        XLA JIT cache invalidations during online replanning.
        """
        orig_rob = bool(getattr(env, "enable_robust_shielding", False))
        orig_r_raw = np.asarray(getattr(env, "tube_obstacle_margin_raw", np.zeros(1, dtype=np.float32))).copy()
        orig_b_raw = np.asarray(getattr(env, "tube_hitch_margin_raw", np.zeros(1, dtype=np.float32))).copy()
        orig_manual = bool(getattr(env, "_robust_margins_manually_set", False))
        orig_cfg = planner.config

        clr_0 = compute_min_obstacle_clearance_np(
            env, x_curr, use_perturbed_geom=False, obs_c=obs_c_np, obs_r=obs_r_np
        )
        hitch_0 = abs(float(_wrap_angle_np(x_curr[2] - x_curr[3])))
        r0_curr = float(getattr(env, "_r_margin_step0", 0.0))
        b0_curr = float(getattr(env, "_b_margin_step0", 0.0))

        need_cap = orig_rob and (clr_0 < r0_curr + 0.005 or hitch_0 > phi_max_f - b0_curr - 0.005)
        capped = False
        try:
            if need_cap and hasattr(env, "set_robust_margins"):
                r_cap = max(0.0, clr_0 - 0.01)
                b_cap = max(0.0, phi_max_f - hitch_0 - 0.01)
                env.set_robust_margins(
                    r_tube=np.minimum(orig_r_raw, r_cap),
                    b_hitch=np.minimum(orig_b_raw, b_cap),
                    enable_robust_shielding=orig_rob,
                )
                capped = True

            xp_new, up_new = planner.plan(x_curr, x_goal, seed=int(orig_cfg.seed + 131 * (attempt_idx + 1)))
            disp = float(np.linalg.norm(xp_new[-1, :2] - x_curr[:2]))
            dist_g = float(np.linalg.norm(x_curr[:2] - x_goal[:2]))
            if disp < 0.25 and dist_g > pos_tol:
                xp_retry, up_retry = planner.plan(
                    x_curr, x_goal, seed=int(orig_cfg.seed + 313 * (attempt_idx + 2))
                )
                if float(np.linalg.norm(xp_retry[-1, :2] - x_curr[:2])) > disp:
                    xp_new, up_new = xp_retry, up_retry
            return xp_new, up_new
        finally:
            if capped and hasattr(env, "set_robust_margins"):
                env.set_robust_margins(r_tube=orig_r_raw, b_hitch=orig_b_raw, enable_robust_shielding=orig_rob)
                env._robust_margins_manually_set = orig_manual

    def _adapt_margins_for_buffer_escape(
        x_eval: np.ndarray,
        x_after: np.ndarray,
        r_arr: np.ndarray,
        b_arr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        When a real-world perturbation has pushed `x_eval` inside the nominal safety buffer
        (`clr_curr < r_arr[1]` or `|hitch_curr| > phi_max - b_arr[1]`), insisting on the full
        uncapped tube at `k=1` deadlocks all escape actions at `v=0`.

        Returns locally feasible margins `(r_eff, b_eff)` ONLY if `x_after` does not worsen
        the violated safety metric and remains strictly physically safe; otherwise returns `None`.
        """
        r1 = float(r_arr[1]) if r_arr.size > 1 else 0.0
        b1 = float(b_arr[1]) if b_arr.size > 1 else 0.0
        if r1 <= 1e-6 and b1 <= 1e-6:
            return r_arr, b_arr

        clr_curr = compute_min_obstacle_clearance_np(
            env, x_eval, use_perturbed_geom=False, obs_c=obs_c_np, obs_r=obs_r_np
        )
        hitch_curr = abs(float(_wrap_angle_np(x_eval[2] - x_eval[3])))
        in_obs_buf = clr_curr < r1 + 0.01
        in_hitch_buf = hitch_curr > phi_max_f - b1 - 0.01
        if not in_obs_buf and not in_hitch_buf:
            return r_arr, b_arr

        r_eff = r_arr.copy()
        b_eff = b_arr.copy()
        if in_obs_buf:
            clr_after = compute_min_obstacle_clearance_np(
                env, x_after, use_perturbed_geom=False, obs_c=obs_c_np, obs_r=obs_r_np
            )
            if clr_after < 0.015 or clr_after < clr_curr - 0.005:
                return None
            r_cap = max(0.01, min(r1, clr_curr - 0.015))
            r_eff[1:] = np.minimum(r_eff[1:], r_cap)

        if in_hitch_buf:
            hitch_after = abs(float(_wrap_angle_np(x_after[2] - x_after[3])))
            if hitch_after > phi_max_f - 0.015 or hitch_after > hitch_curr + 0.005:
                return None
            b_cap = max(0.01, min(b1, phi_max_f - hitch_curr - 0.015))
            b_eff[1:] = np.minimum(b_eff[1:], b_cap)

        r_eff[0] = 0.0
        b_eff[0] = 0.0
        return r_eff, b_eff

    def _try_accept_mpd_action(
        x_eval: np.ndarray,
        u_cand: np.ndarray,
        r_arr: np.ndarray,
        b_arr: np.ndarray,
    ) -> bool:
        x_after = nmpc_backup.step_nominal_np(x_eval, u_cand)

        # Fast-path: when tracking closely inside the nominal tube with ample obstacle AND hitch margin,
        # accept without invoking IPOPT.
        pos_dev = float(np.linalg.norm(x_eval[:2] - x_plan[plan_step, :2]))
        ang_dev = max(
            abs(float(_wrap_angle_np(x_eval[2] - x_plan[plan_step, 2]))),
            abs(float(_wrap_angle_np(x_eval[3] - x_plan[plan_step, 3]))),
        )
        max_r = float(np.max(r_arr))
        max_b = float(np.max(b_arr))
        hitch_after = abs(float(_wrap_angle_np(x_after[2] - x_after[3])))
        if (
            pos_dev <= max(0.02, 0.5 * max_r)
            and ang_dev <= max(0.02, 0.5 * max_b)
            and hitch_after < phi_max_f - max_b - 0.12
        ):
            r_chk = float(r_arr[1]) if r_arr.size > 1 else 0.0
            b_chk = float(b_arr[1]) if b_arr.size > 1 else 0.0
            safe_after = nmpc_backup.check_state_safety_np(x_after, r_chk, b_chk) == (False, False)
            nxt_idx = min(plan_step + 1, len(x_plan) - 1)
            safe_plan_nxt = nmpc_backup.check_state_safety_np(x_plan[nxt_idx], r_chk, b_chk) == (False, False)
            if safe_after and safe_plan_nxt:
                brake_dist = 0.5 * (float(x_after[4]) ** 2) / max(nmpc_backup.a_max, 0.5)
                if max_r <= 1e-6 or (
                    compute_min_obstacle_clearance_np(env, x_after, obs_c=obs_c_np, obs_r=obs_r_np)
                    > max_r + brake_dist + 0.6
                ):
                    return True

        # Adapt margins only if x_eval is already inside the safety buffer and x_after escapes safely
        adapted = _adapt_margins_for_buffer_escape(x_eval, x_after, r_arr, b_arr)
        if adapted is None:
            return False
        r_eff, b_eff = adapted
        return bool(nmpc_backup.is_recoverable(x_after, r_tube=r_eff, b_hitch=b_eff))

    def _decide_rmps_control(x_eval: np.ndarray, r_arr: np.ndarray, b_arr: np.ndarray) -> tuple[np.ndarray, bool]:
        nonlocal plan_step, active_backup
        if not enable_nmpc_shield:
            if plan_step < len(u_plan):
                u_out = np.clip(u_plan[plan_step], -1.0, 1.0)
                plan_step += 1
                return u_out, False
            return nmpc_backup.stabilizing_brake_policy(x_eval), False

        # RMPS Algorithm 1 commitment / hysteresis:
        # Once the NMPC backup shield takes over, commit to bringing the vehicle safely
        # to a stopped equilibrium inside G_e instead of aborting mid-brake and chattering.
        if active_backup:
            return nmpc_backup.step(x_eval), True

        # If current nominal plan is exhausted, brake safely to equilibrium (triggers replan if allowed)
        if plan_step >= len(u_plan):
            nmpc_backup.initialize_backup(x_eval, r_tube=r_arr, b_hitch=b_arr)
            active_backup = True
            return nmpc_backup.step(x_eval), True

        # Propose nominal Safe-MPD action with closed-loop TV-LQR tube tracking feedback
        # (matching the offline tube calibration law in `tube_calibration.py`).
        u_raw = np.clip(u_plan[plan_step], -1.0, 1.0)
        u_mpd = u_raw
        if K_plan is not None and plan_step < len(K_plan):
            err = x_eval - x_plan[plan_step]
            err[2] = float(_wrap_angle_np(err[2]))
            err[3] = float(_wrap_angle_np(err[3]))
            if float(np.linalg.norm(err[:2])) <= 2.5 and max(abs(err[2]), abs(err[3])) <= 0.8:
                u_mpd = np.clip(u_raw + K_plan[plan_step] @ err, -1.0, 1.0)

        # RMPS Algorithm 1: Check recoverability of proposed closed-loop tracking action
        if _try_accept_mpd_action(x_eval, u_mpd, r_arr, b_arr):
            plan_step += 1
            return u_mpd, False

        # Fallback check: if feedback correction was too aggressive, test open-loop u_raw before intervening
        if float(np.linalg.norm(u_mpd - u_raw)) > 0.05 and _try_accept_mpd_action(x_eval, u_raw, r_arr, b_arr):
            plan_step += 1
            return u_raw, False

        # Nominal action is not certifiably recoverable -> NMPC Shield intervenes and commits to stop!
        nmpc_backup.initialize_backup(x_eval, r_tube=r_arr, b_hitch=b_arr)
        active_backup = True
        return nmpc_backup.step(x_eval), True

    # Open-loop variants run for the exact planned horizon len(u_plan); RMPS runs up to max_steps
    effective_max_steps = min(max_steps, len(u_plan)) if not enable_nmpc_shield else max_steps

    t_dec0 = time.perf_counter()
    u_apply, is_intervened = _decide_rmps_control(x0, r_tube_eval, b_hitch_eval)
    t_dec0_elapsed = time.perf_counter() - t_dec0

    for k in range(effective_max_steps):
        step_t0 = time.perf_counter()

        # 1. Execute u_apply in the perturbed environment
        st = step_perturbed_jit(st, jnp.asarray(u_apply, dtype=jnp.float32))
        x_real = np.asarray(st.pipeline_state, dtype=np.float64)
        u_applied_real = np.asarray(st.applied_action, dtype=np.float64)

        col_k = bool(
            env.check_obstacle_collision(
                jnp.asarray(x_real[:4], dtype=jnp.float32),
                env.obs_circles,
                env.obs_rectangles,
                use_perturbed=use_pert_geom,
            )
        )
        hitch_k = bool(env.check_hitch_violation(jnp.asarray(x_real[:4], dtype=jnp.float32)))
        clr_k = compute_min_obstacle_clearance_np(
            env, x_real, use_perturbed_geom=use_pert_geom, obs_c=obs_c_np, obs_r=obs_r_np
        )

        any_collision = any_collision or col_k
        any_hitch_violation = any_hitch_violation or hitch_k

        states_log.append(x_real.copy())
        actions_log.append(u_applied_real.copy())
        shield_interventions.append(bool(is_intervened))
        clearances_log.append(clr_k)

        # 2. Check goal reached
        reached, _, _ = _check_goal_reached(x_real, x_goal, pos_tol, ang_tol, v_tol)
        if reached:
            goal_reached = True
            termination_reason = "goal"
            computation_times.append((time.perf_counter() - step_t0) + (t_dec0_elapsed if k == 0 else 0.0))
            break

        # Clean exit for rmps_no_replan once safely stopped under active backup
        if enable_nmpc_shield and not enable_replanning and active_backup and abs(float(x_real[4])) <= v_replan_tol:
            termination_reason = "stopped_no_replan"
            computation_times.append((time.perf_counter() - step_t0) + (t_dec0_elapsed if k == 0 else 0.0))
            break

        # 3. Replanning trigger from stopped equilibrium set G_e (with cooldown & anti-chatter guard)
        if enable_nmpc_shield and enable_replanning and active_backup and abs(float(x_real[4])) <= v_replan_tol:
            if replan_count >= max_replans:
                replan_skip_counts["max_replans"] += 1
                termination_reason = "max_replans"
                computation_times.append((time.perf_counter() - step_t0) + (t_dec0_elapsed if k == 0 else 0.0))
                break

            if (k + 1) - last_replan_step >= replan_cooldown_steps:
                if last_replan_pos is not None and float(np.linalg.norm(x_real[:2] - last_replan_pos)) < 0.35:
                    stalled_replans += 1
                else:
                    stalled_replans = 0

                if stalled_replans >= 2:
                    replan_skip_counts["stalled"] += 1
                    termination_reason = "max_replans"
                    computation_times.append((time.perf_counter() - step_t0) + (t_dec0_elapsed if k == 0 else 0.0))
                    break

                x_plan, u_plan = _plan_from_stopped_state(x_real, replan_count)
                nominal_plans.append(x_plan.copy())
                K_plan = (
                    _compute_tvlqr_gains_np(env, x_plan, u_plan)
                    if (enable_nmpc_shield and enable_tube_tracking)
                    else None
                )
                replan_count += 1
                replan_steps.append(k + 1)
                last_replan_step = k + 1
                last_replan_pos = x_real[:2].copy()
                plan_step = 0
                active_backup = False
                nmpc_backup.reset()
            else:
                replan_skip_counts["cooldown"] += 1

        # 4. Next-step RMPS switching: ALWAYS evaluate from the true measured state `x_real`
        if k + 1 < effective_max_steps:
            u_apply, is_intervened = _decide_rmps_control(x_real, r_tube_eval, b_hitch_eval)

        computation_times.append((time.perf_counter() - step_t0) + (t_dec0_elapsed if k == 0 else 0.0))

    states_arr = np.asarray(states_log, dtype=np.float64)
    actions_arr = np.asarray(actions_log, dtype=np.float64) if actions_log else np.zeros((0, 2), dtype=np.float64)
    interventions_arr = np.asarray(shield_interventions, dtype=bool)
    clearances_arr = np.asarray(clearances_log, dtype=np.float64)
    comp_times_arr = np.asarray(computation_times, dtype=np.float64)

    final_pos_error = float(np.linalg.norm(states_arr[-1, :2] - x_goal[:2]))
    min_clearance = float(np.min(clearances_arr))

    result: dict[str, Any] = {
        "states": states_arr,
        "actions": actions_arr,
        "nominal_plans": nominal_plans,
        "shield_interventions": interventions_arr,
        "replan_steps": replan_steps,
        "collision": bool(any_collision),
        "hitch_violation": bool(any_hitch_violation),
        "goal_reached": bool(goal_reached),
        "termination_reason": termination_reason,
        "final_pos_error": final_pos_error,
        "min_clearance": min_clearance,
        "clearances": clearances_arr,
        "computation_times": comp_times_arr,
        "initial_plan_time": float(initial_plan_time),
        "intervention_rate": float(np.mean(interventions_arr)) if interventions_arr.size > 0 else 0.0,
        "num_replans": int(replan_count),
        "replan_skip_counts": dict(replan_skip_counts),
        "perturbation_level": pert_level,
    }

    if save_dir is not None:
        save_rmps_results(
            env=env,
            config=planner.config,
            result=result,
            initial_u_plan=initial_u_plan,
            tube_data=tube_data,
            save_dir=save_dir,
            save_plot=save_plot,
        )

    return result


def save_rmps_results(
    env: Any,
    config: Any,
    result: dict[str, Any],
    initial_u_plan: np.ndarray,
    tube_data: dict[str, Any] | None,
    save_dir: str | Path,
    save_plot: bool = True,
) -> Path:
    """Save RMPS episode outputs in the standard `results/` format (compatible with `render.py`)."""
    out_path = Path(save_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    nom_states = np.asarray(result["nominal_plans"][0], dtype=np.float32)
    pert_states = np.asarray(result["states"], dtype=np.float32)
    nom_actions = np.asarray(initial_u_plan, dtype=np.float32)
    pert_actions = np.asarray(result["actions"], dtype=np.float32)

    np.save(out_path / "nominal_trajectory_states.npy", nom_states)
    np.save(out_path / "trajectory_states.npy", nom_states)
    np.save(out_path / "perturbed_trajectory_states.npy", pert_states)
    np.save(out_path / "final_actions.npy", nom_actions)
    np.save(out_path / "perturbed_actions.npy", pert_actions)
    save_env_metadata(env, config, str(out_path))

    if save_plot:
        plot_rmps_diagnostics(env, result, tube_data, out_path / "rmps_diagnostics.png")

    return out_path


def plot_rmps_diagnostics(
    env: Any,
    result: dict[str, Any],
    tube_data: dict[str, Any] | None,
    save_path: str | Path,
) -> Path:
    """Generate a diagnostic figure highlighting NMPC shield takeovers and replanning points."""
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    states = result["states"]
    interventions = result["shield_interventions"]
    replan_steps = result["replan_steps"]
    clearances = result["clearances"]
    dt = float(env.dt)
    t_axis = np.arange(len(states)) * dt

    fig = Figure(figsize=(15, 7))
    FigureCanvasAgg(fig)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1.0], hspace=0.35, wspace=0.25)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_h = fig.add_subplot(gs[1, 1], sharex=ax_v)
    ax_c = fig.add_subplot(gs[2, 1], sharex=ax_v)

    if hasattr(env, "obs_circles") and len(env.obs_circles) > 0:
        for c in np.asarray(env.obs_circles):
            if c[2] > 1e-6:
                ax_map.add_patch(Circle((c[0], c[1]), c[2], color="dimgray", alpha=0.6))
    if hasattr(env, "obs_rectangles") and len(env.obs_rectangles) > 0:
        for r in np.asarray(env.obs_rectangles):
            rx, ry, rw, rh, rang = r
            if rw > 1e-6 and rh > 1e-6:
                rect = Rectangle((-rw / 2, -rh / 2), rw, rh, facecolor="gray", edgecolor="black", alpha=0.65)
                rect.set_transform(Affine2D().rotate(rang).translate(rx, ry) + ax_map.transData)
                ax_map.add_patch(rect)

    for idx, plan in enumerate(result["nominal_plans"]):
        lbl = "Safe-MPD nominal plan(s)" if idx == 0 else None
        ax_map.plot(plan[:, 0], plan[:, 1], "--", color="seagreen", alpha=0.55, linewidth=1.5, label=lbl)

    plotted_mpd = False
    plotted_nmpc = False
    for k in range(len(interventions)):
        is_sh = bool(interventions[k])
        color = "crimson" if is_sh else "royalblue"
        lbl = None
        if is_sh and not plotted_nmpc:
            lbl = "NMPC Shield takeover"
            plotted_nmpc = True
        elif not is_sh and not plotted_mpd:
            lbl = "Safe-MPD execution"
            plotted_mpd = True
        ax_map.plot(states[k : k + 2, 0], states[k : k + 2, 1], "-", color=color, linewidth=2.4, label=lbl)

    if replan_steps:
        rp_pts = states[np.clip(replan_steps, 0, len(states) - 1)]
        ax_map.scatter(rp_pts[:, 0], rp_pts[:, 1], marker="*", s=180, color="gold", edgecolor="black", zorder=6, label="Replan point")

    ax_map.scatter([env.x0[0]], [env.x0[1]], color="green", s=90, zorder=5, label="Start")
    ax_map.scatter([env.xg[0]], [env.xg[1]], color="purple", marker="X", s=110, zorder=5, label="Goal")
    if hasattr(env, "env") and hasattr(env.env, "get_plot_limits"):
        xr, yr = env.env.get_plot_limits()
        ax_map.set_xlim(xr)
        ax_map.set_ylim(yr)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(True, alpha=0.3)
    ax_map.legend(loc="best", fontsize=8)
    ax_map.set_title(
        f"RMPS Episode (Level {result['perturbation_level']} | col={result['collision']}, "
        f"hitch={result['hitch_violation']}, goal={result['goal_reached']})"
    )
    ax_map.set_xlabel("x [m]")
    ax_map.set_ylabel("y [m]")

    ax_v.plot(t_axis, states[:, 4], "b-", linewidth=1.8)
    ax_v.axhline(0.0, color="k", linestyle=":", alpha=0.5)
    ax_v.set_ylabel("v [m/s]")
    ax_v.grid(True, alpha=0.3)

    hitch_deg = np.rad2deg(_wrap_angle_np(states[:, 2] - states[:, 3]))
    phi_max_deg = float(np.rad2deg(float(env.phi_max)))
    ax_h.plot(t_axis, hitch_deg, "m-", linewidth=1.8)
    ax_h.axhline(phi_max_deg, color="r", linestyle="--", linewidth=1.0)
    ax_h.axhline(-phi_max_deg, color="r", linestyle="--", linewidth=1.0)
    if tube_data is not None and "b_hitch" in tube_data:
        bh_deg = float(np.rad2deg(np.max(tube_data["b_hitch"])))
        ax_h.axhline(phi_max_deg - bh_deg, color="orange", linestyle=":", linewidth=1.0)
        ax_h.axhline(-(phi_max_deg - bh_deg), color="orange", linestyle=":", linewidth=1.0)
    ax_h.set_ylabel("hitch [deg]")
    ax_h.grid(True, alpha=0.3)

    ax_c.plot(t_axis, clearances, "g-", linewidth=1.8)
    ax_c.axhline(0.0, color="r", linestyle="--", linewidth=1.0)
    ax_c.set_ylabel("clearance [m]")
    ax_c.set_xlabel("time [s]")
    ax_c.grid(True, alpha=0.3)

    for ax in (ax_v, ax_h, ax_c):
        for k, sh in enumerate(interventions):
            if sh:
                ax.axvspan(k * dt, (k + 1) * dt, color="crimson", alpha=0.14)
        for rs in replan_steps:
            ax.axvline(rs * dt, color="goldenrod", linestyle="--", linewidth=1.3)

    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    return out_file
