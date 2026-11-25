"""
Standalone Nonlinear Model Predictive Control (NMPC) Planner using CasADi + IPOPT.

This module uses the exact same environments (`mbd.envs.Env`) and vehicle classes
(`TractorTrailer2d`, `AccTractorTrailer2d`, `KinematicBicycle2d`, `NTrailer2d`) as
`mbd_planner.py`, without modifying any existing codebase files.

Supports:
1. Full-horizon direct multiple-shooting optimal control (`mpc_mode=False`, default).
2. Receding-horizon closed-loop NMPC (`mpc_mode=True`), re-solving the NLP at each step.
3. All four vehicle dynamics models (`tt2d`, `acc_tt2d`, `kinematic_bicycle2d`, `n_trailer2d`).
4. All four perturbation levels (Levels 0, 1, 2, 3), saving both `nominal_trajectory_states.npy`
   and `perturbed_trajectory_states.npy` under `results/{env_name}/`.
"""

from dataclasses import dataclass
from functools import partial
import heapq
import logging
import os
import time
import casadi as ca
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import mbd
from mbd.planners.mbd_planner import MBDConfig
from mbd.utils import (
    rollout_us_with_terminal,
    rollout_perturbed_us,
    save_env_metadata,
    create_animation,
)


@dataclass
class NMPCConfig(MBDConfig):
    """Configuration for the IPOPT-based NMPC planner (extends MBDConfig for drop-in compatibility)."""

    # NMPC / IPOPT solver settings
    mpc_mode: bool = False                 # False: full-horizon trajectory optimization; True: receding-horizon closed-loop MPC
    mpc_horizon: int = 15                  # Prediction horizon when mpc_mode=True
    ipopt_max_iter: int = 400              # Maximum IPOPT iterations for full-horizon solve
    ipopt_mpc_max_iter: int = 45           # Maximum IPOPT iterations per receding-horizon step
    ipopt_tol: float = 1e-4                # IPOPT convergence tolerance
    ipopt_print_level: int = 0             # IPOPT verbosity (0=silent, 5=detailed)

    # Objective weights
    w_stage_pos: float = 2.0               # Stage position tracking weight
    w_stage_heading: float = 1.5           # Stage heading tracking weight
    w_stage_hitch: float = 0.5             # Stage hitch-angle regularization weight
    w_terminal_pos: float = 120.0          # Terminal position weight
    w_terminal_heading: float = 80.0       # Terminal heading weight
    w_terminal_vel: float = 20.0           # Terminal velocity/steering regularization (for acc_tt2d)
    w_control: float = 0.05                # Control magnitude regularization
    w_control_rate: float = 0.4            # Control rate (smoothness) regularization
    w_obs_slack: float = 5000.0            # L1/L2 penalty on obstacle safety-margin slack

    # Safety margins for smooth collision & hitch constraints
    obstacle_safety_margin: float = 0.25   # Extra clearance [m] around vehicle body discs
    hitch_safety_margin_deg: float = 1.5   # Safety buffer [deg] inside phi_max (40 deg)

    # Disable diffusion-sampler backup shield by default during deterministic NMPC rollout
    enable_shielded_rollout_collision: bool = False
    enable_shielded_rollout_hitch: bool = False


def _wrap_angle_np(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _point_in_obstacles_np(px: float, py: float, obs_circles: np.ndarray, obs_rectangles: np.ndarray, clearance: float) -> bool:
    """Check if a 2D point (px, py) is within `clearance` of any circular or rectangular obstacle."""
    if obs_circles is not None and len(obs_circles) > 0:
        for c in obs_circles:
            cx, cy, cr = float(c[0]), float(c[1]), float(c[2])
            if cr <= 1e-6:
                continue
            if (px - cx) ** 2 + (py - cy) ** 2 < (cr + clearance) ** 2:
                return True

    if obs_rectangles is not None and len(obs_rectangles) > 0:
        for r in obs_rectangles:
            rx, ry, rw, rh, rang = float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])
            dx = px - rx
            dy = py - ry
            cos_a = np.cos(-rang)
            sin_a = np.sin(-rang)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            qx = max(abs(lx) - rw / 2.0, 0.0)
            qy = max(abs(ly) - rh / 2.0, 0.0)
            if abs(lx) <= rw / 2.0 and abs(ly) <= rh / 2.0:
                return True
            if qx * qx + qy * qy < clearance * clearance:
                return True

    return False


def _compute_astar_warmstart(env, H: int) -> np.ndarray:
    """
    Compute a collision-free 2D geometric warm-start trajectory of shape (H + 1, Nx)
    using a coarse 8-connected A* search over the workspace.
    """
    nx = int(env.observation_size)
    x0 = np.asarray(env.x0, dtype=float)
    xg = np.asarray(env.xg, dtype=float)

    obs_circles = np.asarray(env.obs_circles, dtype=float) if hasattr(env, "obs_circles") else np.zeros((0, 3))
    obs_rectangles = np.asarray(env.obs_rectangles, dtype=float) if hasattr(env, "obs_rectangles") else np.zeros((0, 5))

    x_range, y_range = env.env.get_plot_limits()
    min_x = min(float(x_range[0]), float(x0[0]) - 4.0, float(xg[0]) - 4.0)
    max_x = max(float(x_range[1]), float(x0[0]) + 4.0, float(xg[0]) + 4.0)
    min_y = min(float(y_range[0]), float(x0[1]) - 4.0, float(xg[1]) - 4.0)
    max_y = max(float(y_range[1]), float(x0[1]) + 4.0, float(xg[1]) + 4.0)

    res = 0.8  # Grid resolution [m]
    xs_grid = np.arange(min_x, max_x + res, res)
    ys_grid = np.arange(min_y, max_y + res, res)
    W_grid, H_grid = len(xs_grid), len(ys_grid)

    def to_idx(px, py):
        ix = int(np.clip(round((px - min_x) / res), 0, W_grid - 1))
        iy = int(np.clip(round((py - min_y) / res), 0, H_grid - 1))
        return ix, iy

    start_idx = to_idx(x0[0], x0[1])
    goal_idx = to_idx(xg[0], xg[1])

    # Try decreasing clearances if tight parking spaces block the goal cell
    path_xy = None
    base_clearance = max(float(env.tractor_width), float(env.trailer_width)) * 0.55
    for clearance in [base_clearance, base_clearance * 0.75, 0.8, 0.4]:
        occ = np.zeros((W_grid, H_grid), dtype=bool)
        for ix in range(W_grid):
            for iy in range(H_grid):
                occ[ix, iy] = _point_in_obstacles_np(xs_grid[ix], ys_grid[iy], obs_circles, obs_rectangles, clearance)
        occ[start_idx[0], start_idx[1]] = False
        occ[goal_idx[0], goal_idx[1]] = False

        open_heap = [(0.0, start_idx)]
        came_from = {}
        g_score = {start_idx: 0.0}
        neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                     (-1, -1, np.sqrt(2.0)), (-1, 1, np.sqrt(2.0)),
                     (1, -1, np.sqrt(2.0)), (1, 1, np.sqrt(2.0))]

        found = False
        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal_idx:
                found = True
                break
            for dx_i, dy_i, step_cost in neighbors:
                nxt = (current[0] + dx_i, current[1] + dy_i)
                if not (0 <= nxt[0] < W_grid and 0 <= nxt[1] < H_grid):
                    continue
                if occ[nxt[0], nxt[1]]:
                    continue
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(nxt, float("inf")):
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    h = np.hypot(nxt[0] - goal_idx[0], nxt[1] - goal_idx[1])
                    heapq.heappush(open_heap, (tentative_g + h, nxt))

        if found:
            cells = [goal_idx]
            curr = goal_idx
            while curr in came_from:
                curr = came_from[curr]
                cells.append(curr)
            cells.reverse()
            pts = np.array([[xs_grid[ix], ys_grid[iy]] for ix, iy in cells], dtype=float)
            pts[0] = x0[:2]
            pts[-1] = xg[:2]
            path_xy = pts
            break

    if path_xy is None or len(path_xy) < 2:
        path_xy = np.vstack([x0[:2], xg[:2]])

    # Resample path_xy to H + 1 evenly-spaced arc-length points
    diffs = np.diff(path_xy, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = cum_len[-1]

    if total_len < 1e-6:
        interp_xy = np.tile(x0[:2], (H + 1, 1))
    else:
        target_s = np.linspace(0.0, total_len, H + 1)
        interp_x = np.interp(target_s, cum_len, path_xy[:, 0])
        interp_y = np.interp(target_s, cum_len, path_xy[:, 1])
        interp_xy = np.column_stack([interp_x, interp_y])

    # Smooth the resampled 2D path slightly while pinning endpoints
    if H >= 4:
        smoothed = interp_xy.copy()
        for _ in range(3):
            smoothed[1:-1] = 0.25 * smoothed[:-2] + 0.5 * smoothed[1:-1] + 0.25 * smoothed[2:]
        interp_xy = smoothed

    # Construct full state warm-start (H + 1, Nx)
    x_guess = np.zeros((H + 1, nx), dtype=float)
    x_guess[:, :2] = interp_xy
    x_guess[0] = x0

    backward_mode = int(getattr(env, "motion_preference", 0)) in (-1, -2)
    thetas = np.zeros(H + 1, dtype=float)
    thetas[0] = x0[2]
    for k in range(1, H):
        dx = interp_xy[k + 1, 0] - interp_xy[k - 1, 0]
        dy = interp_xy[k + 1, 1] - interp_xy[k - 1, 1]
        if np.hypot(dx, dy) > 1e-4:
            raw_th = np.arctan2(dy, dx)
            if backward_mode:
                raw_th = _wrap_angle_np(raw_th + np.pi)
            # Unwrap relative to previous angle for smooth continuity
            thetas[k] = thetas[k - 1] + _wrap_angle_np(raw_th - thetas[k - 1])
        else:
            thetas[k] = thetas[k - 1]
    thetas[H] = thetas[H - 1] + _wrap_angle_np(xg[2] - thetas[H - 1])

    # Blend last few headings smoothly into goal heading
    blend_steps = min(8, H)
    for idx, k in enumerate(range(H - blend_steps + 1, H + 1)):
        alpha = (idx + 1) / blend_steps
        goal_unwrapped = thetas[k] + _wrap_angle_np(xg[2] - thetas[k])
        thetas[k] = (1.0 - alpha) * thetas[k] + alpha * goal_unwrapped

    x_guess[:, 2] = thetas
    if nx == 4:
        x_guess[:, 3] = thetas
        x_guess[0, 3] = x0[3]
    elif nx == 6:
        x_guess[:, 3] = thetas
        x_guess[0, 3] = x0[3]
        # Estimate longitudinal speed for acc_tt2d
        avg_v = (total_len / max(H * float(env.dt), 1e-3)) * (-1.0 if backward_mode else 1.0)
        avg_v = float(np.clip(avg_v, -float(env.v_max) * 0.8, float(env.v_max) * 0.8))
        x_guess[1:H, 4] = avg_v
        x_guess[0, 4] = x0[4]
        x_guess[H, 4] = 0.0
        x_guess[:, 5] = 0.0
    elif nx > 4:
        # N-trailer
        for j in range(3, nx):
            x_guess[:, j] = thetas
            x_guess[0, j] = x0[j]

    return x_guess


class IPOPTNMPCPlanner:
    """
    CasADi + IPOPT Nonlinear Model Predictive Control (NMPC) solver compatible with
    TractorTrailer2d, AccTractorTrailer2d, KinematicBicycle2d, and NTrailer2d.
    """

    def __init__(self, env, config: NMPCConfig):
        self.env = env
        self.config = config
        self.nx = int(env.observation_size)
        self.nu = int(env.action_size)
        self.dt = float(env.dt)

        cls_name = type(env).__name__
        if cls_name == "KinematicBicycle2d" or self.nx == 3:
            self.model_type = "kinematic_bicycle2d"
        elif cls_name == "AccTractorTrailer2d" or self.nx == 6:
            self.model_type = "acc_tt2d"
        elif cls_name == "NTrailer2d" or self.nx > 4:
            self.model_type = "n_trailer2d"
        else:
            self.model_type = "tt2d"

        self._solver_cache = {}

    def _casadi_step(self, x: ca.MX, u_norm: ca.MX) -> ca.MX:
        """Symbolic Forward-Euler step matching the JAX vehicle dynamics in mbd/robots/."""
        dt = self.dt
        l1 = float(self.env.l1)
        l2 = float(self.env.l2)
        lh = float(self.env.lh)
        v_max = float(self.env.v_max)
        delta_max = float(self.env.delta_max)

        if self.model_type == "kinematic_bicycle2d":
            px, py, theta1 = x[0], x[1], x[2]
            v = u_norm[0] * v_max
            delta = u_norm[1] * delta_max
            px_dot = v * ca.cos(theta1)
            py_dot = v * ca.sin(theta1)
            theta1_dot = (v / l1) * ca.tan(delta)
            return ca.vertcat(px + dt * px_dot, py + dt * py_dot, theta1 + dt * theta1_dot)

        elif self.model_type == "tt2d":
            px, py, theta1, theta2 = x[0], x[1], x[2], x[3]
            v = u_norm[0] * v_max
            delta = u_norm[1] * delta_max
            px_dot = v * ca.cos(theta1)
            py_dot = v * ca.sin(theta1)
            theta1_dot = (v / l1) * ca.tan(delta)
            theta2_dot = (v / l2) * (
                ca.sin(theta1 - theta2) - (lh / l1) * ca.cos(theta1 - theta2) * ca.tan(delta)
            )
            return ca.vertcat(
                px + dt * px_dot,
                py + dt * py_dot,
                theta1 + dt * theta1_dot,
                theta2 + dt * theta2_dot,
            )

        elif self.model_type == "acc_tt2d":
            px, py, theta1, theta2, v, delta = x[0], x[1], x[2], x[3], x[4], x[5]
            a_max = float(getattr(self.env, "a_max", 2.0))
            omega_max = float(getattr(self.env, "omega_max", 1.0))
            a = u_norm[0] * a_max
            omega = u_norm[1] * omega_max

            px_dot = v * ca.cos(theta1)
            py_dot = v * ca.sin(theta1)
            theta1_dot = (v / l1) * ca.tan(delta)
            theta2_dot = (v / l2) * (
                ca.sin(theta1 - theta2) - (lh / l1) * ca.cos(theta1 - theta2) * ca.tan(delta)
            )
            return ca.vertcat(
                px + dt * px_dot,
                py + dt * py_dot,
                theta1 + dt * theta1_dot,
                theta2 + dt * theta2_dot,
                v + dt * a,
                delta + dt * omega,
            )

        else:  # n_trailer2d
            num_trailers = int(self.env.num_trailers)
            trailer_lengths = [float(L) for L in np.asarray(self.env.trailer_lengths)]
            px, py, theta0 = x[0], x[1], x[2]
            v = u_norm[0] * v_max
            delta = u_norm[1] * delta_max

            px_dot = v * ca.cos(theta0)
            py_dot = v * ca.sin(theta0)
            theta0_dot = (v / l1) * ca.tan(delta)

            next_states = [px + dt * px_dot, py + dt * py_dot, theta0 + dt * theta0_dot]
            prev_theta = theta0
            prev_dot = theta0_dot
            prev_axle_speed = v

            for i in range(num_trailers):
                theta_i = x[3 + i]
                L_i = trailer_lengths[i]
                dtheta = prev_theta - theta_i
                if i == 0:
                    theta_i_dot = (v / L_i) * (
                        ca.sin(dtheta) - (lh / l1) * ca.cos(dtheta) * ca.tan(delta)
                    )
                    axle_speed_i = v * ca.cos(dtheta) + lh * theta0_dot * ca.sin(dtheta)
                else:
                    theta_i_dot = (prev_axle_speed / L_i) * ca.sin(dtheta) - (lh / L_i) * prev_dot * ca.cos(dtheta)
                    axle_speed_i = prev_axle_speed * ca.cos(dtheta) + lh * prev_dot * ca.sin(dtheta)

                next_states.append(theta_i + dt * theta_i_dot)
                prev_theta = theta_i
                prev_dot = theta_i_dot
                prev_axle_speed = axle_speed_i

            return ca.vertcat(*next_states)

    def _get_body_discs_symbolic(self, x: ca.MX, extra_margin: float):
        """
        Return a list of (disc_x, disc_y, disc_radius) covering the tractor and all trailer rectangles.
        Each rectangular body is covered by 3 overlapping discs (rear, center, front) aligned with
        the exact length and width of the body plus extra_margin.
        """
        l1 = float(self.env.l1)
        l2 = float(self.env.l2)
        lh = float(self.env.lh)
        lf1 = float(self.env.lf1)
        lr = float(self.env.lr)
        lf2 = float(self.env.lf2)
        lr2 = float(self.env.lr2)
        w_trac = float(self.env.tractor_width)
        w_trail = float(self.env.trailer_width)

        px, py, theta1 = x[0], x[1], x[2]
        c1, s1 = ca.cos(theta1), ca.sin(theta1)

        # Tractor body center & half-length (matches get_tractor_trailer_rectangles in tt2d.py)
        trac_cx = px + 0.5 * (l1 + lf1 - lr) * c1
        trac_cy = py + 0.5 * (l1 + lf1 - lr) * s1
        trac_half_len = 0.5 * (l1 + lf1 + lr)
        trac_r = 0.5 * w_trac + extra_margin
        trac_offset = max(0.0, trac_half_len - 0.5 * w_trac)

        discs = [
            (trac_cx - trac_offset * c1, trac_cy - trac_offset * s1, trac_r),
            (trac_cx, trac_cy, trac_r),
            (trac_cx + trac_offset * c1, trac_cy + trac_offset * s1, trac_r),
        ]

        if self.model_type == "kinematic_bicycle2d":
            return discs

        if self.model_type in ("tt2d", "acc_tt2d"):
            theta2 = x[3]
            c2, s2 = ca.cos(theta2), ca.sin(theta2)
            hx = px - lh * c1
            hy = py - lh * s1
            trail_cx = hx + (0.5 * (lf2 - lr2) - l2) * c2
            trail_cy = hy + (0.5 * (lf2 - lr2) - l2) * s2
            trail_half_len = 0.5 * (lf2 + lr2)
            trail_r = 0.5 * w_trail + extra_margin
            trail_offset = max(0.0, trail_half_len - 0.5 * w_trail)
            discs.extend([
                (trail_cx - trail_offset * c2, trail_cy - trail_offset * s2, trail_r),
                (trail_cx, trail_cy, trail_r),
                (trail_cx + trail_offset * c2, trail_cy + trail_offset * s2, trail_r),
            ])
            return discs

        # n_trailer2d
        num_trailers = int(self.env.num_trailers)
        trailer_lengths = [float(L) for L in np.asarray(self.env.trailer_lengths)]
        front_offsets = [float(f) for f in np.asarray(self.env.trailer_front_offsets)]
        rear_offsets = [float(r) for r in np.asarray(self.env.trailer_rear_offsets)]

        prev_hx = px - lh * c1
        prev_hy = py - lh * s1
        trail_r = 0.5 * w_trail + extra_margin

        for i in range(num_trailers):
            th_i = x[3 + i]
            ci, si = ca.cos(th_i), ca.sin(th_i)
            Lk = trailer_lengths[i]
            lf_k = front_offsets[i]
            lr_k = rear_offsets[i]
            s_center = 0.5 * (lf_k - lr_k) - Lk
            cx = prev_hx + s_center * ci
            cy = prev_hy + s_center * si
            half_len = 0.5 * (lf_k + lr_k)
            offset = max(0.0, half_len - 0.5 * w_trail)
            discs.extend([
                (cx - offset * ci, cy - offset * si, trail_r),
                (cx, cy, trail_r),
                (cx + offset * ci, cy + offset * si, trail_r),
            ])
            rear_x = cx - half_len * ci
            rear_y = cy - half_len * si
            prev_hx = rear_x - lh * ci
            prev_hy = rear_y - lh * si

        return discs

    def _build_solver(self, H: int, safety_margin: float, max_iter: int):
        """Build and cache the CasADi IPOPT NLP solver for horizon H and obstacle safety_margin."""
        cache_key = (H, round(safety_margin, 4), max_iter)
        if cache_key in self._solver_cache:
            return self._solver_cache[cache_key]

        nx, nu = self.nx, self.nu
        cfg = self.config

        # Decision variables: X (nx, H+1), U (nu, H), and one slack S_k >= 0 per step k=1..H
        X = ca.MX.sym("X", nx, H + 1)
        U = ca.MX.sym("U", nu, H)
        S = ca.MX.sym("S", H)

        # Parameters: X0 (nx) and X_ref (nx, H+1)
        P_x0 = ca.MX.sym("P_x0", nx)
        P_xref = ca.MX.sym("P_xref", nx, H + 1)

        g_list = []
        lbg_list = []
        ubg_list = []

        # 1. Initial state constraint: X[:, 0] == P_x0
        g_list.append(X[:, 0] - P_x0)
        lbg_list.extend([0.0] * nx)
        ubg_list.extend([0.0] * nx)

        # Active obstacles
        obs_circles = np.asarray(self.env.obs_circles, dtype=float) if hasattr(self.env, "obs_circles") else np.zeros((0, 3))
        obs_circles = obs_circles[obs_circles[:, 2] > 1e-5] if len(obs_circles) > 0 else np.zeros((0, 3))
        obs_rectangles = np.asarray(self.env.obs_rectangles, dtype=float) if hasattr(self.env, "obs_rectangles") else np.zeros((0, 5))

        phi_max = float(self.env.phi_max) - np.deg2rad(cfg.hitch_safety_margin_deg)
        max_cos_violation = 1.0 - np.cos(phi_max)

        obj = 0.0
        eps_smooth = 1e-3

        def smax0(z):
            return 0.5 * (z + ca.sqrt(z * z + eps_smooth * eps_smooth))

        for k in range(H):
            xk = X[:, k]
            uk = U[:, k]
            x_next = X[:, k + 1]
            xref_next = P_xref[:, k + 1]
            sk = S[k]

            # Multiple-shooting dynamics equality constraint
            x_pred = self._casadi_step(xk, uk)
            g_list.append(x_next - x_pred)
            lbg_list.extend([0.0] * nx)
            ubg_list.extend([0.0] * nx)

            # Stage tracking cost toward reference / goal
            pos_err_sq = (x_next[0] - xref_next[0]) ** 2 + (x_next[1] - xref_next[1]) ** 2
            hdg_err = 1.0 - ca.cos(x_next[2] - xref_next[2])
            if nx >= 4:
                for j in range(3, min(nx, 3 + int(getattr(self.env, "num_trailers", 1)))):
                    hdg_err = hdg_err + (1.0 - ca.cos(x_next[j] - xref_next[j]))

            obj = obj + cfg.w_stage_pos * pos_err_sq + cfg.w_stage_heading * hdg_err

            # Control effort & smoothness
            obj = obj + cfg.w_control * ca.sumsqr(uk)
            if k > 0:
                obj = obj + cfg.w_control_rate * ca.sumsqr(uk - U[:, k - 1])

            # Soft motion preference penalty for +/-1
            pref = int(getattr(self.env, "motion_preference", 0))
            v_var = x_next[4] if self.model_type == "acc_tt2d" else uk[0]
            if pref == 1:
                obj = obj + 2.0 * smax0(-v_var) ** 2
            elif pref == -1:
                obj = obj + 2.0 * smax0(v_var) ** 2

            # Hitch-angle constraints and regularization
            if self.model_type in ("tt2d", "acc_tt2d"):
                hitch_cos = 1.0 - ca.cos(x_next[2] - x_next[3])
                obj = obj + cfg.w_stage_hitch * hitch_cos
                g_list.append(hitch_cos)
                lbg_list.append(-ca.inf)
                ubg_list.append(max_cos_violation)
            elif self.model_type == "n_trailer2d":
                num_tr = int(self.env.num_trailers)
                for j in range(num_tr):
                    hitch_cos = 1.0 - ca.cos(x_next[2 + j] - x_next[3 + j])
                    obj = obj + cfg.w_stage_hitch * hitch_cos
                    g_list.append(hitch_cos)
                    lbg_list.append(-ca.inf)
                    ubg_list.append(max_cos_violation)

            # Full-body obstacle avoidance constraints at x_next
            discs = self._get_body_discs_symbolic(x_next, safety_margin)

            # Circular obstacles: (dx^2 + dy^2) + sk >= (r_obs + r_disc)^2
            for c in obs_circles:
                cx, cy, cr = float(c[0]), float(c[1]), float(c[2])
                for dx_sym, dy_sym, r_disc in discs:
                    dist_sq = (dx_sym - cx) ** 2 + (dy_sym - cy) ** 2
                    min_dist_sq = (cr + r_disc) ** 2
                    g_list.append(dist_sq + sk - min_dist_sq)
                    lbg_list.append(0.0)
                    ubg_list.append(ca.inf)

            # Rectangular obstacles: smooth Euclidean distance squared to oriented box + sk >= r_disc^2
            for r in obs_rectangles:
                rx, ry, rw, rh, rang = float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])
                cos_a = float(np.cos(-rang))
                sin_a = float(np.sin(-rang))
                hw, hh = 0.5 * rw, 0.5 * rh
                for dx_sym, dy_sym, r_disc in discs:
                    rel_x = dx_sym - rx
                    rel_y = dy_sym - ry
                    loc_x = rel_x * cos_a - rel_y * sin_a
                    loc_y = rel_x * sin_a + rel_y * cos_a
                    abs_x = ca.sqrt(loc_x * loc_x + eps_smooth * eps_smooth)
                    abs_y = ca.sqrt(loc_y * loc_y + eps_smooth * eps_smooth)
                    out_x = smax0(abs_x - hw)
                    out_y = smax0(abs_y - hh)
                    box_dist_sq = out_x * out_x + out_y * out_y
                    g_list.append(box_dist_sq + sk - r_disc * r_disc)
                    lbg_list.append(0.0)
                    ubg_list.append(ca.inf)

            # Penalize slack heavily so sk -> 0 whenever feasible
            obj = obj + cfg.w_obs_slack * (sk + 10.0 * sk * sk)

        # Terminal cost at k = H
        x_end = X[:, H]
        xg_sym = P_xref[:, H]
        term_pos_sq = (x_end[0] - xg_sym[0]) ** 2 + (x_end[1] - xg_sym[1]) ** 2
        term_hdg = 1.0 - ca.cos(x_end[2] - xg_sym[2])
        if nx >= 4:
            for j in range(3, min(nx, 3 + int(getattr(self.env, "num_trailers", 1)))):
                term_hdg = term_hdg + (1.0 - ca.cos(x_end[j] - xg_sym[j]))

        obj = obj + cfg.w_terminal_pos * term_pos_sq + cfg.w_terminal_heading * term_hdg
        if self.model_type == "acc_tt2d":
            obj = obj + cfg.w_terminal_vel * (x_end[4] ** 2 + 0.5 * x_end[5] ** 2)

        # Pack decision vector z = [vec(X); vec(U); S]
        z_var = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1), S)
        p_var = ca.vertcat(P_x0, ca.reshape(P_xref, -1, 1))
        g_vec = ca.vertcat(*g_list)

        # Variable bounds lbx, ubx
        n_X = nx * (H + 1)
        n_U = nu * H
        n_S = H

        lbx_X = np.full((nx, H + 1), -ca.inf)
        ubx_X = np.full((nx, H + 1), ca.inf)
        if self.model_type == "acc_tt2d":
            v_max = float(self.env.v_max)
            delta_max = float(self.env.delta_max)
            pref = int(getattr(self.env, "motion_preference", 0))
            lbx_X[4, :] = 0.0 if pref == 2 else -v_max
            ubx_X[4, :] = 0.0 if pref == -2 else v_max
            lbx_X[5, :] = -delta_max
            ubx_X[5, :] = delta_max

        lbx_U = np.full((nu, H), -1.0)
        ubx_U = np.full((nu, H), 1.0)
        pref = int(getattr(self.env, "motion_preference", 0))
        if self.model_type != "acc_tt2d":
            if pref == 2:
                lbx_U[0, :] = 0.0
            elif pref == -2:
                ubx_U[0, :] = 0.0

        lbx_S = np.zeros(n_S)
        ubx_S = np.full(n_S, ca.inf)

        lbx = np.concatenate([lbx_X.reshape(-1, order="F"), lbx_U.reshape(-1, order="F"), lbx_S])
        ubx = np.concatenate([ubx_X.reshape(-1, order="F"), ubx_U.reshape(-1, order="F"), ubx_S])
        lbg = np.array(lbg_list, dtype=float)
        ubg = np.array(ubg_list, dtype=float)

        nlp = {"x": z_var, "p": p_var, "f": obj, "g": g_vec}
        opts = {
            "ipopt.max_iter": int(max_iter),
            "ipopt.tol": float(cfg.ipopt_tol),
            "ipopt.acceptable_tol": float(cfg.ipopt_tol) * 10.0,
            "ipopt.print_level": int(cfg.ipopt_print_level),
            "ipopt.sb": "yes",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.warm_start_init_point": "yes",
            "print_time": 0,
        }
        solver = ca.nlpsol(f"nmpc_ipopt_H{H}", "ipopt", nlp, opts)
        bundle = (solver, lbx, ubx, lbg, ubg, n_X, n_U, n_S)
        self._solver_cache[cache_key] = bundle
        return bundle

    def solve_horizon(
        self,
        x0_val: np.ndarray,
        xref_seq: np.ndarray,
        x_init_guess: np.ndarray,
        u_init_guess: np.ndarray,
        safety_margin: float | None = None,
        max_iter: int | None = None,
    ):
        """Solve the H-step NLP starting from x0_val and return (X_opt, U_opt, stats)."""
        H = int(u_init_guess.shape[0])
        margin = self.config.obstacle_safety_margin if safety_margin is None else safety_margin
        iters = self.config.ipopt_max_iter if max_iter is None else max_iter

        solver, lbx, ubx, lbg, ubg, n_X, n_U, n_S = self._build_solver(H, margin, iters)

        z0 = np.concatenate([
            x_init_guess.T.reshape(-1, order="F"),
            u_init_guess.T.reshape(-1, order="F"),
            np.zeros(n_S),
        ])
        p_val = np.concatenate([
            x0_val.reshape(-1),
            xref_seq.T.reshape(-1, order="F"),
        ])

        sol = solver(x0=z0, p=p_val, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        z_opt = np.asarray(sol["x"]).reshape(-1)

        X_opt = z_opt[:n_X].reshape((self.nx, H + 1), order="F").T
        U_opt = z_opt[n_X:n_X + n_U].reshape((self.nu, H), order="F").T
        U_opt = np.clip(U_opt, -1.0, 1.0)
        return X_opt, U_opt, solver.stats()


def _check_jax_trajectory_violations(env, states_np: np.ndarray) -> tuple[bool, bool]:
    """Verify trajectory against the exact JAX SAT collision and hitch-angle checkers."""
    is_bicycle = int(env.observation_size) == 3
    is_multi = getattr(env, "num_trailers", 1) >= 2
    col_any = False
    hitch_any = False
    for s in states_np:
        s_chk = jnp.asarray(s[:3] if is_bicycle else (s if is_multi else s[:4]))
        if bool(env.check_obstacle_collision(s_chk, env.obs_circles, env.obs_rectangles)):
            col_any = True
        if not is_bicycle and bool(env.check_hitch_violation(s_chk)):
            hitch_any = True
        if col_any and hitch_any:
            break
    return col_any, hitch_any


def run_nmpc(args=None, env=None):
    """
    Run the IPOPT Nonlinear Model Predictive Control (NMPC) planner on `env`.

    Has the exact same arguments and 4-tuple return signature as `run_diffusion(args, env)`:
        rew_final, Y0, trajectory_states, timing_info = run_nmpc(args=config, env=env)
    """
    if args is None:
        args = NMPCConfig()
    elif not isinstance(args, NMPCConfig):
        # Upgrade an MBDConfig or dict into NMPCConfig while preserving all fields
        if isinstance(args, dict):
            from mbd.planners.mbd_planner import dict_to_config_obj
            args = dict_to_config_obj(args)
        args = NMPCConfig(**{k: getattr(args, k) for k in args.__dataclass_fields__ if k in NMPCConfig.__dataclass_fields__})

    total_start_time = time.time()

    # Sync perturbation settings from args to env unless already set directly on env
    if hasattr(env, "set_perturbation") and not getattr(env, "_perturbation_manually_set", False):
        if getattr(args, "perturbation_level", 0) != getattr(env, "perturbation_level", 0):
            env.set_perturbation(
                level=args.perturbation_level,
                seed=args.perturbation_seed,
                control_noise_std=args.control_noise_std,
                param_uncertainty_std=args.param_uncertainty_std,
                non_gaussian_dist=args.non_gaussian_dist,
                non_gaussian_df=args.non_gaussian_df,
                state_disturbance_scale=args.state_disturbance_scale,
                disturbance_clip_factor=args.disturbance_clip_factor,
                perturbed_l1=args.perturbed_l1,
                perturbed_l2=args.perturbed_l2,
                perturbed_lh=args.perturbed_lh,
            )
            env._perturbation_manually_set = False

    H = int(args.Hsample)
    nx = int(env.observation_size)
    nu = int(env.action_size)

    # 1. Generate A* geometric warm-start path
    warmstart_t0 = time.time()
    x_warm = _compute_astar_warmstart(env, H)
    u_warm = np.zeros((H, nu), dtype=float)
    warmstart_time = time.time() - warmstart_t0

    # Build reference sequence (blends A* waypoint guidance into exact terminal goal xg)
    xg_np = np.asarray(env.xg, dtype=float)
    xref_seq = x_warm.copy()
    for k in range(H + 1):
        alpha = (k / max(H, 1)) ** 2
        xref_seq[k, :2] = (1.0 - alpha) * x_warm[k, :2] + alpha * xg_np[:2]
    xref_seq[-1] = xg_np

    # 2. Build and solve IPOPT NLP for the nominal trajectory
    setup_t0 = time.time()
    planner = IPOPTNMPCPlanner(env, args)
    # Trigger solver build so build time is separated from pure solve time
    planner._build_solver(H, args.obstacle_safety_margin, args.ipopt_max_iter)
    setup_time = time.time() - setup_t0

    solve_t0 = time.time()
    x0_np = np.asarray(env.x0, dtype=float)
    X_opt, U_opt, stats = planner.solve_horizon(
        x0_val=x0_np,
        xref_seq=xref_seq,
        x_init_guess=x_warm,
        u_init_guess=u_warm,
        safety_margin=args.obstacle_safety_margin,
        max_iter=args.ipopt_max_iter,
    )

    # Verify against exact JAX SAT collision checker; if a corner touches, refine via gentle margin continuation
    col_found, hitch_found = _check_jax_trajectory_violations(env, X_opt)
    if col_found:
        for extra in [0.08, 0.15, 0.22]:
            X_opt, U_opt, stats = planner.solve_horizon(
                x0_val=x0_np,
                xref_seq=xref_seq,
                x_init_guess=X_opt,
                u_init_guess=U_opt,
                safety_margin=args.obstacle_safety_margin + extra,
                max_iter=150,
            )
            col_found, hitch_found = _check_jax_trajectory_violations(env, X_opt)
            if not col_found:
                break

    pure_solve_time = time.time() - solve_t0

    # 3. Roll out nominal actions Y0 through the exact JAX environment
    post_t0 = time.time()
    want_shield_col = bool(getattr(args, "enable_shielded_rollout_collision", False))
    want_shield_hitch = bool(getattr(args, "enable_shielded_rollout_hitch", False))
    if (
        getattr(env, "enable_shielded_rollout_collision", False) != want_shield_col
        or getattr(env, "enable_shielded_rollout_hitch", False) != want_shield_hitch
    ):
        env.enable_shielded_rollout_collision = want_shield_col
        env.enable_shielded_rollout_hitch = want_shield_hitch
        for cls in type(env).__mro__:
            for fn_name in ("step", "_step_internal"):
                fn = cls.__dict__.get(fn_name)
                if fn is not None and hasattr(fn, "clear_cache"):
                    fn.clear_cache()

    rng = jax.random.PRNGKey(seed=args.seed)
    state_init = env.reset(rng)
    step_env_jit = jax.jit(env.step)
    rollout_terminal_jit = jax.jit(partial(rollout_us_with_terminal, step_env_jit, env))

    Y0 = jnp.asarray(U_opt, dtype=jnp.float32)
    rew_final, _, Y0_applied = rollout_terminal_jit(state_init, Y0)
    if want_shield_col or want_shield_hitch:
        Y0 = Y0_applied

    # Compute full nominal state sequence (H + 1, Nx)
    xs_nom = jnp.zeros((H + 1, nx))
    xs_nom = xs_nom.at[0].set(state_init.pipeline_state)
    st = state_init
    for t in range(H):
        st = step_env_jit(st, jnp.array(Y0[t]))
        xs_nom = xs_nom.at[t + 1].set(st.pipeline_state)
    nominal_trajectory_states = xs_nom
    trajectory_states = nominal_trajectory_states

    # 4. Perturbed trajectory rollout (either open-loop rollout of Y0 or closed-loop receding-horizon MPC)
    step_perturbed_jit = jax.jit(env.step_perturbed)
    state_pert_init = state_init.replace(rng=jax.random.PRNGKey(getattr(env, "perturbation_seed", 123)))

    if args.mpc_mode:
        # Receding-horizon closed-loop NMPC: re-solve IPOPT from current perturbed state at each step t
        H_mpc = min(int(args.mpc_horizon), H)
        planner._build_solver(H_mpc, args.obstacle_safety_margin, args.ipopt_mpc_max_iter)

        xs_pert = np.zeros((H + 1, nx), dtype=float)
        us_pert = np.zeros((H, nu), dtype=float)
        xs_pert[0] = np.asarray(state_pert_init.pipeline_state)

        st_p = state_pert_init
        x_curr_guess = X_opt[: H_mpc + 1].copy()
        u_curr_guess = U_opt[:H_mpc].copy()

        for t in range(H):
            curr_x = np.asarray(st_p.pipeline_state, dtype=float)
            # Build local reference window of length H_mpc + 1
            ref_window = np.zeros((H_mpc + 1, nx), dtype=float)
            for k in range(H_mpc + 1):
                idx = min(t + k, H)
                ref_window[k] = xref_seq[idx]

            X_mpc, U_mpc, _ = planner.solve_horizon(
                x0_val=curr_x,
                xref_seq=ref_window,
                x_init_guess=x_curr_guess,
                u_init_guess=u_curr_guess,
                safety_margin=args.obstacle_safety_margin,
                max_iter=args.ipopt_mpc_max_iter,
            )
            u_cmd = jnp.asarray(U_mpc[0], dtype=jnp.float32)
            st_p = step_perturbed_jit(st_p, u_cmd)
            xs_pert[t + 1] = np.asarray(st_p.pipeline_state)
            us_pert[t] = np.asarray(st_p.applied_action)

            # Shift warm-start for next MPC step
            x_curr_guess[:-1] = X_mpc[1:]
            x_curr_guess[-1] = X_mpc[-1]
            u_curr_guess[:-1] = U_mpc[1:]
            u_curr_guess[-1] = U_mpc[-1]

        perturbed_trajectory_states = jnp.asarray(xs_pert, dtype=jnp.float32)
        perturbed_actions = jnp.asarray(us_pert, dtype=jnp.float32)
    else:
        rollout_perturbed_jit = jax.jit(partial(rollout_perturbed_us, step_perturbed_jit))
        _, perturbed_trajectory_states, perturbed_actions = rollout_perturbed_jit(state_pert_init, Y0)

    env.nominal_trajectory_states = nominal_trajectory_states
    env.perturbed_trajectory_states = perturbed_trajectory_states
    env.perturbed_actions = perturbed_actions

    post_processing_time = time.time() - post_t0
    total_time = time.time() - total_start_time

    timing_info = {
        "total_time": total_time,
        "warmstart_time": warmstart_time,
        "compilation_time": setup_time,
        "pure_diffusion_time": pure_solve_time,  # Kept key name for drop-in compatibility with MBD scripts
        "pure_nmpc_time": pure_solve_time,
        "post_processing_time": post_processing_time,
        "overhead_time": total_time - pure_solve_time,
        "ipopt_return_status": stats.get("return_status", "unknown"),
        "ipopt_iter_count": int(stats.get("iter_count", 0)),
        "perturbation_info": env.get_perturbation_info() if hasattr(env, "get_perturbation_info") else {},
    }

    # 5. Save nominal trajectory and then perturbed trajectory in results/{env_name}
    path = f"{mbd.__path__[0]}/../results/{args.env_name}"
    os.makedirs(path, exist_ok=True)
    Ybars_dummy = jnp.expand_dims(Y0, axis=0)
    jnp.save(f"{path}/mu_0ts.npy", Ybars_dummy)
    jnp.save(f"{path}/final_actions.npy", Y0)
    jnp.save(f"{path}/perturbed_actions.npy", perturbed_actions)
    jnp.save(f"{path}/nominal_trajectory_states.npy", nominal_trajectory_states)
    jnp.save(f"{path}/perturbed_trajectory_states.npy", perturbed_trajectory_states)
    jnp.save(f"{path}/trajectory_states.npy", nominal_trajectory_states)
    save_env_metadata(env, args, path)

    # 6. Render static plot and animations if requested
    if args.render:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        env.render(ax, nominal_trajectory_states)
        pert_level = getattr(env, "perturbation_level", 0)
        if pert_level > 0:
            ax.scatter(
                perturbed_trajectory_states[:, 0],
                perturbed_trajectory_states[:, 1],
                c=range(len(perturbed_trajectory_states)),
                cmap="Blues",
                s=35,
                alpha=0.8,
            )
            ax.plot(
                perturbed_trajectory_states[:, 0],
                perturbed_trajectory_states[:, 1],
                "b--",
                linewidth=2.0,
                label=f"Perturbed path (Level {pert_level})",
                alpha=0.9,
            )

        ax.set_title(f"IPOPT NMPC Rollout ({args.env_name}, Level {pert_level})")
        ax.legend()
        plt.switch_backend("QtAgg")
        plt.draw()
        plt.savefig(f"{path}/rollout.png")
        plt.savefig(f"{path}/rollout.svg")
        plt.show()

        if args.show_animation or args.save_animation:
            plt.close(fig)

            nominal_states_list = [nominal_trajectory_states[i] for i in range(len(nominal_trajectory_states))]
            nominal_actions_list = [Y0[t] for t in range(Y0.shape[0])] + [None]

            create_animation(
                env,
                nominal_states_list,
                nominal_actions_list,
                args,
                video_name="nominal_trajectory_animation.mp4" if args.save_animation else None,
                trajectory_label="Nominal NMPC trajectory",
            )

            if pert_level > 0:
                perturbed_states_list = [perturbed_trajectory_states[i] for i in range(len(perturbed_trajectory_states))]
                perturbed_actions_list = [perturbed_actions[t] for t in range(perturbed_actions.shape[0])] + [None]

                create_animation(
                    env,
                    perturbed_states_list,
                    perturbed_actions_list,
                    args,
                    video_name="perturbed_trajectory_animation.mp4" if args.save_animation else None,
                    nominal_trajectory_overlay=nominal_states_list,
                )

    return rew_final, Y0, trajectory_states, timing_info


if __name__ == "__main__":
    config = NMPCConfig()
    logging.basicConfig(level=logging.DEBUG if config.verbose else logging.INFO)

    env = mbd.envs.get_env(
        config.env_name,
        case=config.case,
        dt=config.dt,
        H=config.Hsample,
        motion_preference=config.motion_preference,
        collision_penalty=config.collision_penalty,
        enable_shielded_rollout_collision=config.enable_shielded_rollout_collision,
        hitch_penalty=config.hitch_penalty,
        enable_shielded_rollout_hitch=config.enable_shielded_rollout_hitch,
        l1=config.l1,
        l2=config.l2,
        lh=config.lh,
        lf1=config.lf1,
        lr=config.lr,
        lf2=config.lf2,
        lr2=config.lr2,
        tractor_width=config.tractor_width,
        trailer_width=config.trailer_width,
        v_max=config.v_max,
        delta_max_deg=config.delta_max_deg,
        a_max=config.a_max,
        omega_max=config.omega_max,
        num_trailers=config.num_trailers,
        perturbation_level=config.perturbation_level,
        perturbation_seed=config.perturbation_seed,
        control_noise_std=config.control_noise_std,
        param_uncertainty_std=config.param_uncertainty_std,
        non_gaussian_dist=config.non_gaussian_dist,
        non_gaussian_df=config.non_gaussian_df,
        state_disturbance_scale=config.state_disturbance_scale,
    )

    env.set_init_pos(dx=3.0, dy=5.0, theta1=0.0, theta2=0.0)
    if config.num_trailers > 1:
        env.set_init_pos(dx=25.0, dy=15.0, theta1=-jnp.pi / 8, theta2=0.0)
    if config.motion_preference in [-1, -2]:
        env.set_goal_pos(theta1=jnp.pi / 2, theta2=jnp.pi / 2)

    rew_final, Y0, trajectory_states, timing_info = run_nmpc(args=config, env=env)
    print(f"NMPC finished: reward={float(rew_final):.3f}, solve_time={timing_info['pure_nmpc_time']:.3f}s, status={timing_info['ipopt_return_status']}")
