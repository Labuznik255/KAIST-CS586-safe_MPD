"""
CPU Tube-Based Robust NMPC Backup Controller (RMPS Algorithms 2 & 3).

Implemented in Pure Python / NumPy + CasADi (`casadi.Opti()` + `ipopt`), strictly
isolated from JAX tracing/vmap:
- `initialize_backup(x_init, r_tube, b_hitch)` -> `(success, x_ref, u_ref)`
- `is_recoverable(x_cand, r_tube, b_hitch)` -> `bool` (with solution caching)
- `step(x_curr)` -> `u0` (Tracking NMPC while moving, stabilizing brake in G_e)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import casadi as ca
import numpy as np


def _wrap_angle_np(angle: float | np.ndarray) -> float | np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _format_tube_schedule(
    val: float | list[float] | np.ndarray | None,
    length: int,
) -> np.ndarray:
    """Normalize scalar or 1D array margin into a non-negative array of shape `(length,)`."""
    if val is None:
        return np.zeros(length, dtype=np.float64)
    arr = np.maximum(0.0, np.asarray(val, dtype=np.float64).reshape(-1))
    if arr.size == 0:
        return np.zeros(length, dtype=np.float64)
    if arr.size == 1:
        return np.full(length, float(arr[0]), dtype=np.float64)
    if arr.size >= length:
        return arr[:length].copy()
    pad = np.full(length - arr.size, float(arr[-1]), dtype=np.float64)
    return np.concatenate([arr, pad], axis=0)


class RobustNMPCBackup:
    """
    Tube-Based Recovery & Tracking NMPC Backup Controller for 6D `AccTractorTrailer2d`.

    Uses reusable parametric `casadi.Opti()` NLP instances with IPOPT (`expand=True`,
    `warm_start_init_point="yes"`, `print_level=0`, `max_iter=40`) so online checks
    and tracking solves run in a few milliseconds without rebuilding symbolic graphs.
    """

    def __init__(
        self,
        env: Any,
        T_bak: int = 12,
        v_eps: float = 0.05,
        max_iter: int = 30,
        disc_safety_pad: float = 0.01,
        hitch_safety_pad: float = 0.01,
        cache_size: int = 128,
    ) -> None:
        self.env = env
        self.T_bak = int(T_bak)
        self.v_eps = float(v_eps)
        self.max_iter = int(max_iter)
        self.disc_safety_pad = float(disc_safety_pad)
        self.hitch_safety_pad = float(hitch_safety_pad)
        self.cache_size = int(cache_size)

        # Extract exact nominal 6D AccTractorTrailer2d physical parameters
        self.nx = 6
        self.nu = 2
        self.dt = float(env.dt)
        self.l1 = float(getattr(env, "l1_nominal", env.l1))
        self.l2 = float(getattr(env, "l2_nominal", env.l2))
        self.lh = float(getattr(env, "lh_nominal", env.lh))
        self.lf1 = float(env.lf1)
        self.lr = float(env.lr)
        self.lf2 = float(env.lf2)
        self.lr2 = float(env.lr2)
        self.tractor_width = float(env.tractor_width)
        self.trailer_width = float(env.trailer_width)
        self.v_max = float(env.v_max)
        self.delta_max = float(env.delta_max)
        self.a_max = float(getattr(env, "a_max", 2.0))
        self.omega_max = float(getattr(env, "omega_max", 1.0))
        self.phi_max = float(env.phi_max)

        # Active backup state (RMPS Algorithms 2 & 3)
        self.x_ref: np.ndarray | None = None
        self.u_ref: np.ndarray | None = None
        self.r_tube_active: np.ndarray = np.zeros(self.T_bak + 1, dtype=np.float64)
        self.b_hitch_active: np.ndarray = np.zeros(self.T_bak + 1, dtype=np.float64)
        self.t_step: int = 0
        self.active: bool = False

        # Solution cache avoiding redundant IPOPT solves between `is_recoverable` and `initialize_backup`
        self._solution_cache: OrderedDict[bytes, tuple[bool, np.ndarray, np.ndarray]] = OrderedDict()
        self.last_cached_solution: tuple[bool, np.ndarray, np.ndarray] | None = None

        # Cached compiled CasADi Opti bundles keyed by obstacle signature
        self._recovery_bundle: tuple[Any, ...] | None = None
        self._tracking_bundle: tuple[Any, ...] | None = None
        self._obs_signature: bytes | None = None

        # Cached NumPy obstacle arrays to avoid repeated JAX -> NumPy device syncs per check
        self._obs_c_ref: Any = None
        self._obs_r_ref: Any = None
        self._cached_obs_c: np.ndarray | None = None
        self._cached_obs_r: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Exact NumPy kinematics & SAT collision / hitch verification
    # ------------------------------------------------------------------
    def step_nominal_np(self, x: np.ndarray, u_norm: np.ndarray) -> np.ndarray:
        """Exact nominal Forward-Euler step matching `AccTractorTrailer2d.tractor_trailer_dynamics`."""
        px, py, theta1, theta2, v, delta = [float(v_i) for v_i in x[:6]]
        u_c = np.clip(np.asarray(u_norm, dtype=np.float64), -1.0, 1.0)
        a = float(u_c[0]) * self.a_max
        omega = float(u_c[1]) * self.omega_max

        v_c = float(np.clip(v, -self.v_max, self.v_max))
        delta_c = float(np.clip(delta, -self.delta_max, self.delta_max))

        c1 = np.cos(theta1)
        s1 = np.sin(theta1)
        tan_d = np.tan(delta_c)
        dtheta = theta1 - theta2

        px_dot = v_c * c1
        py_dot = v_c * s1
        theta1_dot = (v_c / self.l1) * tan_d
        theta2_dot = (v_c / self.l2) * (
            np.sin(dtheta) - (self.lh / self.l1) * np.cos(dtheta) * tan_d
        )
        return np.array(
            [
                px + self.dt * px_dot,
                py + self.dt * py_dot,
                theta1 + self.dt * theta1_dot,
                theta2 + self.dt * theta2_dot,
                v + self.dt * a,
                delta + self.dt * omega,
            ],
            dtype=np.float64,
        )

    def refresh_obstacles(self) -> None:
        """Invalidate cached NumPy obstacle arrays (call if mutating NumPy obstacle arrays in-place)."""
        self._obs_c_ref = None
        self._obs_r_ref = None
        self._cached_obs_c = None
        self._cached_obs_r = None

    def _get_active_obstacles(self) -> tuple[np.ndarray, np.ndarray]:
        obs_c_raw = getattr(self.env, "obs_circles", None)
        obs_r_raw = getattr(self.env, "obs_rectangles", None)
        if (
            self._cached_obs_c is not None
            and self._cached_obs_r is not None
            and obs_c_raw is self._obs_c_ref
            and obs_r_raw is self._obs_r_ref
            and not isinstance(obs_c_raw, np.ndarray)
            and not isinstance(obs_r_raw, np.ndarray)
        ):
            return self._cached_obs_c, self._cached_obs_r

        obs_c = (
            np.asarray(obs_c_raw, dtype=np.float64)
            if obs_c_raw is not None
            else np.zeros((0, 3), dtype=np.float64)
        )
        if obs_c.ndim == 2 and obs_c.shape[0] > 0:
            obs_c = obs_c[obs_c[:, 2] > 1e-6]
        else:
            obs_c = np.zeros((0, 3), dtype=np.float64)

        obs_r = (
            np.asarray(obs_r_raw, dtype=np.float64)
            if obs_r_raw is not None
            else np.zeros((0, 5), dtype=np.float64)
        )
        if obs_r.ndim == 2 and obs_r.shape[0] > 0:
            obs_r = obs_r[(obs_r[:, 2] > 1e-6) & (obs_r[:, 3] > 1e-6)]
        else:
            obs_r = np.zeros((0, 5), dtype=np.float64)

        self._obs_c_ref = obs_c_raw
        self._obs_r_ref = obs_r_raw
        self._cached_obs_c = obs_c
        self._cached_obs_r = obs_r
        return obs_c, obs_r

    @staticmethod
    def _rect_corners_np(cx: float, cy: float, half_len: float, half_w: float, ang: float) -> np.ndarray:
        local = np.array(
            [
                [-half_len, -half_w],
                [-half_len, half_w],
                [half_len, half_w],
                [half_len, -half_w],
            ],
            dtype=np.float64,
        )
        c, s = np.cos(ang), np.sin(ang)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        return local @ rot.T + np.array([cx, cy], dtype=np.float64)

    @staticmethod
    def _rect_circle_collide_np(corners: np.ndarray, center: np.ndarray, radius: float) -> bool:
        if radius <= 0.0:
            return False
        # Also handle circle center strictly inside rectangle via local projection
        e0 = corners[3] - corners[0]
        e1 = corners[1] - corners[0]
        rel = center - corners[0]
        l0_sq = float(np.dot(e0, e0))
        l1_sq = float(np.dot(e1, e1))
        if l0_sq > 1e-8 and l1_sq > 1e-8:
            u0 = float(np.dot(rel, e0)) / l0_sq
            u1 = float(np.dot(rel, e1)) / l1_sq
            if 0.0 <= u0 <= 1.0 and 0.0 <= u1 <= 1.0:
                return True
        edges = np.roll(corners, -1, axis=0) - corners
        to_c = center[None, :] - corners
        len_sq = np.maximum(np.sum(edges * edges, axis=1), 1e-8)
        t_val = np.clip(np.sum(to_c * edges, axis=1) / len_sq, 0.0, 1.0)
        closest = corners + t_val[:, None] * edges
        min_d2 = float(np.min(np.sum((center[None, :] - closest) ** 2, axis=1)))
        return min_d2 <= radius * radius

    @staticmethod
    def _rect_rect_collide_np(c1: np.ndarray, c2: np.ndarray) -> bool:
        for corners in (c1, c2):
            edges = np.roll(corners, -1, axis=0) - corners
            for idx in (0, 1):
                nx_a, ny_a = -edges[idx, 1], edges[idx, 0]
                norm = np.hypot(nx_a, ny_a)
                if norm < 1e-8:
                    continue
                axis = np.array([nx_a / norm, ny_a / norm], dtype=np.float64)
                p1 = c1 @ axis
                p2 = c2 @ axis
                if np.max(p1) < np.min(p2) or np.max(p2) < np.min(p1):
                    return False
        return True

    def check_state_safety_np(
        self,
        x: np.ndarray,
        r_margin: float = 0.0,
        b_margin: float = 0.0,
    ) -> tuple[bool, bool]:
        """
        Exact CPU NumPy collision & hitch checker matching `TractorTrailer2d`.
        Returns `(obstacle_collision, hitch_violation)`.
        """
        px, py, theta1, theta2 = float(x[0]), float(x[1]), float(x[2]), float(x[3])
        hitch = abs(float(_wrap_angle_np(theta1 - theta2)))
        eff_phi = max(0.0, self.phi_max - float(b_margin))
        hitch_viol = hitch > eff_phi + 1e-5

        c1, s1 = np.cos(theta1), np.sin(theta1)
        trac_cx = px + 0.5 * (self.l1 + self.lf1 - self.lr) * c1
        trac_cy = py + 0.5 * (self.l1 + self.lf1 - self.lr) * s1
        trac_corners = self._rect_corners_np(
            trac_cx, trac_cy, 0.5 * (self.l1 + self.lf1 + self.lr), 0.5 * self.tractor_width, theta1
        )

        hx = px - self.lh * c1
        hy = py - self.lh * s1
        c2, s2 = np.cos(theta2), np.sin(theta2)
        trail_cx = hx + (0.5 * (self.lf2 - self.lr2) - self.l2) * c2
        trail_cy = hy + (0.5 * (self.lf2 - self.lr2) - self.l2) * s2
        trail_corners = self._rect_corners_np(
            trail_cx, trail_cy, 0.5 * (self.lf2 + self.lr2), 0.5 * self.trailer_width, theta2
        )

        obs_c, obs_r = self._get_active_obstacles()
        for cir in obs_c:
            rad = float(cir[2]) + float(r_margin)
            center = cir[:2]
            if self._rect_circle_collide_np(trac_corners, center, rad) or self._rect_circle_collide_np(
                trail_corners, center, rad
            ):
                return True, hitch_viol

        for rect in obs_r:
            rx, ry, rw, rh, rang = [float(v) for v in rect]
            obs_corners = self._rect_corners_np(
                rx, ry, 0.5 * rw + float(r_margin), 0.5 * rh + float(r_margin), rang
            )
            if self._rect_rect_collide_np(trac_corners, obs_corners) or self._rect_rect_collide_np(
                trail_corners, obs_corners
            ):
                return True, hitch_viol

        return False, hitch_viol

    def _verify_trajectory_np(
        self,
        xs: np.ndarray,
        r_sched: np.ndarray,
        b_sched: np.ndarray,
        require_stopped: bool = True,
    ) -> bool:
        if not np.all(np.isfinite(xs)):
            return False
        if require_stopped and abs(float(xs[-1, 4])) > self.v_eps + 1e-3:
            return False
        # k = 0 is the already-measured current state: verify physical safety;
        # enforce inflated disturbance tube margins for all future steps k >= 1.
        col0, hitch0 = self.check_state_safety_np(xs[0], 0.0, 0.0)
        if col0 or hitch0:
            return False
        for k in range(1, xs.shape[0]):
            col, hitch = self.check_state_safety_np(xs[k], float(r_sched[k]), float(b_sched[k]))
            if col or hitch:
                return False
        return True

    # ------------------------------------------------------------------
    # Symbolic CasADi Opti builders (compiled once, reused parametrically)
    # ------------------------------------------------------------------
    def _casadi_step_expr(self, x: ca.MX, u_norm: ca.MX) -> ca.MX:
        px, py, theta1, theta2, v, delta = x[0], x[1], x[2], x[3], x[4], x[5]
        v_c = ca.fmin(ca.fmax(v, -self.v_max), self.v_max)
        delta_c = ca.fmin(ca.fmax(delta, -self.delta_max), self.delta_max)
        a = u_norm[0] * self.a_max
        omega = u_norm[1] * self.omega_max

        c1 = ca.cos(theta1)
        s1 = ca.sin(theta1)
        tan_d = ca.tan(delta_c)
        dtheta = theta1 - theta2

        px_dot = v_c * c1
        py_dot = v_c * s1
        theta1_dot = (v_c / self.l1) * tan_d
        theta2_dot = (v_c / self.l2) * (
            ca.sin(dtheta) - (self.lh / self.l1) * ca.cos(dtheta) * tan_d
        )
        return ca.vertcat(
            px + self.dt * px_dot,
            py + self.dt * py_dot,
            theta1 + self.dt * theta1_dot,
            theta2 + self.dt * theta2_dot,
            v + self.dt * a,
            delta + self.dt * omega,
        )

    def _vehicle_boxes_expr(self, x: ca.MX) -> list[tuple[ca.MX, ca.MX, float, float, ca.MX, ca.MX]]:
        """
        Return exact `(cx, cy, half_len, half_width, cos_th, sin_th)` for tractor and trailer rectangles.
        """
        px, py, theta1, theta2 = x[0], x[1], x[2], x[3]
        c1, s1 = ca.cos(theta1), ca.sin(theta1)
        c2, s2 = ca.cos(theta2), ca.sin(theta2)

        trac_cx = px + 0.5 * (self.l1 + self.lf1 - self.lr) * c1
        trac_cy = py + 0.5 * (self.l1 + self.lf1 - self.lr) * s1
        trac_hl = 0.5 * (self.l1 + self.lf1 + self.lr)
        trac_hw = 0.5 * self.tractor_width

        hx = px - self.lh * c1
        hy = py - self.lh * s1
        trail_cx = hx + (0.5 * (self.lf2 - self.lr2) - self.l2) * c2
        trail_cy = hy + (0.5 * (self.lf2 - self.lr2) - self.l2) * s2
        trail_hl = 0.5 * (self.lf2 + self.lr2)
        trail_hw = 0.5 * self.trailer_width

        return [
            (trac_cx, trac_cy, trac_hl, trac_hw, c1, s1),
            (trail_cx, trail_cy, trail_hl, trail_hw, c2, s2),
        ]

    def _ensure_solvers_built(self) -> None:
        obs_c, obs_r = self._get_active_obstacles()
        sig = obs_c.tobytes() + b"|" + obs_r.tobytes()
        if self._obs_signature == sig and self._recovery_bundle is not None and self._tracking_bundle is not None:
            return

        if self._obs_signature is not None and self._obs_signature != sig:
            self._solution_cache.clear()
        self._obs_signature = sig
        self._recovery_bundle = self._build_opti_problem(obs_c, obs_r, tracking_mode=False)
        self._tracking_bundle = self._build_opti_problem(obs_c, obs_r, tracking_mode=True)

    def _build_opti_problem(
        self,
        obs_c: np.ndarray,
        obs_r: np.ndarray,
        tracking_mode: bool,
    ) -> tuple[Any, ...]:
        H = self.T_bak
        opti = ca.Opti()

        X = opti.variable(self.nx, H + 1)
        U = opti.variable(self.nu, H)
        S_obs = opti.variable(H) if tracking_mode else None

        P_x0 = opti.parameter(self.nx)
        P_r_tube = opti.parameter(H + 1)
        P_b_hitch = opti.parameter(H + 1)
        P_disc_pad = opti.parameter()

        P_xref = opti.parameter(self.nx, H + 1) if tracking_mode else None
        P_uref = opti.parameter(self.nu, H) if tracking_mode else None

        opti.subject_to(X[:, 0] == P_x0)

        eps_sm = 1e-3

        def _smax0(z: ca.MX) -> ca.MX:
            return 0.5 * (z + ca.sqrt(z * z + eps_sm * eps_sm))

        obj = 0.0

        for k in range(H):
            xk = X[:, k]
            uk = U[:, k]
            x_next = X[:, k + 1]

            # Normalized control box bounds & dynamics
            opti.subject_to(opti.bounded(-1.0, uk[0], 1.0))
            opti.subject_to(opti.bounded(-1.0, uk[1], 1.0))
            opti.subject_to(x_next == self._casadi_step_expr(xk, uk))

            # State velocity and steering angle bounds for k >= 1
            opti.subject_to(opti.bounded(-self.v_max, x_next[4], self.v_max))
            opti.subject_to(opti.bounded(-self.delta_max, x_next[5], self.delta_max))

            # Hitch constraint: 1 - cos(theta1 - theta2) <= 1 - cos(phi_eff)
            hitch_cos = 1.0 - ca.cos(x_next[2] - x_next[3])
            phi_eff = ca.fmax(0.05, self.phi_max - P_b_hitch[k + 1] - self.hitch_safety_pad)

            if tracking_mode:
                sk = S_obs[k]
                opti.subject_to(sk >= 0.0)
                # Tighten hitch limit via slack + penalize articulation to prevent jackknifing
                opti.subject_to(hitch_cos <= (1.0 - ca.cos(phi_eff)) + sk)
                xr_next = P_xref[:, k + 1]
                ur_k = P_uref[:, k]
                pos_err = (x_next[0] - xr_next[0]) ** 2 + (x_next[1] - xr_next[1]) ** 2
                hdg_err = (1.0 - ca.cos(x_next[2] - xr_next[2])) + (1.0 - ca.cos(x_next[3] - xr_next[3]))
                vel_err = (x_next[4] - xr_next[4]) ** 2 + 0.5 * (x_next[5] - xr_next[5]) ** 2
                u_err = ca.sumsqr(uk - ur_k)
                obj = (
                    obj
                    + 12.0 * pos_err
                    + 10.0 * hdg_err
                    + 6.0 * vel_err
                    + 15.0 * hitch_cos
                    + 1.5 * u_err
                    + 20000.0 * (sk + 10.0 * sk * sk)
                )
            else:
                opti.subject_to(hitch_cos <= 1.0 - ca.cos(phi_eff))
                # Strongly discourage accelerating during emergency recovery and encourage hitch straightening
                speed_inc = _smax0(x_next[4] ** 2 - P_x0[4] ** 2)
                obj = (
                    obj
                    + 6.0 * (x_next[4] ** 2)
                    + 120.0 * speed_inc
                    + 0.3 * ca.sumsqr(uk)
                    + 12.0 * hitch_cos
                )
                if k > 0:
                    obj = obj + 0.5 * ca.sumsqr(uk - U[:, k - 1])

            # Exact smooth vehicle-rectangle vs obstacle distance constraints at x_next
            r_inf = P_r_tube[k + 1]
            veh_boxes = self._vehicle_boxes_expr(x_next)

            # 1. Exact circle-to-oriented-vehicle-rectangle distance in vehicle local frame
            for c_row in obs_c:
                cx, cy, cr = float(c_row[0]), float(c_row[1]), float(c_row[2])
                req_r = cr + r_inf + P_disc_pad
                for v_cx, v_cy, v_hl, v_hw, v_c, v_s in veh_boxes:
                    rel_x = cx - v_cx
                    rel_y = cy - v_cy
                    lx = rel_x * v_c + rel_y * v_s
                    ly = -rel_x * v_s + rel_y * v_c
                    ox = _smax0(ca.sqrt(lx * lx + eps_sm * eps_sm) - v_hl)
                    oy = _smax0(ca.sqrt(ly * ly + eps_sm * eps_sm) - v_hw)
                    dist_sq = ox * ox + oy * oy
                    if tracking_mode:
                        opti.subject_to(dist_sq + S_obs[k] >= req_r * req_r)
                    else:
                        opti.subject_to(dist_sq >= req_r * req_r)

            # 2. Vehicle boundary points vs inflated obstacle rectangle + obstacle corners vs vehicle rectangle
            for r_row in obs_r:
                rx, ry, rw, rh, rang = [float(v) for v in r_row]
                cos_o = float(np.cos(-rang))
                sin_o = float(np.sin(-rang))
                c_pos = float(np.cos(rang))
                s_pos = float(np.sin(rang))
                hw = 0.5 * rw + r_inf
                hh = 0.5 * rh + r_inf
                obs_corners = [
                    (rx + sx * 0.5 * rw * c_pos - sy * 0.5 * rh * s_pos, ry + sx * 0.5 * rw * s_pos + sy * 0.5 * rh * c_pos)
                    for sx in (-1.0, 1.0)
                    for sy in (-1.0, 1.0)
                ]
                for v_cx, v_cy, v_hl, v_hw, v_c, v_s in veh_boxes:
                    # 6 boundary points on vehicle rectangle (4 corners + 2 side midpoints)
                    veh_pts = [
                        (v_cx + sx * v_hl * v_c - sy * v_hw * v_s, v_cy + sx * v_hl * v_s + sy * v_hw * v_c)
                        for sx in (-1.0, 0.0, 1.0)
                        for sy in (-1.0, 1.0)
                    ]
                    for px_v, py_v in veh_pts:
                        rel_x = px_v - rx
                        rel_y = py_v - ry
                        lx = rel_x * cos_o - rel_y * sin_o
                        ly = rel_x * sin_o + rel_y * cos_o
                        ox = _smax0(ca.sqrt(lx * lx + eps_sm * eps_sm) - hw)
                        oy = _smax0(ca.sqrt(ly * ly + eps_sm * eps_sm) - hh)
                        d2 = ox * ox + oy * oy
                        if tracking_mode:
                            opti.subject_to(d2 + S_obs[k] >= P_disc_pad * P_disc_pad)
                        else:
                            opti.subject_to(d2 >= P_disc_pad * P_disc_pad)

                    # 4 obstacle corners in vehicle local frame
                    v_hl_inf = v_hl + r_inf
                    v_hw_inf = v_hw + r_inf
                    for ox_c, oy_c in obs_corners:
                        rel_x = ox_c - v_cx
                        rel_y = oy_c - v_cy
                        lx = rel_x * v_c + rel_y * v_s
                        ly = -rel_x * v_s + rel_y * v_c
                        ox = _smax0(ca.sqrt(lx * lx + eps_sm * eps_sm) - v_hl_inf)
                        oy = _smax0(ca.sqrt(ly * ly + eps_sm * eps_sm) - v_hw_inf)
                        d2 = ox * ox + oy * oy
                        if tracking_mode:
                            opti.subject_to(d2 + S_obs[k] >= P_disc_pad * P_disc_pad)
                        else:
                            opti.subject_to(d2 >= P_disc_pad * P_disc_pad)

        # Terminal equilibrium condition at k = H
        if tracking_mode:
            obj = obj + 80.0 * (X[4, H] ** 2)
        else:
            opti.subject_to(opti.bounded(-self.v_eps, X[4, H], self.v_eps))
            obj = obj + 50.0 * (X[4, H] ** 2) + 5.0 * (1.0 - ca.cos(X[2, H] - X[3, H]))

        opti.minimize(obj)

        p_opts = {"expand": True, "print_time": 0}
        s_opts = {
            "print_level": 0,
            "sb": "yes",
            "mu_strategy": "adaptive",
            "warm_start_init_point": "yes" if tracking_mode else "no",
            "max_iter": int(self.max_iter),
            "tol": 1e-3,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 2,
        }
        opti.solver("ipopt", p_opts, s_opts)
        return (opti, X, U, S_obs, P_x0, P_r_tube, P_b_hitch, P_disc_pad, P_xref, P_uref)

    # ------------------------------------------------------------------
    # Warm-start rollout helpers & cache keying
    # ------------------------------------------------------------------
    def _rollout_braking_guess(
        self,
        x0: np.ndarray,
        steer_bias: float | str = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        xs = np.zeros((self.T_bak + 1, self.nx), dtype=np.float64)
        us = np.zeros((self.T_bak, self.nu), dtype=np.float64)
        xs[0] = np.asarray(x0[:6], dtype=np.float64)
        for k in range(self.T_bak):
            vk = xs[k, 4]
            dk = xs[k, 5]
            u_a = float(np.clip(-vk / (self.dt * self.a_max), -1.0, 1.0))
            if steer_bias == "straighten":
                phi = float(_wrap_angle_np(xs[k, 2] - xs[k, 3]))
                # When reversing, steering with +phi counteracts unstable trailer jackknifing;
                # when moving forward, centering delta naturally straightens the hitch.
                delta_des = float(np.clip(1.6 * phi, -self.delta_max, self.delta_max)) if vk < -0.05 else 0.0
                u_w = float(np.clip((delta_des - dk) / (self.dt * self.omega_max), -1.0, 1.0))
            elif isinstance(steer_bias, (int, float)) and abs(float(steer_bias)) > 1e-6 and k < max(2, self.T_bak // 2):
                u_w = float(np.clip(float(steer_bias), -1.0, 1.0))
            else:
                u_w = float(np.clip(-dk / (self.dt * self.omega_max), -1.0, 1.0))
            us[k] = [u_a, u_w]
            xs[k + 1] = self.step_nominal_np(xs[k], us[k])
        return xs, us

    def _make_cache_key(self, x_cand: np.ndarray, r_sched: np.ndarray, b_sched: np.ndarray) -> bytes:
        q_r = np.round(np.asarray(x_cand[:6], dtype=np.float64), 5)
        r_r = np.round(r_sched, 4)
        b_r = np.round(b_sched, 4)
        return q_r.tobytes() + r_r.tobytes() + b_r.tobytes()

    def _store_cache(self, key: bytes, val: tuple[bool, np.ndarray, np.ndarray]) -> None:
        self._solution_cache[key] = val
        self._solution_cache.move_to_end(key)
        while len(self._solution_cache) > self.cache_size:
            self._solution_cache.popitem(last=False)
        self.last_cached_solution = val

    def _effective_stopping_schedules(
        self,
        x0: np.ndarray,
        r_sched: np.ndarray,
        b_sched: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Freeze tube growth past the kinematic stopping time `k_stop(v0)` so slow
        parking maneuvers (`|v| <= 1 m/s`, stopping in 1-2 steps) are not rejected
        by multi-second high-speed cruise tubes while stationary inside G_e.
        """
        v0_abs = abs(float(x0[4]))
        k_stop = int(np.ceil(v0_abs / max(self.a_max * self.dt, 1e-4))) + 1
        k_stop = max(2, min(self.T_bak, k_stop))
        r_eff = np.minimum(r_sched, float(r_sched[k_stop]))
        b_eff = np.minimum(b_sched, float(b_sched[k_stop]))
        r_eff[0] = 0.0
        b_eff[0] = 0.0
        return r_eff, b_eff

    def _solve_recovery_nlp(
        self,
        x_init: np.ndarray,
        r_sched: np.ndarray,
        b_sched: np.ndarray,
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        """Solve (or fetch cached) T_bak safe-stopping NLP with symmetry-breaking warm-starts."""
        x0 = np.asarray(x_init[:6], dtype=np.float64)
        r_sched, b_sched = self._effective_stopping_schedules(x0, r_sched, b_sched)

        # Invalidate solution cache only if obstacle set actually changed
        obs_c, obs_r = self._get_active_obstacles()
        sig = obs_c.tobytes() + b"|" + obs_r.tobytes()
        if self._obs_signature is not None and self._obs_signature != sig:
            self._solution_cache.clear()
            self._recovery_bundle = None
            self._tracking_bundle = None
            self._obs_signature = None

        key = self._make_cache_key(x0, r_sched, b_sched)
        if key in self._solution_cache:
            cached = self._solution_cache[key]
            self.last_cached_solution = cached
            return cached[0], cached[1].copy(), cached[2].copy()

        # Check immediate t=0 physical safety first (current measured state has zero future drift)
        col0, hitch0 = self.check_state_safety_np(x0, 0.0, 0.0)
        x_brk, u_brk = self._rollout_braking_guess(x0, steer_bias="straighten")
        if col0 or hitch0:
            res = (False, x_brk, u_brk)
            self._store_cache(key, res)
            return False, x_brk.copy(), u_brk.copy()

        # Fast-path: test hitch-straightening + straight + steered braking BEFORE building/invoking IPOPT (~0.3 ms)
        for fast_bias in ("straighten", 0.0, 0.55, -0.55):
            xg_fast, ug_fast = (
                (x_brk, u_brk) if fast_bias == "straighten" else self._rollout_braking_guess(x0, steer_bias=fast_bias)
            )
            if self._verify_trajectory_np(xg_fast, r_sched, b_sched, require_stopped=True):
                res = (True, xg_fast, ug_fast)
                self._store_cache(key, res)
                return True, xg_fast.copy(), ug_fast.copy()

        # Lazily compile CasADi Opti graphs only when IPOPT recovery is genuinely required
        self._ensure_solvers_built()
        opti, X, U, _, P_x0, P_r_tube, P_b_hitch, P_disc_pad, _, _ = self._recovery_bundle
        opti.set_value(P_x0, x0)
        opti.set_value(P_r_tube, r_sched)
        opti.set_value(P_b_hitch, b_sched)

        base_pad = 0.0 if float(np.max(r_sched)) <= 1e-6 else self.disc_safety_pad
        for pad_extra in (0.0, 0.14):
            opti.set_value(P_disc_pad, base_pad + pad_extra)
            had_unverified_feasible = False
            for bias in ("straighten", 0.65, -0.65):
                xg_init, ug_init = (
                    (x_brk, u_brk) if bias == "straighten" else self._rollout_braking_guess(x0, steer_bias=bias)
                )
                opti.set_initial(X, xg_init.T)
                opti.set_initial(U, ug_init.T)
                u_raw = None
                converged = False
                try:
                    sol = opti.solve()
                    converged = True
                    u_raw = np.asarray(sol.value(U), dtype=np.float64).T
                except RuntimeError:
                    # At max_iter, IPOPT may already have a 100% primal-feasible recovery trajectory
                    status = opti.return_status()
                    if status in ("Solved_To_Acceptable_Level", "Feasible_Point_Found"):
                        converged = True
                    if status in ("Maximum_Iterations_Exceeded", "Solved_To_Acceptable_Level", "Feasible_Point_Found"):
                        try:
                            u_raw = np.asarray(opti.debug.value(U), dtype=np.float64).T
                        except Exception:
                            u_raw = None

                if u_raw is not None and np.all(np.isfinite(u_raw)):
                    u_opt = np.clip(u_raw, -1.0, 1.0)
                    # Roll out exact nominal step for bit-exact kinematic consistency & strict verification
                    x_opt = np.zeros((self.T_bak + 1, self.nx), dtype=np.float64)
                    x_opt[0] = x0
                    for k in range(self.T_bak):
                        x_opt[k + 1] = self.step_nominal_np(x_opt[k], u_opt[k])
                    if self._verify_trajectory_np(x_opt, r_sched, b_sched, require_stopped=True):
                        res = (True, x_opt, u_opt)
                        self._store_cache(key, res)
                        return True, x_opt.copy(), u_opt.copy()
                    if converged:
                        had_unverified_feasible = True
            if not had_unverified_feasible:
                # Only retry larger smooth-distance padding if IPOPT converged but clipped an exact SAT corner
                break

        res = (False, x_brk, u_brk)
        self._store_cache(key, res)
        return False, x_brk.copy(), u_brk.copy()

    # ------------------------------------------------------------------
    # Public RMPS API (Algorithms 2 & 3)
    # ------------------------------------------------------------------
    def is_recoverable(
        self,
        x_cand: np.ndarray,
        r_tube: float | np.ndarray = 0.0,
        b_hitch: float | np.ndarray = 0.0,
        return_solution: bool = False,
    ) -> bool | tuple[bool, np.ndarray, np.ndarray]:
        """
        Check feasibility of steering `x_cand` safely to a stopped equilibrium inside G_e
        within `T_bak` steps under tightened tube margins `(r_tube, b_hitch)`.

        Caches `(feasible, x_sol, u_sol)` internally so a subsequent `initialize_backup`
        on the same state reuses the solution with zero redundant IPOPT solves.
        """
        r_sched = _format_tube_schedule(r_tube, self.T_bak + 1)
        b_sched = _format_tube_schedule(b_hitch, self.T_bak + 1)
        feasible, x_sol, u_sol = self._solve_recovery_nlp(x_cand, r_sched, b_sched)
        if return_solution:
            return feasible, x_sol, u_sol
        return feasible

    def get_cached_solution(
        self,
        x_cand: np.ndarray,
        r_tube: float | np.ndarray = 0.0,
        b_hitch: float | np.ndarray = 0.0,
    ) -> tuple[bool, np.ndarray, np.ndarray] | None:
        """Retrieve cached `(feasible, x_ref, u_ref)` for `(x_cand, r_tube, b_hitch)` if available."""
        x0 = np.asarray(x_cand[:6], dtype=np.float64)
        r_sched = _format_tube_schedule(r_tube, self.T_bak + 1)
        b_sched = _format_tube_schedule(b_hitch, self.T_bak + 1)
        r_sched, b_sched = self._effective_stopping_schedules(x0, r_sched, b_sched)
        key = self._make_cache_key(x0, r_sched, b_sched)
        cached = self._solution_cache.get(key)
        if cached is None:
            return None
        return cached[0], cached[1].copy(), cached[2].copy()

    def initialize_backup(
        self,
        x_init: np.ndarray,
        r_tube: float | np.ndarray = 0.0,
        b_hitch: float | np.ndarray = 0.0,
    ) -> tuple[bool, np.ndarray, np.ndarray]:
        """
        RMPS Algorithm 2: Solve finite-horizon recovery NLP over `T_bak` to bring the
        tractor-trailer to a safe stopped equilibrium (`|v[T_bak]| <= v_eps`), store
        internal backup state `(x_ref, u_ref, t_step=0)`, and return `(success, x_ref, u_ref)`.
        If strict tube margins are infeasible from a perturbed state, falls back
        progressively to 50% and physical (0.0) margins to always find the safest stop.
        """
        r_sched = _format_tube_schedule(r_tube, self.T_bak + 1)
        b_sched = _format_tube_schedule(b_hitch, self.T_bak + 1)
        success, x_sol, u_sol = self._solve_recovery_nlp(x_init, r_sched, b_sched)
        if not success and (float(np.max(r_sched)) > 1e-6 or float(np.max(b_sched)) > 1e-6):
            for scale in (0.5, 0.0):
                s_fb, x_fb, u_fb = self._solve_recovery_nlp(x_init, r_sched * scale, b_sched * scale)
                if s_fb:
                    success, x_sol, u_sol = True, x_fb, u_fb
                    break

        self.x_ref = x_sol.copy()
        self.u_ref = u_sol.copy()
        self.r_tube_active = r_sched.copy()
        self.b_hitch_active = b_sched.copy()
        self.t_step = 0
        self.active = True
        return success, self.x_ref.copy(), self.u_ref.copy()

    def stabilizing_brake_policy(self, x_curr: np.ndarray) -> np.ndarray:
        """Equilibrium stabilizing policy `pi_stable` inside G_e keeping `v = 0`."""
        v_curr = float(x_curr[4])
        u_a = float(np.clip(-v_curr / (self.dt * self.a_max), -1.0, 1.0))
        return np.array([u_a, 0.0], dtype=np.float64)

    def in_equilibrium_set(self, x_curr: np.ndarray) -> bool:
        """Check whether `x_curr` is safely stopped inside the equilibrium set G_e (`|v| <= v_eps`)."""
        return abs(float(x_curr[4])) <= self.v_eps

    def reset(self) -> None:
        """Clear active backup trajectory state."""
        self.x_ref = None
        self.u_ref = None
        self.t_step = 0
        self.active = False

    def step(self, x_curr: np.ndarray) -> np.ndarray:
        """
        RMPS Algorithm 3: Execute one step of the active backup controller.
        - If `t_step < T_bak` and `|v| > v_eps`: solves Tracking NMPC following
          `(x_ref[t_step:], u_ref[t_step:])` (padded with stopped equilibrium) and
          increments `t_step`.
        - Otherwise (inside equilibrium set G_e where `|v| <= v_eps`): returns `pi_stable`.
        """
        x0 = np.asarray(x_curr[:6], dtype=np.float64)
        if self.x_ref is None or self.u_ref is None or self.t_step >= self.T_bak or abs(float(x0[4])) <= self.v_eps:
            return self.stabilizing_brake_policy(x0)

        # Fast-path: on the exact initialization state (`t_step == 0` with zero deviation)
        # or when tracking `x_ref[t_step]` very closely, execute `u_ref[t_step]` directly
        # without incurring a CasADi Opti compile / Tracking IPOPT solve.
        x_target = self.x_ref[self.t_step]
        pos_err = float(np.linalg.norm(x0[:2] - x_target[:2]))
        ang_err = max(
            abs(float(_wrap_angle_np(x0[2] - x_target[2]))),
            abs(float(_wrap_angle_np(x0[3] - x_target[3]))),
        )
        vel_err = abs(float(x0[4] - x_target[4]))
        if pos_err <= 0.03 and ang_err <= 0.03 and vel_err <= 0.06:
            u_fast = np.clip(self.u_ref[self.t_step], -1.0, 1.0)
            self.t_step += 1
            return u_fast

        self._ensure_solvers_built()
        H = self.T_bak
        rem_x = self.x_ref[self.t_step :]  # length <= H + 1
        rem_u = self.u_ref[self.t_step :]  # length <= H

        # Pad remaining reference tail with the stopped terminal equilibrium state
        xref_win = np.zeros((H + 1, self.nx), dtype=np.float64)
        uref_win = np.zeros((H, self.nu), dtype=np.float64)
        xref_win[: len(rem_x)] = rem_x
        if len(rem_x) < H + 1:
            xref_win[len(rem_x) :] = rem_x[-1]
        uref_win[: len(rem_u)] = rem_u

        # Remaining tube margins (softened via slack S_obs so perturbed states inside the tube stay feasible)
        rem_r = self.r_tube_active[self.t_step :]
        rem_b = self.b_hitch_active[self.t_step :]
        r_win = _format_tube_schedule(rem_r, H + 1) * 0.5
        b_win = _format_tube_schedule(rem_b, H + 1) * 0.5

        opti, X, U, S_obs, P_x0, P_r_tube, P_b_hitch, P_disc_pad, P_xref, P_uref = self._tracking_bundle
        opti.set_value(P_x0, x0)
        opti.set_value(P_r_tube, r_win)
        opti.set_value(P_b_hitch, b_win)
        opti.set_value(P_disc_pad, self.disc_safety_pad * 0.5)
        opti.set_value(P_xref, xref_win.T)
        opti.set_value(P_uref, uref_win.T)

        opti.set_initial(X, xref_win.T)
        opti.set_initial(U, uref_win.T)
        opti.set_initial(S_obs, np.zeros(H, dtype=np.float64))
        opti.set_initial(opti.lam_g, 0.0)

        u0 = None
        try:
            sol = opti.solve()
            u_opt = np.clip(np.asarray(sol.value(U), dtype=np.float64).T, -1.0, 1.0)
            u0 = u_opt[0].copy()
        except RuntimeError:
            if opti.return_status() in ("Maximum_Iterations_Exceeded", "Solved_To_Acceptable_Level", "Feasible_Point_Found"):
                try:
                    u_dbg = np.asarray(opti.debug.value(U), dtype=np.float64).T
                    if np.all(np.isfinite(u_dbg)):
                        u0 = np.clip(u_dbg[0], -1.0, 1.0)
                except Exception:
                    u0 = None
            if u0 is None:
                # Safe fallback: proportional velocity/steering tracking toward uref_win[0]
                v_err = float(x0[4] - xref_win[0, 4])
                d_err = float(_wrap_angle_np(x0[5] - xref_win[0, 5]))
                u0 = np.clip(uref_win[0] + np.array([-0.8 * v_err, -0.8 * d_err]), -1.0, 1.0)

        self.t_step += 1
        return u0
