"""
Pure-JAX, Branchless, Vectorized Constrained iLQR Backup Solver for AccTractorTrailer2d.

Designed as Stage 2 of the two-stage GPU backup shield inside `AccTractorTrailer2d.shielded_rollout`:
- Fixed-horizon (`T_B`), fixed-iteration (`jax.lax.scan`, `N_iters=4`) Gauss-Newton iLQR.
- Smooth squared-hinge obstacle & hitch-angle barriers using tube-tightened margins.
- Closed-form 2x2 eigenvalue-clamped Levenberg-Marquardt regularization on `Q_uu`.
- Fixed parallel line-search over `alphas = [1.0, 0.5, 0.1]` selected via `jnp.argmin`.
- Fully compatible with `jax.jit` and `jax.vmap` (static shapes, zero Python branching on tracers).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class ILQRBackupParams(NamedTuple):
    """Static/traced physical and cost parameters for 6D AccTractorTrailer2d iLQR backup."""

    dt: float = 0.25
    l1: float = 3.23
    l2: float = 2.9
    lh: float = 1.15
    lf1: float = 0.9
    lr: float = 1.848
    lf2: float = 1.42
    lr2: float = 3.225
    tractor_width: float = 2.0
    trailer_width: float = 2.5
    v_max: float = 3.0
    delta_max: float = 0.9599310885968813  # 55 deg in rad
    a_max: float = 2.0
    omega_max: float = 1.0
    phi_max: float = 0.6981317007977318    # 40 deg in rad
    # Stage & terminal weights (residual squares: cost = 0.5 * sum(r^2))
    w_u: float = 0.25
    w_v_stage: float = 0.4
    w_obs: float = 120.0
    w_hitch: float = 100.0
    w_v_term: float = 200.0
    w_hitch_term: float = 15.0
    lm_reg: float = 1e-2


def _wrap_angle(angle: jax.Array) -> jax.Array:
    """Smooth branchless angle wrap to (-pi, pi]."""
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def _splus(z: jax.Array, eps: float = 1e-2) -> jax.Array:
    """Smooth C^infty approximation to max(0, z) with non-vanishing curvature near 0."""
    return 0.5 * (z + jnp.sqrt(z * z + eps * eps))


def acc_tt2d_step(x: jax.Array, u_norm: jax.Array, params: ILQRBackupParams) -> jax.Array:
    """Exact nominal Forward-Euler step for 6D AccTractorTrailer2d."""
    px, py, theta1, theta2, v, delta = x
    u_c = jnp.clip(u_norm, -1.0, 1.0)
    a = u_c[0] * params.a_max
    omega = u_c[1] * params.omega_max

    v_c = jnp.clip(v, -params.v_max, params.v_max)
    delta_c = jnp.clip(delta, -params.delta_max, params.delta_max)

    c1 = jnp.cos(theta1)
    s1 = jnp.sin(theta1)
    tan_d = jnp.tan(delta_c)
    dtheta = theta1 - theta2

    px_dot = v_c * c1
    py_dot = v_c * s1
    theta1_dot = (v_c / params.l1) * tan_d
    theta2_dot = (v_c / params.l2) * (
        jnp.sin(dtheta) - (params.lh / params.l1) * jnp.cos(dtheta) * tan_d
    )

    dt = params.dt
    return jnp.array(
        [
            px + dt * px_dot,
            py + dt * py_dot,
            theta1 + dt * theta1_dot,
            theta2 + dt * theta2_dot,
            v + dt * a,
            delta + dt * omega,
        ],
        dtype=x.dtype,
    )


def acc_tt2d_jacobians(
    x: jax.Array,
    u_norm: jax.Array,
    params: ILQRBackupParams,
) -> tuple[jax.Array, jax.Array]:
    """Closed-form discrete-time Jacobians (F_x, F_u) of `acc_tt2d_step`."""
    _, _, theta1, theta2, v, delta = x
    v_c = jnp.clip(v, -params.v_max, params.v_max)
    delta_c = jnp.clip(delta, -params.delta_max, params.delta_max)

    c1 = jnp.cos(theta1)
    s1 = jnp.sin(theta1)
    tan_d = jnp.tan(delta_c)
    sec2_d = 1.0 + tan_d * tan_d

    dtheta = theta1 - theta2
    cd = jnp.cos(dtheta)
    sd = jnp.sin(dtheta)

    dt = params.dt
    l1 = params.l1
    l2 = params.l2
    lh = params.lh

    term = sd - (lh / l1) * cd * tan_d
    dterm = cd + (lh / l1) * sd * tan_d

    F_x = jnp.array(
        [
            [1.0, 0.0, -dt * v_c * s1, 0.0, dt * c1, 0.0],
            [0.0, 1.0, dt * v_c * c1, 0.0, dt * s1, 0.0],
            [0.0, 0.0, 1.0, 0.0, dt * (tan_d / l1), dt * (v_c / l1) * sec2_d],
            [
                0.0,
                0.0,
                dt * (v_c / l2) * dterm,
                1.0 - dt * (v_c / l2) * dterm,
                dt * (term / l2),
                -dt * (v_c / l2) * (lh / l1) * cd * sec2_d,
            ],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=x.dtype,
    )

    F_u = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [dt * params.a_max, 0.0],
            [0.0, dt * params.omega_max],
        ],
        dtype=x.dtype,
    )
    return F_x, F_u


def _vehicle_body_discs(
    x: jax.Array,
    params: ILQRBackupParams,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute 6 covering disc centers (6, 2) and base radii (6,) for tractor + trailer.
    """
    px, py, theta1, theta2 = x[0], x[1], x[2], x[3]
    c1, s1 = jnp.cos(theta1), jnp.sin(theta1)
    c2, s2 = jnp.cos(theta2), jnp.sin(theta2)

    trac_cx = px + 0.5 * (params.l1 + params.lf1 - params.lr) * c1
    trac_cy = py + 0.5 * (params.l1 + params.lf1 - params.lr) * s1
    trac_half_len = 0.5 * (params.l1 + params.lf1 + params.lr)
    trac_r_base = 0.5 * params.tractor_width + 0.12
    trac_off = jnp.maximum(0.0, trac_half_len - 0.5 * params.tractor_width)

    hx = px - params.lh * c1
    hy = py - params.lh * s1
    trail_cx = hx + (0.5 * (params.lf2 - params.lr2) - params.l2) * c2
    trail_cy = hy + (0.5 * (params.lf2 - params.lr2) - params.l2) * s2
    trail_half_len = 0.5 * (params.lf2 + params.lr2)
    trail_r_base = 0.5 * params.trailer_width + 0.12
    trail_off = jnp.maximum(0.0, trail_half_len - 0.5 * params.trailer_width)

    centers = jnp.array(
        [
            [trac_cx - trac_off * c1, trac_cy - trac_off * s1],
            [trac_cx, trac_cy],
            [trac_cx + trac_off * c1, trac_cy + trac_off * s1],
            [trail_cx - trail_off * c2, trail_cy - trail_off * s2],
            [trail_cx, trail_cy],
            [trail_cx + trail_off * c2, trail_cy + trail_off * s2],
        ],
        dtype=x.dtype,
    )
    radii = jnp.array(
        [trac_r_base, trac_r_base, trac_r_base, trail_r_base, trail_r_base, trail_r_base],
        dtype=x.dtype,
    )
    return centers, radii


def _state_barrier_residuals(
    x: jax.Array,
    obs_circles: jax.Array,
    obs_rectangles: jax.Array,
    r_margin: jax.Array,
    b_margin: jax.Array,
    params: ILQRBackupParams,
) -> jax.Array:
    """Smooth residual vector for tightened obstacle and hitch-angle barriers."""
    dtheta = _wrap_angle(x[2] - x[3])
    abs_dtheta = jnp.sqrt(dtheta * dtheta + 1e-6)
    phi_lim = jnp.maximum(0.05, params.phi_max - b_margin - 0.02)
    res_hitch = jnp.sqrt(params.w_hitch) * _splus(abs_dtheta - phi_lim)

    parts = [jnp.atleast_1d(res_hitch)]
    centers, radii = _vehicle_body_discs(x, params)
    sqrt_w_obs = jnp.sqrt(params.w_obs)

    if obs_circles.shape[0] > 0:
        # (N_cir, 6, 2)
        diffs = centers[None, :, :] - obs_circles[:, None, :2]
        dists = jnp.sqrt(jnp.sum(diffs * diffs, axis=-1) + 1e-6)
        active_c = obs_circles[:, 2:3] > 0.0
        req_dist = obs_circles[:, 2:3] + r_margin + radii[None, :]
        pen_c = jnp.where(active_c, sqrt_w_obs * _splus(req_dist - dists), 0.0)
        parts.append(pen_c.reshape(-1))

    if obs_rectangles.shape[0] > 0:
        rx = obs_rectangles[:, 0:1]
        ry = obs_rectangles[:, 1:2]
        rw = obs_rectangles[:, 2:3]
        rh = obs_rectangles[:, 3:4]
        rang = obs_rectangles[:, 4:5]

        dx = centers[None, :, 0] - rx
        dy = centers[None, :, 1] - ry
        cos_a = jnp.cos(-rang)
        sin_a = jnp.sin(-rang)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a

        hw = 0.5 * rw + r_margin
        hh = 0.5 * rh + r_margin
        abs_lx = jnp.sqrt(lx * lx + 1e-4)
        abs_ly = jnp.sqrt(ly * ly + 1e-4)

        out_x = _splus(abs_lx - hw)
        out_y = _splus(abs_ly - hh)
        d_out = jnp.sqrt(out_x * out_x + out_y * out_y + 1e-6)
        d_in = jnp.minimum(hw - abs_lx, hh - abs_ly)
        signed_dist = jnp.where((abs_lx < hw) & (abs_ly < hh), -_splus(d_in), d_out)

        active_r = (rw > 0.0) & (rh > 0.0)
        pen_r = jnp.where(active_r, sqrt_w_obs * _splus(radii[None, :] - signed_dist), 0.0)
        parts.append(pen_r.reshape(-1))

    return jnp.concatenate(parts, axis=0)


def _stage_state_residuals(
    x: jax.Array,
    obs_circles: jax.Array,
    obs_rectangles: jax.Array,
    r_margin: jax.Array,
    b_margin: jax.Array,
    params: ILQRBackupParams,
) -> jax.Array:
    res_v = jnp.atleast_1d(jnp.sqrt(params.w_v_stage) * x[4])
    res_bar = _state_barrier_residuals(x, obs_circles, obs_rectangles, r_margin, b_margin, params)
    return jnp.concatenate([res_v, res_bar], axis=0)


def _terminal_state_residuals(
    x: jax.Array,
    obs_circles: jax.Array,
    obs_rectangles: jax.Array,
    r_margin: jax.Array,
    b_margin: jax.Array,
    params: ILQRBackupParams,
) -> jax.Array:
    dtheta = _wrap_angle(x[2] - x[3])
    res_term = jnp.array(
        [
            jnp.sqrt(params.w_v_term) * x[4],
            jnp.sqrt(params.w_hitch_term) * dtheta,
        ],
        dtype=x.dtype,
    )
    res_bar = _state_barrier_residuals(x, obs_circles, obs_rectangles, r_margin, b_margin, params)
    return jnp.concatenate([res_term, res_bar], axis=0)


def _trajectory_cost(
    xs: jax.Array,
    us: jax.Array,
    obs_circles: jax.Array,
    obs_rectangles: jax.Array,
    r_margins: jax.Array,
    b_margins: jax.Array,
    params: ILQRBackupParams,
) -> jax.Array:
    """Evaluate total smooth trajectory cost 0.5 * sum(residuals^2)."""
    res_x = jax.vmap(
        lambda xk, rk, bk: _stage_state_residuals(xk, obs_circles, obs_rectangles, rk, bk, params)
    )(xs[:-1], r_margins[:-1], b_margins[:-1])
    stage_x_cost = 0.5 * jnp.sum(res_x * res_x)
    stage_u_cost = 0.5 * params.w_u * jnp.sum(us * us)

    res_T = _terminal_state_residuals(
        xs[-1], obs_circles, obs_rectangles, r_margins[-1], b_margins[-1], params
    )
    term_cost = 0.5 * jnp.sum(res_T * res_T)
    total = stage_x_cost + stage_u_cost + term_cost
    return jnp.nan_to_num(total, nan=1e12, posinf=1e12, neginf=1e12)


def generate_braking_warmstart(
    x0: jax.Array,
    params: ILQRBackupParams,
    horizon: int = 8,
) -> tuple[jax.Array, jax.Array]:
    """
    Generate a smooth, deadbeat-braking + steering-centering warm-start (xs_init, us_init)
    of shapes (horizon + 1, 6) and (horizon, 2).
    """
    dt = params.dt

    def _brake_step(x_curr: jax.Array, _: None) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        v_curr = x_curr[4]
        delta_curr = x_curr[5]
        u_a = jnp.clip(-v_curr / (dt * params.a_max), -1.0, 1.0)
        u_w = jnp.clip(-delta_curr / (dt * params.omega_max), -1.0, 1.0)
        u_k = jnp.array([u_a, u_w], dtype=x0.dtype)
        x_next = acc_tt2d_step(x_curr, u_k, params)
        return x_next, (x_next, u_k)

    _, (xs_tail, us_init) = jax.lax.scan(_brake_step, x0, None, length=horizon)
    xs_init = jnp.concatenate([x0[None, :], xs_tail], axis=0)
    return xs_init, us_init


def _regularize_quu_2x2(Q_uu: jax.Array, lm_reg: float) -> jax.Array:
    """Closed-form 2x2 eigenvalue floor ensuring Q_uu >= lm_reg * I_2 without branching."""
    Q_sym = 0.5 * (Q_uu + Q_uu.T)
    a = Q_sym[0, 0]
    b = Q_sym[0, 1]
    d = Q_sym[1, 1]
    disc = jnp.sqrt((a - d) * (a - d) + 4.0 * b * b + 1e-12)
    min_eig = 0.5 * (a + d - disc)
    shift = jnp.maximum(lm_reg, lm_reg - min_eig)
    return Q_sym + shift * jnp.eye(2, dtype=Q_uu.dtype)


@partial(jax.jit, static_argnames=("n_iters", "horizon"))
def solve_ilqr_backup(
    x0: jax.Array,
    obs_circles: jax.Array,
    obs_rectangles: jax.Array,
    r_margins: jax.Array,
    b_margins: jax.Array,
    params: ILQRBackupParams,
    n_iters: int = 4,
    horizon: int = 8,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Run fixed-iteration, branchless Gauss-Newton iLQR backup trajectory optimization.

    Args:
        x0: Initial 6D state [px, py, theta1, theta2, v, delta].
        obs_circles: (N_c, 3) circular obstacles.
        obs_rectangles: (N_r, 5) oriented rectangular obstacles.
        r_margins: (horizon + 1,) spatial obstacle inflation schedule [m].
        b_margins: (horizon + 1,) hitch-angle tightening schedule [rad].
        params: Physical & cost parameters (`ILQRBackupParams`).
        n_iters: Fixed number of iLQR iterations (`jax.lax.scan`).
        horizon: Backup horizon `T_B`.

    Returns:
        xs_opt: Optimized backup states of shape (horizon + 1, 6).
        us_opt: Optimized normalized controls of shape (horizon, 2) in [-1, 1]^2.
        cost_opt: Scalar final smooth cost.
    """
    xs_init, us_init = generate_braking_warmstart(x0, params, horizon=horizon)
    cost_init = _trajectory_cost(
        xs_init, us_init, obs_circles, obs_rectangles, r_margins, b_margins, params
    )

    def _stage_quad(xk: jax.Array, uk: jax.Array, rk: jax.Array, bk: jax.Array):
        res_fn = lambda z: _stage_state_residuals(z, obs_circles, obs_rectangles, rk, bk, params)
        res = res_fn(xk)
        J_x = jax.jacfwd(res_fn)(xk)
        l_x = J_x.T @ res
        l_xx = J_x.T @ J_x + 1e-5 * jnp.eye(6, dtype=xk.dtype)
        l_u = params.w_u * uk
        l_uu = params.w_u * jnp.eye(2, dtype=xk.dtype)
        l_ux = jnp.zeros((2, 6), dtype=xk.dtype)
        return l_x, l_u, l_xx, l_uu, l_ux

    def _term_quad(xT: jax.Array, rT: jax.Array, bT: jax.Array):
        res_fn = lambda z: _terminal_state_residuals(z, obs_circles, obs_rectangles, rT, bT, params)
        res = res_fn(xT)
        J_x = jax.jacfwd(res_fn)(xT)
        V_x = J_x.T @ res
        V_xx = J_x.T @ J_x + 1e-4 * jnp.eye(6, dtype=xT.dtype)
        return V_x, V_xx

    alphas = jnp.array([1.0, 0.5, 0.1], dtype=x0.dtype)

    def _ilqr_iter(carry, _: None):
        xs, us, curr_cost = carry

        F_x_seq, F_u_seq = jax.vmap(lambda xk, uk: acc_tt2d_jacobians(xk, uk, params))(xs[:-1], us)
        l_x_seq, l_u_seq, l_xx_seq, l_uu_seq, l_ux_seq = jax.vmap(_stage_quad)(
            xs[:-1], us, r_margins[:-1], b_margins[:-1]
        )
        V_x_T, V_xx_T = _term_quad(xs[-1], r_margins[-1], b_margins[-1])

        def _backward_step(V_carry, step_data):
            V_x, V_xx = V_carry
            F_x, F_u, l_x, l_u, l_xx, l_uu, l_ux = step_data

            Q_x = l_x + F_x.T @ V_x
            Q_u = l_u + F_u.T @ V_x
            Q_xx = l_xx + F_x.T @ V_xx @ F_x
            Q_ux = l_ux + F_u.T @ V_xx @ F_x
            Q_uu_raw = l_uu + F_u.T @ V_xx @ F_u
            Q_uu_reg = _regularize_quu_2x2(Q_uu_raw, params.lm_reg)

            k_ff = -jnp.linalg.solve(Q_uu_reg, Q_u)
            K_fb = -jnp.linalg.solve(Q_uu_reg, Q_ux)

            V_x_prev = Q_x + K_fb.T @ Q_uu_reg @ k_ff + K_fb.T @ Q_u + Q_ux.T @ k_ff
            V_xx_prev = Q_xx + K_fb.T @ Q_uu_reg @ K_fb + K_fb.T @ Q_ux + Q_ux.T @ K_fb
            V_xx_prev = 0.5 * (V_xx_prev + V_xx_prev.T)
            return (V_x_prev, V_xx_prev), (k_ff, K_fb)

        _, (k_seq, K_seq) = jax.lax.scan(
            _backward_step,
            (V_x_T, V_xx_T),
            (F_x_seq, F_u_seq, l_x_seq, l_u_seq, l_xx_seq, l_uu_seq, l_ux_seq),
            reverse=True,
        )

        def _eval_alpha(alpha: jax.Array):
            def _fwd_step(x_curr, inp):
                x_nom, u_nom, k_ff, K_fb = inp
                dx = x_curr - x_nom
                dx = dx.at[2].set(_wrap_angle(dx[2]))
                dx = dx.at[3].set(_wrap_angle(dx[3]))
                u_new = jnp.clip(u_nom + alpha * k_ff + K_fb @ dx, -1.0, 1.0)
                x_next = acc_tt2d_step(x_curr, u_new, params)
                return x_next, (x_next, u_new)

            _, (xs_tail, us_cand) = jax.lax.scan(_fwd_step, x0, (xs[:-1], us, k_seq, K_seq))
            xs_cand = jnp.concatenate([x0[None, :], xs_tail], axis=0)
            c_cand = _trajectory_cost(
                xs_cand, us_cand, obs_circles, obs_rectangles, r_margins, b_margins, params
            )
            return xs_cand, us_cand, c_cand

        xs_cands, us_cands, c_cands = jax.vmap(_eval_alpha)(alphas)
        best_idx = jnp.argmin(c_cands)
        best_xs = xs_cands[best_idx]
        best_us = us_cands[best_idx]
        best_c = c_cands[best_idx]

        accept = best_c < curr_cost
        next_xs = jnp.where(accept, best_xs, xs)
        next_us = jnp.where(accept, best_us, us)
        next_c = jnp.where(accept, best_c, curr_cost)
        return (next_xs, next_us, next_c), next_c

    (xs_opt, us_opt, cost_opt), _ = jax.lax.scan(_ilqr_iter, (xs_init, us_init, cost_init), None, length=n_iters)
    return xs_opt, us_opt, cost_opt
