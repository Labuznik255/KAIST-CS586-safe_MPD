"""
Offline Reachable Disturbance Tube Calibration (RMPS Algorithm 4 + Section IV-E).

Precomputes axis-aligned reachable error tubes B(t) and conservative constraint
tightening margins (spatial inflation r_tube[t], hitch margin b_hitch[t], and
velocity margin b_v[t]) for AccTractorTrailer2d across Perturbation Levels 0-3.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import mbd


def _wrap_angle_jnp(angle: jax.Array) -> jax.Array:
    """Wrap angle to (-pi, pi] in pure, branchless JAX."""
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def generate_nominal_maneuver(
    env: Any,
    T_horizon: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a representative curved reversing nominal maneuver (x_ref, u_ref)
    for AccTractorTrailer2d where trailer articulation dynamics are sensitive.

    Args:
        env: AccTractorTrailer2d environment instance.
        T_horizon: Number of control steps in the calibration maneuver.

    Returns:
        x_ref: Nominal state trajectory of shape (T_horizon + 1, 6).
        u_ref: Normalized control sequence of shape (T_horizon, 2) in [-1, 1]^2.
    """
    if T_horizon <= 0:
        raise ValueError(f"T_horizon must be positive; got {T_horizon}")

    u_ref = np.zeros((T_horizon, 2), dtype=np.float32)
    n_ramp = max(1, T_horizon // 4)
    # Reverse acceleration -> constant reverse cruise -> brake back toward zero
    u_ref[:n_ramp, 0] = -0.35
    u_ref[T_horizon - n_ramp :, 0] = 0.35

    # Smooth sinusoidal steering-rate excitation for an S-curve reversing path (~23 deg max hitch < phi_max=40 deg)
    t_idx = np.arange(T_horizon, dtype=np.float32)
    u_ref[:, 1] = 0.07 * np.sin(2.0 * np.pi * t_idx / float(T_horizon))

    dt = float(env.dt)
    u_jnp = jnp.asarray(u_ref, dtype=jnp.float32)
    x0 = jnp.zeros(6, dtype=jnp.float32)

    def _nom_step(x: jax.Array, u_norm: jax.Array) -> tuple[jax.Array, jax.Array]:
        u_scaled = env.input_scaler(u_norm)
        x_next = x + dt * env.tractor_trailer_dynamics(x, u_scaled)
        return x_next, x_next

    _, xs_tail = jax.lax.scan(_nom_step, x0, u_jnp)
    x_ref = np.vstack([np.zeros((1, 6), dtype=np.float32), np.asarray(xs_tail, dtype=np.float32)])
    return x_ref, u_ref


def compute_tvlqr_gains(
    env: Any,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    Q_terminal: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute finite-horizon time-varying discrete LQR feedback gains K_fb[t]
    around the nominal trajectory (x_ref, u_ref) via JAX autodiff linearization.

    Args:
        env: AccTractorTrailer2d environment instance.
        x_ref: Nominal states of shape (T + 1, 6).
        u_ref: Nominal normalized controls of shape (T, 2).
        Q: Optional (6, 6) stage state weight matrix.
        R: Optional (2, 2) stage control weight matrix.
        Q_terminal: Optional (6, 6) terminal state weight matrix.

    Returns:
        K_fb: Feedback gain array of shape (T, 2, 6) for u = clip(u_ref + K_fb @ e, -1, 1).
    """
    T = int(u_ref.shape[0])
    nx = int(x_ref.shape[1])
    nu = int(u_ref.shape[1])
    dt = float(env.dt)

    if Q is None:
        Q = np.diag([2.0, 2.0, 5.0, 8.0, 2.0, 1.0])
    if R is None:
        R = np.diag([0.5, 0.5])
    if Q_terminal is None:
        Q_terminal = 5.0 * Q

    def _f_nom(x: jax.Array, u_norm: jax.Array) -> jax.Array:
        return x + dt * env.tractor_trailer_dynamics(x, env.input_scaler(u_norm))

    jac_x_fn = jax.jit(jax.vmap(jax.jacfwd(_f_nom, argnums=0)))
    jac_u_fn = jax.jit(jax.vmap(jax.jacfwd(_f_nom, argnums=1)))

    x_t_jnp = jnp.asarray(x_ref[:-1], dtype=jnp.float32)
    u_t_jnp = jnp.asarray(u_ref, dtype=jnp.float32)
    A_all = np.asarray(jac_x_fn(x_t_jnp, u_t_jnp), dtype=np.float64)
    B_all = np.asarray(jac_u_fn(x_t_jnp, u_t_jnp), dtype=np.float64)

    K_fb = np.zeros((T, nu, nx), dtype=np.float32)
    P = Q_terminal.astype(np.float64).copy()

    for t in range(T - 1, -1, -1):
        A_t = A_all[t]
        B_t = B_all[t]
        BtP = B_t.T @ P
        G = R + BtP @ B_t
        K_t = -np.linalg.solve(G, BtP @ A_t)
        K_fb[t] = K_t.astype(np.float32)
        P = Q + A_t.T @ P @ (A_t + B_t @ K_t)

    return K_fb


def _rollout_mc_stochastic(
    env: Any,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    K_fb: np.ndarray,
    N_mc: int,
    seed: int,
) -> np.ndarray:
    """
    Simulate N_mc closed-loop perturbed rollouts for Levels 0, 1, or 3 using env.step_perturbed.
    Returns state trajectories of shape (N_mc, T + 1, 6).
    """
    x0 = jnp.asarray(x_ref[0], dtype=jnp.float32)
    x_ref_jnp = jnp.asarray(x_ref[:-1], dtype=jnp.float32)
    u_ref_jnp = jnp.asarray(u_ref, dtype=jnp.float32)
    K_fb_jnp = jnp.asarray(K_fb, dtype=jnp.float32)

    master_key = jax.random.PRNGKey(seed)
    keys = jax.random.split(master_key, N_mc)

    def _single_rollout(rng_key: jax.Array) -> jax.Array:
        st0 = env.reset(rng_key).replace(pipeline_state=x0, obs=x0)

        def _scan_body(st: Any, step_inputs: tuple[jax.Array, jax.Array, jax.Array]):
            xr_t, ur_t, K_t = step_inputs
            x_curr = st.pipeline_state
            err = x_curr - xr_t
            err = err.at[2].set(_wrap_angle_jnp(err[2]))
            err = err.at[3].set(_wrap_angle_jnp(err[3]))
            u_fb = jnp.clip(ur_t + K_t @ err, -1.0, 1.0)
            st_next = env.step_perturbed(st, u_fb)
            return st_next, st_next.pipeline_state

        _, xs_tail = jax.lax.scan(_scan_body, st0, (x_ref_jnp, u_ref_jnp, K_fb_jnp))
        return jnp.concatenate([x0[None, :], xs_tail], axis=0)

    trajs = jax.jit(jax.vmap(_single_rollout))(keys)
    return np.asarray(trajs, dtype=np.float64)


def _rollout_mc_level2_params(
    env: Any,
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    K_fb: np.ndarray,
    N_mc: int,
    seed: int,
    param_uncertainty_std: float,
) -> np.ndarray:
    """
    Simulate N_mc closed-loop rollouts under Level 2 time-invariant model parameter uncertainty.
    Each Monte Carlo sample draws an independent (l1, l2, lh) tuple matching TractorTrailer2d._init_perturbation_params.
    """
    dt = float(env.dt)
    l1_nom = float(env.l1_nominal)
    l2_nom = float(env.l2_nominal)
    lh_nom = float(env.lh_nominal)
    v_max = float(env.v_max)
    delta_max = float(env.delta_max)

    master_key = jax.random.PRNGKey(seed)
    eps = jax.random.normal(master_key, shape=(N_mc, 3)) * float(param_uncertainty_std)
    l1_samples = jnp.clip(l1_nom * (1.0 + eps[:, 0]), 0.2 * l1_nom, 5.0 * l1_nom)
    l2_samples = jnp.clip(l2_nom * (1.0 + eps[:, 1]), 0.2 * l2_nom, 5.0 * l2_nom)
    lh_samples = jnp.clip(lh_nom * (1.0 + eps[:, 2]), 0.2 * lh_nom, 5.0 * lh_nom)

    x0 = jnp.asarray(x_ref[0], dtype=jnp.float32)
    x_ref_jnp = jnp.asarray(x_ref[:-1], dtype=jnp.float32)
    u_ref_jnp = jnp.asarray(u_ref, dtype=jnp.float32)
    K_fb_jnp = jnp.asarray(K_fb, dtype=jnp.float32)

    def _single_param_rollout(l1_p: jax.Array, l2_p: jax.Array, lh_p: jax.Array) -> jax.Array:
        def _scan_body(x_curr: jax.Array, step_inputs: tuple[jax.Array, jax.Array, jax.Array]):
            xr_t, ur_t, K_t = step_inputs
            err = x_curr - xr_t
            err = err.at[2].set(_wrap_angle_jnp(err[2]))
            err = err.at[3].set(_wrap_angle_jnp(err[3]))
            u_fb = jnp.clip(ur_t + K_t @ err, -1.0, 1.0)
            u_scaled = env.input_scaler(u_fb)

            px, py, theta1, theta2, v, delta = x_curr
            a, omega = u_scaled
            v_c = jnp.clip(v, -v_max, v_max)
            delta_c = jnp.clip(delta, -delta_max, delta_max)

            px_dot = v_c * jnp.cos(theta1)
            py_dot = v_c * jnp.sin(theta1)
            theta1_dot = (v_c / l1_p) * jnp.tan(delta_c)
            theta2_dot = (v_c / l2_p) * (
                jnp.sin(theta1 - theta2) - (lh_p / l1_p) * jnp.cos(theta1 - theta2) * jnp.tan(delta_c)
            )
            x_dot = jnp.array([px_dot, py_dot, theta1_dot, theta2_dot, a, omega], dtype=x_curr.dtype)
            x_next = x_curr + dt * x_dot
            return x_next, x_next

        _, xs_tail = jax.lax.scan(_scan_body, x0, (x_ref_jnp, u_ref_jnp, K_fb_jnp))
        return jnp.concatenate([x0[None, :], xs_tail], axis=0)

    trajs = jax.jit(jax.vmap(_single_param_rollout))(l1_samples, l2_samples, lh_samples)
    return np.asarray(trajs, dtype=np.float64)


def default_tube_path(
    level: int,
    model_name: str = "acc_tt2d",
    base_dir: str | Path = "results/tubes",
) -> Path:
    """Return default .npz path for calibrated reachable tubes."""
    return Path(base_dir) / f"{model_name}_level{int(level)}.npz"


def save_tubes(tube_data: dict[str, Any], path: str | Path) -> Path:
    """Save calibrated tube dictionary to a compressed .npz archive."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for k, v in tube_data.items():
        if v is None:
            continue
        if isinstance(v, (np.ndarray, list, tuple)):
            payload[k] = np.asarray(v)
        elif isinstance(v, (int, float, bool, np.number, np.bool_, str)):
            payload[k] = np.asarray(v)
    np.savez(out_path, **payload)
    return out_path


def load_tubes(path: str | Path) -> dict[str, Any]:
    """Load calibrated tube dictionary from an .npz archive."""
    in_path = Path(path)
    data = np.load(in_path, allow_pickle=False)
    out: dict[str, Any] = {}
    scalar_int_keys = {"level", "N_mc", "T_horizon", "seed"}
    scalar_float_keys = {"dt", "quantile", "param_uncertainty_std", "non_gaussian_df", "state_disturbance_scale"}
    scalar_str_keys = {"non_gaussian_dist"}

    for k in data.files:
        arr = data[k]
        if k in scalar_int_keys:
            out[k] = int(arr.item())
        elif k in scalar_float_keys:
            out[k] = float(arr.item())
        elif k in scalar_str_keys:
            out[k] = str(arr.item())
        else:
            out[k] = arr
    return out


def calibrate_tubes(
    env: Any,
    level: int,
    N_mc: int = 1000,
    T_horizon: int = 25,
    seed: int = 42,
    quantile: float = 0.99,
    x_ref: np.ndarray | None = None,
    u_ref: np.ndarray | None = None,
    K_fb: np.ndarray | None = None,
    save_path: str | Path | None = None,
    **perturbation_kwargs: Any,
) -> dict[str, Any]:
    """
    Calibrate axis-aligned disturbance reachable tubes B(t) and conservative
    constraint tightening margins for AccTractorTrailer2d (RMPS Algorithm 4).

    Args:
        env: AccTractorTrailer2d environment instance.
        level: Perturbation level (0, 1, 2, or 3).
        N_mc: Number of Monte Carlo closed-loop rollouts.
        T_horizon: Tube horizon length in discrete steps.
        seed: PRNG seed for reproducibility.
        quantile: Quantile in (0, 1] used to bound deviations (1.0 uses worst-case max).
        x_ref: Optional nominal reference states (T + 1, 6).
        u_ref: Optional nominal normalized controls (T, 2).
        K_fb: Optional feedback gain schedule (T, 2, 6).
        save_path: Optional path to save the calibrated .npz archive.
        **perturbation_kwargs: Optional overrides passed to `env.set_perturbation(...)`.

    Returns:
        Dictionary containing reachable tubes, scalar tightening margins, reference
        trajectory, TV-LQR feedback gains, and perturbation metadata.
    """
    if level not in (0, 1, 2, 3):
        raise ValueError(f"level must be 0, 1, 2, or 3; got {level}")
    if not (0.0 < quantile <= 1.0):
        raise ValueError(f"quantile must be in (0, 1]; got {quantile}")

    # Snapshot caller's perturbation configuration to restore on exit
    prev_manual_flag = bool(getattr(env, "_perturbation_manually_set", False))
    prev_info = env.get_perturbation_info()

    try:
        env.set_perturbation(level=level, seed=seed, **perturbation_kwargs)
        active_info = env.get_perturbation_info()

        if x_ref is None or u_ref is None:
            x_ref, u_ref = generate_nominal_maneuver(env, T_horizon=T_horizon)
        else:
            x_ref = np.asarray(x_ref, dtype=np.float32)
            u_ref = np.asarray(u_ref, dtype=np.float32)
            T_horizon = int(u_ref.shape[0])

        if K_fb is None:
            K_fb = compute_tvlqr_gains(env, x_ref, u_ref)
        else:
            K_fb = np.asarray(K_fb, dtype=np.float32)

        if level == 2:
            trajs = _rollout_mc_level2_params(
                env=env,
                x_ref=x_ref,
                u_ref=u_ref,
                K_fb=K_fb,
                N_mc=N_mc,
                seed=seed,
                param_uncertainty_std=float(active_info["param_uncertainty_std"]),
            )
        else:
            trajs = _rollout_mc_stochastic(
                env=env,
                x_ref=x_ref,
                u_ref=u_ref,
                K_fb=K_fb,
                N_mc=N_mc,
                seed=seed,
            )

        # Compute wrapped deviations e_i(t) = x_i(t) - x_ref(t) of shape (N_mc, T + 1, 6)
        errors = trajs - x_ref[None, :, :].astype(np.float64)
        errors[:, :, 2] = np.arctan2(np.sin(errors[:, :, 2]), np.cos(errors[:, :, 2]))
        errors[:, :, 3] = np.arctan2(np.sin(errors[:, :, 3]), np.cos(errors[:, :, 3]))

        if quantile >= 1.0:
            box_halfwidths = np.max(np.abs(errors), axis=0)
            box_min = np.min(errors, axis=0)
            box_max = np.max(errors, axis=0)
        else:
            box_halfwidths = np.quantile(np.abs(errors), quantile, axis=0)
            q_low = (1.0 - quantile) * 0.5
            q_high = 1.0 - q_low
            box_min = np.quantile(errors, q_low, axis=0)
            box_max = np.quantile(errors, q_high, axis=0)

        r_tube = np.sqrt(box_halfwidths[:, 0] ** 2 + box_halfwidths[:, 1] ** 2)
        b_hitch = box_halfwidths[:, 2] + box_halfwidths[:, 3]
        b_v = box_halfwidths[:, 4]

        result: dict[str, Any] = {
            "level": int(level),
            "N_mc": int(N_mc),
            "T_horizon": int(T_horizon),
            "dt": float(env.dt),
            "seed": int(seed),
            "quantile": float(quantile),
            "x_ref": np.asarray(x_ref, dtype=np.float32),
            "u_ref": np.asarray(u_ref, dtype=np.float32),
            "K_fb": np.asarray(K_fb, dtype=np.float32),
            "box_halfwidths": np.asarray(box_halfwidths, dtype=np.float32),
            "box_min": np.asarray(box_min, dtype=np.float32),
            "box_max": np.asarray(box_max, dtype=np.float32),
            "r_tube": np.asarray(r_tube, dtype=np.float32),
            "b_hitch": np.asarray(b_hitch, dtype=np.float32),
            "b_v": np.asarray(b_v, dtype=np.float32),
            "control_noise_std": np.asarray(active_info["control_noise_std"], dtype=np.float32),
            "param_uncertainty_std": float(active_info["param_uncertainty_std"]),
            "non_gaussian_dist": str(active_info["non_gaussian_dist"]),
            "non_gaussian_df": float(active_info["non_gaussian_df"]),
            "state_disturbance_scale": float(active_info["state_disturbance_scale"]),
        }

        if save_path is not None:
            save_tubes(result, save_path)

        return result
    finally:
        env.set_perturbation(
            level=prev_info["perturbation_level"],
            seed=prev_info["perturbation_seed"],
            control_noise_std=prev_info["control_noise_std"],
            param_uncertainty_std=prev_info["param_uncertainty_std"],
            non_gaussian_dist=prev_info["non_gaussian_dist"],
            non_gaussian_df=prev_info["non_gaussian_df"],
            state_disturbance_scale=prev_info["state_disturbance_scale"],
            disturbance_clip_factor=prev_info["disturbance_clip_factor"],
        )
        env._perturbation_manually_set = prev_manual_flag


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RMPS reachable tube calibration for AccTractorTrailer2d.")
    parser.add_argument("--N-mc", type=int, default=1000, help="Number of Monte Carlo rollouts per level")
    parser.add_argument("--horizon", type=int, default=25, help="Tube prediction horizon T")
    parser.add_argument("--dt", type=float, default=0.25, help="Simulation timestep dt [s]")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quantile", type=float, default=0.99, help="Empirical bounding quantile in (0, 1]")
    parser.add_argument("--out-dir", type=str, default="results/tubes", help="Directory for saved .npz tubes")
    args = parser.parse_args()

    env = mbd.envs.get_env("acc_tt2d", case="navigation", dt=args.dt, H=50)
    x_ref, u_ref = generate_nominal_maneuver(env, T_horizon=args.horizon)
    nom_max_hitch_deg = float(np.rad2deg(np.max(np.abs(np.arctan2(np.sin(x_ref[:, 2] - x_ref[:, 3]), np.cos(x_ref[:, 2] - x_ref[:, 3]))))))
    assert nom_max_hitch_deg < float(np.rad2deg(env.phi_max)), (
        f"Nominal calibration maneuver exceeded phi_max: {nom_max_hitch_deg:.2f} deg"
    )
    print(f"Nominal calibration maneuver: T={args.horizon}, dt={args.dt}s, max |hitch|={nom_max_hitch_deg:.2f} deg")
    print("=" * 84)
    print(f"{'Level':<7} | {'max r_tube [m]':<16} | {'max b_hitch [rad / deg]':<26} | {'max b_v [m/s]':<14}")
    print("-" * 84)

    for level in (0, 1, 2, 3):
        save_file = default_tube_path(level=level, model_name="acc_tt2d", base_dir=args.out_dir)
        tube = calibrate_tubes(
            env=env,
            level=level,
            N_mc=args.N_mc,
            T_horizon=args.horizon,
            seed=args.seed,
            quantile=args.quantile,
            save_path=save_file,
        )
        loaded = load_tubes(save_file)
        assert np.allclose(loaded["r_tube"], tube["r_tube"])
        assert np.allclose(loaded["b_hitch"], tube["b_hitch"])
        assert np.allclose(loaded["b_v"], tube["b_v"])

        max_r = float(np.max(tube["r_tube"]))
        max_bh = float(np.max(tube["b_hitch"]))
        max_bv = float(np.max(tube["b_v"]))

        assert np.all(np.isfinite(tube["r_tube"]))
        assert np.all(np.isfinite(tube["b_hitch"]))
        assert np.all(np.isfinite(tube["b_v"]))

        if level == 0:
            assert max_r < 1e-6 and max_bh < 1e-6 and max_bv < 1e-6, (
                f"Expected ~0 margins at Level 0, got r={max_r}, bh={max_bh}, bv={max_bv}"
            )
        else:
            assert max_r > 0.0 and max_bh > 0.0, f"Expected positive margins at Level {level}"

        print(
            f"Level {level} | {max_r:<16.4f} | {max_bh:>7.4f} rad ({np.rad2deg(max_bh):>6.2f} deg)  | {max_bv:<14.4f}"
        )

    print("=" * 84)
    print(f"Saved calibrated tubes to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
