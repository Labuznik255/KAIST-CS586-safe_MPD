"""
Standalone perturbation testing script for Safe Model Predictive Diffusion (Safe MPD).

This script loads an already-generated nominal action trajectory (`final_actions.npy`)
from a results directory (without re-running the diffusion planner), applies the
selected perturbation mode (Level 0, 1, 2, or 3), rolls out the perturbed trajectory,
saves both `nominal_trajectory_states.npy` and `perturbed_trajectory_states.npy`,
and reports trajectory deviation and safety (collision / jackknife) metrics.

Usage examples:
    # Run with parameters configured in main() below:
    python tests/test_perturbations.py

    # Run from CLI on a specific results directory and perturbation level:
    python tests/test_perturbations.py results/acc_tt2d --level 1 --control-noise-std 0.08 0.08
    python tests/test_perturbations.py results/acc_tt2d --level 2 --param-uncertainty-std 0.15 --seed 42
    python tests/test_perturbations.py results/acc_tt2d --level 3 --dist student_t --df 2.5 --state-disturbance-scale 0.05
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mbd
from mbd.utils import (
    rollout_perturbed_us,
    load_env_from_results,
    save_env_metadata,
    create_animation,
)


def check_trajectory_safety(env, states_np: np.ndarray, use_perturbed_geom: bool = False):
    """Check whether any state along the trajectory collides with obstacles or violates hitch limits."""
    is_bicycle = env.observation_size == 3
    is_multi_trailer = getattr(env, "num_trailers", 1) >= 2

    collision_steps = []
    hitch_steps = []

    for t, state in enumerate(states_np):
        if is_bicycle:
            s_chk = jnp.asarray(state[:3])
        elif is_multi_trailer:
            s_chk = jnp.asarray(state)
        else:
            s_chk = jnp.asarray(state[:4])

        col = bool(env.check_obstacle_collision(s_chk, env.obs_circles, env.obs_rectangles, use_perturbed=use_perturbed_geom))
        hitch = False if is_bicycle else bool(env.check_hitch_violation(s_chk))
        if col:
            collision_steps.append(t)
        if hitch:
            hitch_steps.append(t)

    return {
        "collision": len(collision_steps) > 0,
        "collision_steps": collision_steps,
        "hitch_violation": len(hitch_steps) > 0,
        "hitch_steps": hitch_steps,
    }


def perturb_existing_trajectory(
    results_dir: str | Path = "results/acc_tt2d",
    perturbation_level: int = 1,
    perturbation_seed: int | None = None,
    control_noise_std=(0.05, 0.05),
    param_uncertainty_std: float = 0.10,
    non_gaussian_dist: str = "student_t",
    non_gaussian_df: float = 2.5,
    state_disturbance_scale: float = 0.05,
    disturbance_clip_factor: float = 5.0,
    perturbed_l1: float | None = None,
    perturbed_l2: float | None = None,
    perturbed_lh: float | None = None,
    visualize: bool = True,
):
    """
    Load an already-generated nominal action sequence `final_actions.npy` from `results_dir`,
    simulate the perturbed rollout under the specified perturbation mode, and save the results.
    """
    results_path = Path(results_dir)
    actions_file = results_path / "final_actions.npy"
    if not actions_file.exists():
        raise FileNotFoundError(
            f"Could not find '{actions_file}'. Run a planner script (e.g. `python tests/test_mbd.py`) first to generate nominal actions."
        )

    # Load nominal actions
    nominal_actions = jnp.asarray(np.load(actions_file), dtype=jnp.float32)

    # Reconstruct environment and configuration from results_dir
    env, config, _ = load_env_from_results(results_path)

    # Load or compute nominal trajectory states
    nom_file = results_path / "nominal_trajectory_states.npy"
    leg_file = results_path / "trajectory_states.npy"
    if nom_file.exists():
        nominal_states = jnp.asarray(np.load(nom_file), dtype=jnp.float32)
    elif leg_file.exists():
        nominal_states = jnp.asarray(np.load(leg_file), dtype=jnp.float32)
    else:
        # Roll out nominal actions on Level 0 nominal model
        env.set_perturbation(level=0, seed=0)
        state_nom_init = env.reset(jax.random.PRNGKey(0))
        _, nominal_states, _ = rollout_perturbed_us(jax.jit(env.step_perturbed), state_nom_init, nominal_actions)

    # Ensure initial state x0 matches the start of the nominal trajectory
    env.x0 = jnp.asarray(nominal_states[0], dtype=jnp.float32)

    # Configure requested perturbation mode & parameters on the environment
    pert_info = env.set_perturbation(
        level=perturbation_level,
        seed=perturbation_seed,
        control_noise_std=control_noise_std,
        param_uncertainty_std=param_uncertainty_std,
        non_gaussian_dist=non_gaussian_dist,
        non_gaussian_df=non_gaussian_df,
        state_disturbance_scale=state_disturbance_scale,
        disturbance_clip_factor=disturbance_clip_factor,
        perturbed_l1=perturbed_l1,
        perturbed_l2=perturbed_l2,
        perturbed_lh=perturbed_lh,
    )

    # Roll out the nominal actions in the perturbed environment
    state_pert_init = env.reset(jax.random.PRNGKey(env.perturbation_seed))
    step_perturbed_jit = jax.jit(env.step_perturbed)
    perturbed_rewards, perturbed_states, perturbed_actions = rollout_perturbed_us(
        step_perturbed_jit, state_pert_init, nominal_actions
    )

    env.nominal_trajectory_states = nominal_states
    env.perturbed_trajectory_states = perturbed_states
    env.perturbed_actions = perturbed_actions

    # Save nominal trajectory and then perturbed trajectory
    jnp.save(str(results_path / "nominal_trajectory_states.npy"), nominal_states)
    jnp.save(str(results_path / "perturbed_trajectory_states.npy"), perturbed_states)
    jnp.save(str(results_path / "trajectory_states.npy"), nominal_states)
    jnp.save(str(results_path / "perturbed_actions.npy"), perturbed_actions)
    save_env_metadata(env, config, str(results_path))

    # Compute diagnostics
    nom_np = np.asarray(nominal_states)
    pert_np = np.asarray(perturbed_states)
    act_nom_np = np.asarray(nominal_actions)
    act_pert_np = np.asarray(perturbed_actions)
    action_noise = act_pert_np - act_nom_np

    pos_dev_per_step = np.linalg.norm(pert_np[:, :2] - nom_np[:, :2], axis=1)
    nom_safety = check_trajectory_safety(env, nom_np, use_perturbed_geom=False)
    pert_safety = check_trajectory_safety(env, pert_np, use_perturbed_geom=(perturbation_level == 2))

    print("=" * 65)
    print(f"PERTURBATION ROLLOUT SUMMARY ({results_path})")
    print("=" * 65)
    print(f"Robot / Environment : {config.env_name} (state dim={nom_np.shape[1]}, horizon={act_nom_np.shape[0]})")
    print(f"Perturbation Level  : Level {pert_info['perturbation_level']}")
    print(f"Perturbation Seed   : {pert_info['perturbation_seed']}")
    if perturbation_level in (1, 3):
        print(f"Control Noise Std   : {pert_info['control_noise_std']}")
        print(f"Sampled Step Noise  : mean={action_noise.mean(axis=0).round(4)}, std={action_noise.std(axis=0).round(4)}")
    if perturbation_level == 2:
        print(f"Nominal Params      : {pert_info['nominal_params']}")
        print(f"Perturbed Params    : {pert_info['perturbed_params']} (time-invariant)")
    if perturbation_level == 3:
        print(f"Non-Gaussian Dist   : {pert_info['non_gaussian_dist']} (df={pert_info['non_gaussian_df']}, state_scale={pert_info['state_disturbance_scale']})")
    print("-" * 65)
    print(f"Nominal Final Pose  : {nom_np[-1, :min(4, nom_np.shape[1])].round(4)}")
    print(f"Perturbed Final Pose: {pert_np[-1, :min(4, pert_np.shape[1])].round(4)}")
    print(f"Max (x,y) Deviation : {pos_dev_per_step.max():.4f} m")
    print(f"End (x,y) Deviation : {pos_dev_per_step[-1]:.4f} m")
    print(f"Nominal Safety      : collision={nom_safety['collision']}, hitch_violation={nom_safety['hitch_violation']}")
    print(f"Perturbed Safety    : collision={pert_safety['collision']} (steps {pert_safety['collision_steps']}), "
          f"hitch_violation={pert_safety['hitch_violation']} (steps {pert_safety['hitch_steps']})")
    print(f"Saved Files         : {results_path / 'nominal_trajectory_states.npy'}")
    print(f"                      {results_path / 'perturbed_trajectory_states.npy'}")
    print("=" * 65)

    if visualize:
        from render import visualize_results
        visualize_results(results_path, animate=True)

    return nominal_states, perturbed_states, perturbed_actions, pert_info


def main():
    # =========================================================================
    # Default Perturbation Settings (edit directly here or pass CLI flags)
    # =========================================================================
    DEFAULT_RESULTS_DIR = "results/acc_tt2d"
    DEFAULT_LEVEL = 1                     # 0: Nominal, 1: Gaussian Control Noise, 2: Param Uncertainty, 3: Non-Gaussian
    DEFAULT_SEED = None                   # None = random seed each run; or set int (e.g. 42) for reproducible noise
    DEFAULT_CONTROL_NOISE_STD = (0.05, 0.05)
    DEFAULT_PARAM_UNCERTAINTY_STD = 0.15  # Time-invariant relative link-length std for Level 2
    DEFAULT_NON_GAUSSIAN_DIST = "student_t"  # "uniform", "student_t", "cauchy", "laplace"
    DEFAULT_NON_GAUSSIAN_DF = 2.5
    DEFAULT_STATE_DISTURBANCE_SCALE = 0.05
    DEFAULT_VISUALIZE = True

    parser = argparse.ArgumentParser(description="Perturb an already-generated nominal action trajectory.")
    parser.add_argument("results_dir", nargs="?", default=DEFAULT_RESULTS_DIR, help="Path to results directory containing final_actions.npy")
    parser.add_argument("--level", type=int, default=DEFAULT_LEVEL, choices=[0, 1, 2, 3], help="Perturbation level (0, 1, 2, 3)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for perturbation (default: random each run)")
    parser.add_argument("--control-noise-std", type=float, nargs="+", default=list(DEFAULT_CONTROL_NOISE_STD), help="Control noise std (1 or 2 floats)")
    parser.add_argument("--param-uncertainty-std", type=float, default=DEFAULT_PARAM_UNCERTAINTY_STD, help="Relative std for Level 2 link-length uncertainty")
    parser.add_argument("--dist", type=str, default=DEFAULT_NON_GAUSSIAN_DIST, choices=["uniform", "student_t", "cauchy", "laplace"], help="Non-Gaussian distribution for Level 3")
    parser.add_argument("--df", type=float, default=DEFAULT_NON_GAUSSIAN_DF, help="Degrees of freedom for student_t in Level 3")
    parser.add_argument("--state-disturbance-scale", type=float, default=DEFAULT_STATE_DISTURBANCE_SCALE, help="External state-rate disturbance scale for Level 3")
    parser.add_argument("--perturbed-l1", type=float, default=None, help="Explicit override for perturbed l1 in Level 2")
    parser.add_argument("--perturbed-l2", type=float, default=None, help="Explicit override for perturbed l2 in Level 2")
    parser.add_argument("--perturbed-lh", type=float, default=None, help="Explicit override for perturbed lh in Level 2")
    parser.add_argument("--no-visualize", action="store_true", help="Disable plotting and animation")
    args = parser.parse_args()

    perturb_existing_trajectory(
        results_dir=args.results_dir,
        perturbation_level=args.level,
        perturbation_seed=args.seed,
        control_noise_std=args.control_noise_std,
        param_uncertainty_std=args.param_uncertainty_std,
        non_gaussian_dist=args.dist,
        non_gaussian_df=args.df,
        state_disturbance_scale=args.state_disturbance_scale,
        perturbed_l1=args.perturbed_l1,
        perturbed_l2=args.perturbed_l2,
        perturbed_lh=args.perturbed_lh,
        visualize=(DEFAULT_VISUALIZE and not args.no_visualize),
    )


if __name__ == "__main__":
    main()
