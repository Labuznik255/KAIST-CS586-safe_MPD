"""
Test script for the standalone CasADi + IPOPT Nonlinear Model Predictive Control (NMPC) planner.

This script uses the exact same environments (`mbd.envs.Env`), obstacles, vehicle classes
(`TractorTrailer2d`, `AccTractorTrailer2d`, `KinematicBicycle2d`, `NTrailer2d`), and
perturbation modes (Levels 0, 1, 2, 3) as `tests/test_mbd.py`.

Usage examples:
    # Run the default custom navigation scenario (mirrors tests/test_mbd.py):
    python tests/test_nmpc.py

    # Run with a specific vehicle, scenario, or perturbation level:
    python tests/test_nmpc.py --env tt2d --case navigation --level 1
    python tests/test_nmpc.py --env acc_tt2d --case parking --level 2
    python tests/test_nmpc.py --env acc_tt2d --mpc-mode --level 1

    # Run verification across all 4 vehicle models:
    python tests/test_nmpc.py --all-vehicles --no-render
"""

import argparse
from pathlib import Path
import sys
import jax.numpy as jnp
import numpy as np

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mbd
from mbd.envs.env import Env
from mbd.planners.nmpc_planner import NMPCConfig, run_nmpc, _check_jax_trajectory_violations


def build_navigation_environment(config: NMPCConfig):
    """Build the custom navigation environment identical to tests/test_mbd.py."""
    env_config = Env(width=32.0, height=32.0, case="navigation")
    env = mbd.envs.get_env(
        config.env_name,
        case="navigation",
        env_config=env_config,
        dt=config.dt,
        H=config.Hsample,
        num_trailers=config.num_trailers,
        perturbation_level=config.perturbation_level,
        perturbation_seed=config.perturbation_seed,
        control_noise_std=config.control_noise_std,
        param_uncertainty_std=config.param_uncertainty_std,
        non_gaussian_dist=config.non_gaussian_dist,
        non_gaussian_df=config.non_gaussian_df,
        state_disturbance_scale=config.state_disturbance_scale,
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


def build_parking_environment(config: NMPCConfig):
    """Build the standard parking lot environment."""
    env = mbd.envs.get_env(
        config.env_name,
        case="parking",
        dt=config.dt,
        H=config.Hsample,
        num_trailers=config.num_trailers,
        perturbation_level=config.perturbation_level,
        perturbation_seed=config.perturbation_seed,
        control_noise_std=config.control_noise_std,
        param_uncertainty_std=config.param_uncertainty_std,
        non_gaussian_dist=config.non_gaussian_dist,
        non_gaussian_df=config.non_gaussian_df,
        state_disturbance_scale=config.state_disturbance_scale,
    )
    if config.num_trailers > 1 or config.env_name == "n_trailer2d":
        env.set_init_pos(dx=12.0, dy=8.0, theta1=0.0, theta2=0.0)
    else:
        env.set_init_pos(dx=3.0, dy=5.0, theta1=0.0, theta2=0.0)
    env.set_goal_pos(theta1=-jnp.pi / 2, theta2=-jnp.pi / 2)
    return env


def run_single_test(config: NMPCConfig):
    """Run a single NMPC test and print comprehensive diagnostics."""
    if config.case == "navigation":
        env = build_navigation_environment(config)
    else:
        env = build_parking_environment(config)

    pert_info = env.get_perturbation_info()
    print("=" * 68)
    print(f"Running IPOPT NMPC Planner ({config.env_name}, case={config.case}, mpc_mode={config.mpc_mode})")
    print("=" * 68)
    print(f"  Initial position:   {env.x0}")
    print(f"  Goal position:      {env.xg}")
    print(f"  Perturbation level: {pert_info['perturbation_level']}")
    print(f"  Nominal params:     {pert_info['nominal_params']}")
    print(f"  Perturbed params:   {pert_info['perturbed_params']}")

    rew_final, Y0, trajectory_states, timing_info = run_nmpc(args=config, env=env)

    nom_traj = np.asarray(env.nominal_trajectory_states)
    pert_traj = np.asarray(env.perturbed_trajectory_states)
    goal_xy = np.asarray(env.xg[:2])

    nom_goal_err = float(np.linalg.norm(nom_traj[-1, :2] - goal_xy))
    pert_goal_err = float(np.linalg.norm(pert_traj[-1, :2] - goal_xy))
    xy_dev = float(np.linalg.norm(pert_traj[-1, :2] - nom_traj[-1, :2]))

    nom_col, nom_hitch = _check_jax_trajectory_violations(env, nom_traj)
    pert_col, pert_hitch = _check_jax_trajectory_violations(env, pert_traj)

    print("\nNMPC Planning Completed!")
    print(f"  IPOPT Status:                 {timing_info['ipopt_return_status']} ({timing_info['ipopt_iter_count']} iters)")
    print(f"  Final nominal reward:         {float(rew_final):.3f}")
    print(f"  Final nominal state:          {nom_traj[-1].round(4)}")
    print(f"  Final perturbed state:        {pert_traj[-1].round(4)}")
    print(f"  Nominal goal (x,y) error [m]: {nom_goal_err:.4f}")
    print(f"  Perturbed goal error [m]:     {pert_goal_err:.4f}")
    print(f"  Final (x, y) deviation [m]:   {xy_dev:.4f}")
    print(f"  Nominal violations:           collision={nom_col}, hitch={nom_hitch}")
    print(f"  Perturbed violations:         collision={pert_col}, hitch={pert_hitch}")
    print(f"  Pure NMPC solve time:         {timing_info['pure_nmpc_time']:.3f}s")
    print(f"  Total time:                   {timing_info['total_time']:.3f}s")
    print("=" * 68)

    return rew_final, Y0, trajectory_states, timing_info, (nom_col, nom_hitch, nom_goal_err)


def run_all_vehicles_suite(render: bool = False):
    """Verify IPOPT NMPC across all 4 vehicle types in the repository."""
    specs = [
        ("kinematic_bicycle2d", 0),
        ("tt2d", 1),
        ("acc_tt2d", 1),
        ("n_trailer2d", 2),
    ]
    print("\nRunning IPOPT NMPC Multi-Vehicle Verification Suite...")
    for env_name, num_trailers in specs:
        cfg = NMPCConfig(
            env_name=env_name,
            case="navigation",
            num_trailers=num_trailers,
            Hsample=50,
            dt=0.25,
            render=render,
            show_animation=render,
            save_animation=False,
            perturbation_level=1,
            perturbation_seed=42,
            control_noise_std=(0.05, 0.05),
        )
        _, _, _, timing, (nom_col, nom_hitch, nom_goal_err) = run_single_test(cfg)
        assert not nom_col, f"Nominal trajectory for {env_name} had an obstacle collision!"
        assert not nom_hitch, f"Nominal trajectory for {env_name} had a hitch violation!"
        assert nom_goal_err < 1.5, f"Nominal trajectory for {env_name} did not reach goal (err={nom_goal_err:.3f}m)!"
    print("\nALL 4 VEHICLE MODELS PASSED IPOPT NMPC VERIFICATION!")


def main():
    parser = argparse.ArgumentParser(description="Test standalone CasADi + IPOPT NMPC planner.")
    parser.add_argument("--env", type=str, default="acc_tt2d", choices=["tt2d", "acc_tt2d", "kinematic_bicycle2d", "n_trailer2d"], help="Vehicle model")
    parser.add_argument("--case", type=str, default="navigation", choices=["navigation", "parking"], help="Scenario type")
    parser.add_argument("--num-trailers", type=int, default=1, help="Number of trailers (for tt2d / n_trailer2d)")
    parser.add_argument("--horizon", type=int, default=50, help="Planning horizon Hsample")
    parser.add_argument("--level", type=int, default=1, choices=[0, 1, 2, 3], help="Perturbation level (0..3)")
    parser.add_argument("--seed", type=int, default=None, help="Perturbation seed (None = random each run)")
    parser.add_argument("--mpc-mode", action="store_true", help="Enable receding-horizon closed-loop NMPC during perturbed rollout")
    parser.add_argument("--no-render", action="store_true", help="Disable GUI plots and animations")
    parser.add_argument("--all-vehicles", action="store_true", help="Run verification across all 4 vehicle models")
    args = parser.parse_args()

    if args.all_vehicles:
        run_all_vehicles_suite(render=not args.no_render)
        return

    num_trailers = args.num_trailers
    if args.env == "kinematic_bicycle2d":
        num_trailers = 0
    elif args.env == "n_trailer2d" and num_trailers < 2:
        num_trailers = 2

    config = NMPCConfig(
        env_name=args.env,
        case=args.case,
        num_trailers=num_trailers,
        Hsample=args.horizon,
        dt=0.25,
        mpc_mode=args.mpc_mode,
        render=not args.no_render,
        show_animation=not args.no_render,
        save_animation=False,
        # --- Perturbation parameters ---
        perturbation_level=args.level,
        perturbation_seed=args.seed,
        control_noise_std=(0.05, 0.05),
        param_uncertainty_std=0.10,
        non_gaussian_dist="student_t",
        non_gaussian_df=2.5,
        state_disturbance_scale=0.05,
    )

    run_single_test(config)


if __name__ == "__main__":
    main()
