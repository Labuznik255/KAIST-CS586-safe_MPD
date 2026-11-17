"""
Minimal example for running Safe Model Predictive Diffusion (Safe MPD) planner
with a custom navigation environment.

This script demonstrates how to:
1. Create a custom environment with navigation mode
2. Add rectangular and circular obstacles
3. Set initial and goal positions
4. Run the diffusion planner
"""

import jax.numpy as jnp
import mbd
from mbd.planners.mbd_planner import MBDConfig, run_diffusion
from mbd.envs.env import Env


def main():
    # ============================================================
    # Step 1: Create configuration (using defaults, modify as needed)
    # ============================================================
    # Perturbation modes:
    #   Level 0 - Nominal Environment (perfect model knowledge, no disturbances)
    #   Level 1 - Gaussian Control Noise (imperfect actuation / execution errors)
    #   Level 2 - Gaussian Model Parameter Uncertainty (time-invariant link-length mismatch)
    #   Level 3 - Non-Gaussian Disturbances (uniform or heavy-tailed student_t / cauchy / laplace)
    config = MBDConfig(
        env_name="acc_tt2d",       # "tt2d", "acc_tt2d", "kinematic_bicycle2d", or "n_trailer2d"
        case="navigation",         # Use "navigation" for custom environments
        Nsample=20000,             # Number of samples (reduce for faster testing)
        Hsample=50,                # Planning horizon
        Ndiffuse=100,              # Diffusion steps (reduce for faster testing)
        render=True,
        save_animation=False,
        show_animation=True,
        save_denoising_animation=False,
        # --- Perturbation parameters (can also be set via env.set_perturbation(...)) ---
        perturbation_level=1,             # 0: Nominal, 1: Gaussian Control Noise, 2: Param Uncertainty, 3: Non-Gaussian
        perturbation_seed=None,           # None = random seed each run; or set int (e.g. 42) for reproducibility
        control_noise_std=(0.05, 0.05),   # Std/scale of control noise in normalized [-1, 1]^2 space (Levels 1 & 3)
        param_uncertainty_std=0.10,       # Relative std of time-invariant link-length uncertainty (Level 2)
        non_gaussian_dist="student_t",    # "uniform", "student_t", "cauchy", or "laplace" (Level 3)
        non_gaussian_df=2.5,              # Degrees of freedom for student_t (Level 3)
        state_disturbance_scale=0.05,     # Scale of external process disturbance on state rates (Level 3)
    )
    
    # ============================================================
    # Step 2: Create custom environment with 'navigation' mode (or, use 'parking' mode for predefined parking scenario)
    # ============================================================
    # Create base environment
    env_config = Env(width=32.0, height=32.0, case="navigation")
    
    # ============================================================
    # Step 3: Create the robot environment with the custom config
    # ============================================================
    env = mbd.envs.get_env(
        config.env_name,
        case=config.case,
        env_config=env_config,
        dt=config.dt,
        H=config.Hsample,
        perturbation_level=config.perturbation_level,
        perturbation_seed=config.perturbation_seed,
        control_noise_std=config.control_noise_std,
        param_uncertainty_std=config.param_uncertainty_std,
        non_gaussian_dist=config.non_gaussian_dist,
        non_gaussian_df=config.non_gaussian_df,
        state_disturbance_scale=config.state_disturbance_scale,
    )

    # Optionally override perturbation mode/parameters directly on the environment:
    # env.set_perturbation(
    #     level=2,
    #     seed=123,
    #     param_uncertainty_std=0.15,
    # )
    
    # ============================================================
    # Step 4: Define obstacles
    # ============================================================
    # Parking area dimensions (two car-sized spaces)
    parking_space_width = 2.5   # Width of each parking space (meters)
    parking_space_length = 5.0  # Length of each parking space (meters)
    parking_x = 5.0             # X-center of parking area
    parking_y = -5.0            # Y-center of parking area
    
    # Add rectangular obstacles (left side of parking area)
    # Format: [x_center, y_center, width, height, rotation]
    rectangular_obstacles = [
        # Wall/obstacle to the left of parking area
        [parking_x - parking_space_width - 1.0, parking_y, 1.0, parking_space_length * 2, 0.0],
        # Bottom boundary wall
        [parking_x, parking_y - parking_space_length - 1.0, parking_space_width * 2 + 4.0, 1.0, 0.0],
        [parking_x - parking_space_width - 6.0, parking_y, 1.0, parking_space_length * 2, -jnp.pi/4],
        [parking_x - parking_space_width - 6.0, parking_y + 18.0, 1.2, parking_space_length * 2, jnp.pi/4],
        [parking_x - parking_space_width - 12.0, parking_y + 18.0, 0.8, parking_space_length, -jnp.pi/6],
    ]
    env.set_rectangle_obs(rectangular_obstacles, coordinate_mode="center", padding=0.0)
    
    # Add circular obstacles (right side of parking area)
    # Format: [x_center, y_center, radius]
    circular_obstacles = [
        [parking_x + parking_space_width + 0.0, parking_y + 3.0, 1.0], 
        [parking_x + parking_space_width + 0.0, parking_y - 3.0, 2.0], 
        [parking_x + parking_space_width + 1.0, parking_y, 1.5], 
        [parking_x + parking_space_width + 1.0, parking_y + 12.0, 3.5], 
    ]
    env.set_circle_obs(circular_obstacles, padding=0.0)
    
    # ============================================================
    # Step 5: Set initial and goal positions
    # ============================================================
    # Initial position: start from the left side, facing right
    env.set_init_pos(
        x=-10.0,              # X position
        y=5.0,                # Y position
        theta1=0.0,           # Tractor heading (0 = facing right)
        theta2=0.0,           # Trailer heading
    )
    
    # Goal position: enter the parking space
    env.set_goal_pos(
        x=parking_x,                    # X position (center of parking area)
        y=parking_y,                    # Y position
        theta1=-jnp.pi/2,               # Tractor heading (facing down into parking space)
        theta2=-jnp.pi/2,               # Trailer heading
    )
    
    # Optionally set custom plot limits
    #env.env.set_plot_limits(x_range=(-16, 16), y_range=(-16, 16))
    
    # ============================================================
    # Step 6: Run the diffusion planner
    # ============================================================
    pert_info = env.get_perturbation_info()
    print("Running Safe MPD planner...")
    print(f"  Initial position:   {env.x0}")
    print(f"  Goal position:      {env.xg}")
    print(f"  Perturbation level: {pert_info['perturbation_level']}")
    print(f"  Nominal params:     {pert_info['nominal_params']}")
    print(f"  Perturbed params:   {pert_info['perturbed_params']}")
    
    rew_final, Y0, trajectory_states, timing_info = run_diffusion(args=config, env=env)
    nominal_traj = env.nominal_trajectory_states
    perturbed_traj = env.perturbed_trajectory_states
    state_Deviation = float(jnp.linalg.norm(perturbed_traj[-1, :2] - nominal_traj[-1, :2]))
    
    print(f"\nPlanning completed!")
    print(f"  Final nominal reward:         {rew_final:.3f}")
    print(f"  Final nominal state:          {nominal_traj[-1]}")
    print(f"  Final perturbed state:        {perturbed_traj[-1]}")
    print(f"  Final (x, y) deviation [m]:   {state_Deviation:.4f}")
    print(f"  Pure diffusion time:          {timing_info['pure_diffusion_time']:.2f}s")
    print(f"  Total time:                   {timing_info['total_time']:.2f}s")


if __name__ == "__main__":
    main()

