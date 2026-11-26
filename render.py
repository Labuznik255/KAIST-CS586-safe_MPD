"""
Standalone visualizer for any already-generated trajectory in Safe MPD.

Can be run on any results directory (e.g. `results/acc_tt2d`, `results/tt2d`,
`results/kinematic_bicycle2d`, `results/n_trailer2d`) or `.npz` test archive.

Usage examples:
    python render.py results/acc_tt2d
    python render.py results/tt2d --mode both
    python render.py results/acc_tt2d --mode perturbed
    python render.py results/acc_tt2d --no-animate
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mbd
from mbd.utils import load_env_from_results, create_animation


def _load_trajectories_from_target(target_path: Path):
    """Load nominal/perturbed states and actions from a directory or .npz file."""
    if target_path.is_file() and target_path.suffix == ".npz":
        data = np.load(target_path)
        nominal_states = data["nominal_states"] if "nominal_states" in data else data["states"]
        perturbed_states = data["perturbed_states"] if "perturbed_states" in data else None
        actions = data["actions"]
        perturbed_actions = data["perturbed_actions"] if "perturbed_actions" in data else None
        mu_0ts = data["mu_0ts"] if "mu_0ts" in data else None
        results_dir = target_path.parent
        return results_dir, nominal_states, perturbed_states, actions, perturbed_actions, mu_0ts

    results_dir = target_path
    nominal_path = results_dir / "nominal_trajectory_states.npy"
    perturbed_path = results_dir / "perturbed_trajectory_states.npy"
    legacy_path = results_dir / "trajectory_states.npy"
    actions_path = results_dir / "final_actions.npy"
    pert_actions_path = results_dir / "perturbed_actions.npy"
    mu_path = results_dir / "mu_0ts.npy"

    if nominal_path.exists():
        nominal_states = np.load(nominal_path)
    elif legacy_path.exists():
        nominal_states = np.load(legacy_path)
    else:
        raise FileNotFoundError(f"No trajectory state files found in '{results_dir}'.")

    perturbed_states = np.load(perturbed_path) if perturbed_path.exists() else None
    actions = np.load(actions_path) if actions_path.exists() else np.zeros((max(1, len(nominal_states) - 1), 2))
    perturbed_actions = np.load(pert_actions_path) if pert_actions_path.exists() else None
    mu_0ts = np.load(mu_path) if mu_path.exists() else None

    return results_dir, nominal_states, perturbed_states, actions, perturbed_actions, mu_0ts


def visualize_results(
    target_path: str | Path = "results/acc_tt2d",
    mode: str = "both",
    animate: bool = True,
    save_animation: bool = False,
    show_static: bool = True,
):
    """
    Visualize nominal and/or perturbed trajectories from `target_path`.

    Args:
        target_path: Directory containing `.npy` files (e.g. `results/acc_tt2d`) or path to `.npz`.
        mode: "both", "nominal", or "perturbed".
        animate: Whether to run the rigid-body vehicle animation after the static plot.
        save_animation: Whether to export MP4 video(s) of the animation.
        show_static: Whether to display the static trajectory + control figure.
    """
    target_path = Path(target_path)
    if not target_path.exists() and target_path == Path("."):
        if (Path("results/acc_tt2d") / "final_actions.npy").exists():
            target_path = Path("results/acc_tt2d")

    results_dir, nominal_states, perturbed_states, actions, perturbed_actions, mu_0ts = (
        _load_trajectories_from_target(target_path)
    )

    print(f"Loaded from: {results_dir}")
    print(f"  nominal_trajectory_states:   {nominal_states.shape}")
    if perturbed_states is not None:
        print(f"  perturbed_trajectory_states: {perturbed_states.shape}")
    print(f"  final_actions:               {actions.shape}")
    if perturbed_actions is not None:
        print(f"  perturbed_actions:           {perturbed_actions.shape}")
    if mu_0ts is not None:
        print(f"  mu_0ts:                      {mu_0ts.shape}")

    # Reconstruct environment for obstacle/map rendering and rigid-body animation
    env, config, meta = load_env_from_results(results_dir)
    pert_info = meta.get("perturbation_info", {})
    pert_level = pert_info.get("perturbation_level", 0)
    env.perturbation_level = pert_level

    has_perturbed = perturbed_states is not None and (
        nominal_states.shape != perturbed_states.shape
        or not np.allclose(nominal_states, perturbed_states)
    )

    # 1. Static figure: Environment + (x, y) trajectories on left, Controls on right
    if show_static:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Left subplot: full environment (obstacles, parking spaces) + trajectories
        ax = axes[0]
        if mode == "perturbed" and perturbed_states is not None:
            orig_H = env.H
            env.H = len(perturbed_states) - 1
            try:
                env.render(ax, perturbed_states)
            finally:
                env.H = orig_H
        else:
            env.render(ax, nominal_states)

        # Start and Goal markers
        ax.scatter(env.x0[0], env.x0[1], color="green", s=110, marker="o", edgecolor="black", zorder=6, label="Start")
        ax.scatter(env.xg[0], env.xg[1], color="gold", s=140, marker="*", edgecolor="black", zorder=6, label="Goal")

        # Heading arrows for nominal
        if mode in ("both", "nominal") and nominal_states.shape[1] >= 3:
            theta_nom = nominal_states[:, 2]
            step = max(1, len(nominal_states) // 15)
            ax.quiver(
                nominal_states[::step, 0],
                nominal_states[::step, 1],
                np.cos(theta_nom[::step]),
                np.sin(theta_nom[::step]),
                color="red",
                angles="xy",
                scale_units="xy",
                scale=1.2,
                alpha=0.6,
                width=0.004,
            )

        # Overlay perturbed trajectory if requested
        if mode in ("both", "perturbed") and perturbed_states is not None:
            x_pert, y_pert = perturbed_states[:, 0], perturbed_states[:, 1]
            ax.scatter(x_pert, y_pert, c=range(len(x_pert)), cmap="Blues", s=35, alpha=0.85, zorder=4)
            ax.plot(
                x_pert,
                y_pert,
                "b--",
                linewidth=2.0,
                label=f"Perturbed path (Level {pert_level})",
                alpha=0.9,
                zorder=5,
            )
            if perturbed_states.shape[1] >= 3:
                theta_pert = perturbed_states[:, 2]
                step = max(1, len(perturbed_states) // 15)
                ax.quiver(
                    x_pert[::step],
                    y_pert[::step],
                    np.cos(theta_pert[::step]),
                    np.sin(theta_pert[::step]),
                    color="blue",
                    angles="xy",
                    scale_units="xy",
                    scale=1.2,
                    alpha=0.6,
                    width=0.004,
                )

        ax.set_title(f"Trajectory Rollout ({config.env_name}, Level {pert_level})")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

        # Right subplot: Control inputs over horizon
        ax_u = axes[1]
        if mode in ("both", "nominal"):
            for i in range(actions.shape[1]):
                ax_u.plot(actions[:, i], linewidth=2, label=f"Nominal u[{i}]")
        if mode in ("both", "perturbed") and perturbed_actions is not None:
            for i in range(perturbed_actions.shape[1]):
                ax_u.plot(perturbed_actions[:, i], "--", linewidth=1.5, alpha=0.85, label=f"Perturbed u[{i}]")
        ax_u.set_title("Control Inputs Over Horizon")
        ax_u.set_xlabel("Horizon Step")
        ax_u.set_ylabel("Normalized Control [-1, 1]")
        ax_u.grid(True, alpha=0.3)
        ax_u.legend(loc="best")

        plt.tight_layout()
        if results_dir.is_dir():
            plt.savefig(results_dir / "rollout.png")
            plt.savefig(results_dir / "rollout.svg")
        plt.show()
        plt.close(fig)

    # 2. Rigid-body vehicle animation
    if animate or save_animation:
        config.show_animation = animate
        config.save_animation = save_animation

        if mode in ("both", "nominal"):
            nom_states_list = [nominal_states[i] for i in range(len(nominal_states))]
            nom_actions_list = [actions[t] for t in range(actions.shape[0])] + [None]
            create_animation(
                env,
                nom_states_list,
                nom_actions_list,
                config,
                video_name="nominal_trajectory_animation.mp4" if save_animation else None,
                trajectory_label="Nominal trajectory",
            )

        if mode == "perturbed" or (mode == "both" and (has_perturbed or pert_level > 0)):
            if perturbed_states is not None:
                pert_states_list = [perturbed_states[i] for i in range(len(perturbed_states))]
                act_seq = perturbed_actions if perturbed_actions is not None else actions
                pert_actions_list = [act_seq[t] for t in range(act_seq.shape[0])] + [None]
                nom_overlay = [nominal_states[i] for i in range(len(nominal_states))] if mode == "both" else None
                create_animation(
                    env,
                    pert_states_list,
                    pert_actions_list,
                    config,
                    video_name="perturbed_trajectory_animation.mp4" if save_animation else None,
                    nominal_trajectory_overlay=nom_overlay,
                )


def main():
    parser = argparse.ArgumentParser(description="Standalone visualizer for generated Safe-MPD trajectories.")
    parser.add_argument(
        "target",
        nargs="?",
        default="results/acc_tt2d" if Path("results/acc_tt2d").exists() else ".",
        help="Path to results directory (e.g. results/acc_tt2d) or .npz file",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "nominal", "perturbed"],
        default="both",
        help="Which trajectory to visualize (default: both)",
    )
    parser.add_argument("--no-animate", action="store_true", help="Show only the static plot without animation")
    parser.add_argument("--save-animation", action="store_true", help="Export animation(s) to MP4")
    parser.add_argument("--no-static", action="store_true", help="Skip static plot and go straight to animation")
    args = parser.parse_args()

    visualize_results(
        target_path=args.target,
        mode=args.mode,
        animate=not args.no_animate,
        save_animation=args.save_animation,
        show_static=not args.no_static,
    )


if __name__ == "__main__":
    main()
