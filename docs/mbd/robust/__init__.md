# `mbd/robust/__init__.py`

**Source file:** [mbd/robust/__init__.py](../../../mbd/robust/__init__.py)

## Overview
Subpackage entry point exporting the **Robust Safe-MPD (RMPS)** toolchain implemented across Stages 1–4 for `AccTractorTrailer2d`:

- **Stage 1 (`tube_calibration.py`)**: `calibrate_tubes`, `compute_tvlqr_gains`, `default_tube_path`, `generate_nominal_maneuver`, `load_tubes`, `save_tubes`.
- **Stage 2 (`ilqr_backup.py`)**: `ILQRBackupParams`, `acc_tt2d_jacobians`, `acc_tt2d_step`, `generate_braking_warmstart`, `solve_ilqr_backup`.
- **Stage 3 (`planner_wrapper.py`, `nmpc_backup.py`)**: `SafeMPDPlanner`, `RobustNMPCBackup`.
- **Stage 4 (`rmps_runner.py`)**: `compute_min_obstacle_clearance_np`, `plot_rmps_diagnostics`, `run_rmps_episode`, `save_rmps_results`.
