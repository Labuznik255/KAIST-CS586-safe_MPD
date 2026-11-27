# `tests/test_nmpc.py`

**Source file:** [tests/test_nmpc.py](../../tests/test_nmpc.py)

## Overview
CLI runner and verification script for the standalone CasADi + IPOPT NMPC planner ([mbd/planners/nmpc_planner.py](../mbd/planners/nmpc_planner.md)) across the same navigation/parking scenarios, 4 vehicle models (`tt2d`, `acc_tt2d`, `kinematic_bicycle2d`, `n_trailer2d`), and Perturbation Levels 0–3 as [tests/test_mbd.py](test_mbd.md).

---

## Functions & CLI
- **`build_navigation_environment(config)`** & **`build_parking_environment(config)`**: Construct the exact navigation and parking benchmark environments for `NMPCConfig`.
- **CLI flags**: `--env`, `--case {navigation,parking}`, `--level {0,1,2,3}`, `--mpc-mode`, `--all-vehicles`, `--no-render`, `--seed`, `--horizon`, `--dt`.
