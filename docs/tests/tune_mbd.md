# `tests/tune_mbd.py`

**Source file:** [tests/tune_mbd.py](../../tests/tune_mbd.py)

## Overview
Optuna-based hyperparameter tuner using a Gaussian Process sampler (`GPSampler`) and `MedianPruner`, optionally logging trial metrics and heat maps to Weights & Biases (`wandb`), maximizing `success_rate` from `stat_mbd.run_statistical_evaluation`.

---

## Class: `DiffusionOptimizer` (lines 39–458)

### `__init__(self, base_config: MBDConfig, num_trials_per_eval: int = 20, study_name: str = "diffusion_optimization", use_wandb: bool = True)` (lines 42–72)
Stores configuration and initializes a W&B run (`project="mbd-diffusion-optimization"`) if `use_wandb=True`.

### `_config_to_dict(self, config: MBDConfig) -> Dict[str, Any]` (lines 74–91)
Extracts a subset of `MBDConfig` fields into a dictionary for W&B metadata logging.

### `create_heat_map(self, results: StatisticalResults, config: MBDConfig, trial_number: int)` (lines 93–132)
Builds a sample environment via `create_trial_environment` and generates a `Figure` via `create_heat_map_visualization(..., return_fig=True)` for W&B image upload.

### `define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]` (lines 135–170)
Defines Optuna search variables; currently tunes `d_thr_factor` and `k_switch` in `[0.5, 5.0]` (step `0.5`), with other reward/diffusion parameters commented out for easy activation.

### `create_config_with_params(self, params: Dict[str, Any]) -> MBDConfig` (lines 172–258)
Constructs a new `MBDConfig` keeping fixed parameters from `self.base_config` and substituting tuned `params` (`d_thr_factor`, `k_switch`), with rendering disabled.

### `objective_function(self, trial: optuna.Trial) -> float` (lines 260–346)
1. Samples `params` and builds `config`.
2. Calls `clear_jit_cache()`.
3. Runs `run_statistical_evaluation(config=config, num_trials=self.num_trials_per_eval, seed=42, verbose=False)`.
4. Reports `success_rate` to Optuna for pruning (`trial.report(success_rate, 0)` / `trial.should_prune()`).
5. Logs metrics + heat map to W&B if enabled and returns `success_rate`.

### `run_optimization(self, n_trials: int = 50, timeout: Optional[int] = None, n_jobs: int = 1) -> optuna.Study` (lines 348–441)
Creates an Optuna maximization study with `GPSampler(seed=42)` and `MedianPruner(n_startup_trials=15, n_warmup_steps=1, interval_steps=1)`, runs `study.optimize(...)`, prints best trial summary, and logs final counts to W&B.

### `save_results(self, study: optuna.Study, filepath: str = "optimization_results.json")` (lines 443–458)
Serializes study metadata, best parameters, and `self.optimization_results` to JSON.

---

## Module-Level Functions

### `create_base_config() -> MBDConfig` (lines 461–500)
Returns a base `"tt2d"` `"parking"` `MBDConfig` with rendering disabled.

### `main()` (lines 503–531)
Instantiates `DiffusionOptimizer` (`num_trials_per_eval=100`, `use_wandb=True`), runs `n_trials=100` (`timeout=72000`), saves `mbd_optimization_results.json`, and finishes W&B.

---

## Extension Notes
- **New vehicle class / Perturbations**: Update `create_base_config()` (line 461), `_config_to_dict()` (line 74), and `create_config_with_params()` (line 172) so any new `MBDConfig` fields (e.g., vehicle dynamics or perturbation parameters) are preserved across Optuna trials.
