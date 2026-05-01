import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import optuna
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "Optuna is not installed. Install it with `pip install optuna` in the active environment."
    ) from exc

from train import (
    CONFIG_DIR,
    DEFAULT_CONFIG_PATH,
    PROJECT_DIR,
    TrainConfig,
    build_config_from_sources,
    ensure_dataset_prepared,
    train,
)

DEFAULT_OPTUNA_CONFIG_PATH = CONFIG_DIR / "optuna.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna tuning for the CAMELS simplified TFT.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the base training config JSON file.",
    )
    parser.add_argument(
        "--optuna-config",
        type=Path,
        default=DEFAULT_OPTUNA_CONFIG_PATH,
        help="Path to the Optuna search-space config JSON file.",
    )
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def resolve_config_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(payload).__name__}")
    return payload


def load_base_config(config_path: Path) -> TrainConfig:
    return build_config_from_sources(config_path, argparse.Namespace())


def load_optuna_config(path: Path) -> Dict[str, Any]:
    payload = load_json_file(path)
    payload.setdefault("study_name", "camels_simple_tft")
    payload.setdefault("n_trials", 20)
    payload.setdefault("timeout", None)
    payload.setdefault("storage", None)
    payload.setdefault("output_root", "outputs/optuna")
    payload.setdefault("epochs", None)
    payload.setdefault("objective_metric", "best_val_median_nse")
    payload.setdefault("search_space", {})
    payload.setdefault(
        "pruner",
        {
            "type": "median",
            "n_startup_trials": 5,
            "n_warmup_steps": 3,
            "interval_steps": 1,
        },
    )
    return payload


def ensure_list(value: Iterable[Any]) -> List[Any]:
    return list(value)


def suggest_param(
    trial: optuna.Trial,
    name: str,
    spec: Dict[str, Any],
    chosen_values: Dict[str, Any],
) -> Any:
    spec_type = spec.get("type")
    if spec_type == "float":
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=bool(spec.get("log", False)),
            step=spec.get("step"),
        )
    if spec_type == "int":
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=int(spec.get("step", 1)),
            log=bool(spec.get("log", False)),
        )
    if spec_type == "categorical":
        return trial.suggest_categorical(name, ensure_list(spec["choices"]))
    if spec_type == "divisors":
        base_name = spec["base"]
        if base_name not in chosen_values:
            raise ValueError(
                f"Search space for {name!r} depends on {base_name!r}, but {base_name!r} has not been chosen yet."
            )
        allowed_choices = [
            choice
            for choice in ensure_list(spec["choices"])
            if int(chosen_values[base_name]) % int(choice) == 0
        ]
        if not allowed_choices:
            raise ValueError(
                f"No valid choices remain for {name!r} after applying divisor constraint on {base_name!r}."
            )
        return trial.suggest_categorical(name, allowed_choices)
    raise ValueError(
        f"Unsupported search space type {spec_type!r} for parameter {name!r}. "
        "Expected one of: float, int, categorical, divisors."
    )


def apply_search_space(
    trial: optuna.Trial,
    cfg: TrainConfig,
    search_space: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    chosen_values: Dict[str, Any] = {}
    for name, spec in search_space.items():
        value = suggest_param(trial, name, spec, chosen_values)
        if not hasattr(cfg, name):
            raise ValueError(f"TrainConfig has no field named {name!r}")
        setattr(cfg, name, value)
        chosen_values[name] = value
    return chosen_values


def build_pruner(pruner_cfg: Dict[str, Any]) -> Optional[optuna.pruners.BasePruner]:
    pruner_type = str(pruner_cfg.get("type", "median")).lower()
    if pruner_type in {"none", "disabled"}:
        return None
    if pruner_type == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 3)),
            interval_steps=int(pruner_cfg.get("interval_steps", 1)),
        )
    raise ValueError(f"Unsupported pruner type {pruner_type!r}. Expected 'median' or 'none'.")


def objective_factory(
    base_cfg: TrainConfig,
    output_root: Path,
    search_space: Dict[str, Dict[str, Any]],
    objective_metric: str,
):
    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)
        chosen_values = apply_search_space(trial, cfg, search_space)
        cfg.rebuild_dataset = False
        cfg.output_dir = output_root / f"trial_{trial.number:03d}"

        try:
            metrics = train(cfg, trial=trial)
        except optuna.TrialPruned:
            trial.set_user_attr("output_dir", str(cfg.output_dir))
            trial.set_user_attr("sampled_params", chosen_values)
            raise

        best_val_metrics = metrics.get("best_val_metrics") or {}
        trial.set_user_attr("output_dir", metrics["output_dir"])
        trial.set_user_attr("sampled_params", chosen_values)
        trial.set_user_attr("best_val_metrics", best_val_metrics)
        trial.set_user_attr("test_metrics", metrics["test_metrics"])
        objective_value = metrics.get(objective_metric)
        if objective_value is None:
            raise ValueError(
                f"Objective metric {objective_metric!r} was not returned by train(). "
                f"Available keys: {sorted(metrics.keys())}"
            )
        return float(objective_value)

    return objective


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    optuna_config_path = resolve_config_path(args.optuna_config)

    base_cfg = load_base_config(config_path)
    optuna_cfg = load_optuna_config(optuna_config_path)

    study_name = args.study_name or optuna_cfg["study_name"]
    n_trials = args.n_trials if args.n_trials is not None else int(optuna_cfg["n_trials"])
    timeout = args.timeout if args.timeout is not None else optuna_cfg["timeout"]
    storage = args.storage if args.storage is not None else optuna_cfg["storage"]
    epochs_override = args.epochs if args.epochs is not None else optuna_cfg.get("epochs")
    objective_metric = str(optuna_cfg.get("objective_metric", "best_val_median_nse"))
    output_root_value = args.output_root if args.output_root is not None else Path(optuna_cfg["output_root"])
    output_root = (
        output_root_value if output_root_value.is_absolute() else PROJECT_DIR / output_root_value
    )
    output_root.mkdir(parents=True, exist_ok=True)

    if epochs_override is not None:
        base_cfg.epochs = int(epochs_override)
    base_cfg.rebuild_dataset = False

    search_space = optuna_cfg.get("search_space", {})
    if not search_space:
        raise ValueError(
            f"No search_space found in {optuna_config_path}. Add at least one parameter to tune."
        )

    print("Ensuring the prepared dataset is available before Optuna trials...", flush=True)
    ensure_dataset_prepared(base_cfg)

    pruner = build_pruner(optuna_cfg.get("pruner", {}))
    study_kwargs: Dict[str, Any] = {
        "study_name": study_name,
        "direction": "maximize",
        "load_if_exists": True,
    }
    if storage:
        study_kwargs["storage"] = storage
    if pruner is not None:
        study_kwargs["pruner"] = pruner

    study = optuna.create_study(**study_kwargs)
    objective = objective_factory(
        base_cfg=base_cfg,
        output_root=output_root,
        search_space=search_space,
        objective_metric=objective_metric,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_summary = {
        "study_name": study.study_name,
        "best_trial_number": study.best_trial.number,
        "best_value": study.best_value,
        "objective_metric": objective_metric,
        "best_params": study.best_trial.params,
        "best_user_attrs": study.best_trial.user_attrs,
        "n_trials": len(study.trials),
        "optuna_config_path": str(optuna_config_path),
        "train_config_path": str(config_path),
    }
    summary_path = output_root / "best_trial.json"
    summary_path.write_text(json.dumps(best_summary, indent=2, default=str) + "\n", encoding="utf-8")

    print("Optuna tuning finished.", flush=True)
    print(f"Best trial: {study.best_trial.number}", flush=True)
    print(f"Best objective ({objective_metric}): {study.best_value:.6f}", flush=True)
    print(f"Best params: {study.best_trial.params}", flush=True)
    print(f"Saved summary to: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
