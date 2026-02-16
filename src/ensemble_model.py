"""Stacked gradient boosting ensemble with Bayesian hyperparameter optimization.

Trains LightGBM + XGBoost + CatBoost on provider spending prediction,
uses Optuna for hyperparameter search, stacks with Ridge meta-learner,
and computes conformal prediction intervals.
"""

import argparse
import gc
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_ensemble_features(config: dict) -> Tuple[pd.DataFrame, np.ndarray, list]:
    """Prepare feature matrix with base features + SVD embedding components."""
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    providers = pd.read_parquet(processed_dir / "provider_features.parquet")

    base_features = [
        "total_claims", "total_beneficiaries",
        "n_unique_hcpcs", "n_servicing_npis",
        "n_years_active", "n_months_active",
        "claims_per_beneficiary",
    ]
    base_features = [c for c in base_features if c in providers.columns]

    # Add SVD embedding components if available
    svd_path = processed_dir / "provider_embeddings_svd.parquet"
    if svd_path.exists():
        svd_df = pd.read_parquet(svd_path)
        # Use top 10 SVD components
        svd_cols = [f"svd_{i}" for i in range(min(10, sum(1 for c in svd_df.columns if c.startswith("svd_"))))]
        svd_cols = [c for c in svd_cols if c in svd_df.columns]
        providers = providers.merge(svd_df[["billing_npi"] + svd_cols], on="billing_npi", how="left")
        feature_cols = base_features + svd_cols
        logger.info(f"Added {len(svd_cols)} SVD embedding features")
    else:
        feature_cols = base_features
        logger.warning("SVD embeddings not found, using base features only")

    # Filter valid rows
    target = "total_paid"
    valid = providers.dropna(subset=feature_cols + [target])
    valid = valid[valid[target] > 0].copy()

    X = valid[feature_cols].fillna(0).values.astype(np.float32)
    y = np.log1p(valid[target].values)

    logger.info(f"Ensemble feature matrix: {X.shape}, features: {feature_cols}")
    return valid, y, feature_cols


# ---------------------------------------------------------------------------
# Optuna hyperparameter optimization
# ---------------------------------------------------------------------------

def optimize_lightgbm(X: np.ndarray, y: np.ndarray, n_trials: int, n_folds: int, seed: int) -> dict:
    """Bayesian optimization for LightGBM hyperparameters."""
    import optuna
    import lightgbm as lgb

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = []
        for train_idx, val_idx in kf.split(X):
            model = lgb.LGBMRegressor(
                **params, random_state=seed, n_jobs=-1, verbose=-1
            )
            model.fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50, verbose=False)],
            )
            pred = model.predict(X[val_idx])
            scores.append(r2_score(y[val_idx], pred))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"LightGBM best R2: {study.best_value:.4f} (trial {study.best_trial.number})")
    return study.best_params, study


def optimize_xgboost(X: np.ndarray, y: np.ndarray, n_trials: int, n_folds: int, seed: int) -> dict:
    """Bayesian optimization for XGBoost hyperparameters."""
    import optuna
    import xgboost as xgb

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        }

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = []
        for train_idx, val_idx in kf.split(X):
            model = xgb.XGBRegressor(
                **params, random_state=seed, n_jobs=-1, verbosity=0,
                tree_method="hist",
                early_stopping_rounds=50,
            )
            model.fit(
                X[train_idx], y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                verbose=False,
            )
            pred = model.predict(X[val_idx])
            scores.append(r2_score(y[val_idx], pred))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"XGBoost best R2: {study.best_value:.4f} (trial {study.best_trial.number})")
    return study.best_params, study


def optimize_catboost(X: np.ndarray, y: np.ndarray, n_trials: int, n_folds: int, seed: int) -> dict:
    """Bayesian optimization for CatBoost hyperparameters."""
    import optuna
    from catboost import CatBoostRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "border_count": trial.suggest_int("border_count", 32, 255),
        }

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = []
        for train_idx, val_idx in kf.split(X):
            model = CatBoostRegressor(
                **params, random_seed=seed, verbose=0,
                early_stopping_rounds=50,
            )
            model.fit(
                X[train_idx], y[train_idx],
                eval_set=(X[val_idx], y[val_idx]),
                verbose=0,
            )
            pred = model.predict(X[val_idx])
            scores.append(r2_score(y[val_idx], pred))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"CatBoost best R2: {study.best_value:.4f} (trial {study.best_trial.number})")
    return study.best_params, study


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------

def train_stacked_ensemble(
    X: np.ndarray, y: np.ndarray,
    lgbm_params: dict, xgb_params: dict, cb_params: dict,
    n_folds: int, seed: int,
) -> Tuple[np.ndarray, dict, list]:
    """Train 3 base models with OOF stacking + Ridge meta-learner."""
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor

    n = len(X)
    oof_lgbm = np.zeros(n)
    oof_xgb = np.zeros(n)
    oof_cb = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    models = {"lgbm": [], "xgb": [], "catboost": []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        logger.info(f"  Stacking fold {fold + 1}/{n_folds}...")

        # LightGBM
        m_lgb = lgb.LGBMRegressor(
            **lgbm_params, random_state=seed, n_jobs=-1, verbose=-1
        )
        m_lgb.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(50, verbose=False)],
        )
        oof_lgbm[val_idx] = m_lgb.predict(X[val_idx])
        models["lgbm"].append(m_lgb)

        # XGBoost
        m_xgb = xgb.XGBRegressor(
            **xgb_params, random_state=seed, n_jobs=-1, verbosity=0,
            tree_method="hist", early_stopping_rounds=50,
        )
        m_xgb.fit(
            X[train_idx], y[train_idx],
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )
        oof_xgb[val_idx] = m_xgb.predict(X[val_idx])
        models["xgb"].append(m_xgb)

        # CatBoost
        m_cb = CatBoostRegressor(
            **cb_params, random_seed=seed, verbose=0, early_stopping_rounds=50,
        )
        m_cb.fit(
            X[train_idx], y[train_idx],
            eval_set=(X[val_idx], y[val_idx]),
            verbose=0,
        )
        oof_cb[val_idx] = m_cb.predict(X[val_idx])
        models["catboost"].append(m_cb)

    # Stack OOF predictions
    oof_stack = np.column_stack([oof_lgbm, oof_xgb, oof_cb])

    # Ridge meta-learner
    meta = Ridge(alpha=1.0)
    meta.fit(oof_stack, y)
    oof_meta = meta.predict(oof_stack)

    # Metrics for each model
    metrics = {}
    for name, oof in [("lgbm", oof_lgbm), ("xgb", oof_xgb),
                       ("catboost", oof_cb), ("stacked", oof_meta)]:
        oof_actual = np.expm1(oof)
        y_actual = np.expm1(y)
        metrics[name] = {
            "r2_log": float(r2_score(y, oof)),
            "r2_actual": float(r2_score(y_actual, oof_actual)),
            "mae_log": float(mean_absolute_error(y, oof)),
            "mae_actual": float(mean_absolute_error(y_actual, oof_actual)),
            "rmse_log": float(np.sqrt(mean_squared_error(y, oof))),
        }
        logger.info(f"  {name}: R2_log={metrics[name]['r2_log']:.4f}, "
                    f"R2_actual={metrics[name]['r2_actual']:.4f}, "
                    f"MAE_actual={format_currency(metrics[name]['mae_actual'])}")

    return oof_meta, metrics, models, meta


# ---------------------------------------------------------------------------
# Conformal Prediction
# ---------------------------------------------------------------------------

def compute_conformal_intervals(
    X: np.ndarray, y: np.ndarray,
    models: dict, meta: Ridge,
    alpha: float = 0.1, seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Split conformal prediction intervals (1-alpha coverage)."""
    import lightgbm as lgb

    # Use a 50/50 split for calibration
    n = len(X)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    cal_idx = perm[:n // 2]
    test_idx = perm[n // 2:]

    # Get ensemble predictions on calibration set
    def get_ensemble_pred(indices):
        preds = []
        for name, model_list in models.items():
            fold_preds = np.column_stack([m.predict(X[indices]) for m in model_list])
            preds.append(fold_preds.mean(axis=1))
        stack = np.column_stack(preds)
        return meta.predict(stack)

    cal_preds = get_ensemble_pred(cal_idx)
    test_preds = get_ensemble_pred(test_idx)

    # Calibration residuals
    cal_residuals = np.abs(y[cal_idx] - cal_preds)
    q = np.quantile(cal_residuals, 1 - alpha)

    # Prediction intervals
    lower = test_preds - q
    upper = test_preds + q

    # Actual coverage
    coverage = np.mean((y[test_idx] >= lower) & (y[test_idx] <= upper))
    logger.info(f"Conformal prediction: target coverage={1-alpha:.0%}, "
                f"actual coverage={coverage:.1%}, q={q:.4f}")

    # Return full-dataset intervals
    full_preds = get_ensemble_pred(np.arange(n))
    full_lower = full_preds - q
    full_upper = full_preds + q

    return full_lower, full_upper, float(coverage)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_ensemble_results(
    y: np.ndarray, oof_preds: np.ndarray,
    lower: np.ndarray, upper: np.ndarray,
    metrics: dict, feature_cols: list,
    models: dict, config: dict,
    optuna_studies: dict = None,
) -> None:
    """Generate ensemble model visualizations."""
    setup_plotting(config)

    # 1. Predicted vs actual
    fig, ax = plt.subplots(figsize=(10, 10))
    y_actual = np.expm1(y)
    pred_actual = np.expm1(oof_preds)
    idx = np.random.RandomState(42).choice(len(y), min(50000, len(y)), replace=False)
    ax.scatter(y_actual[idx], pred_actual[idx], s=1, alpha=0.15, rasterized=True)
    max_val = max(y_actual[idx].max(), pred_actual[idx].max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect")
    ax.set_xlabel("Actual Total Spending ($)")
    ax.set_ylabel("Predicted Total Spending ($)")
    stacked_r2 = metrics.get("stacked", {}).get("r2_actual", 0)
    ax.set_title(f"Stacked Ensemble: Predicted vs Actual (R²={stacked_r2:.4f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    save_figure(fig, "ensemble_predicted_vs_actual", config)
    plt.close(fig)

    # 2. Model comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    model_names = list(metrics.keys())
    r2_vals = [metrics[m]["r2_log"] for m in model_names]
    mae_vals = [metrics[m]["mae_log"] for m in model_names]

    colors = ["#1976D2", "#388E3C", "#F57C00", "#D32F2F"]
    axes[0].bar(model_names, r2_vals, color=colors[:len(model_names)], edgecolor="white")
    axes[0].set_ylabel("R² (log scale)")
    axes[0].set_title("Model Comparison: R² Score")
    axes[0].set_ylim(min(r2_vals) * 0.95, max(r2_vals) * 1.02)
    for i, v in enumerate(r2_vals):
        axes[0].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

    axes[1].bar(model_names, mae_vals, color=colors[:len(model_names)], edgecolor="white")
    axes[1].set_ylabel("MAE (log scale)")
    axes[1].set_title("Model Comparison: MAE")
    for i, v in enumerate(mae_vals):
        axes[1].text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)

    plt.tight_layout()
    save_figure(fig, "ensemble_model_comparison", config)
    plt.close(fig)

    # 3. Prediction intervals
    fig, ax = plt.subplots(figsize=(14, 6))
    sort_idx = np.argsort(y)
    sample = sort_idx[np.linspace(0, len(sort_idx)-1, 1000, dtype=int)]
    ax.fill_between(range(len(sample)),
                    np.expm1(lower[sample]),
                    np.expm1(upper[sample]),
                    alpha=0.3, color="steelblue", label="90% Prediction Interval")
    ax.scatter(range(len(sample)), np.expm1(y[sample]),
               s=3, color="red", alpha=0.5, label="Actual", zorder=5)
    ax.set_xlabel("Provider (sorted by spending)")
    ax.set_ylabel("Total Spending ($)")
    ax.set_title("Conformal Prediction Intervals (90% Coverage)")
    ax.set_yscale("log")
    ax.legend()
    save_figure(fig, "ensemble_prediction_intervals", config)
    plt.close(fig)

    # 4. SHAP summary (LightGBM)
    try:
        import shap
        shap_sample_size = config.get("ensemble", {}).get("shap_sample_size", 5000)
        lgbm_model = models["lgbm"][0]  # Use first fold model
        explainer = shap.TreeExplainer(lgbm_model)
        sample_idx = np.random.RandomState(42).choice(
            len(y), min(shap_sample_size, len(y)), replace=False
        )
        from sklearn.model_selection import KFold  # just for X access

        # We need the feature matrix - reconstruct from providers
        root = get_project_root()
        processed_dir = root / config["paths"]["processed_dir"]
        providers = pd.read_parquet(processed_dir / "provider_features.parquet")
        svd_path = processed_dir / "provider_embeddings_svd.parquet"
        if svd_path.exists():
            svd_df = pd.read_parquet(svd_path)
            svd_cols = [f"svd_{i}" for i in range(min(10, sum(1 for c in svd_df.columns if c.startswith("svd_"))))]
            svd_cols = [c for c in svd_cols if c in svd_df.columns]
            providers = providers.merge(svd_df[["billing_npi"] + svd_cols], on="billing_npi", how="left")

        valid = providers.dropna(subset=feature_cols + ["total_paid"])
        valid = valid[valid["total_paid"] > 0]
        X_shap = valid[feature_cols].fillna(0).values[sample_idx].astype(np.float32)
        X_shap_df = pd.DataFrame(X_shap, columns=feature_cols)

        shap_values = explainer.shap_values(X_shap_df)
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_shap_df, show=False, plot_size=(12, 8))
        save_figure(plt.gcf(), "ensemble_shap_summary", config)
        plt.close("all")
    except Exception as e:
        logger.warning(f"SHAP analysis skipped: {e}")

    # 5. Optuna optimization history
    if optuna_studies:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, (name, study) in zip(axes, optuna_studies.items()):
            trials = study.trials
            values = [t.value for t in trials if t.value is not None]
            ax.plot(range(len(values)), values, "o-", markersize=3, alpha=0.6)
            ax.axhline(study.best_value, color="red", linestyle="--", alpha=0.7,
                       label=f"Best: {study.best_value:.4f}")
            ax.set_xlabel("Trial")
            ax.set_ylabel("CV R² Score")
            ax.set_title(f"{name} Optimization")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig, "ensemble_optuna_optimization", config)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_ensemble_model(config: Optional[dict] = None) -> dict:
    """Run the full stacked ensemble pipeline."""
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    ens_cfg = config.get("ensemble", {})
    n_folds = ens_cfg.get("n_folds", 5)
    n_trials = ens_cfg.get("n_optuna_trials", 50)
    seed = config["analysis"]["random_seed"]

    # Prepare features
    providers, y, feature_cols = prepare_ensemble_features(config)
    X = providers[feature_cols].fillna(0).values.astype(np.float32)

    # Optuna optimization for each model
    logger.info(f"Running Optuna optimization ({n_trials} trials per model)...")

    logger.info("Optimizing LightGBM...")
    lgbm_params, lgbm_study = optimize_lightgbm(X, y, n_trials, n_folds, seed)

    logger.info("Optimizing XGBoost...")
    xgb_params, xgb_study = optimize_xgboost(X, y, n_trials, n_folds, seed)

    logger.info("Optimizing CatBoost...")
    cb_params, cb_study = optimize_catboost(X, y, n_trials, n_folds, seed)

    # Save best params
    best_params = {
        "lgbm": lgbm_params,
        "xgb": xgb_params,
        "catboost": cb_params,
    }
    with open(root / config["paths"]["tables_dir"] / "ensemble_optuna_best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Train stacked ensemble
    logger.info("Training stacked ensemble with optimized parameters...")
    oof_preds, metrics, models, meta = train_stacked_ensemble(
        X, y, lgbm_params, xgb_params, cb_params, n_folds, seed
    )

    # Conformal prediction intervals
    logger.info("Computing conformal prediction intervals...")
    conformal_alpha = ens_cfg.get("conformal_alpha", 0.1)
    lower, upper, coverage = compute_conformal_intervals(
        X, y, models, meta, alpha=conformal_alpha, seed=seed
    )

    # Save predictions
    pred_df = providers[["billing_npi"]].copy()
    pred_df["actual_log"] = y
    pred_df["predicted_log"] = oof_preds
    pred_df["actual_paid"] = np.expm1(y)
    pred_df["predicted_paid"] = np.expm1(oof_preds)
    pred_df["residual"] = y - oof_preds
    pred_df["lower_90"] = np.expm1(lower)
    pred_df["upper_90"] = np.expm1(upper)
    pred_df.to_parquet(processed_dir / "provider_ensemble_predictions.parquet", index=False)
    logger.info(f"Saved ensemble predictions: {pred_df.shape}")

    # Save metrics
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = "model"
    save_table(metrics_df, "ensemble_model_metrics", config)

    # Feature importance table
    import lightgbm as lgb
    lgbm_imp = pd.Series(
        models["lgbm"][0].feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    imp_df = pd.DataFrame({"feature": lgbm_imp.index, "importance": lgbm_imp.values})
    save_table(imp_df, "ensemble_feature_importance", config)

    # Save models
    import joblib
    models_dir = root / config["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models["lgbm"][0], models_dir / "lgbm_optimized.joblib")
    joblib.dump(models["xgb"][0], models_dir / "xgb_optimized.joblib")
    models["catboost"][0].save_model(str(models_dir / "catboost_optimized.cbm"))
    joblib.dump(meta, models_dir / "ridge_metalearner.joblib")

    # Plots
    logger.info("Generating ensemble plots...")
    optuna_studies = {"LightGBM": lgbm_study, "XGBoost": xgb_study, "CatBoost": cb_study}
    plot_ensemble_results(
        y, oof_preds, lower, upper, metrics, feature_cols,
        models, config, optuna_studies
    )

    # Add conformal info to metrics
    metrics["conformal"] = {
        "target_coverage": 1 - conformal_alpha,
        "actual_coverage": coverage,
    }

    logger.info("Stacked ensemble pipeline complete")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stacked ensemble model")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_ensemble_model(config)
