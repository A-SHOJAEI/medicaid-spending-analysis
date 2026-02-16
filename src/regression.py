"""Predictive and explanatory modeling for provider spending."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


def train_spending_predictor(config: Optional[dict] = None) -> dict:
    """Train a model to predict provider total spending.

    Uses LightGBM with SHAP explanations.

    Args:
        config: Configuration dictionary.

    Returns:
        Dictionary with model performance metrics.
    """
    if config is None:
        config = load_config()

    import lightgbm as lgb

    root = get_project_root()
    providers = pd.read_parquet(
        root / config["paths"]["processed_dir"] / "provider_features.parquet"
    )

    # Feature and target
    feature_cols = ["total_claims", "total_beneficiaries",
                    "n_unique_hcpcs", "n_servicing_npis",
                    "n_years_active", "n_months_active",
                    "claims_per_beneficiary"]
    feature_cols = [c for c in feature_cols if c in providers.columns]

    target = "total_paid"
    providers = providers.dropna(subset=feature_cols + [target])
    providers = providers[providers[target] > 0]

    X = providers[feature_cols].copy()
    y = np.log1p(providers[target].values)  # Log-transform target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config["analysis"]["random_seed"]
    )

    # LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config["analysis"]["random_seed"],
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(0)],
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_actual = np.expm1(y_pred)
    y_test_actual = np.expm1(y_test)

    metrics = {
        "r2_log": r2_score(y_test, y_pred),
        "r2_actual": r2_score(y_test_actual, y_pred_actual),
        "mae_log": mean_absolute_error(y_test, y_pred),
        "mae_actual": mean_absolute_error(y_test_actual, y_pred_actual),
        "rmse_log": np.sqrt(mean_squared_error(y_test, y_pred)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    logger.info(f"LightGBM R^2 (log): {metrics['r2_log']:.4f}")
    logger.info(f"LightGBM R^2 (actual): {metrics['r2_actual']:.4f}")
    logger.info(f"LightGBM MAE (actual): {format_currency(metrics['mae_actual'])}")

    # Feature importance
    setup_plotting(config)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Built-in importance
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
    importance.plot(kind="barh", ax=axes[0], color="#1976D2", alpha=0.85)
    axes[0].set_title("LightGBM Feature Importance (Split)")
    axes[0].set_xlabel("Importance")

    # SHAP values
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        # Use subsample for SHAP
        X_sample = X_test.sample(min(5000, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(X_sample)

        shap_importance = pd.Series(
            np.abs(shap_values).mean(axis=0), index=feature_cols
        ).sort_values()
        shap_importance.plot(kind="barh", ax=axes[1], color="#E64A19", alpha=0.85)
        axes[1].set_title("SHAP Feature Importance (Mean |SHAP|)")
        axes[1].set_xlabel("Mean |SHAP value|")

        save_figure(fig, "feature_importance", config)

        # SHAP summary plot
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False, plot_size=(12, 8))
        save_figure(plt.gcf(), "shap_summary", config)

    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")
        save_figure(fig, "feature_importance", config)

    # Prediction vs actual scatter
    fig, ax = plt.subplots(figsize=(10, 10))
    sample_idx = np.random.choice(len(y_test_actual), min(10000, len(y_test_actual)), replace=False)
    ax.scatter(y_test_actual[sample_idx], y_pred_actual[sample_idx], s=2, alpha=0.2)
    max_val = max(y_test_actual[sample_idx].max(), y_pred_actual[sample_idx].max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1)
    ax.set_xlabel("Actual Total Spending ($)")
    ax.set_ylabel("Predicted Total Spending ($)")
    ax.set_title(f"Predicted vs Actual Provider Spending (R^2={metrics['r2_actual']:.3f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    save_figure(fig, "prediction_vs_actual", config)

    # Save model
    import joblib
    models_dir = root / config["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "lgbm_spending_predictor.joblib")

    save_table(pd.DataFrame([metrics]), "model_metrics", config)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run regression modeling")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train_spending_predictor(config)
