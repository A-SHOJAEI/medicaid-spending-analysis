"""Unified provider risk scoring with calibrated probabilities.

Aggregates all anomaly signals into a single calibrated risk score
per provider using semi-supervised pseudo-labels and Platt scaling.
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, brier_score_loss,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


# ---------------------------------------------------------------------------
# Signal aggregation
# ---------------------------------------------------------------------------

def aggregate_anomaly_signals(config: dict) -> pd.DataFrame:
    """Load and merge all available anomaly signals into a single DataFrame."""
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Start with provider features + existing anomaly scores
    anomaly_path = processed_dir / "provider_anomaly_scores.parquet"
    if anomaly_path.exists():
        df = pd.read_parquet(anomaly_path)
        logger.info(f"Loaded anomaly scores: {df.shape}")
    else:
        df = pd.read_parquet(processed_dir / "provider_features.parquet")
        logger.info(f"Loaded provider features: {df.shape}")

    # Autoencoder reconstruction error
    ae_path = processed_dir / "provider_ae_anomaly.parquet"
    if ae_path.exists():
        ae_df = pd.read_parquet(ae_path)
        df = df.merge(ae_df, on="billing_npi", how="left")
        logger.info(f"  + autoencoder anomaly ({ae_df.shape[0]} rows)")

    # Embedding k-NN anomaly
    emb_path = processed_dir / "provider_embedding_anomaly.parquet"
    if emb_path.exists():
        emb_df = pd.read_parquet(emb_path)
        df = df.merge(emb_df, on="billing_npi", how="left")
        logger.info(f"  + embedding anomaly ({emb_df.shape[0]} rows)")

    # HDBSCAN cluster info (noise flag, low probability)
    hdb_path = processed_dir / "provider_advanced_clustered.parquet"
    if hdb_path.exists():
        hdb_df = pd.read_parquet(hdb_path, columns=[
            "billing_npi", "hdbscan_cluster", "hdbscan_probability"
        ])
        hdb_df["hdbscan_noise"] = (hdb_df["hdbscan_cluster"] == -1).astype(int)
        hdb_df["hdbscan_low_prob"] = (hdb_df["hdbscan_probability"] < 0.3).astype(int)
        df = df.merge(hdb_df[["billing_npi", "hdbscan_noise", "hdbscan_low_prob"]], on="billing_npi", how="left")
        logger.info(f"  + HDBSCAN cluster flags ({hdb_df.shape[0]} rows)")

    # Trajectory features
    traj_path = processed_dir / "provider_trajectory_features.parquet"
    if traj_path.exists():
        traj_df = pd.read_parquet(traj_path)
        traj_cols = [c for c in traj_df.columns if c != "billing_npi"]
        df = df.merge(traj_df, on="billing_npi", how="left")
        logger.info(f"  + trajectory features ({len(traj_cols)} cols)")

    # Ensemble anomaly ranks
    ens_path = processed_dir / "provider_ensemble_anomaly.parquet"
    if ens_path.exists():
        ens_df = pd.read_parquet(ens_path)
        # Keep only columns not already in df
        ens_cols = [c for c in ens_df.columns if c not in df.columns and c != "billing_npi"]
        if ens_cols:
            df = df.merge(ens_df[["billing_npi"] + ens_cols], on="billing_npi", how="left")
            logger.info(f"  + ensemble anomaly ranks ({len(ens_cols)} cols)")

    logger.info(f"Aggregated signals: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Pseudo-labeling
# ---------------------------------------------------------------------------

def create_pseudo_labels(
    df: pd.DataFrame,
    agreement_threshold: int = 3,
) -> pd.Series:
    """Create semi-supervised pseudo-labels based on multi-method agreement.

    High-confidence anomaly: flagged by >= agreement_threshold methods
    High-confidence normal: flagged by 0 methods AND bottom 50% scores
    """
    # Count how many methods flag each provider as anomalous
    flag_count = pd.Series(0, index=df.index, dtype=int)

    # Isolation Forest anomaly
    if "isolation_forest_label" in df.columns:
        flag_count += (df["isolation_forest_label"] == -1).astype(int)

    # LOF anomaly (scores below -1.5 typically)
    if "lof_score" in df.columns:
        flag_count += (df["lof_score"] < np.percentile(df["lof_score"].dropna(), 1)).astype(int)

    # Autoencoder top 1%
    if "ae_reconstruction_error" in df.columns:
        ae_vals = df["ae_reconstruction_error"].dropna()
        if len(ae_vals) > 0:
            flag_count += (df["ae_reconstruction_error"] >= np.percentile(ae_vals, 99)).astype(int)

    # Embedding anomaly top 1%
    if "svd_knn_anomaly" in df.columns:
        svd_vals = df["svd_knn_anomaly"].dropna()
        if len(svd_vals) > 0:
            flag_count += (df["svd_knn_anomaly"] >= np.percentile(svd_vals, 99)).astype(int)

    # Statistical outlier top 1%
    if "statistical_outlier_score" in df.columns:
        stat_vals = df["statistical_outlier_score"].dropna()
        if len(stat_vals) > 0:
            flag_count += (df["statistical_outlier_score"] >= np.percentile(stat_vals, 99)).astype(int)

    # Fraud flags >= 3
    if "fraud_flags_count" in df.columns:
        flag_count += (df["fraud_flags_count"] >= 3).astype(int)

    # HDBSCAN noise
    if "hdbscan_noise" in df.columns:
        flag_count += df["hdbscan_noise"].fillna(0).astype(int)

    # Create labels: 1 = anomaly, 0 = normal, NaN = uncertain
    labels = pd.Series(np.nan, index=df.index)

    # High confidence anomaly
    labels[flag_count >= agreement_threshold] = 1

    # High confidence normal: zero flags AND below median for all score columns
    zero_flags = flag_count == 0
    score_cols = [c for c in ["isolation_forest_score", "statistical_outlier_score",
                              "ae_reconstruction_error", "svd_knn_anomaly"]
                  if c in df.columns]
    if score_cols:
        # For each score, check if provider is in bottom 50%
        below_median = pd.Series(True, index=df.index)
        for col in score_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                # For isolation_forest_score, higher is more normal
                if col == "isolation_forest_score":
                    below_median &= df[col].fillna(0) >= vals.median()
                else:
                    below_median &= df[col].fillna(0) <= vals.median()
        labels[zero_flags & below_median] = 0

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    n_unk = labels.isna().sum()
    logger.info(f"Pseudo-labels: {n_pos:,} anomaly, {n_neg:,} normal, {n_unk:,} uncertain")

    return labels


# ---------------------------------------------------------------------------
# Risk model
# ---------------------------------------------------------------------------

def train_risk_model(
    df: pd.DataFrame, labels: pd.Series, config: dict,
) -> tuple:
    """Train calibrated LightGBM classifier on pseudo-labels."""
    import lightgbm as lgb

    risk_cfg = config.get("risk_scoring", {})

    # Select features for risk model
    feature_candidates = [
        "total_paid", "total_claims", "total_beneficiaries",
        "paid_per_claim", "paid_per_beneficiary", "claims_per_beneficiary",
        "n_unique_hcpcs", "n_servicing_npis",
        "n_years_active", "n_months_active",
        "neg_paid_ratio",
    ]
    # Add anomaly scores as features
    score_cols = [c for c in df.columns if any(
        c.endswith(s) for s in [
            "_score", "_error", "_anomaly", "_knn_anomaly",
            "fraud_flags_count", "hdbscan_noise", "hdbscan_low_prob",
        ]
    )]
    # Add trajectory features
    traj_cols = [c for c in df.columns if c.startswith(("trend_", "volatility", "max_drawdown",
                                                         "growth_", "spending_entropy",
                                                         "autocorrelation", "covid_"))]

    feature_cols = [c for c in feature_candidates + score_cols + traj_cols if c in df.columns]
    feature_cols = list(dict.fromkeys(feature_cols))  # deduplicate while preserving order

    # Filter to labeled rows
    labeled_mask = labels.notna()
    X_labeled = df.loc[labeled_mask, feature_cols].fillna(0).values.astype(np.float32)
    y_labeled = labels[labeled_mask].values.astype(int)

    # Log-transform skewed features
    log_cols = ["total_paid", "total_claims", "total_beneficiaries",
                "paid_per_claim", "paid_per_beneficiary"]
    log_indices = [i for i, c in enumerate(feature_cols) if c in log_cols]
    X_labeled_transformed = X_labeled.copy()
    for idx in log_indices:
        X_labeled_transformed[:, idx] = np.log1p(np.clip(X_labeled_transformed[:, idx], 0, None))

    logger.info(f"Training risk model: {X_labeled_transformed.shape}, "
                f"{len(feature_cols)} features, "
                f"pos_rate={y_labeled.mean():.4f}")

    # Train LightGBM with Platt scaling via CalibratedClassifierCV
    base_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float((y_labeled == 0).sum() / max((y_labeled == 1).sum(), 1)),
        random_state=config["analysis"]["random_seed"],
        n_jobs=-1,
        verbose=-1,
    )

    calibrated_model = CalibratedClassifierCV(
        base_model, cv=5, method="sigmoid"
    )
    calibrated_model.fit(X_labeled_transformed, y_labeled)

    # Evaluate
    probs = calibrated_model.predict_proba(X_labeled_transformed)[:, 1]
    auc = roc_auc_score(y_labeled, probs)
    ap = average_precision_score(y_labeled, probs)
    brier = brier_score_loss(y_labeled, probs)
    logger.info(f"Risk model: AUC={auc:.4f}, AP={ap:.4f}, Brier={brier:.4f}")

    # Score ALL providers
    X_all = df[feature_cols].fillna(0).values.astype(np.float32)
    X_all_transformed = X_all.copy()
    for idx in log_indices:
        X_all_transformed[:, idx] = np.log1p(np.clip(X_all_transformed[:, idx], 0, None))

    all_probs = calibrated_model.predict_proba(X_all_transformed)[:, 1]

    model_metrics = {
        "auc_roc": float(auc),
        "average_precision": float(ap),
        "brier_score": float(brier),
        "n_labeled": int(len(y_labeled)),
        "n_positive": int(y_labeled.sum()),
        "n_negative": int((y_labeled == 0).sum()),
        "n_features": len(feature_cols),
    }

    return calibrated_model, all_probs, feature_cols, model_metrics


# ---------------------------------------------------------------------------
# Risk tiers
# ---------------------------------------------------------------------------

def assign_risk_tiers(probs: np.ndarray, thresholds: list = None) -> np.ndarray:
    """Assign risk tiers 1-5 based on calibrated probabilities.

    Tier 1: Low risk (below threshold[0])
    Tier 2: Below average risk
    Tier 3: Average risk
    Tier 4: Elevated risk
    Tier 5: High risk (above threshold[3])
    """
    if thresholds is None:
        thresholds = [0.05, 0.15, 0.35, 0.65]

    tiers = np.ones(len(probs), dtype=int)
    tiers[probs >= thresholds[0]] = 2
    tiers[probs >= thresholds[1]] = 3
    tiers[probs >= thresholds[2]] = 4
    tiers[probs >= thresholds[3]] = 5

    for t in range(1, 6):
        logger.info(f"  Tier {t}: {(tiers == t).sum():,} providers "
                    f"({(tiers == t).mean()*100:.1f}%)")

    return tiers


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_risk_scoring(
    df: pd.DataFrame, probs: np.ndarray, tiers: np.ndarray,
    model_metrics: dict, config: dict,
) -> None:
    """Generate risk scoring visualizations."""
    setup_plotting(config)

    # 1. Risk tier distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tier_counts = pd.Series(tiers).value_counts().sort_index()
    colors = ["#4CAF50", "#8BC34A", "#FFC107", "#FF9800", "#F44336"]
    tier_labels = ["Tier 1\n(Low)", "Tier 2\n(Below Avg)", "Tier 3\n(Average)",
                   "Tier 4\n(Elevated)", "Tier 5\n(High)"]
    axes[0].bar(range(1, 6), [tier_counts.get(t, 0) for t in range(1, 6)],
                color=colors, edgecolor="white")
    axes[0].set_xticks(range(1, 6))
    axes[0].set_xticklabels(tier_labels)
    axes[0].set_ylabel("Number of Providers")
    axes[0].set_title("Risk Tier Distribution")
    for i in range(5):
        c = tier_counts.get(i + 1, 0)
        axes[0].text(i + 1, c + 100, f"{c:,}", ha="center", fontsize=9)

    # Spending by tier
    spending_by_tier = []
    for t in range(1, 6):
        mask = tiers == t
        if mask.sum() > 0:
            spending_by_tier.append(df.loc[mask, "total_paid"].median())
        else:
            spending_by_tier.append(0)
    axes[1].bar(range(1, 6), spending_by_tier, color=colors, edgecolor="white")
    axes[1].set_xticks(range(1, 6))
    axes[1].set_xticklabels(tier_labels)
    axes[1].set_ylabel("Median Total Spending ($)")
    axes[1].set_title("Median Spending by Risk Tier")
    axes[1].set_yscale("log")
    for i, v in enumerate(spending_by_tier):
        axes[1].text(i + 1, v * 1.2, format_currency(v), ha="center", fontsize=9)

    plt.tight_layout()
    save_figure(fig, "risk_tier_distribution", config)
    plt.close(fig)

    # 2. Probability distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(probs, bins=200, edgecolor="none", alpha=0.7, color="steelblue")
    ax.set_xlabel("Calibrated Risk Probability")
    ax.set_ylabel("Count")
    ax.set_title("Risk Probability Distribution")
    ax.set_yscale("log")
    risk_thresholds = config.get("risk_scoring", {}).get("risk_tiers", [0.05, 0.15, 0.35, 0.65])
    for t_val in risk_thresholds:
        ax.axvline(t_val, color="red", linestyle="--", alpha=0.7)
        ax.text(t_val, ax.get_ylim()[1] * 0.3, f"{t_val:.0%}", color="red",
                fontsize=9, rotation=90)
    save_figure(fig, "risk_probability_distribution", config)
    plt.close(fig)

    # 3. Spending vs claims by tier
    fig, ax = plt.subplots(figsize=(12, 10))
    for t in range(1, 6):
        mask = tiers == t
        if mask.sum() > 0:
            idx = np.where(mask)[0]
            sample = idx[np.random.RandomState(42).choice(len(idx), min(5000, len(idx)), replace=False)]
            ax.scatter(
                np.log1p(df.iloc[sample]["total_claims"].values),
                np.log1p(df.iloc[sample]["total_paid"].values),
                s=3, alpha=0.3, label=f"Tier {t} (n={mask.sum():,})",
                color=colors[t - 1], rasterized=True,
            )
    ax.set_xlabel("log(Total Claims)")
    ax.set_ylabel("log(Total Spending)")
    ax.set_title("Provider Risk Tiers: Spending vs Claims")
    ax.legend(markerscale=5)
    save_figure(fig, "risk_heatmap_spending_claims", config)
    plt.close(fig)

    # 4. Calibration curve
    fig, ax = plt.subplots(figsize=(8, 8))
    # Use pseudo-labels for calibration check
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    # Bin predicted probabilities and plot observed frequency
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # We can only assess calibration on pseudo-labeled samples
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positive (Pseudo-Label)")
    ax.set_title(f"Risk Model Calibration (AUC={model_metrics['auc_roc']:.3f})")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_figure(fig, "risk_calibration_curve", config)
    plt.close(fig)

    # 5. SHAP for risk model
    try:
        import shap
        import lightgbm as lgb

        # Extract a base estimator from the calibrated model
        base = None
        if hasattr(config, "_risk_model"):
            base = config._risk_model
        else:
            # Try to get first calibrated classifier's base
            cal_model = config.get("_cal_model", None)
            if cal_model and hasattr(cal_model, "calibrated_classifiers_"):
                base = cal_model.calibrated_classifiers_[0].estimator

        if base is not None:
            explainer = shap.TreeExplainer(base)
            sample_idx = np.random.RandomState(42).choice(
                len(df), min(5000, len(df)), replace=False
            )
            feature_cols = config.get("_risk_features", [])
            X_sample = df.iloc[sample_idx][feature_cols].fillna(0).values.astype(np.float32)
            shap_values = explainer.shap_values(X_sample)
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values, pd.DataFrame(X_sample, columns=feature_cols),
                              show=False, plot_size=(12, 8))
            save_figure(plt.gcf(), "risk_shap_summary", config)
            plt.close("all")
    except Exception as e:
        logger.warning(f"Risk SHAP analysis skipped: {e}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_risk_scoring(config: Optional[dict] = None) -> dict:
    """Run the full risk scoring pipeline."""
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    risk_cfg = config.get("risk_scoring", {})

    # Step 1: Aggregate all signals
    logger.info("Aggregating anomaly signals...")
    df = aggregate_anomaly_signals(config)

    # Step 2: Create pseudo-labels
    logger.info("Creating pseudo-labels...")
    agreement_threshold = risk_cfg.get("agreement_threshold", 3)
    labels = create_pseudo_labels(df, agreement_threshold)

    # Step 3: Train calibrated risk model
    logger.info("Training calibrated risk model...")
    model, probs, feature_cols, model_metrics = train_risk_model(df, labels, config)

    # Step 4: Assign risk tiers
    logger.info("Assigning risk tiers...")
    thresholds = risk_cfg.get("risk_tiers", [0.05, 0.15, 0.35, 0.65])
    tiers = assign_risk_tiers(probs, thresholds)

    # Step 5: Save results
    result_df = df[["billing_npi"]].copy()
    result_df["risk_probability"] = probs
    result_df["risk_tier"] = tiers

    # Add key anomaly signals for reference
    signal_cols = [c for c in [
        "isolation_forest_score", "lof_score", "statistical_outlier_score",
        "fraud_flags_count", "ae_reconstruction_error",
        "svd_knn_anomaly", "nmf_knn_anomaly",
        "hdbscan_noise", "hdbscan_low_prob",
        "total_paid", "total_claims", "total_beneficiaries",
    ] if c in df.columns]
    for col in signal_cols:
        result_df[col] = df[col].values

    result_df.to_parquet(processed_dir / "provider_risk_scores.parquet", index=False)
    logger.info(f"Risk scores saved: {result_df.shape}")

    # Save model
    import joblib
    models_dir = root / config["paths"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "risk_model_calibrated.joblib")

    # Tier summary table
    tier_summary = []
    for t in range(1, 6):
        mask = tiers == t
        if mask.sum() > 0:
            tier_data = df[mask]
            tier_summary.append({
                "tier": t,
                "n_providers": int(mask.sum()),
                "pct_of_total": float(mask.mean() * 100),
                "mean_risk_prob": float(probs[mask].mean()),
                "median_spending": float(tier_data["total_paid"].median()),
                "total_spending": float(tier_data["total_paid"].sum()),
                "mean_claims": float(tier_data["total_claims"].mean()),
                "mean_fraud_flags": float(tier_data["fraud_flags_count"].mean()) if "fraud_flags_count" in tier_data.columns else 0,
            })
    tier_summary_df = pd.DataFrame(tier_summary)
    save_table(tier_summary_df, "risk_tier_summary", config)

    # Top N risk profiles
    top_n = risk_cfg.get("top_n_profiles", 50)
    top_risk = result_df.nlargest(top_n, "risk_probability")
    save_table(top_risk, "risk_top50_profiles", config)

    # Model metrics
    save_table(pd.DataFrame([model_metrics]), "risk_model_metrics", config)

    # Plots
    logger.info("Generating risk scoring plots...")
    plot_risk_scoring(df, probs, tiers, model_metrics, config)

    logger.info("Risk scoring pipeline complete")
    return {
        "model_metrics": model_metrics,
        "tier_counts": {int(t): int((tiers == t).sum()) for t in range(1, 6)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run unified risk scoring")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_risk_scoring(config)
