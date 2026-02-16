"""Phase 8E: Cross-Method Anomaly Consensus.

Unifies anomaly signals from multiple detection methods (Phases 4A, 6C, 6H,
7A, 7B) into a single consensus framework using a LightGBM meta-learner.

Methods:
    - Score aggregation from 6 anomaly detection methods
    - LightGBM stacking meta-learner with semi-supervised pseudo-labels
    - Consensus classification: unanimous, majority, contested, normal
    - SHAP analysis of meta-learner to explain which methods drive alerts
    - Method agreement analysis and Venn-style overlaps
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


# ── Score Collection ─────────────────────────────────────────────────


def collect_anomaly_scores(config: dict) -> pd.DataFrame:
    """Collect and normalize anomaly scores from all detection methods."""
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Base: provider features
    base = pd.read_parquet(
        processed_dir / "provider_features.parquet",
        columns=["billing_npi", "total_paid"]
    )

    score_sources = {
        # Phase 4A: Statistical anomaly detection
        "provider_anomaly_scores.parquet": {
            "columns": ["billing_npi", "isolation_forest_score", "lof_score"],
            "rename": {},
        },
        # Phase 6C: Autoencoder
        "provider_ae_anomaly.parquet": {
            "columns": ["billing_npi", "ae_reconstruction_error"],
            "rename": {},
        },
        # Phase 6H: Risk scoring
        "provider_risk_scores.parquet": {
            "columns": ["billing_npi", "risk_probability", "fraud_flags_count"],
            "rename": {},
        },
        # Phase 7A: TDA
        "provider_tda_scores.parquet": {
            "columns": ["billing_npi", "tda_anomaly_score"],
            "rename": {},
        },
        # Phase 7B: VAE
        "provider_vae_scores.parquet": {
            "columns": ["billing_npi", "vae_anomaly_score"],
            "rename": {},
        },
        # Phase 6A: Embedding anomaly
        "provider_embedding_anomaly.parquet": {
            "columns": ["billing_npi", "svd_knn_anomaly"],
            "rename": {},
        },
    }

    merged = base.copy()
    available_methods = []

    for filename, spec in score_sources.items():
        path = processed_dir / filename
        if not path.exists():
            logger.warning(f"Missing {filename}, skipping")
            continue

        try:
            df = pd.read_parquet(path)
            cols_available = [c for c in spec["columns"] if c in df.columns]
            if len(cols_available) < 2:  # need at least billing_npi + 1 score
                continue
            df = df[cols_available]
            for old_name, new_name in spec.get("rename", {}).items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            merged = merged.merge(df, on="billing_npi", how="left")
            score_cols = [c for c in cols_available if c != "billing_npi"]
            available_methods.extend(score_cols)
            logger.info(f"Loaded {filename}: {score_cols}")
        except Exception as e:
            logger.warning(f"Error loading {filename}: {e}")

    logger.info(f"Collected {len(available_methods)} anomaly score columns "
                f"from {len(score_sources)} sources")
    return merged, available_methods


def normalize_scores(df: pd.DataFrame, score_cols: list) -> pd.DataFrame:
    """Normalize all scores to [0, 1] range using percentile ranking."""
    df = df.copy()
    for col in score_cols:
        if col not in df.columns:
            continue
        vals = df[col].fillna(0)
        df[f"{col}_norm"] = sp_stats.rankdata(vals) / len(vals)
    return df


# ── Consensus Classification ────────────────────────────────────────


def classify_consensus(df: pd.DataFrame, score_cols: list,
                       threshold: float = 0.95) -> pd.DataFrame:
    """Classify providers based on agreement across methods.

    Categories:
        - unanimous: flagged by ALL methods (all scores > threshold)
        - majority: flagged by >50% of methods
        - contested: flagged by 1-50% of methods
        - normal: not flagged by any method
    """
    df = df.copy()
    norm_cols = [f"{c}_norm" for c in score_cols if f"{c}_norm" in df.columns]

    # Count how many methods flag each provider
    flags = df[norm_cols] > threshold
    df["n_methods_flagging"] = flags.sum(axis=1)
    df["pct_methods_flagging"] = df["n_methods_flagging"] / len(norm_cols)

    conditions = [
        df["n_methods_flagging"] == len(norm_cols),
        df["pct_methods_flagging"] > 0.5,
        df["n_methods_flagging"] > 0,
    ]
    labels = ["unanimous", "majority", "contested"]
    df["consensus_category"] = np.select(conditions, labels, default="normal")

    # Mean and max of normalized scores
    df["consensus_mean_score"] = df[norm_cols].mean(axis=1)
    df["consensus_max_score"] = df[norm_cols].max(axis=1)

    logger.info(f"Consensus classification:\n"
                f"  unanimous: {(df['consensus_category'] == 'unanimous').sum()}\n"
                f"  majority:  {(df['consensus_category'] == 'majority').sum()}\n"
                f"  contested: {(df['consensus_category'] == 'contested').sum()}\n"
                f"  normal:    {(df['consensus_category'] == 'normal').sum()}")

    return df


# ── Meta-Learner ─────────────────────────────────────────────────────


def train_meta_learner(df: pd.DataFrame, score_cols: list,
                       config: dict) -> tuple:
    """Train a LightGBM meta-learner to combine anomaly scores."""
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("LightGBM not available, skipping meta-learner")
        return None, None

    norm_cols = [f"{c}_norm" for c in score_cols if f"{c}_norm" in df.columns]
    if len(norm_cols) < 3:
        logger.warning("Too few methods for meta-learner")
        return None, None

    X = df[norm_cols].fillna(0).values

    # Semi-supervised: use top-5% as positive, bottom-50% as negative
    mean_score = df["consensus_mean_score"].values
    y = np.zeros(len(df))
    y[mean_score > np.percentile(mean_score, 95)] = 1
    y[mean_score < np.percentile(mean_score, 50)] = 0
    # Exclude ambiguous middle range
    mask = (y == 1) | (mean_score < np.percentile(mean_score, 50))

    X_train = X[mask]
    y_train = y[mask]

    logger.info(f"Meta-learner: {mask.sum()} samples, "
                f"{y_train.sum():.0f} positive, {(1 - y_train).sum():.0f} negative")

    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=config.get("analysis", {}).get("random_seed", 42),
        verbose=-1,
    )

    # Cross-validated predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_probs = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")
    auc = roc_auc_score(y_train, cv_probs[:, 1])
    ap = average_precision_score(y_train, cv_probs[:, 1])
    logger.info(f"Meta-learner CV: AUC={auc:.4f}, AP={ap:.4f}")

    # Fit on all labeled data
    model.fit(X_train, y_train)

    # Predict on all providers
    all_probs = model.predict_proba(X)[:, 1]
    df["meta_score"] = all_probs

    # Feature importance
    importance = pd.DataFrame({
        "method": norm_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    meta_results = {
        "auc": float(auc),
        "average_precision": float(ap),
        "n_train": int(mask.sum()),
        "n_positive": int(y_train.sum()),
    }

    return model, importance, meta_results


# ── SHAP Analysis ────────────────────────────────────────────────────


def compute_shap_analysis(model, df: pd.DataFrame, score_cols: list,
                          config: dict) -> None:
    """SHAP analysis of meta-learner."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not available, skipping")
        return

    setup_plotting(config)
    norm_cols = [f"{c}_norm" for c in score_cols if f"{c}_norm" in df.columns]

    # Sample for SHAP
    n_sample = min(5000, len(df))
    sample_idx = np.random.RandomState(42).choice(len(df), n_sample, replace=False)
    X_sample = df.iloc[sample_idx][norm_cols].fillna(0).values

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # positive class

        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample,
                         feature_names=[c.replace("_norm", "") for c in norm_cols],
                         show=False)
        plt.title("SHAP Feature Importance for Anomaly Meta-Learner",
                  fontsize=14, fontweight="bold")
        save_figure(plt.gcf(), "consensus_shap_summary", config)
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}")


# ── Visualization ────────────────────────────────────────────────────


def plot_method_agreement(df: pd.DataFrame, score_cols: list,
                          config: dict) -> None:
    """Heatmap of pairwise method agreement."""
    setup_plotting(config)
    norm_cols = [f"{c}_norm" for c in score_cols if f"{c}_norm" in df.columns]

    # Compute pairwise correlation
    corr = df[norm_cols].corr()
    labels = [c.replace("_norm", "") for c in norm_cols]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdYlBu_r", vmin=-0.3, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    # Annotate
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(corr.values[i, j]) > 0.5 else "black")

    plt.colorbar(im, ax=ax, label="Spearman Correlation")
    ax.set_title("Anomaly Method Agreement (Score Correlation)",
                 fontsize=14, fontweight="bold")
    save_figure(fig, "consensus_method_agreement", config)


def plot_consensus_distribution(df: pd.DataFrame, config: dict) -> None:
    """Plot consensus category distribution and score distributions."""
    setup_plotting(config)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Category bar plot
    cats = df["consensus_category"].value_counts()
    colors = {"normal": "#4CAF50", "contested": "#FF9800",
              "majority": "#F44336", "unanimous": "#9C27B0"}
    cat_colors = [colors.get(c, "gray") for c in cats.index]
    axes[0, 0].bar(cats.index, cats.values, color=cat_colors, alpha=0.8)
    axes[0, 0].set_ylabel("Number of Providers")
    axes[0, 0].set_title("Consensus Classification Distribution")
    for i, (c, v) in enumerate(cats.items()):
        axes[0, 0].text(i, v + cats.max() * 0.01, f"{v:,}", ha="center")

    # Mean consensus score distribution
    axes[0, 1].hist(df["consensus_mean_score"], bins=100, color="#2196F3",
                    alpha=0.7, density=True)
    axes[0, 1].set_xlabel("Mean Consensus Score")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Consensus Score Distribution")
    axes[0, 1].axvline(0.95, color="red", ls="--", label="Flag threshold (0.95)")
    axes[0, 1].legend()

    # Number of methods flagging
    flag_counts = df["n_methods_flagging"].value_counts().sort_index()
    axes[1, 0].bar(flag_counts.index, flag_counts.values, color="#FF9800", alpha=0.8)
    axes[1, 0].set_xlabel("Number of Methods Flagging")
    axes[1, 0].set_ylabel("Number of Providers")
    axes[1, 0].set_title("Methods Flagging Distribution")

    # Score vs spending
    sample = df.sample(min(10000, len(df)), random_state=42)
    axes[1, 1].scatter(
        np.log10(sample["total_paid"].clip(lower=1)),
        sample["consensus_mean_score"],
        alpha=0.1, s=3,
        c=sample["consensus_category"].map(colors).fillna("gray"),
    )
    axes[1, 1].set_xlabel("Log10(Total Spending)")
    axes[1, 1].set_ylabel("Consensus Score")
    axes[1, 1].set_title("Consensus Score vs Spending")

    fig.suptitle("Cross-Method Anomaly Consensus", fontsize=14, fontweight="bold")
    save_figure(fig, "consensus_distribution", config)


def plot_method_overlap(df: pd.DataFrame, score_cols: list,
                        config: dict, threshold: float = 0.95) -> None:
    """UpSet-style overlap plot of methods flagging providers."""
    setup_plotting(config)
    norm_cols = [f"{c}_norm" for c in score_cols if f"{c}_norm" in df.columns]

    flags = (df[norm_cols] > threshold).astype(int)
    flags.columns = [c.replace("_norm", "") for c in norm_cols]

    # Compute per-method flag counts
    method_counts = flags.sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(method_counts)), method_counts.values, color="#2196F3", alpha=0.8)
    ax.set_yticks(range(len(method_counts)))
    ax.set_yticklabels(method_counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Flagged Providers (>95th percentile)")
    ax.set_title("Providers Flagged by Each Method",
                 fontsize=14, fontweight="bold")
    for i, v in enumerate(method_counts.values):
        ax.text(v + method_counts.max() * 0.01, i, f"{v:,}",
                va="center", fontsize=9)
    save_figure(fig, "consensus_method_overlap", config)


def plot_meta_importance(importance: pd.DataFrame, config: dict) -> None:
    """Plot meta-learner feature importance."""
    if importance is None or importance.empty:
        return
    setup_plotting(config)

    fig, ax = plt.subplots(figsize=(10, 6))
    imp = importance.sort_values("importance", ascending=True)
    ax.barh(range(len(imp)), imp["importance"].values, color="#4CAF50", alpha=0.8)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp["method"].str.replace("_norm", "").values)
    ax.set_xlabel("Feature Importance (split-based)")
    ax.set_title("Meta-Learner Method Importance",
                 fontsize=14, fontweight="bold")
    save_figure(fig, "consensus_meta_importance", config)


# ── Main Pipeline ────────────────────────────────────────────────────


def run_anomaly_consensus(config: dict) -> None:
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    tables_dir = root / config["paths"]["tables_dir"]

    cfg = config.get("consensus", {})
    threshold = cfg.get("threshold", 0.95)

    # Collect scores
    logger.info("=== Collecting anomaly scores ===")
    df, score_cols = collect_anomaly_scores(config)
    logger.info(f"Score matrix: {df.shape}, methods: {score_cols}")

    # Normalize
    df = normalize_scores(df, score_cols)

    # Consensus classification
    logger.info("=== Consensus classification ===")
    df = classify_consensus(df, score_cols, threshold=threshold)

    # Meta-learner
    logger.info("=== Training meta-learner ===")
    result = train_meta_learner(df, score_cols, config)
    if result[0] is not None:
        model, importance, meta_results = result
        save_table(importance, "consensus_meta_importance", config)
    else:
        model, importance, meta_results = None, None, {}

    # Save consensus scores
    output_cols = ["billing_npi", "total_paid", "n_methods_flagging",
                   "pct_methods_flagging", "consensus_category",
                   "consensus_mean_score", "consensus_max_score"]
    if "meta_score" in df.columns:
        output_cols.append("meta_score")
    # Add normalized scores
    for col in score_cols:
        norm_col = f"{col}_norm"
        if norm_col in df.columns:
            output_cols.append(norm_col)

    df[output_cols].to_parquet(
        processed_dir / "provider_consensus_scores.parquet", index=False
    )

    # Top flagged providers
    top_flagged = df.nlargest(100, "consensus_mean_score")[
        ["billing_npi", "total_paid", "consensus_category",
         "consensus_mean_score", "n_methods_flagging"]
    ]
    save_table(top_flagged, "consensus_top_flagged", config)

    # Visualizations
    logger.info("=== Generating consensus plots ===")
    plot_method_agreement(df, score_cols, config)
    plot_consensus_distribution(df, config)
    plot_method_overlap(df, score_cols, config, threshold)
    plot_meta_importance(importance, config)
    if model is not None:
        compute_shap_analysis(model, df, score_cols, config)

    # Summary
    summary = {
        "n_providers": len(df),
        "n_methods": len(score_cols),
        "methods": score_cols,
        "consensus_counts": df["consensus_category"].value_counts().to_dict(),
        "mean_methods_flagging": float(df["n_methods_flagging"].mean()),
        "unanimous_count": int((df["consensus_category"] == "unanimous").sum()),
        "majority_count": int((df["consensus_category"] == "majority").sum()),
        "meta_learner": meta_results,
        "threshold": threshold,
    }

    with open(tables_dir / "consensus_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Consensus: {summary['unanimous_count']} unanimous, "
                f"{summary['majority_count']} majority flagged")
    gc.collect()
    logger.info("Phase 8E: Anomaly Consensus complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8E: Cross-Method Anomaly Consensus")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_anomaly_consensus(config)
