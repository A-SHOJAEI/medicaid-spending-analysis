"""Phase 8C: Provider Phenotyping via Gaussian Mixture Models.

Discovers latent provider archetypes using soft clustering with GMM,
going beyond hard K-Means assignments to capture mixed membership.

Methods:
    - GMM with BIC/AIC model selection (2-15 components)
    - Soft cluster assignment (posterior probabilities)
    - Phenotype profiling with feature importance
    - HCPCS code enrichment analysis per phenotype
    - Radar/spider plots of phenotype profiles
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
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


# ── Model Selection ──────────────────────────────────────────────────


def select_gmm_components(X: np.ndarray, k_range: range,
                          random_state: int = 42) -> dict:
    """Evaluate BIC/AIC for different numbers of GMM components."""
    results = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                               random_state=random_state, n_init=3, max_iter=200)
        gmm.fit(X)
        results.append({
            "k": k,
            "bic": gmm.bic(X),
            "aic": gmm.aic(X),
            "log_likelihood": gmm.score(X) * len(X),
            "converged": gmm.converged_,
        })
        logger.info(f"  k={k}: BIC={results[-1]['bic']:.0f}, "
                    f"AIC={results[-1]['aic']:.0f}")

    results_df = pd.DataFrame(results)
    best_bic_k = int(results_df.loc[results_df["bic"].idxmin(), "k"])
    best_aic_k = int(results_df.loc[results_df["aic"].idxmin(), "k"])
    logger.info(f"Best k: BIC={best_bic_k}, AIC={best_aic_k}")

    return {
        "results_df": results_df,
        "best_bic_k": best_bic_k,
        "best_aic_k": best_aic_k,
    }


# ── Phenotype Assignment ─────────────────────────────────────────────


def fit_gmm_phenotypes(X: np.ndarray, n_components: int,
                       random_state: int = 42) -> tuple:
    """Fit GMM and return soft assignments."""
    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                           random_state=random_state, n_init=5, max_iter=300)
    gmm.fit(X)
    probs = gmm.predict_proba(X)  # (n_samples, n_components)
    labels = gmm.predict(X)

    # Compute entropy of soft assignment (how "mixed" each provider is)
    assignment_entropy = -np.sum(probs * np.log2(probs + 1e-10), axis=1)

    return gmm, labels, probs, assignment_entropy


# ── Phenotype Profiling ──────────────────────────────────────────────


def build_phenotype_profiles(df: pd.DataFrame, labels: np.ndarray,
                             feature_cols: list) -> pd.DataFrame:
    """Build statistical profiles for each phenotype."""
    df = df.copy()
    df["phenotype"] = labels
    profiles = df.groupby("phenotype")[feature_cols].agg(["mean", "median", "std"])
    profiles.columns = [f"{col}_{stat}" for col, stat in profiles.columns]

    # Add size
    sizes = df.groupby("phenotype").size().rename("n_providers")
    profiles = profiles.join(sizes)
    profiles["pct_of_total"] = profiles["n_providers"] / len(df) * 100

    return profiles


def compute_hcpcs_enrichment(config: dict, labels: np.ndarray,
                              billing_npis: np.ndarray) -> pd.DataFrame:
    """Compute top HCPCS codes enriched in each phenotype."""
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Map NPI → phenotype
    npi_to_pheno = dict(zip(billing_npis, labels))

    # Sample a recent year
    path = processed_dir / "medicaid_2023.parquet"
    if not path.exists():
        path = processed_dir / "medicaid_2024.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path, columns=["BILLING_PROVIDER_NPI_NUM", "HCPCS_CODE",
                                         "TOTAL_PAID"])
    df["phenotype"] = df["BILLING_PROVIDER_NPI_NUM"].map(npi_to_pheno)
    df = df.dropna(subset=["phenotype"])
    df["phenotype"] = df["phenotype"].astype(int)

    # Overall HCPCS distribution
    overall_share = df.groupby("HCPCS_CODE")["TOTAL_PAID"].sum()
    overall_share = overall_share / overall_share.sum()

    # Per-phenotype HCPCS distribution
    enrichment_rows = []
    for pheno in sorted(df["phenotype"].unique()):
        pheno_df = df[df["phenotype"] == pheno]
        pheno_share = pheno_df.groupby("HCPCS_CODE")["TOTAL_PAID"].sum()
        pheno_share = pheno_share / pheno_share.sum()

        # Enrichment = phenotype share / overall share
        enrichment = pheno_share / overall_share.reindex(pheno_share.index).fillna(1e-10)
        top_enriched = enrichment.nlargest(10)

        for code, enrich_val in top_enriched.items():
            enrichment_rows.append({
                "phenotype": pheno,
                "hcpcs_code": code,
                "enrichment_ratio": float(enrich_val),
                "phenotype_share": float(pheno_share.get(code, 0)),
                "overall_share": float(overall_share.get(code, 0)),
            })

    del df
    gc.collect()
    return pd.DataFrame(enrichment_rows)


# ── Visualization ────────────────────────────────────────────────────


def plot_bic_aic(selection_results: pd.DataFrame, best_k: int,
                 config: dict) -> None:
    """Plot BIC/AIC curves with optimal k."""
    setup_plotting(config)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(selection_results["k"], selection_results["bic"],
                 "o-", color="#2196F3", lw=2)
    axes[0].axvline(best_k, color="red", ls="--",
                    label=f"Optimal k={best_k}")
    axes[0].set_xlabel("Number of Components")
    axes[0].set_ylabel("BIC")
    axes[0].set_title("Bayesian Information Criterion")
    axes[0].legend()

    axes[1].plot(selection_results["k"], selection_results["aic"],
                 "o-", color="#4CAF50", lw=2)
    axes[1].axvline(best_k, color="red", ls="--",
                    label=f"Optimal k={best_k}")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("AIC")
    axes[1].set_title("Akaike Information Criterion")
    axes[1].legend()

    fig.suptitle("GMM Model Selection", fontsize=14, fontweight="bold")
    save_figure(fig, "phenotype_bic_aic", config)


def plot_phenotype_sizes(labels: np.ndarray, config: dict) -> None:
    """Plot phenotype size distribution."""
    setup_plotting(config)
    unique, counts = np.unique(labels, return_counts=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
    axes[0].bar(unique, counts, color=colors, edgecolor="white")
    axes[0].set_xlabel("Phenotype")
    axes[0].set_ylabel("Number of Providers")
    axes[0].set_title("Phenotype Size Distribution")
    for i, (u, c) in enumerate(zip(unique, counts)):
        axes[0].text(u, c + counts.max() * 0.01, f"{c:,}",
                     ha="center", va="bottom", fontsize=8)

    # Pie chart
    pct = counts / counts.sum() * 100
    axes[1].pie(counts, labels=[f"P{u} ({p:.1f}%)" for u, p in zip(unique, pct)],
                colors=colors, startangle=90)
    axes[1].set_title("Phenotype Proportions")

    fig.suptitle("Provider Phenotype Assignment", fontsize=14, fontweight="bold")
    save_figure(fig, "phenotype_sizes", config)


def plot_radar_profiles(profiles: pd.DataFrame, feature_cols: list,
                        config: dict) -> None:
    """Spider/radar chart comparing phenotype feature profiles."""
    setup_plotting(config)

    # Normalize mean features to [0, 1] range
    n_phenos = len(profiles)
    if n_phenos > 8:
        # Only plot top-8 largest phenotypes
        top_phenos = profiles.nlargest(8, "n_providers").index
    else:
        top_phenos = profiles.index

    # Select features for radar
    radar_features = [f"{c}_mean" for c in feature_cols[:8]
                      if f"{c}_mean" in profiles.columns]
    if len(radar_features) < 3:
        return

    values_matrix = profiles.loc[top_phenos, radar_features].values
    # Normalize columns
    col_min = values_matrix.min(axis=0)
    col_max = values_matrix.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    values_norm = (values_matrix - col_min) / col_range

    angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_phenos)))

    for i, (pheno_idx, color) in enumerate(zip(top_phenos, colors)):
        vals = values_norm[i].tolist()
        vals += vals[:1]
        ax.fill(angles, vals, alpha=0.1, color=color)
        ax.plot(angles, vals, "o-", color=color, lw=2, markersize=5,
                label=f"Phenotype {pheno_idx}")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.replace("_mean", "") for f in radar_features],
                       fontsize=8)
    ax.set_title("Phenotype Feature Profiles (Radar)", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    save_figure(fig, "phenotype_radar", config)


def plot_soft_assignment_entropy(entropy: np.ndarray, labels: np.ndarray,
                                 config: dict) -> None:
    """Plot distribution of soft assignment entropy."""
    setup_plotting(config)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].hist(entropy, bins=100, color="#2196F3", alpha=0.7, density=True)
    axes[0].set_xlabel("Assignment Entropy (bits)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Soft Assignment Uncertainty Distribution")
    axes[0].axvline(np.median(entropy), color="red", ls="--",
                    label=f"Median = {np.median(entropy):.3f}")
    axes[0].legend()

    # Per-phenotype entropy
    unique_labels = np.unique(labels)
    pheno_medians = [np.median(entropy[labels == l]) for l in unique_labels]
    axes[1].bar(unique_labels, pheno_medians, color="#4CAF50", alpha=0.8)
    axes[1].set_xlabel("Phenotype")
    axes[1].set_ylabel("Median Assignment Entropy")
    axes[1].set_title("Assignment Certainty by Phenotype")

    fig.suptitle("GMM Soft Clustering Quality", fontsize=14, fontweight="bold")
    save_figure(fig, "phenotype_soft_entropy", config)


def plot_hcpcs_enrichment(enrichment_df: pd.DataFrame, config: dict) -> None:
    """Heatmap of top HCPCS enrichment by phenotype."""
    if enrichment_df.empty:
        return
    setup_plotting(config)

    # Pivot to get top-5 per phenotype
    top_codes = enrichment_df.groupby("phenotype").head(5)
    pivot = top_codes.pivot_table(index="hcpcs_code", columns="phenotype",
                                   values="enrichment_ratio", fill_value=0)
    # Keep top-20 most enriched codes overall
    top_20 = enrichment_df.groupby("hcpcs_code")["enrichment_ratio"].max().nlargest(20).index
    pivot = pivot.loc[pivot.index.isin(top_20)]

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(12, len(pivot.columns) * 1.5),
                                     max(8, len(pivot) * 0.5)))
    im = ax.imshow(np.log1p(pivot.values), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"P{c}" for c in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("HCPCS Code")
    ax.set_title("HCPCS Code Enrichment by Provider Phenotype",
                 fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax, label="log(1 + enrichment ratio)")
    save_figure(fig, "phenotype_hcpcs_enrichment", config)


def plot_phenotype_spending(df: pd.DataFrame, labels: np.ndarray,
                            config: dict) -> None:
    """Box plot of spending distributions by phenotype."""
    setup_plotting(config)

    plot_df = pd.DataFrame({
        "phenotype": labels,
        "log_spending": np.log10(df["total_paid"].clip(lower=1).values),
    })

    fig, ax = plt.subplots(figsize=(14, 7))
    phenotypes = sorted(plot_df["phenotype"].unique())
    data = [plot_df[plot_df["phenotype"] == p]["log_spending"].values
            for p in phenotypes]
    bp = ax.boxplot(data, labels=[f"P{p}" for p in phenotypes],
                    patch_artist=True, showfliers=False)
    colors = plt.cm.Set3(np.linspace(0, 1, len(phenotypes)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax.set_xlabel("Phenotype")
    ax.set_ylabel("Log10(Total Spending)")
    ax.set_title("Spending Distribution by Provider Phenotype",
                 fontsize=14, fontweight="bold")
    save_figure(fig, "phenotype_spending_boxplot", config)


# ── Main Pipeline ────────────────────────────────────────────────────


def run_provider_phenotyping(config: dict) -> None:
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    tables_dir = root / config["paths"]["tables_dir"]

    cfg = config.get("phenotyping", {})
    k_min = cfg.get("k_min", 2)
    k_max = cfg.get("k_max", 15)
    seed = config.get("analysis", {}).get("random_seed", 42)

    # Load provider features
    logger.info("=== Loading provider features ===")
    feats = pd.read_parquet(processed_dir / "provider_features.parquet")
    feature_cols = ["total_paid", "total_claims", "total_beneficiaries",
                    "paid_per_claim", "paid_per_beneficiary", "claims_per_beneficiary",
                    "n_years_active", "n_months_active", "n_unique_hcpcs",
                    "n_servicing_npis"]
    feature_cols = [c for c in feature_cols if c in feats.columns]

    X_raw = feats[feature_cols].fillna(0).values
    # Log-transform skewed features
    X_log = np.log1p(np.abs(X_raw)) * np.sign(X_raw)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_log)
    logger.info(f"Feature matrix: {X.shape}")

    # Model selection
    logger.info("=== GMM model selection ===")
    selection = select_gmm_components(X, range(k_min, k_max + 1), seed)
    save_table(selection["results_df"], "phenotype_model_selection", config)
    best_k = selection["best_bic_k"]

    # Fit final model
    logger.info(f"=== Fitting GMM with k={best_k} components ===")
    gmm, labels, probs, entropy = fit_gmm_phenotypes(X, best_k, seed)

    feats["phenotype"] = labels
    feats["phenotype_entropy"] = entropy
    for i in range(best_k):
        feats[f"phenotype_{i}_prob"] = probs[:, i]

    feats.to_parquet(processed_dir / "provider_phenotypes.parquet", index=False)

    # Profiles
    logger.info("=== Building phenotype profiles ===")
    profiles = build_phenotype_profiles(feats, labels, feature_cols)
    save_table(profiles, "phenotype_profiles", config)

    # HCPCS enrichment
    logger.info("=== Computing HCPCS enrichment ===")
    enrichment_df = compute_hcpcs_enrichment(
        config, labels, feats["billing_npi"].values
    )
    if not enrichment_df.empty:
        save_table(enrichment_df, "phenotype_hcpcs_enrichment", config)

    # Visualizations
    logger.info("=== Generating plots ===")
    plot_bic_aic(selection["results_df"], best_k, config)
    plot_phenotype_sizes(labels, config)
    plot_radar_profiles(profiles, feature_cols, config)
    plot_soft_assignment_entropy(entropy, labels, config)
    plot_hcpcs_enrichment(enrichment_df, config)
    plot_phenotype_spending(feats, labels, config)

    # Summary
    summary = {
        "n_providers": len(feats),
        "n_phenotypes": int(best_k),
        "best_bic": float(selection["results_df"].loc[
            selection["results_df"]["k"] == best_k, "bic"].values[0]),
        "phenotype_sizes": {
            int(k): int(v)
            for k, v in zip(*np.unique(labels, return_counts=True))
        },
        "mean_assignment_entropy": float(entropy.mean()),
        "high_uncertainty_pct": float((entropy > 1.0).mean() * 100),
        "features_used": feature_cols,
    }

    with open(tables_dir / "phenotype_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Phenotyping summary: {best_k} phenotypes, "
                f"mean entropy={entropy.mean():.3f}")
    gc.collect()
    logger.info("Phase 8C: Provider Phenotyping complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8C: Provider Phenotyping")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_provider_phenotyping(config)
