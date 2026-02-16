"""Unsupervised clustering and segmentation of providers and procedure codes."""

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
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


def find_optimal_k(X_scaled: np.ndarray, k_range: range, seed: int = 42) -> dict:
    """Find optimal number of clusters using multiple criteria.

    Args:
        X_scaled: Scaled feature matrix.
        k_range: Range of k values to try.
        seed: Random seed.

    Returns:
        Dictionary with scores for each k.
    """
    results = {"k": [], "inertia": [], "silhouette": [], "bic": []}

    for k in k_range:
        # K-Means
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        results["k"].append(k)
        results["inertia"].append(km.inertia_)

        if k > 1:
            sil = silhouette_score(X_scaled, labels, sample_size=min(10000, len(X_scaled)))
            results["silhouette"].append(sil)
        else:
            results["silhouette"].append(0)

        # GMM BIC
        gmm = GaussianMixture(n_components=k, random_state=seed, max_iter=200)
        gmm.fit(X_scaled)
        results["bic"].append(gmm.bic(X_scaled))

    return results


def cluster_providers(config: Optional[dict] = None) -> pd.DataFrame:
    """Cluster providers by their billing feature vectors.

    Args:
        config: Configuration dictionary.

    Returns:
        Provider DataFrame with cluster assignments.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    providers = pd.read_parquet(
        root / config["paths"]["processed_dir"] / "provider_features.parquet"
    )

    # Feature selection
    feature_cols = ["total_paid", "total_claims", "paid_per_claim",
                    "paid_per_beneficiary", "n_unique_hcpcs",
                    "n_servicing_npis", "n_years_active", "n_months_active"]
    feature_cols = [c for c in feature_cols if c in providers.columns]

    X = providers[feature_cols].fillna(0).copy()
    # Log transform highly skewed features
    for col in ["total_paid", "total_claims", "paid_per_claim", "paid_per_beneficiary"]:
        if col in X.columns:
            X[col] = np.log1p(np.abs(X[col]))

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Use subsample for silhouette/elbow (full set is too large)
    np.random.seed(config["analysis"]["random_seed"])
    n_sample = min(50000, len(X_scaled))
    idx = np.random.choice(len(X_scaled), n_sample, replace=False)
    X_sample = X_scaled[idx]

    # Find optimal k
    k_min, k_max = config["analysis"]["n_clusters_range"]
    k_range = range(k_min, min(k_max + 1, 12))
    logger.info(f"Finding optimal k in range {list(k_range)}...")
    opt_results = find_optimal_k(X_sample, k_range, config["analysis"]["random_seed"])

    # Plot elbow and silhouette
    setup_plotting(config)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(opt_results["k"], opt_results["inertia"], "bo-", linewidth=2)
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")

    axes[1].plot(opt_results["k"], opt_results["silhouette"], "go-", linewidth=2)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score")

    axes[2].plot(opt_results["k"], opt_results["bic"], "ro-", linewidth=2)
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("BIC")
    axes[2].set_title("GMM BIC")

    save_figure(fig, "clustering_optimal_k", config)

    # Choose optimal k (best silhouette)
    best_k = opt_results["k"][np.argmax(opt_results["silhouette"])]
    logger.info(f"Optimal k = {best_k} (silhouette = {max(opt_results['silhouette']):.3f})")

    # Fit final K-Means on full data
    km_final = KMeans(n_clusters=best_k, random_state=config["analysis"]["random_seed"],
                       n_init=10, max_iter=300)
    providers["cluster_kmeans"] = km_final.fit_predict(X_scaled)

    # GMM clustering
    gmm_final = GaussianMixture(n_components=best_k,
                                 random_state=config["analysis"]["random_seed"])
    providers["cluster_gmm"] = gmm_final.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=config["analysis"]["random_seed"])
    pca_coords = pca.fit_transform(X_scaled)
    providers["pca_1"] = pca_coords[:, 0]
    providers["pca_2"] = pca_coords[:, 1]

    # Plot PCA with clusters (subsample for visibility)
    sample = providers.sample(min(50000, len(providers)), random_state=42)
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(sample["pca_1"], sample["pca_2"],
                         c=sample["cluster_kmeans"], cmap="tab10",
                         s=2, alpha=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"Provider Clusters (k={best_k}, K-Means)")
    plt.colorbar(scatter, ax=ax, label="Cluster")
    save_figure(fig, "provider_clusters_pca", config)

    # Cluster profiles
    profiles = providers.groupby("cluster_kmeans")[feature_cols].agg(["mean", "median", "count"])
    profiles.columns = ["_".join(c) for c in profiles.columns]
    save_table(profiles, "cluster_profiles", config)

    # Print cluster summaries
    for k in range(best_k):
        cluster = providers[providers["cluster_kmeans"] == k]
        logger.info(f"Cluster {k}: n={len(cluster):,}, "
                    f"median_paid={format_currency(cluster['total_paid'].median())}, "
                    f"median_hcpcs={cluster['n_unique_hcpcs'].median():.0f}")

    # Save
    output_path = root / config["paths"]["processed_dir"] / "provider_clustered.parquet"
    providers.to_parquet(output_path, index=False)

    return providers


def cluster_hcpcs(config: Optional[dict] = None) -> pd.DataFrame:
    """Cluster HCPCS codes by utilization and cost patterns.

    Args:
        config: Configuration dictionary.

    Returns:
        HCPCS DataFrame with cluster assignments.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    hcpcs = pd.read_parquet(
        root / config["paths"]["processed_dir"] / "hcpcs_features.parquet"
    )

    feature_cols = ["total_paid", "total_claims", "avg_paid_per_claim",
                    "n_providers", "n_years"]
    X = hcpcs[feature_cols].fillna(0).copy()
    for col in ["total_paid", "total_claims", "avg_paid_per_claim", "n_providers"]:
        X[col] = np.log1p(X[col])

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means with k=5 for procedure categories
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    hcpcs["cluster"] = km.fit_predict(X_scaled)

    # Cluster profiles
    for k in range(5):
        cluster = hcpcs[hcpcs["cluster"] == k]
        logger.info(f"HCPCS Cluster {k}: n={len(cluster)}, "
                    f"avg cost/claim={format_currency(cluster['avg_paid_per_claim'].median())}, "
                    f"median providers={cluster['n_providers'].median():.0f}")

    save_table(
        hcpcs.groupby("cluster")[feature_cols].median(),
        "hcpcs_cluster_profiles",
        config,
    )

    output_path = root / config["paths"]["processed_dir"] / "hcpcs_clustered.parquet"
    hcpcs.to_parquet(output_path, index=False)

    return hcpcs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run clustering analysis")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--providers", action="store_true")
    parser.add_argument("--hcpcs", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.all or args.providers:
        cluster_providers(config)
    if args.all or args.hcpcs:
        cluster_hcpcs(config)
