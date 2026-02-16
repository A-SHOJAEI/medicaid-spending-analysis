"""Advanced clustering via UMAP dimensionality reduction and HDBSCAN.

Produces density-based clusters that handle noise explicitly,
find variable-density clusters, and don't require specifying k.
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency, format_number,
)

logger = setup_logging()


def compute_umap_projection(
    embeddings: np.ndarray,
    n_neighbors: int = 30,
    min_dist: float = 0.0,
    metric: str = "cosine",
    random_state: int = 42,
    landmark_size: int = 100_000,
) -> np.ndarray:
    """Project high-dimensional embeddings to 2D using UMAP.

    For large datasets (>landmark_size), uses a landmark approach:
    fit UMAP on a subsample, then transform remaining points.
    Also applies PCA pre-reduction to speed up NN graph construction.
    """
    import umap
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    n_total = embeddings.shape[0]
    embeddings = embeddings.astype(np.float32)

    # L2-normalize for cosine equivalence
    if metric == "cosine":
        logger.info("L2-normalizing embeddings (cosine→euclidean equivalence)")
        embeddings = normalize(embeddings, norm="l2")
        effective_metric = "euclidean"
    else:
        effective_metric = metric

    # PCA pre-reduction from 64 → 20 dims to speed up NN search
    pca_target = min(20, embeddings.shape[1])
    if embeddings.shape[1] > pca_target:
        logger.info(f"PCA pre-reduction: {embeddings.shape[1]} → {pca_target} dimensions")
        pca = PCA(n_components=pca_target, random_state=random_state)
        embeddings = pca.fit_transform(embeddings)
        logger.info(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    if n_total > landmark_size:
        # Landmark approach: fit on subsample, transform rest
        logger.info(f"UMAP landmark approach: fit on {landmark_size:,}, "
                    f"transform {n_total:,} total points")
        rng = np.random.RandomState(random_state)
        landmark_idx = rng.choice(n_total, landmark_size, replace=False)
        other_idx = np.setdiff1d(np.arange(n_total), landmark_idx)

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=effective_metric,
            n_components=2,
            low_memory=True,
            n_jobs=-1,
            verbose=True,
            transform_seed=random_state,
        )
        logger.info(f"Fitting UMAP on {landmark_size:,} landmarks...")
        landmark_proj = reducer.fit_transform(embeddings[landmark_idx])
        logger.info(f"Transforming remaining {len(other_idx):,} points...")
        other_proj = reducer.transform(embeddings[other_idx])

        # Reassemble in original order
        projection = np.empty((n_total, 2), dtype=np.float32)
        projection[landmark_idx] = landmark_proj
        projection[other_idx] = other_proj
    else:
        logger.info(f"UMAP: {n_total:,} points, d={embeddings.shape[1]}, "
                    f"n_neighbors={n_neighbors}, metric={effective_metric}")
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=effective_metric,
            n_components=2,
            low_memory=True,
            n_jobs=-1,
            verbose=True,
        )
        projection = reducer.fit_transform(embeddings)

    logger.info(f"UMAP projection complete: {projection.shape}")
    return projection


def run_hdbscan_clustering(
    projection: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 10,
    cluster_selection_method: str = "eom",
) -> Tuple[np.ndarray, np.ndarray, object]:
    """Cluster UMAP-projected data using HDBSCAN."""
    import hdbscan
    logger.info(f"HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=-1,
        prediction_data=True,
    )
    labels = clusterer.fit_predict(projection)
    probabilities = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise:,} noise points "
                f"({n_noise/len(labels)*100:.1f}%)")

    return labels, probabilities, clusterer


def generate_cluster_profiles(
    providers: pd.DataFrame,
    cluster_col: str,
    feature_cols: list,
) -> pd.DataFrame:
    """Generate statistical profiles for each cluster."""
    records = []
    for cluster_id in sorted(providers[cluster_col].unique()):
        mask = providers[cluster_col] == cluster_id
        subset = providers.loc[mask, feature_cols]
        record = {"cluster": cluster_id, "count": int(mask.sum())}
        for col in feature_cols:
            record[f"{col}_median"] = subset[col].median()
            record[f"{col}_mean"] = subset[col].mean()
            record[f"{col}_std"] = subset[col].std()
        records.append(record)

    profiles = pd.DataFrame(records)

    # Kruskal-Wallis test for each feature across clusters
    kw_results = {}
    non_noise = providers[providers[cluster_col] >= 0]
    for col in feature_cols:
        groups = [g[col].dropna().values for _, g in non_noise.groupby(cluster_col)]
        if len(groups) >= 2:
            stat, p = stats.kruskal(*groups)
            kw_results[col] = {"H_statistic": stat, "p_value": p}
    logger.info(f"Kruskal-Wallis tests: {len(kw_results)} features, "
                f"all significant: {all(r['p_value'] < 0.05 for r in kw_results.values())}")

    return profiles


def plot_advanced_clustering(
    projection: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    providers: pd.DataFrame,
    profiles: pd.DataFrame,
    config: dict,
) -> None:
    """Generate advanced clustering visualizations."""
    setup_plotting(config)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # 1. UMAP scatter colored by HDBSCAN cluster
    fig, ax = plt.subplots(figsize=(14, 12))
    noise_mask = labels == -1
    # Plot noise first (gray)
    if noise_mask.any():
        ax.scatter(projection[noise_mask, 0], projection[noise_mask, 1],
                   c="lightgray", s=0.5, alpha=0.1, label=f"Noise ({noise_mask.sum():,})",
                   rasterized=True)
    # Plot clusters
    unique_labels = sorted(set(labels) - {-1})
    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 2))
    for i, cl in enumerate(unique_labels):
        mask = labels == cl
        ax.scatter(projection[mask, 0], projection[mask, 1],
                   c=[cmap(i)], s=1, alpha=0.3, label=f"Cluster {cl} ({mask.sum():,})",
                   rasterized=True)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(f"UMAP + HDBSCAN: {n_clusters} Clusters (617K Providers)")
    if n_clusters <= 15:
        ax.legend(markerscale=10, fontsize=8, loc="upper right")
    save_figure(fig, "umap_hdbscan_clusters", config)
    plt.close(fig)

    # 2. UMAP colored by log spending
    fig, ax = plt.subplots(figsize=(14, 12))
    log_spending = np.log1p(providers["total_paid"].clip(lower=0).values)
    idx = np.random.RandomState(42).choice(len(projection), min(100000, len(projection)), replace=False)
    sc = ax.scatter(projection[idx, 0], projection[idx, 1],
                    c=log_spending[idx], cmap="magma", s=0.5, alpha=0.3, rasterized=True)
    plt.colorbar(sc, ax=ax, label="log(Total Spending)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP Embedding — Colored by Provider Spending")
    save_figure(fig, "umap_spending_heatmap", config)
    plt.close(fig)

    # 3. Cluster sizes
    non_noise_labels = labels[labels >= 0]
    if len(non_noise_labels) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        cluster_sizes = pd.Series(non_noise_labels).value_counts().sort_index()
        ax.bar(cluster_sizes.index, cluster_sizes.values, color="steelblue", edgecolor="white")
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Provider Count")
        ax.set_title(f"HDBSCAN Cluster Sizes ({n_clusters} clusters + {(labels==-1).sum():,} noise)")
        ax.set_yscale("log")
        for i, (cl, cnt) in enumerate(cluster_sizes.items()):
            ax.text(cl, cnt * 1.1, f"{cnt:,}", ha="center", fontsize=7)
        save_figure(fig, "hdbscan_cluster_sizes", config)
        plt.close(fig)

    # 4. Cluster probability distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(probabilities[labels >= 0], bins=50, edgecolor="none", alpha=0.7, color="steelblue")
    ax.set_xlabel("HDBSCAN Membership Probability")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Membership Confidence Distribution")
    ax.axvline(probabilities[labels >= 0].mean(), color="red", linestyle="--",
               label=f"Mean: {probabilities[labels >= 0].mean():.3f}")
    ax.legend()
    save_figure(fig, "hdbscan_persistence_diagram", config)
    plt.close(fig)

    # 5. Radar/spider chart of cluster profiles (top 8 clusters by size)
    feature_cols = ["total_paid_median", "total_claims_median", "paid_per_claim_median",
                    "n_unique_hcpcs_median", "n_servicing_npis_median", "n_months_active_median"]
    display_names = ["Spending", "Claims", "$/Claim", "HCPCS Codes", "Servicing NPIs", "Months Active"]

    non_noise_profiles = profiles[profiles["cluster"] >= 0].nlargest(8, "count")
    if len(non_noise_profiles) >= 2:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(feature_cols), endpoint=False).tolist()
        angles += angles[:1]

        cmap = plt.cm.get_cmap("tab10", len(non_noise_profiles))
        for idx, (_, row) in enumerate(non_noise_profiles.iterrows()):
            vals = []
            for fc in feature_cols:
                if fc in row.index:
                    vals.append(row[fc])
                else:
                    vals.append(0)
            # Normalize to 0-1 range across clusters for visualization
            vals_norm = np.array(vals)
            max_vals = non_noise_profiles[feature_cols].max().values
            max_vals = np.where(max_vals == 0, 1, max_vals)
            vals_norm = vals_norm / max_vals
            vals_norm = vals_norm.tolist() + vals_norm[:1].tolist()
            ax.plot(angles, vals_norm, "o-", color=cmap(idx), label=f"C{int(row['cluster'])} (n={int(row['count']):,})")
            ax.fill(angles, vals_norm, alpha=0.1, color=cmap(idx))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(display_names, fontsize=9)
        ax.set_title("Cluster Profiles (Normalized Medians)", fontsize=14, pad=20)
        ax.legend(bbox_to_anchor=(1.15, 1.0), fontsize=8)
        save_figure(fig, "cluster_radar_profiles", config)
        plt.close(fig)


def run_advanced_clustering(config: Optional[dict] = None) -> pd.DataFrame:
    """Run the full UMAP + HDBSCAN pipeline."""
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    ac_cfg = config.get("advanced_clustering", {})
    seed = config["analysis"]["random_seed"]

    # Load SVD embeddings
    emb_path = processed_dir / "provider_embeddings_svd.parquet"
    logger.info(f"Loading SVD embeddings from {emb_path}...")
    emb_df = pd.read_parquet(emb_path)
    npi_index = emb_df["billing_npi"].values
    svd_cols = [c for c in emb_df.columns if c.startswith("svd_")]
    embeddings = emb_df[svd_cols].values
    del emb_df
    gc.collect()

    # Step 1: UMAP
    logger.info("Computing UMAP projection on full dataset...")
    projection = compute_umap_projection(
        embeddings,
        n_neighbors=ac_cfg.get("umap_n_neighbors", 30),
        min_dist=ac_cfg.get("umap_min_dist", 0.0),
        metric=ac_cfg.get("umap_metric", "cosine"),
        random_state=seed,
    )

    # Step 2: HDBSCAN
    logger.info("Running HDBSCAN clustering...")
    labels, probabilities, clusterer = run_hdbscan_clustering(
        projection,
        min_cluster_size=ac_cfg.get("hdbscan_min_cluster_size", 100),
        min_samples=ac_cfg.get("hdbscan_min_samples", 10),
        cluster_selection_method=ac_cfg.get("cluster_selection_method", "eom"),
    )

    # Step 3: Load provider features and merge
    providers = pd.read_parquet(processed_dir / "provider_features.parquet")
    providers["umap_1"] = projection[:, 0]
    providers["umap_2"] = projection[:, 1]
    providers["hdbscan_cluster"] = labels
    providers["hdbscan_probability"] = probabilities

    # Step 4: Compare with K-Means
    try:
        clustered = pd.read_parquet(
            processed_dir / "provider_clustered.parquet",
            columns=["billing_npi", "cluster_kmeans"],
        )
        km_labels = clustered.set_index("billing_npi").loc[npi_index, "cluster_kmeans"].values
        # Only compare non-noise points
        valid = labels >= 0
        ari = adjusted_rand_score(km_labels[valid], labels[valid])
        nmi = normalized_mutual_info_score(km_labels[valid], labels[valid])
        logger.info(f"HDBSCAN vs K-Means: ARI={ari:.4f}, NMI={nmi:.4f}")
        comparison = {"ARI": ari, "NMI": nmi, "n_hdbscan_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
                      "n_noise": int((labels == -1).sum())}
    except Exception as e:
        logger.warning(f"Could not compare with K-Means: {e}")
        comparison = {}

    # Step 5: Cluster profiles
    feature_cols = ["total_paid", "total_claims", "paid_per_claim",
                    "paid_per_beneficiary", "n_unique_hcpcs",
                    "n_servicing_npis", "n_years_active", "n_months_active"]
    profiles = generate_cluster_profiles(providers, "hdbscan_cluster", feature_cols)
    save_table(profiles, "advanced_cluster_profiles", config)

    # Save comparison metrics
    comparison_df = pd.DataFrame([comparison])
    save_table(comparison_df, "cluster_comparison_metrics", config)

    # Step 6: Plots
    logger.info("Generating advanced clustering plots...")
    plot_advanced_clustering(projection, labels, probabilities, providers, profiles, config)

    # Step 7: Save
    output_path = processed_dir / "provider_advanced_clustered.parquet"
    providers.to_parquet(output_path, index=False)
    logger.info(f"Advanced clustering saved: {output_path} ({providers.shape})")

    return providers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UMAP + HDBSCAN clustering")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_advanced_clustering(config)
