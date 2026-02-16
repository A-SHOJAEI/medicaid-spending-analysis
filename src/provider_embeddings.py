"""Provider embedding learning via sparse matrix factorization.

Builds a full provider-HCPCS co-occurrence matrix from ALL year-partitioned
parquet files, then computes dense embeddings via Truncated SVD and NMF.
Also computes cosine k-NN anomaly scores in embedding space.
"""

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def build_cooccurrence_matrix(
    config: Optional[dict] = None,
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Build sparse provider-HCPCS co-occurrence matrix from all parquet files."""
    if config is None:
        config = load_config()
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Index mappings from existing feature files
    providers_df = pd.read_parquet(
        processed_dir / "provider_features.parquet", columns=["billing_npi"]
    )
    npi_index = providers_df["billing_npi"].values
    npi_to_idx = {npi: i for i, npi in enumerate(npi_index)}
    del providers_df

    hcpcs_df = pd.read_parquet(
        processed_dir / "hcpcs_features.parquet", columns=["hcpcs_code"]
    )
    hcpcs_index = hcpcs_df["hcpcs_code"].values
    hcpcs_to_idx = {code: i for i, code in enumerate(hcpcs_index)}
    del hcpcs_df

    # Accumulate COO triplets from all year files
    rows_list, cols_list, data_list = [], [], []

    for pf in sorted(processed_dir.glob("medicaid_*.parquet")):
        logger.info(f"Building co-occurrence from {pf.name}...")
        df = pd.read_parquet(pf, columns=[
            "BILLING_PROVIDER_NPI_NUM", "HCPCS_CODE", "TOTAL_PAID"
        ])
        # Aggregate per (provider, hcpcs) within this year file
        agg = (
            df.groupby(["BILLING_PROVIDER_NPI_NUM", "HCPCS_CODE"], observed=True)
            ["TOTAL_PAID"].sum().reset_index()
        )
        del df
        gc.collect()

        # Map to integer indices
        valid_mask = (
            agg["BILLING_PROVIDER_NPI_NUM"].isin(npi_to_idx)
            & agg["HCPCS_CODE"].isin(hcpcs_to_idx)
        )
        agg = agg[valid_mask]

        row_idx = agg["BILLING_PROVIDER_NPI_NUM"].map(npi_to_idx).values.astype(np.int32)
        col_idx = agg["HCPCS_CODE"].map(hcpcs_to_idx).values.astype(np.int32)
        vals = agg["TOTAL_PAID"].values.astype(np.float64)

        rows_list.append(row_idx)
        cols_list.append(col_idx)
        data_list.append(vals)
        del agg, row_idx, col_idx, vals
        gc.collect()

    # Construct sparse matrix (COO -> CSR, duplicates auto-summed)
    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_data = np.concatenate(data_list)
    del rows_list, cols_list, data_list
    gc.collect()

    matrix = sparse.coo_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(len(npi_index), len(hcpcs_index)),
    ).tocsr()
    del all_rows, all_cols, all_data
    gc.collect()

    logger.info(
        f"Co-occurrence matrix: {matrix.shape}, "
        f"nnz={matrix.nnz:,} "
        f"({matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100:.4f}% dense), "
        f"memory={matrix.data.nbytes / 1e6:.1f} MB"
    )

    return matrix, npi_index, hcpcs_index


def compute_svd_embeddings(
    matrix: sparse.csr_matrix,
    n_components: int = 64,
    random_state: int = 42,
) -> Tuple[np.ndarray, TruncatedSVD]:
    """Compute Truncated SVD embeddings from the co-occurrence matrix."""
    # Log1p transform to reduce extreme dynamic range
    matrix_log = matrix.copy()
    matrix_log.data = np.log1p(np.abs(matrix_log.data)) * np.sign(matrix_log.data)

    svd = TruncatedSVD(
        n_components=n_components,
        algorithm="randomized",
        n_iter=10,
        random_state=random_state,
    )
    embeddings = svd.fit_transform(matrix_log)

    total_var = svd.explained_variance_ratio_.sum()
    logger.info(
        f"SVD embeddings: {embeddings.shape}, "
        f"explained variance: {total_var:.4f} ({total_var*100:.1f}%)"
    )

    return embeddings, svd


def compute_nmf_embeddings(
    matrix: sparse.csr_matrix,
    n_components: int = 64,
    random_state: int = 42,
) -> Tuple[np.ndarray, NMF]:
    """Compute NMF embeddings from the co-occurrence matrix."""
    # NMF requires non-negative input
    matrix_nn = matrix.copy()
    matrix_nn.data = np.clip(matrix_nn.data, 0, None)
    matrix_nn.data = np.log1p(matrix_nn.data)

    nmf = NMF(
        n_components=n_components,
        init="nndsvda",
        solver="mu",
        max_iter=200,
        random_state=random_state,
        l1_ratio=0.5,
        alpha_W=0.01,
        alpha_H=0.01,
        verbose=0,
    )
    embeddings = nmf.fit_transform(matrix_nn)

    logger.info(
        f"NMF embeddings: {embeddings.shape}, "
        f"reconstruction error: {nmf.reconstruction_err_:.4f}"
    )

    return embeddings, nmf


def embedding_nearest_neighbor_anomaly(
    embeddings: np.ndarray,
    k: int = 10,
    batch_size: int = 5000,
) -> np.ndarray:
    """Compute anomaly scores based on cosine distance to k nearest neighbors.

    Uses sklearn NearestNeighbors with ball_tree for memory efficiency.
    """
    from sklearn.neighbors import NearestNeighbors

    logger.info(f"  Fitting NearestNeighbors (k={k}, metric=cosine, n={embeddings.shape[0]:,})...")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(embeddings.astype(np.float32))

    logger.info("  Querying nearest neighbors...")
    distances, _ = nn.kneighbors(embeddings.astype(np.float32))
    # Column 0 is self (distance ~0), skip it
    anomaly_scores = distances[:, 1:].mean(axis=1)

    logger.info(f"  k-NN anomaly scoring complete: "
                f"mean={anomaly_scores.mean():.4f}, "
                f"p99={np.percentile(anomaly_scores, 99):.4f}")

    return anomaly_scores


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_embedding_analysis(
    svd_embeddings: np.ndarray,
    nmf_embeddings: np.ndarray,
    npi_index: np.ndarray,
    svd_anomaly: np.ndarray,
    svd_model: TruncatedSVD,
    config: dict,
) -> None:
    """Generate embedding visualizations."""
    setup_plotting(config)

    # 1. Scree plot - SVD explained variance
    fig, ax = plt.subplots(figsize=(12, 6))
    cumvar = np.cumsum(svd_model.explained_variance_ratio_)
    ax.bar(range(1, len(svd_model.explained_variance_ratio_) + 1),
           svd_model.explained_variance_ratio_, alpha=0.6, label="Individual")
    ax.plot(range(1, len(cumvar) + 1), cumvar, "r-o", markersize=3, label="Cumulative")
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(f"SVD Explained Variance (total: {cumvar[-1]:.1%})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_figure(fig, "embedding_scree_plot", config)
    plt.close(fig)

    # 2. SVD first 2 components scatter, colored by anomaly score
    fig, ax = plt.subplots(figsize=(12, 10))
    # Subsample for plotting speed
    n = len(svd_embeddings)
    idx = np.random.RandomState(42).choice(n, min(50000, n), replace=False)
    sc = ax.scatter(
        svd_embeddings[idx, 0], svd_embeddings[idx, 1],
        c=svd_anomaly[idx], cmap="YlOrRd", s=1, alpha=0.3,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="k-NN Anomaly Score")
    ax.set_xlabel("SVD Component 1")
    ax.set_ylabel("SVD Component 2")
    ax.set_title("Provider Embeddings (SVD) — Colored by Anomaly Score")
    save_figure(fig, "embedding_svd_scatter", config)
    plt.close(fig)

    # 3. NMF component activation heatmap for top 30 providers by spending
    providers = pd.read_parquet(
        get_project_root() / config["paths"]["processed_dir"] / "provider_features.parquet",
        columns=["billing_npi", "total_paid"],
    )
    top_idx = providers.nlargest(30, "total_paid").index.values
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(nmf_embeddings[top_idx, :30], aspect="auto", cmap="YlOrRd")
    ax.set_xlabel("NMF Component")
    ax.set_ylabel("Provider (ranked by spending)")
    ax.set_title("NMF Component Activations — Top 30 Providers")
    plt.colorbar(im, ax=ax, label="Activation")
    save_figure(fig, "embedding_nmf_activations", config)
    plt.close(fig)

    # 4. Anomaly score distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(svd_anomaly, bins=100, edgecolor="none", alpha=0.7)
    axes[0].set_xlabel("SVD k-NN Anomaly Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("SVD Embedding Anomaly Distribution")
    axes[0].set_yscale("log")

    # Percentile markers
    for pct in [90, 95, 99]:
        val = np.percentile(svd_anomaly, pct)
        axes[0].axvline(val, color="red", linestyle="--", alpha=0.7)
        axes[0].text(val, axes[0].get_ylim()[1] * 0.5, f"p{pct}", color="red", fontsize=9)

    axes[1].hist(svd_anomaly, bins=100, edgecolor="none", alpha=0.7, cumulative=True, density=True)
    axes[1].set_xlabel("SVD k-NN Anomaly Score")
    axes[1].set_ylabel("Cumulative Fraction")
    axes[1].set_title("Cumulative Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "embedding_anomaly_distribution", config)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_provider_embeddings(config: Optional[dict] = None) -> dict:
    """Run the full embedding pipeline."""
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    emb_cfg = config.get("embeddings", {})
    n_components = emb_cfg.get("n_components_svd", 64)
    n_components_nmf = emb_cfg.get("n_components_nmf", 64)
    knn_k = emb_cfg.get("knn_k", 10)
    seed = config["analysis"]["random_seed"]

    # Step 1: Build co-occurrence matrix
    logger.info("Building provider-HCPCS co-occurrence matrix...")
    matrix, npi_index, hcpcs_index = build_cooccurrence_matrix(config)

    # Save sparse matrix
    sparse.save_npz(processed_dir / "cooccurrence_matrix.npz", matrix)

    # Step 2: SVD embeddings
    logger.info(f"Computing Truncated SVD (d={n_components})...")
    svd_embeddings, svd_model = compute_svd_embeddings(
        matrix, n_components=n_components, random_state=seed
    )

    # Save SVD embeddings
    svd_df = pd.DataFrame(
        svd_embeddings,
        columns=[f"svd_{i}" for i in range(n_components)],
    )
    svd_df.insert(0, "billing_npi", npi_index)
    svd_df.to_parquet(processed_dir / "provider_embeddings_svd.parquet", index=False)
    logger.info(f"SVD embeddings saved: {svd_df.shape}")
    del svd_df

    # Step 3: NMF embeddings
    logger.info(f"Computing NMF (d={n_components_nmf})...")
    nmf_embeddings, nmf_model = compute_nmf_embeddings(
        matrix, n_components=n_components_nmf, random_state=seed
    )

    nmf_df = pd.DataFrame(
        nmf_embeddings,
        columns=[f"nmf_{i}" for i in range(n_components_nmf)],
    )
    nmf_df.insert(0, "billing_npi", npi_index)
    nmf_df.to_parquet(processed_dir / "provider_embeddings_nmf.parquet", index=False)
    logger.info(f"NMF embeddings saved: {nmf_df.shape}")
    del nmf_df

    # Step 4: k-NN anomaly scoring in SVD space
    logger.info(f"Computing k-NN anomaly scores (k={knn_k})...")
    svd_anomaly = embedding_nearest_neighbor_anomaly(
        svd_embeddings, k=knn_k,
        batch_size=emb_cfg.get("batch_size", 5000),
    )

    # Also compute NMF anomaly
    logger.info("Computing NMF k-NN anomaly scores...")
    nmf_anomaly = embedding_nearest_neighbor_anomaly(
        nmf_embeddings, k=knn_k,
        batch_size=emb_cfg.get("batch_size", 5000),
    )

    # Save anomaly scores
    anomaly_df = pd.DataFrame({
        "billing_npi": npi_index,
        "svd_knn_anomaly": svd_anomaly,
        "nmf_knn_anomaly": nmf_anomaly,
    })
    anomaly_df.to_parquet(processed_dir / "provider_embedding_anomaly.parquet", index=False)
    logger.info(f"Embedding anomaly scores saved: {anomaly_df.shape}")

    # Top anomalies table
    top_anomalies = anomaly_df.nlargest(100, "svd_knn_anomaly")
    save_table(top_anomalies, "embedding_top_anomalies", config)

    # Metadata
    metadata = {
        "matrix_shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
        "density_pct": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1]) * 100),
        "svd_n_components": n_components,
        "svd_explained_variance": float(svd_model.explained_variance_ratio_.sum()),
        "nmf_n_components": n_components_nmf,
        "nmf_reconstruction_error": float(nmf_model.reconstruction_err_),
        "knn_k": knn_k,
        "n_providers": int(len(npi_index)),
        "n_hcpcs": int(len(hcpcs_index)),
    }
    import json
    with open(root / config["paths"]["tables_dir"] / "embedding_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Step 5: Plots
    logger.info("Generating embedding plots...")
    plot_embedding_analysis(
        svd_embeddings, nmf_embeddings, npi_index, svd_anomaly, svd_model, config
    )

    logger.info("Provider embedding pipeline complete")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute provider embeddings")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_provider_embeddings(config)
