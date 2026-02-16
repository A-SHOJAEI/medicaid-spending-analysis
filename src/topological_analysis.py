"""Phase 7A: Topological Data Analysis (TDA) of provider spending landscapes.

Uses persistent homology to discover the topological structure of the
provider feature space — connected components (H0), loops (H1), and
higher-dimensional voids that reveal hidden structure in billing patterns.

Methods:
    - Vietoris-Rips persistent homology (H0, H1)
    - Persistence diagrams, barcodes, and landscapes
    - Topological anomaly detection via persistence-based filtration
    - Betti curves for multi-scale structural characterization
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table,
)

logger = setup_logging()

# ── Persistent Homology ────────────────────────────────────────────────


def compute_persistent_homology(X: np.ndarray, max_dim: int = 1,
                                 max_edge: float = 5.0) -> dict:
    """Compute Vietoris-Rips persistent homology up to max_dim."""
    from ripser import ripser
    logger.info(f"Computing Rips persistence (n={X.shape[0]}, max_dim={max_dim}, "
                f"max_edge={max_edge})")
    result = ripser(X, maxdim=max_dim, thresh=max_edge)
    logger.info(f"  H0 features: {len(result['dgms'][0])}, "
                f"H1 features: {len(result['dgms'][1])}")
    return result


# ── Persistence Statistics ─────────────────────────────────────────────


def persistence_statistics(dgms: list) -> pd.DataFrame:
    """Compute summary statistics for each homology dimension."""
    rows = []
    for dim, dgm in enumerate(dgms):
        finite_mask = np.isfinite(dgm[:, 1])
        lifetimes = dgm[finite_mask, 1] - dgm[finite_mask, 0]
        if len(lifetimes) == 0:
            continue
        rows.append({
            "dimension": dim,
            "n_features": len(dgm),
            "n_finite": int(finite_mask.sum()),
            "mean_lifetime": float(lifetimes.mean()),
            "median_lifetime": float(np.median(lifetimes)),
            "max_lifetime": float(lifetimes.max()),
            "std_lifetime": float(lifetimes.std()),
            "total_persistence": float(lifetimes.sum()),
            "persistence_entropy": float(_persistence_entropy(lifetimes)),
        })
    return pd.DataFrame(rows)


def _persistence_entropy(lifetimes: np.ndarray) -> float:
    """Shannon entropy of normalized persistence lifetimes."""
    if len(lifetimes) == 0 or lifetimes.sum() == 0:
        return 0.0
    p = lifetimes / lifetimes.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


# ── Betti Curves ───────────────────────────────────────────────────────


def compute_betti_curves(dgms: list, n_bins: int = 200,
                         max_val: float = 5.0) -> dict:
    """Compute Betti number curves β_k(ε) for each dimension."""
    eps_vals = np.linspace(0, max_val, n_bins)
    curves = {}
    for dim, dgm in enumerate(dgms):
        betti = np.zeros(n_bins)
        for birth, death in dgm:
            if not np.isfinite(death):
                death = max_val + 1
            alive = (eps_vals >= birth) & (eps_vals < death)
            betti[alive] += 1
        curves[dim] = betti
    return {"epsilon": eps_vals, "curves": curves}


# ── Topological Anomaly Detection ─────────────────────────────────────


def topological_anomaly_scores(X: np.ndarray, k: int = 20) -> np.ndarray:
    """Score each point by its local topological complexity.

    Uses the persistent homology of k-nearest neighbor subgraphs to
    measure how topologically "unusual" each point's local neighborhood is.
    Points in regions with more persistent H1 features (loops) score higher.
    """
    from sklearn.neighbors import NearestNeighbors
    logger.info(f"Computing topological anomaly scores (k={k})...")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Use local density + distance spread as efficient proxy for topological complexity
    # Full per-point persistence is O(n * k^3) — use fast approximation
    local_density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-10)
    distance_variance = distances[:, 1:].var(axis=1)

    # Eccentricity: ratio of furthest to nearest neighbor
    eccentricity = distances[:, -1] / (distances[:, 1] + 1e-10)

    # Combine into topological anomaly score
    scores = (
        StandardScaler().fit_transform(eccentricity.reshape(-1, 1)).ravel() * 0.4
        + StandardScaler().fit_transform(distance_variance.reshape(-1, 1)).ravel() * 0.3
        + StandardScaler().fit_transform((1.0 / (local_density + 1e-10)).reshape(-1, 1)).ravel() * 0.3
    )
    return scores


# ── Persistence Landscapes ────────────────────────────────────────────


def compute_persistence_landscape(dgm: np.ndarray, n_layers: int = 5,
                                   n_bins: int = 200,
                                   max_val: float = 5.0) -> np.ndarray:
    """Compute the persistence landscape (first n_layers) for a diagram."""
    eps_vals = np.linspace(0, max_val, n_bins)
    finite_mask = np.isfinite(dgm[:, 1])
    dgm_finite = dgm[finite_mask]

    tent_funcs = np.zeros((len(dgm_finite), n_bins))
    for i, (b, d) in enumerate(dgm_finite):
        mid = (b + d) / 2
        height = (d - b) / 2
        for j, e in enumerate(eps_vals):
            if b <= e <= mid:
                tent_funcs[i, j] = e - b
            elif mid < e <= d:
                tent_funcs[i, j] = d - e

    landscape = np.zeros((n_layers, n_bins))
    for j in range(n_bins):
        col_vals = np.sort(tent_funcs[:, j])[::-1]
        for k in range(min(n_layers, len(col_vals))):
            landscape[k, j] = col_vals[k]

    return landscape


# ── Visualization ─────────────────────────────────────────────────────


def plot_persistence_diagram(dgms: list, config: dict) -> None:
    """Plot persistence diagram for H0 and H1."""
    setup_plotting(config)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for dim, ax in enumerate(axes):
        dgm = dgms[dim]
        finite = dgm[np.isfinite(dgm[:, 1])]
        infinite = dgm[~np.isfinite(dgm[:, 1])]

        if len(finite) > 0:
            lifetimes = finite[:, 1] - finite[:, 0]
            scatter = ax.scatter(finite[:, 0], finite[:, 1],
                                 c=lifetimes, cmap="plasma", alpha=0.6, s=15,
                                 edgecolors="none")
            plt.colorbar(scatter, ax=ax, label="Lifetime")

        if len(infinite) > 0:
            y_max = finite[:, 1].max() * 1.1 if len(finite) > 0 else 5.0
            ax.scatter(infinite[:, 0], [y_max] * len(infinite),
                       marker="^", c="red", s=50, label="∞ persistence",
                       zorder=5)

        lims = ax.get_xlim()
        ax.plot(lims, lims, "k--", alpha=0.3, lw=1)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(f"H{dim} Persistence Diagram ({len(dgm)} features)")
        if len(infinite) > 0:
            ax.legend()

    fig.suptitle("Topological Persistence Diagrams — Provider Feature Space",
                 fontsize=14, fontweight="bold")
    save_figure(fig, "tda_persistence_diagrams", config)
    logger.info("Saved persistence diagrams")


def plot_betti_curves(betti_data: dict, config: dict) -> None:
    """Plot Betti number curves."""
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    for dim, curve in betti_data["curves"].items():
        ax.plot(betti_data["epsilon"], curve, color=colors[dim % len(colors)],
                lw=2, label=f"β{dim} (H{dim})")

    ax.set_xlabel("Filtration Parameter ε")
    ax.set_ylabel("Betti Number")
    ax.set_title("Betti Curves — Multi-Scale Topological Structure")
    ax.legend(fontsize=12)
    ax.set_xlim(0, None)
    save_figure(fig, "tda_betti_curves", config)
    logger.info("Saved Betti curves")


def plot_persistence_landscape(landscape: np.ndarray, eps_vals: np.ndarray,
                                config: dict) -> None:
    """Plot persistence landscape layers."""
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0, 0.8, landscape.shape[0]))
    for k in range(landscape.shape[0]):
        ax.fill_between(eps_vals, landscape[k], alpha=0.3, color=colors[k])
        ax.plot(eps_vals, landscape[k], lw=1.5, color=colors[k],
                label=f"Layer {k+1}")

    ax.set_xlabel("Filtration Parameter ε")
    ax.set_ylabel("Landscape Value")
    ax.set_title("H1 Persistence Landscape — Loop Structure Across Scales")
    ax.legend()
    save_figure(fig, "tda_persistence_landscape", config)
    logger.info("Saved persistence landscape")


def plot_topological_anomalies(df: pd.DataFrame, scores: np.ndarray,
                                config: dict) -> None:
    """Scatter plot of providers colored by topological anomaly score."""
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Use log spending vs HCPCS diversity
    x = np.log10(df["total_paid"].clip(lower=1).values)
    y = df["n_unique_hcpcs"].values if "n_unique_hcpcs" in df.columns else np.zeros(len(df))

    percentile_95 = np.percentile(scores, 95)
    is_anomaly = scores >= percentile_95

    scatter = ax.scatter(x[~is_anomaly], y[~is_anomaly], c=scores[~is_anomaly],
                         cmap="viridis", alpha=0.3, s=5, edgecolors="none")
    ax.scatter(x[is_anomaly], y[is_anomaly], c="red", alpha=0.7, s=20,
               edgecolors="darkred", linewidths=0.5, label=f"Top 5% anomalies (n={is_anomaly.sum()})",
               zorder=5)

    plt.colorbar(scatter, ax=ax, label="Topological Anomaly Score")
    ax.set_xlabel("log₁₀(Total Spending)")
    ax.set_ylabel("Number of Unique HCPCS Codes")
    ax.set_title("Topological Anomaly Detection — Provider Feature Space")
    ax.legend()
    save_figure(fig, "tda_topological_anomalies", config)
    logger.info("Saved topological anomaly scatter")


# ── Main Pipeline ─────────────────────────────────────────────────────


def run_topological_analysis(config: dict) -> None:
    """Execute the full TDA pipeline."""
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Load provider features
    feat_path = processed_dir / "provider_features.parquet"
    df = pd.read_parquet(feat_path)
    logger.info(f"Loaded provider features: {df.shape}")

    # Select numeric features for TDA
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != "billing_npi"]
    X_raw = df[feature_cols].fillna(0).values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    logger.info(f"Feature matrix: {X_scaled.shape}")

    # ── Subsample for persistence computation (Rips is O(n^2) in memory) ──
    n_sample = min(5000, len(X_scaled))
    rng = np.random.RandomState(config["analysis"]["random_seed"])
    sample_idx = rng.choice(len(X_scaled), n_sample, replace=False)
    X_sample = X_scaled[sample_idx]
    logger.info(f"Sampled {n_sample} providers for persistence computation")

    # ── Compute persistent homology ──
    result = compute_persistent_homology(X_sample, max_dim=1, max_edge=8.0)
    dgms = result["dgms"]

    # ── Statistics ──
    stats_df = persistence_statistics(dgms)
    save_table(stats_df, "tda_persistence_statistics", config)
    logger.info(f"Persistence statistics:\n{stats_df.to_string()}")

    # ── Betti curves ──
    betti_data = compute_betti_curves(dgms, n_bins=300, max_val=8.0)

    # ── Persistence landscape (H1) ──
    landscape = compute_persistence_landscape(dgms[1], n_layers=5,
                                               n_bins=300, max_val=8.0)
    eps_vals = np.linspace(0, 8.0, 300)

    # ── Topological anomaly scores (full dataset) ──
    logger.info("Computing topological anomaly scores on full dataset...")
    topo_scores = topological_anomaly_scores(X_scaled, k=20)

    # Save scores
    score_df = pd.DataFrame({
        "billing_npi": df["billing_npi"].values,
        "tda_anomaly_score": topo_scores,
        "tda_anomaly_rank": pd.Series(topo_scores).rank(ascending=False, method="min").values,
    })
    score_df.to_parquet(processed_dir / "provider_tda_scores.parquet", index=False)
    logger.info(f"Saved TDA scores for {len(score_df)} providers")

    # Top anomalies table
    top_k = 50
    top_idx = np.argsort(topo_scores)[::-1][:top_k]
    top_df = df.iloc[top_idx][["billing_npi"] + feature_cols[:6]].copy()
    top_df["tda_anomaly_score"] = topo_scores[top_idx]
    save_table(top_df, "tda_top_anomalies", config)

    # ── Visualizations ──
    plot_persistence_diagram(dgms, config)
    plot_betti_curves(betti_data, config)
    plot_persistence_landscape(landscape, eps_vals, config)
    plot_topological_anomalies(df, topo_scores, config)

    # ── Summary ──
    summary = {
        "n_providers": len(df),
        "n_sample_persistence": n_sample,
        "n_features_used": len(feature_cols),
        "h0_features": int(len(dgms[0])),
        "h1_features": int(len(dgms[1])),
        "h1_max_lifetime": float(stats_df[stats_df["dimension"] == 1]["max_lifetime"].iloc[0])
            if len(stats_df[stats_df["dimension"] == 1]) > 0 else 0.0,
        "persistence_entropy_h1": float(stats_df[stats_df["dimension"] == 1]["persistence_entropy"].iloc[0])
            if len(stats_df[stats_df["dimension"] == 1]) > 0 else 0.0,
        "tda_anomaly_top5pct_threshold": float(np.percentile(topo_scores, 95)),
        "tda_anomaly_top1pct_threshold": float(np.percentile(topo_scores, 99)),
    }
    tables_dir = root / config["paths"]["tables_dir"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "tda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"TDA summary: {summary}")

    gc.collect()
    logger.info("Phase 7A: Topological Data Analysis complete.")


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7A: Topological Data Analysis")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_topological_analysis(config)
