"""Provider spending trajectory analysis and temporal pattern mining.

Builds monthly trajectories for ALL 617K providers, extracts temporal
features, classifies trajectory archetypes, and applies K-Shape clustering.
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

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency, format_number,
)

logger = setup_logging()


def build_monthly_trajectories(
    config: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Build monthly spending trajectories for ALL 617K providers."""
    if config is None:
        config = load_config()
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Load NPI index from provider features
    pf = pd.read_parquet(processed_dir / "provider_features.parquet", columns=["billing_npi"])
    npi_index = pf["billing_npi"].values
    npi_to_idx = {npi: i for i, npi in enumerate(npi_index)}
    n_providers = len(npi_index)
    del pf

    # Build month index (2018-01 to 2024-12 = 84 months)
    month_index = pd.date_range("2018-01-01", "2024-12-01", freq="MS")
    n_months = len(month_index)
    month_str_to_idx = {m.strftime("%Y-%m"): i for i, m in enumerate(month_index)}

    # Initialize trajectory matrix with NaN
    trajectories = np.full((n_providers, n_months), np.nan, dtype=np.float64)

    # Accumulate from year-partitioned parquets
    for pf in sorted(processed_dir.glob("medicaid_*.parquet")):
        logger.info(f"Building trajectories from {pf.name}...")
        df = pd.read_parquet(pf, columns=[
            "BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH", "TOTAL_PAID"
        ])
        # Aggregate per (provider, month)
        agg = (
            df.groupby(["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH"], observed=True)
            ["TOTAL_PAID"].sum().reset_index()
        )
        del df
        gc.collect()

        for _, row in agg.iterrows():
            npi = row["BILLING_PROVIDER_NPI_NUM"]
            month = row["CLAIM_FROM_MONTH"]
            if npi in npi_to_idx and month in month_str_to_idx:
                trajectories[npi_to_idx[npi], month_str_to_idx[month]] = row["TOTAL_PAID"]

        # Vectorized approach (much faster)
        valid = agg["BILLING_PROVIDER_NPI_NUM"].isin(npi_to_idx) & agg["CLAIM_FROM_MONTH"].isin(month_str_to_idx)
        agg_valid = agg[valid]
        row_idx = agg_valid["BILLING_PROVIDER_NPI_NUM"].map(npi_to_idx).values
        col_idx = agg_valid["CLAIM_FROM_MONTH"].map(month_str_to_idx).values
        # Overwrite with vectorized assignment
        trajectories[row_idx, col_idx] = agg_valid["TOTAL_PAID"].values
        del agg, agg_valid
        gc.collect()

    # Impute active gaps: between first and last non-NaN, fill NaN with 0
    logger.info("Imputing active-period gaps with 0...")
    for i in range(n_providers):
        row = trajectories[i]
        non_nan = np.where(~np.isnan(row))[0]
        if len(non_nan) >= 2:
            first, last = non_nan[0], non_nan[-1]
            gap_mask = np.isnan(row[first:last + 1])
            row[first:last + 1] = np.where(gap_mask, 0.0, row[first:last + 1])

    logger.info(f"Trajectory matrix: {trajectories.shape}, "
                f"NaN fraction: {np.isnan(trajectories).mean():.3f}")

    return trajectories, npi_index, month_index


def extract_trajectory_features(
    trajectories: np.ndarray,
    npi_index: np.ndarray,
    month_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Extract time-series features from each provider's trajectory."""
    n_providers = trajectories.shape[0]
    features = {
        "billing_npi": npi_index,
        "trend_slope": np.full(n_providers, np.nan),
        "trend_r2": np.full(n_providers, np.nan),
        "volatility": np.full(n_providers, np.nan),
        "max_drawdown": np.full(n_providers, np.nan),
        "growth_ratio": np.full(n_providers, np.nan),
        "months_active": np.zeros(n_providers, dtype=int),
        "first_month_idx": np.full(n_providers, -1, dtype=int),
        "last_month_idx": np.full(n_providers, -1, dtype=int),
        "gap_count": np.zeros(n_providers, dtype=int),
        "peak_month_idx": np.full(n_providers, -1, dtype=int),
        "spending_entropy": np.full(n_providers, np.nan),
        "autocorrelation_lag1": np.full(n_providers, np.nan),
        "mean_monthly_spending": np.full(n_providers, np.nan),
        "covid_growth_pct": np.full(n_providers, np.nan),
    }

    n_months = trajectories.shape[1]
    covid_start = 26  # March 2020 = month index 26

    for i in range(n_providers):
        row = trajectories[i]
        non_nan_idx = np.where(~np.isnan(row))[0]

        if len(non_nan_idx) == 0:
            continue

        active_values = row[non_nan_idx]
        features["months_active"][i] = len(non_nan_idx)
        features["first_month_idx"][i] = non_nan_idx[0]
        features["last_month_idx"][i] = non_nan_idx[-1]
        features["mean_monthly_spending"][i] = active_values.mean()

        # Peak
        features["peak_month_idx"][i] = non_nan_idx[np.argmax(active_values)]

        # Gap count (NaN within active period)
        if len(non_nan_idx) >= 2:
            span = non_nan_idx[-1] - non_nan_idx[0] + 1
            features["gap_count"][i] = span - len(non_nan_idx)

        if len(non_nan_idx) < 3:
            continue

        mean_val = active_values.mean()
        if mean_val > 0:
            features["volatility"][i] = active_values.std() / mean_val

        # Trend (OLS on active months)
        x = non_nan_idx.astype(float)
        if len(x) >= 3 and active_values.std() > 0:
            slope, intercept, r_value, _, _ = stats.linregress(x, active_values)
            features["trend_slope"][i] = slope / max(mean_val, 1)  # normalized
            features["trend_r2"][i] = r_value ** 2

        # Max drawdown
        cummax = np.maximum.accumulate(active_values)
        drawdowns = (cummax - active_values) / np.where(cummax > 0, cummax, 1)
        features["max_drawdown"][i] = drawdowns.max()

        # Growth ratio (last 12 vs first 12 months)
        if len(non_nan_idx) >= 24:
            first_12 = active_values[:12].mean()
            last_12 = active_values[-12:].mean()
            if first_12 > 0:
                features["growth_ratio"][i] = last_12 / first_12

        # Spending entropy
        pos_vals = active_values[active_values > 0]
        if len(pos_vals) > 1:
            probs = pos_vals / pos_vals.sum()
            features["spending_entropy"][i] = stats.entropy(probs)

        # Autocorrelation lag-1
        if len(active_values) >= 4:
            try:
                ac = np.corrcoef(active_values[:-1], active_values[1:])[0, 1]
                features["autocorrelation_lag1"][i] = ac
            except:
                pass

        # COVID growth
        pre_covid = row[:covid_start]
        post_covid = row[covid_start:covid_start + 24]
        pre_vals = pre_covid[~np.isnan(pre_covid)]
        post_vals = post_covid[~np.isnan(post_covid)]
        if len(pre_vals) >= 6 and len(post_vals) >= 6:
            pre_mean = pre_vals.mean()
            if pre_mean > 0:
                features["covid_growth_pct"][i] = (post_vals.mean() - pre_mean) / pre_mean * 100

        if i % 100000 == 0 and i > 0:
            logger.info(f"  Feature extraction: {i:,}/{n_providers:,}")

    return pd.DataFrame(features)


def classify_trajectory_archetypes(features: pd.DataFrame) -> pd.DataFrame:
    """Classify each provider into a trajectory archetype."""
    n = len(features)
    archetypes = np.full(n, "other", dtype=object)

    # Priority ordering (first match wins)
    conditions = [
        ("new_entrant", features["first_month_idx"] > 24),
        ("exiting", (features["last_month_idx"] >= 0) & (features["last_month_idx"] < 60)),
        ("spike_then_decline", (features["max_drawdown"] > 0.7) & (features["peak_month_idx"] < 56)),
        ("growing", (features["trend_slope"] > 0.03) & (features["growth_ratio"] > 1.5)),
        ("declining", (features["trend_slope"] < -0.03) & (features["growth_ratio"] < 0.5)),
        ("volatile", features["volatility"] > 1.0),
        ("stable", (features["trend_slope"].abs() < 0.01) & (features["volatility"] < 0.3)),
    ]

    assigned = np.zeros(n, dtype=bool)
    for name, mask in conditions:
        mask = mask.fillna(False).values & ~assigned
        archetypes[mask] = name
        assigned = assigned | mask

    features = features.copy()
    features["archetype"] = archetypes
    return features


def kshape_clustering(
    trajectories: np.ndarray,
    features: pd.DataFrame,
    n_clusters: int = 8,
    top_n: int = 50000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """K-Shape time series clustering on top providers by activity."""
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMeanVariance

    # Select top providers by months active
    active_counts = features["months_active"].values
    top_idx = np.argsort(active_counts)[-top_n:]

    # Extract and prepare trajectories
    traj_subset = trajectories[top_idx].copy()
    # Fill NaN with 0 for clustering
    traj_subset = np.nan_to_num(traj_subset, nan=0.0)

    # Z-normalize each time series
    scaler = TimeSeriesScalerMeanVariance()
    traj_3d = traj_subset.reshape(top_n, -1, 1)
    traj_scaled = scaler.fit_transform(traj_3d)
    # Replace any NaN from zero-variance series
    traj_scaled = np.nan_to_num(traj_scaled, nan=0.0)

    logger.info(f"K-Shape clustering: {top_n:,} providers, {n_clusters} clusters...")
    km = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="softdtw",
        metric_params={"gamma": 0.1},
        max_iter=15,
        n_init=1,
        random_state=random_state,
        verbose=1,
        n_jobs=-1,
    )
    labels = km.fit_predict(traj_scaled)
    centroids = km.cluster_centers_.squeeze()

    logger.info(f"K-Shape: {n_clusters} clusters, inertia={km.inertia_:.2f}")
    return labels, centroids, top_idx


def plot_trajectory_analysis(
    trajectories: np.ndarray,
    features: pd.DataFrame,
    centroids: Optional[np.ndarray],
    month_index: pd.DatetimeIndex,
    kshape_labels: Optional[np.ndarray],
    kshape_idx: Optional[np.ndarray],
    config: dict,
) -> None:
    """Generate trajectory analysis visualizations."""
    setup_plotting(config)

    # 1. Archetype distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    archetype_counts = features["archetype"].value_counts()
    colors = plt.cm.Set2(np.linspace(0, 1, len(archetype_counts)))
    bars = ax.barh(archetype_counts.index, archetype_counts.values, color=colors, edgecolor="white")
    ax.set_xlabel("Provider Count")
    ax.set_title("Provider Trajectory Archetype Distribution")
    for bar, val in zip(bars, archetype_counts.values):
        ax.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                f"{val:,} ({val/len(features)*100:.1f}%)", va="center", fontsize=9)
    plt.tight_layout()
    save_figure(fig, "trajectory_archetype_distribution", config)
    plt.close(fig)

    # 2. Spaghetti plots per archetype
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=True)
    axes = axes.ravel()
    archetypes = features["archetype"].unique()
    rng = np.random.RandomState(42)
    dates = month_index.to_pydatetime()

    for i, arch in enumerate(sorted(archetypes)[:9]):
        ax = axes[i] if i < 9 else None
        if ax is None:
            break
        mask = features["archetype"] == arch
        idx = features.index[mask].values
        sample = rng.choice(idx, min(50, len(idx)), replace=False)
        for s in sample:
            traj = trajectories[s]
            valid = ~np.isnan(traj)
            if valid.sum() > 0:
                ax.plot(dates[valid], traj[valid] / 1e6, alpha=0.3, linewidth=0.5)
        ax.set_title(f"{arch} (n={mask.sum():,})", fontsize=10)
        ax.set_ylabel("$ Millions")
        ax.tick_params(axis="x", rotation=45, labelsize=7)

    for j in range(len(archetypes), 9):
        axes[j].set_visible(False)

    plt.suptitle("Provider Spending Trajectories by Archetype (sample of 50 each)", fontsize=14)
    plt.tight_layout()
    save_figure(fig, "trajectory_spaghetti_by_archetype", config)
    plt.close(fig)

    # 3. K-Shape cluster centroids
    if centroids is not None:
        fig, ax = plt.subplots(figsize=(14, 8))
        n_c = centroids.shape[0]
        cmap = plt.cm.get_cmap("tab10", n_c)
        for c in range(n_c):
            cluster_count = (kshape_labels == c).sum()
            ax.plot(dates, centroids[c], color=cmap(c), linewidth=2,
                    label=f"Cluster {c} (n={cluster_count:,})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Normalized Spending")
        ax.set_title("K-Shape Cluster Centroids (Temporal Spending Patterns)")
        ax.legend(loc="upper left", fontsize=9)
        ax.axvline(pd.Timestamp("2020-03-01"), color="red", linestyle="--", alpha=0.5, label="COVID onset")
        ax.grid(True, alpha=0.3)
        save_figure(fig, "trajectory_kshape_centroids", config)
        plt.close(fig)

    # 4. DTW distance heatmap between centroids
    if centroids is not None:
        from tslearn.metrics import dtw as tslearn_dtw
        n_c = centroids.shape[0]
        dtw_matrix = np.zeros((n_c, n_c))
        for a in range(n_c):
            for b in range(a + 1, n_c):
                d = tslearn_dtw(centroids[a], centroids[b])
                dtw_matrix[a, b] = d
                dtw_matrix[b, a] = d

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(dtw_matrix, cmap="YlOrRd")
        ax.set_xticks(range(n_c))
        ax.set_yticks(range(n_c))
        ax.set_xticklabels([f"C{i}" for i in range(n_c)])
        ax.set_yticklabels([f"C{i}" for i in range(n_c)])
        for i in range(n_c):
            for j in range(n_c):
                ax.text(j, i, f"{dtw_matrix[i,j]:.1f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="DTW Distance")
        ax.set_title("DTW Distance Between K-Shape Centroids")
        save_figure(fig, "trajectory_dtw_heatmap", config)
        plt.close(fig)

        dtw_df = pd.DataFrame(dtw_matrix, columns=[f"C{i}" for i in range(n_c)],
                              index=[f"C{i}" for i in range(n_c)])
        save_table(dtw_df.reset_index(), "trajectory_dtw_distances", config)

    # 5. Growth vs volatility scatter
    fig, ax = plt.subplots(figsize=(12, 8))
    valid = features["growth_ratio"].notna() & features["volatility"].notna()
    for arch in sorted(features["archetype"].unique()):
        mask = valid & (features["archetype"] == arch)
        if mask.sum() > 0:
            sample = features[mask].sample(min(5000, mask.sum()), random_state=42)
            ax.scatter(sample["growth_ratio"].clip(-5, 20),
                       sample["volatility"].clip(0, 5),
                       s=3, alpha=0.3, label=arch)
    ax.set_xlabel("Growth Ratio (last 12m / first 12m)")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Provider Growth vs Volatility by Archetype")
    ax.legend(fontsize=8, markerscale=5)
    ax.grid(True, alpha=0.3)
    save_figure(fig, "trajectory_growth_vs_volatility", config)
    plt.close(fig)

    # 6. COVID impact by archetype
    valid_covid = features["covid_growth_pct"].notna()
    if valid_covid.sum() > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        arch_covid = features[valid_covid].groupby("archetype")["covid_growth_pct"].agg(["median", "mean", "count"])
        arch_covid = arch_covid[arch_covid["count"] >= 50].sort_values("median")
        ax.barh(arch_covid.index, arch_covid["median"], color="steelblue", edgecolor="white")
        ax.set_xlabel("Median COVID Growth (%)")
        ax.set_title("COVID Spending Impact by Provider Archetype")
        ax.axvline(0, color="black", linewidth=0.5)
        for i, (idx, row) in enumerate(arch_covid.iterrows()):
            ax.text(row["median"] + 1, i, f"{row['median']:.1f}% (n={int(row['count']):,})",
                    va="center", fontsize=9)
        plt.tight_layout()
        save_figure(fig, "trajectory_covid_impact_by_archetype", config)
        plt.close(fig)


def run_trajectory_analysis(config: Optional[dict] = None) -> pd.DataFrame:
    """Run the full trajectory analysis pipeline."""
    if config is None:
        config = load_config()

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    traj_cfg = config.get("trajectory", {})
    seed = config["analysis"]["random_seed"]

    # Step 1: Build trajectories
    logger.info("Building monthly trajectories for all providers...")
    trajectories, npi_index, month_index = build_monthly_trajectories(config)

    # Save trajectories
    traj_df = pd.DataFrame(trajectories, columns=[m.strftime("%Y-%m") for m in month_index])
    traj_df.insert(0, "billing_npi", npi_index)
    traj_df.to_parquet(processed_dir / "provider_trajectories.parquet", index=False)
    logger.info(f"Trajectories saved: {traj_df.shape}")
    del traj_df
    gc.collect()

    # Step 2: Extract features
    logger.info("Extracting trajectory features...")
    features = extract_trajectory_features(trajectories, npi_index, month_index)
    logger.info(f"Trajectory features: {features.shape}")

    # Step 3: Classify archetypes
    logger.info("Classifying trajectory archetypes...")
    features = classify_trajectory_archetypes(features)
    archetype_counts = features["archetype"].value_counts()
    for arch, count in archetype_counts.items():
        logger.info(f"  {arch}: {count:,} providers ({count/len(features)*100:.1f}%)")

    # Save features
    features.to_parquet(processed_dir / "provider_trajectory_features.parquet", index=False)

    # Archetype summary
    summary = features.groupby("archetype").agg(
        count=("billing_npi", "count"),
        median_spending=("mean_monthly_spending", "median"),
        median_volatility=("volatility", "median"),
        median_growth=("growth_ratio", "median"),
        median_covid_growth=("covid_growth_pct", "median"),
    ).reset_index()
    save_table(summary, "trajectory_archetype_summary", config)

    # Step 4: K-Shape clustering
    kshape_labels, centroids, kshape_idx = None, None, None
    top_n = traj_cfg.get("kshape_top_n", 50000)
    n_clusters = traj_cfg.get("kshape_n_clusters", 8)

    try:
        logger.info(f"Running K-Shape clustering (top {top_n:,} providers, k={n_clusters})...")
        kshape_labels, centroids, kshape_idx = kshape_clustering(
            trajectories, features, n_clusters=n_clusters, top_n=top_n, random_state=seed,
        )

        # Save K-Shape results
        ks_df = pd.DataFrame({
            "billing_npi": npi_index[kshape_idx],
            "kshape_cluster": kshape_labels,
        })
        ks_df.to_parquet(processed_dir / "provider_kshape_clusters.parquet", index=False)
    except Exception as e:
        logger.warning(f"K-Shape clustering failed: {e}. Continuing without it.")

    # Step 5: Plots
    logger.info("Generating trajectory plots...")
    plot_trajectory_analysis(
        trajectories, features, centroids, month_index,
        kshape_labels, kshape_idx, config,
    )

    logger.info("Trajectory analysis complete")
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run trajectory analysis")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_trajectory_analysis(config)
