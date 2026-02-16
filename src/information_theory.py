"""Phase 7E: Information-Theoretic Analysis of Medicaid spending patterns.

Applies information theory to quantify the complexity, predictability, and
information flow in provider billing patterns.

Methods:
    - Shannon entropy of billing code distributions per provider
    - Mutual information between provider features and spending outcomes
    - KL divergence between provider-level and population-level distributions
    - Transfer entropy for directed information flow in time series
    - Jensen-Shannon divergence for cluster/group comparisons
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
from scipy import stats as sp_stats
from scipy.special import rel_entr
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


# ── Entropy Measures ──────────────────────────────────────────────────


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy H(X) = -Σ p(x) log₂ p(x)."""
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def compute_provider_entropy(config: dict) -> pd.DataFrame:
    """Compute the billing code distribution entropy for each provider.

    Higher entropy → more diverse billing patterns.
    Lower entropy → concentrated on fewer codes (specialist).
    """
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    logger.info("Computing provider-level billing entropy...")
    entropies = []

    for year in range(2018, 2025):
        path = processed_dir / f"medicaid_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["BILLING_PROVIDER_NPI_NUM", "HCPCS_CODE",
                                             "TOTAL_CLAIMS"])
        # Group by provider → HCPCS claim distribution
        provider_hcpcs = df.groupby(["BILLING_PROVIDER_NPI_NUM", "HCPCS_CODE"])["TOTAL_CLAIMS"].sum()
        provider_hcpcs = provider_hcpcs.reset_index()

        for npi, group in provider_hcpcs.groupby("BILLING_PROVIDER_NPI_NUM"):
            claims = group["TOTAL_CLAIMS"].values.astype(float)
            entropies.append({
                "billing_npi": npi,
                "year": year,
                "billing_entropy": shannon_entropy(claims),
                "n_codes": len(claims),
                "total_claims": float(claims.sum()),
                "max_code_share": float(claims.max() / (claims.sum() + 1e-10)),
            })
        del df, provider_hcpcs
        gc.collect()
        logger.info(f"  {year}: computed entropy for {len(entropies)} provider-years")

    ent_df = pd.DataFrame(entropies)

    # Aggregate across years
    agg_df = ent_df.groupby("billing_npi").agg(
        mean_entropy=("billing_entropy", "mean"),
        std_entropy=("billing_entropy", "std"),
        max_entropy=("billing_entropy", "max"),
        mean_n_codes=("n_codes", "mean"),
        mean_max_code_share=("max_code_share", "mean"),
        n_years=("year", "nunique"),
    ).reset_index()
    agg_df["std_entropy"] = agg_df["std_entropy"].fillna(0)

    logger.info(f"Provider entropy: {len(agg_df)} providers, "
                f"mean H = {agg_df['mean_entropy'].mean():.3f} bits")
    return agg_df, ent_df


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) with smoothing to avoid infinities."""
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    # Add small constant for numerical stability
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(rel_entr(p, q)))


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence: JSD(p||q) = 0.5*KL(p||m) + 0.5*KL(q||m)."""
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# ── Mutual Information ────────────────────────────────────────────────


def compute_mutual_information(config: dict) -> pd.DataFrame:
    """Compute MI between each provider feature and log-spending."""
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    df = pd.read_parquet(processed_dir / "provider_features.parquet")
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != "billing_npi" and c != "total_paid"]

    X = df[feature_cols].fillna(0).values
    y = np.log(df["total_paid"].clip(lower=1).values)

    logger.info(f"Computing MI between {len(feature_cols)} features and log-spending...")
    mi_scores = mutual_info_regression(X, y, n_neighbors=5, random_state=42)

    mi_df = pd.DataFrame({
        "feature": feature_cols,
        "mutual_information": mi_scores,
    }).sort_values("mutual_information", ascending=False)
    mi_df["mi_normalized"] = mi_df["mutual_information"] / mi_df["mutual_information"].max()

    logger.info(f"Top MI features:\n{mi_df.head(10).to_string()}")
    return mi_df


# ── Transfer Entropy ──────────────────────────────────────────────────


def transfer_entropy(source: np.ndarray, target: np.ndarray,
                      lag: int = 1, n_bins: int = 10) -> float:
    """Estimate transfer entropy TE(source → target) via binned estimator.

    TE(X→Y) = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})
    """
    n = min(len(source), len(target))
    if n <= lag:
        return 0.0

    # Discretize
    src_binned = np.digitize(source[:n], np.linspace(source.min(), source.max(), n_bins))
    tgt_binned = np.digitize(target[:n], np.linspace(target.min(), target.max(), n_bins))

    # Build joint distributions
    tgt_future = tgt_binned[lag:]
    tgt_past = tgt_binned[:n - lag]
    src_past = src_binned[:n - lag]

    # H(Y_t | Y_{t-1}) via joint entropy
    def conditional_entropy(x, y):
        """H(X|Y) = H(X,Y) - H(Y)."""
        xy = np.column_stack([x, y])
        _, counts_xy = np.unique(xy, axis=0, return_counts=True)
        _, counts_y = np.unique(y, return_counts=True)
        p_xy = counts_xy / counts_xy.sum()
        p_y = counts_y / counts_y.sum()
        h_xy = -np.sum(p_xy * np.log2(p_xy + 1e-10))
        h_y = -np.sum(p_y * np.log2(p_y + 1e-10))
        return h_xy - h_y

    h_y_given_ypast = conditional_entropy(tgt_future, tgt_past)
    h_y_given_ypast_xpast = conditional_entropy(
        tgt_future, np.column_stack([tgt_past, src_past])
    )

    return max(0.0, h_y_given_ypast - h_y_given_ypast_xpast)


def compute_transfer_entropy_matrix(config: dict) -> pd.DataFrame:
    """Compute pairwise transfer entropy between spending time series."""
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    ts_path = processed_dir / "monthly_time_series.parquet"
    if not ts_path.exists():
        logger.warning("Monthly time series not found, skipping transfer entropy")
        return pd.DataFrame()

    ts = pd.read_parquet(ts_path)
    logger.info(f"Monthly time series: {ts.shape}")

    # Compute TE between spending and claims
    metrics = ["total_paid", "total_claims", "total_beneficiaries"]
    available = [m for m in metrics if m in ts.columns]

    results = []
    for src in available:
        for tgt in available:
            if src == tgt:
                continue
            te = transfer_entropy(ts[src].values, ts[tgt].values, lag=1)
            results.append({
                "source": src,
                "target": tgt,
                "transfer_entropy": te,
            })

    te_df = pd.DataFrame(results)
    logger.info(f"Transfer entropy:\n{te_df.to_string()}")
    return te_df


# ── JSD Between Groups ───────────────────────────────────────────────


def compute_group_divergences(config: dict) -> pd.DataFrame:
    """Compute JSD between spending distributions of different provider clusters."""
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Load clustered providers
    cluster_path = processed_dir / "provider_clustered.parquet"
    if not cluster_path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(cluster_path)
    if "cluster" not in df.columns:
        return pd.DataFrame()

    clusters = sorted(df["cluster"].unique())
    n_bins = 100
    all_spending = np.log10(df["total_paid"].clip(lower=1).values)
    bins = np.linspace(all_spending.min(), all_spending.max(), n_bins + 1)

    # Build histograms per cluster
    cluster_hists = {}
    for c in clusters:
        vals = all_spending[df["cluster"] == c]
        h, _ = np.histogram(vals, bins=bins, density=True)
        cluster_hists[c] = h

    # Pairwise JSD
    rows = []
    for i, c1 in enumerate(clusters):
        for c2 in clusters[i + 1:]:
            jsd = jensen_shannon_divergence(cluster_hists[c1], cluster_hists[c2])
            rows.append({
                "cluster_1": c1,
                "cluster_2": c2,
                "jsd": jsd,
                "n_cluster_1": int((df["cluster"] == c1).sum()),
                "n_cluster_2": int((df["cluster"] == c2).sum()),
            })

    return pd.DataFrame(rows)


# ── Visualization ─────────────────────────────────────────────────────


def plot_entropy_distribution(ent_df: pd.DataFrame, config: dict) -> None:
    setup_plotting(config)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(ent_df["mean_entropy"], bins=100, color="#2196F3", alpha=0.7,
                 density=True)
    axes[0].set_xlabel("Mean Billing Entropy (bits)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Provider Billing Entropy Distribution")
    axes[0].axvline(ent_df["mean_entropy"].median(), color="red", ls="--",
                    label=f"Median = {ent_df['mean_entropy'].median():.2f}")
    axes[0].legend()

    axes[1].scatter(ent_df["mean_n_codes"], ent_df["mean_entropy"],
                    alpha=0.2, s=3, color="#4CAF50")
    axes[1].set_xlabel("Mean Number of HCPCS Codes")
    axes[1].set_ylabel("Mean Billing Entropy (bits)")
    axes[1].set_title("HCPCS Diversity vs Billing Entropy")

    axes[2].scatter(ent_df["mean_max_code_share"], ent_df["mean_entropy"],
                    alpha=0.2, s=3, color="#FF9800")
    axes[2].set_xlabel("Mean Top Code Share")
    axes[2].set_ylabel("Mean Billing Entropy (bits)")
    axes[2].set_title("Concentration vs Entropy")

    fig.suptitle("Information-Theoretic Provider Characterization",
                 fontsize=14, fontweight="bold")
    save_figure(fig, "info_entropy_distribution", config)


def plot_mutual_information(mi_df: pd.DataFrame, config: dict) -> None:
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(12, 8))

    top_n = min(20, len(mi_df))
    top = mi_df.head(top_n)

    colors = plt.cm.viridis(top["mi_normalized"].values)
    ax.barh(range(top_n), top["mutual_information"].values, color=colors,
            alpha=0.8, height=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Mutual Information (nats)")
    ax.set_title("Feature-Spending Mutual Information Ranking",
                 fontsize=13, fontweight="bold")

    save_figure(fig, "info_mutual_information", config)


def plot_entropy_time_evolution(ent_yearly: pd.DataFrame, config: dict) -> None:
    """Plot how provider entropy evolves over years."""
    setup_plotting(config)

    yearly_stats = ent_yearly.groupby("year").agg(
        mean_entropy=("billing_entropy", "mean"),
        median_entropy=("billing_entropy", "median"),
        std_entropy=("billing_entropy", "std"),
        n_providers=("billing_npi", "nunique"),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].plot(yearly_stats["year"], yearly_stats["mean_entropy"],
                 "o-", color="#2196F3", lw=2, label="Mean")
    axes[0].fill_between(
        yearly_stats["year"],
        yearly_stats["mean_entropy"] - yearly_stats["std_entropy"],
        yearly_stats["mean_entropy"] + yearly_stats["std_entropy"],
        alpha=0.2, color="#2196F3",
    )
    axes[0].plot(yearly_stats["year"], yearly_stats["median_entropy"],
                 "s--", color="#FF5722", lw=2, label="Median")
    axes[0].axvline(2020, color="gray", ls=":", label="COVID onset")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Billing Entropy (bits)")
    axes[0].set_title("Average Provider Billing Entropy Over Time")
    axes[0].legend()

    axes[1].bar(yearly_stats["year"], yearly_stats["n_providers"],
                color="#4CAF50", alpha=0.7)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Number of Active Providers")
    axes[1].set_title("Provider Count Over Time")

    fig.suptitle("Temporal Evolution of Billing Pattern Complexity",
                 fontsize=14, fontweight="bold")
    save_figure(fig, "info_entropy_evolution", config)


def plot_transfer_entropy(te_df: pd.DataFrame, config: dict) -> None:
    if te_df.empty:
        return
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = [f"{r['source']}\n→ {r['target']}" for _, r in te_df.iterrows()]
    colors = ["#2196F3" if te > 0.1 else "#90CAF9" for te in te_df["transfer_entropy"]]
    ax.barh(labels, te_df["transfer_entropy"], color=colors, alpha=0.8, height=0.5)
    ax.set_xlabel("Transfer Entropy (bits)")
    ax.set_title("Directed Information Flow Between Spending Metrics",
                 fontsize=13, fontweight="bold")

    save_figure(fig, "info_transfer_entropy", config)


# ── Main Pipeline ─────────────────────────────────────────────────────


def run_information_theory(config: dict) -> None:
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # ── Provider entropy ──
    logger.info("=== Computing provider billing entropy ===")
    ent_agg, ent_yearly = compute_provider_entropy(config)

    ent_agg.to_parquet(processed_dir / "provider_entropy.parquet", index=False)
    save_table(ent_agg.describe().T, "info_entropy_statistics", config)

    # ── Mutual information ──
    logger.info("=== Computing mutual information ===")
    mi_df = compute_mutual_information(config)
    save_table(mi_df, "info_mutual_information", config)

    # ── Transfer entropy ──
    logger.info("=== Computing transfer entropy ===")
    te_df = compute_transfer_entropy_matrix(config)
    if not te_df.empty:
        save_table(te_df, "info_transfer_entropy", config)

    # ── Group divergences ──
    logger.info("=== Computing cluster divergences ===")
    jsd_df = compute_group_divergences(config)
    if not jsd_df.empty:
        save_table(jsd_df, "info_cluster_jsd", config)
        logger.info(f"Cluster JSD:\n{jsd_df.to_string()}")

    # ── Visualizations ──
    plot_entropy_distribution(ent_agg, config)
    plot_mutual_information(mi_df, config)
    plot_entropy_time_evolution(ent_yearly, config)
    plot_transfer_entropy(te_df, config)

    # ── Summary ──
    summary = {
        "n_providers": len(ent_agg),
        "mean_billing_entropy": float(ent_agg["mean_entropy"].mean()),
        "median_billing_entropy": float(ent_agg["mean_entropy"].median()),
        "max_billing_entropy": float(ent_agg["mean_entropy"].max()),
        "top_mi_feature": mi_df.iloc[0]["feature"] if len(mi_df) > 0 else None,
        "top_mi_value": float(mi_df.iloc[0]["mutual_information"]) if len(mi_df) > 0 else None,
        "n_transfer_entropy_pairs": len(te_df),
        "max_transfer_entropy": float(te_df["transfer_entropy"].max()) if not te_df.empty else None,
    }
    tables_dir = root / config["paths"]["tables_dir"]
    with open(tables_dir / "info_theory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Info theory summary: {json.dumps(summary, indent=2)}")

    gc.collect()
    logger.info("Phase 7E: Information-Theoretic Analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7E: Information-Theoretic Analysis")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_information_theory(config)
