"""Phase 7C: Optimal Transport Analysis of Medicaid spending distributions.

Applies optimal transport (OT) theory to quantify how spending distributions
shift across time and provider segments.  Uses Wasserstein distances,
Sinkhorn divergences, and transport plans to map how healthcare dollars
flow between cost regimes.

Methods:
    - Wasserstein-1 and Wasserstein-2 distances between annual distributions
    - Sinkhorn divergence (entropic regularization) for distributional shifts
    - Transport plans: pre-COVID → post-COVID spending flow
    - Displacement interpolation (McCann, 1997) for distributional barycenters
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
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


# ── Wasserstein Distances ─────────────────────────────────────────────


def compute_wasserstein_matrix(annual_distributions: dict) -> pd.DataFrame:
    """Compute pairwise Wasserstein-1 distances between annual distributions."""
    years = sorted(annual_distributions.keys())
    n = len(years)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = stats.wasserstein_distance(
                annual_distributions[years[i]],
                annual_distributions[years[j]],
            )
            W[i, j] = W[j, i] = d
    return pd.DataFrame(W, index=years, columns=years)


def compute_sinkhorn_divergences(annual_dists: dict, reg: float = 0.1,
                                  n_bins: int = 200) -> pd.DataFrame:
    """Compute Sinkhorn divergences between binned annual distributions."""
    import ot

    years = sorted(annual_dists.keys())
    # Create shared histogram bins across all years
    all_vals = np.concatenate([annual_dists[y] for y in years])
    bins = np.linspace(np.percentile(all_vals, 0.1), np.percentile(all_vals, 99.9), n_bins)

    # Histograms as probability vectors
    hists = {}
    for y in years:
        h, _ = np.histogram(annual_dists[y], bins=bins, density=True)
        h = h / (h.sum() + 1e-10)
        hists[y] = h

    # Cost matrix (bin-to-bin squared distance)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    M = ot.dist(bin_centers.reshape(-1, 1), bin_centers.reshape(-1, 1), metric="sqeuclidean")
    M /= M.max()

    n = len(years)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = hists[years[i]], hists[years[j]]
            s = ot.sinkhorn2(a, b, M, reg=reg)
            S[i, j] = S[j, i] = float(s)

    return pd.DataFrame(S, index=years, columns=years), bin_centers, hists, M


def compute_transport_plan(source: np.ndarray, target: np.ndarray,
                            n_bins: int = 100, reg: float = 0.05) -> dict:
    """Compute regularized OT plan between two 1D distributions."""
    import ot

    all_vals = np.concatenate([source, target])
    bins = np.linspace(np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5), n_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    h_src, _ = np.histogram(source, bins=bins, density=True)
    h_tgt, _ = np.histogram(target, bins=bins, density=True)
    h_src = h_src / (h_src.sum() + 1e-10)
    h_tgt = h_tgt / (h_tgt.sum() + 1e-10)

    M = ot.dist(bin_centers.reshape(-1, 1), bin_centers.reshape(-1, 1), metric="sqeuclidean")
    M /= M.max()

    T = ot.sinkhorn(h_src, h_tgt, M, reg=reg)
    cost = float(np.sum(T * M))

    return {
        "transport_plan": T,
        "source_hist": h_src,
        "target_hist": h_tgt,
        "bin_centers": bin_centers,
        "cost": cost,
    }


def compute_displacement_interpolation(source: np.ndarray, target: np.ndarray,
                                         n_steps: int = 5,
                                         n_bins: int = 100) -> list:
    """Compute McCann displacement interpolation between distributions."""
    import ot

    all_vals = np.concatenate([source, target])
    bins = np.linspace(np.percentile(all_vals, 0.5), np.percentile(all_vals, 99.5), n_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    h_src, _ = np.histogram(source, bins=bins, density=True)
    h_tgt, _ = np.histogram(target, bins=bins, density=True)
    h_src = h_src / (h_src.sum() + 1e-10)
    h_tgt = h_tgt / (h_tgt.sum() + 1e-10)

    M = ot.dist(bin_centers.reshape(-1, 1), bin_centers.reshape(-1, 1))

    interpolations = []
    for t in np.linspace(0, 1, n_steps):
        if t == 0:
            interpolations.append(h_src.copy())
        elif t == 1:
            interpolations.append(h_tgt.copy())
        else:
            # Approximate barycenter at interpolation point
            weights = np.array([1 - t, t])
            bary = ot.bregman.barycenter(
                np.column_stack([h_src, h_tgt]), M, reg=0.01,
                weights=weights, numItermax=200,
            )
            interpolations.append(bary)

    return interpolations, bin_centers


# ── Year-over-Year Flow Analysis ──────────────────────────────────────


def compute_yoy_flows(annual_dists: dict) -> pd.DataFrame:
    """Compute year-over-year Wasserstein distances and flow statistics."""
    years = sorted(annual_dists.keys())
    rows = []
    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        d1, d2 = annual_dists[y1], annual_dists[y2]
        w1 = stats.wasserstein_distance(d1, d2)
        ks_stat, ks_p = stats.ks_2samp(d1, d2)
        rows.append({
            "year_from": y1,
            "year_to": y2,
            "wasserstein_1": w1,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_p,
            "mean_shift": float(np.mean(d2) - np.mean(d1)),
            "median_shift": float(np.median(d2) - np.median(d1)),
            "std_ratio": float(np.std(d2) / (np.std(d1) + 1e-10)),
        })
    return pd.DataFrame(rows)


# ── Visualization ─────────────────────────────────────────────────────


def plot_wasserstein_heatmap(W: pd.DataFrame, config: dict) -> None:
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(W.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(W.columns)))
    ax.set_xticklabels(W.columns, rotation=45)
    ax.set_yticks(range(len(W.index)))
    ax.set_yticklabels(W.index)

    for i in range(len(W)):
        for j in range(len(W)):
            ax.text(j, i, f"{W.values[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if W.values[i,j] < W.values.max()*0.6 else "white")

    plt.colorbar(im, ax=ax, label="Wasserstein-1 Distance")
    ax.set_title("Pairwise Wasserstein Distances Between Annual Spending Distributions",
                 fontsize=13, fontweight="bold")
    save_figure(fig, "ot_wasserstein_heatmap", config)
    logger.info("Saved Wasserstein heatmap")


def plot_sinkhorn_heatmap(S: pd.DataFrame, config: dict) -> None:
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(S.values, cmap="PuBuGn", aspect="auto")
    ax.set_xticks(range(len(S.columns)))
    ax.set_xticklabels(S.columns, rotation=45)
    ax.set_yticks(range(len(S.index)))
    ax.set_yticklabels(S.index)

    for i in range(len(S)):
        for j in range(len(S)):
            ax.text(j, i, f"{S.values[i,j]:.4f}", ha="center", va="center",
                    fontsize=8, color="black" if S.values[i,j] < S.values.max()*0.6 else "white")

    plt.colorbar(im, ax=ax, label="Sinkhorn Divergence")
    ax.set_title("Pairwise Sinkhorn Divergences Between Annual Spending Distributions",
                 fontsize=13, fontweight="bold")
    save_figure(fig, "ot_sinkhorn_heatmap", config)


def plot_transport_plan(tp: dict, year_from: str, year_to: str,
                         config: dict) -> None:
    setup_plotting(config)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Transport matrix
    T = tp["transport_plan"]
    T_display = T / (T.max() + 1e-10)
    axes[0].imshow(T_display, cmap="hot", aspect="auto",
                   norm=mcolors.LogNorm(vmin=T_display[T_display > 0].min(),
                                         vmax=T_display.max()))
    axes[0].set_xlabel(f"Target bins ({year_to})")
    axes[0].set_ylabel(f"Source bins ({year_from})")
    axes[0].set_title(f"Optimal Transport Plan: {year_from} → {year_to}")

    # Source and target marginals with flow arrows
    bc = tp["bin_centers"]
    axes[1].fill_between(bc, tp["source_hist"], alpha=0.4, color="#2196F3",
                         label=f"{year_from}")
    axes[1].fill_between(bc, tp["target_hist"], alpha=0.4, color="#FF5722",
                         label=f"{year_to}")
    axes[1].plot(bc, tp["source_hist"], color="#2196F3", lw=2)
    axes[1].plot(bc, tp["target_hist"], color="#FF5722", lw=2)
    axes[1].set_xlabel("log₁₀(Provider Spending)")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Spending Distribution Shift (OT cost = {tp['cost']:.4f})")
    axes[1].legend(fontsize=12)

    fig.suptitle(f"Optimal Transport: {year_from} → {year_to} Spending Flow",
                 fontsize=14, fontweight="bold")
    save_figure(fig, f"ot_transport_plan_{year_from}_{year_to}", config)


def plot_displacement_interpolation(interpolations: list, bin_centers: np.ndarray,
                                     year_from: str, year_to: str,
                                     config: dict) -> None:
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(14, 7))

    n_steps = len(interpolations)
    colors = plt.cm.coolwarm(np.linspace(0, 1, n_steps))

    for i, (interp, c) in enumerate(zip(interpolations, colors)):
        t = i / (n_steps - 1)
        label = f"t={t:.1f}" + (" (" + year_from + ")" if t == 0 else
                                " (" + year_to + ")" if t == 1 else "")
        ax.plot(bin_centers, interp, color=c, lw=2 if t in [0, 1] else 1.2,
                alpha=1 if t in [0, 1] else 0.7, label=label)

    ax.set_xlabel("log₁₀(Provider Spending)")
    ax.set_ylabel("Density")
    ax.set_title(f"McCann Displacement Interpolation: {year_from} → {year_to}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    save_figure(fig, "ot_displacement_interpolation", config)


def plot_yoy_wasserstein(yoy_df: pd.DataFrame, config: dict) -> None:
    setup_plotting(config)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    labels = [f"{r['year_from']}→{r['year_to']}" for _, r in yoy_df.iterrows()]

    axes[0].bar(labels, yoy_df["wasserstein_1"], color="#2196F3", alpha=0.8)
    axes[0].set_ylabel("Wasserstein-1 Distance")
    axes[0].set_title("Year-over-Year Distributional Shift (Wasserstein)")
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(labels, yoy_df["mean_shift"], color="#4CAF50", alpha=0.8, label="Mean shift")
    axes[1].bar(labels, yoy_df["median_shift"], color="#FF9800", alpha=0.6, label="Median shift")
    axes[1].set_ylabel("Shift in log₁₀(Spending)")
    axes[1].set_title("Mean and Median Spending Distribution Shifts")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend()

    fig.suptitle("Optimal Transport — Year-over-Year Spending Flow Analysis",
                 fontsize=14, fontweight="bold")
    save_figure(fig, "ot_yoy_wasserstein", config)


# ── Main Pipeline ─────────────────────────────────────────────────────


def run_optimal_transport(config: dict) -> None:
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Load provider features for annual spending
    df = pd.read_parquet(processed_dir / "provider_features.parquet")
    logger.info(f"Loaded {len(df)} providers")

    # We need per-year spending. Load from monthly time series or yearly parquets.
    # Compute log-spending distributions per year from provider features
    # Since provider_features has aggregate spending, load year-partitioned data
    annual_dists = {}
    for year in range(2018, 2025):
        parquet_path = processed_dir / f"medicaid_{year}.parquet"
        if not parquet_path.exists():
            continue
        year_df = pd.read_parquet(parquet_path,
                                   columns=["BILLING_PROVIDER_NPI_NUM", "TOTAL_PAID"])
        # Aggregate to provider level per year
        provider_spending = year_df.groupby("BILLING_PROVIDER_NPI_NUM")["TOTAL_PAID"].sum()
        # Filter positive and take log
        pos = provider_spending[provider_spending > 0]
        annual_dists[year] = np.log10(pos.values)
        logger.info(f"  {year}: {len(pos)} providers, "
                    f"median={format_currency(10**np.median(annual_dists[year]))}")
        del year_df, provider_spending
        gc.collect()

    if len(annual_dists) < 2:
        logger.error("Need at least 2 years of data for OT analysis")
        return

    # ── Wasserstein distance matrix ──
    logger.info("Computing Wasserstein distance matrix...")
    W = compute_wasserstein_matrix(annual_dists)
    save_table(W, "ot_wasserstein_distances", config)
    logger.info(f"Wasserstein matrix:\n{W.to_string()}")

    # ── Sinkhorn divergences ──
    logger.info("Computing Sinkhorn divergences...")
    S, bin_centers, hists, M = compute_sinkhorn_divergences(annual_dists)
    save_table(S, "ot_sinkhorn_divergences", config)

    # ── Year-over-year flows ──
    yoy = compute_yoy_flows(annual_dists)
    save_table(yoy, "ot_yoy_flows", config)
    logger.info(f"YoY flows:\n{yoy.to_string()}")

    # ── Transport plan: pre-COVID (2019) → post-COVID (2021) ──
    if 2019 in annual_dists and 2021 in annual_dists:
        logger.info("Computing transport plan: 2019 → 2021...")
        tp = compute_transport_plan(annual_dists[2019], annual_dists[2021])
        plot_transport_plan(tp, "2019", "2021", config)

    # ── Transport plan: peak (2023) → unwinding (2024) ──
    if 2023 in annual_dists and 2024 in annual_dists:
        logger.info("Computing transport plan: 2023 → 2024...")
        tp2 = compute_transport_plan(annual_dists[2023], annual_dists[2024])
        plot_transport_plan(tp2, "2023", "2024", config)

    # ── Displacement interpolation: 2019 → 2022 ──
    if 2019 in annual_dists and 2022 in annual_dists:
        logger.info("Computing displacement interpolation: 2019 → 2022...")
        interps, bc = compute_displacement_interpolation(
            annual_dists[2019], annual_dists[2022], n_steps=7
        )
        plot_displacement_interpolation(interps, bc, "2019", "2022", config)

    # ── Visualizations ──
    plot_wasserstein_heatmap(W, config)
    plot_sinkhorn_heatmap(S, config)
    plot_yoy_wasserstein(yoy, config)

    # ── Summary ──
    years = sorted(annual_dists.keys())
    max_shift_idx = yoy["wasserstein_1"].idxmax()
    summary = {
        "years_analyzed": years,
        "n_year_pairs": len(yoy),
        "max_wasserstein_shift": {
            "from": int(yoy.loc[max_shift_idx, "year_from"]),
            "to": int(yoy.loc[max_shift_idx, "year_to"]),
            "distance": float(yoy.loc[max_shift_idx, "wasserstein_1"]),
        },
        "pre_post_covid_wasserstein": float(W.loc[2019, 2021]) if 2019 in W.index and 2021 in W.columns else None,
        "total_cumulative_shift": float(yoy["wasserstein_1"].sum()),
    }
    tables_dir = root / config["paths"]["tables_dir"]
    with open(tables_dir / "ot_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"OT summary: {summary}")

    gc.collect()
    logger.info("Phase 7C: Optimal Transport Analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 7C: Optimal Transport Analysis")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_optimal_transport(config)
