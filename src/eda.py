"""Comprehensive Exploratory Data Analysis with publication-quality visualizations."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency, format_number,
)

logger = setup_logging()


def plot_annual_spending(ts: pd.DataFrame, config: dict) -> None:
    """Plot national total spending by year.

    Args:
        ts: Monthly time series DataFrame.
        config: Configuration dictionary.
    """
    ts["year"] = ts["date"].dt.year
    annual = ts.groupby("year").agg(
        total_paid=("total_paid", "sum"),
        total_claims=("total_claims", "sum"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 7))
    bars = ax1.bar(annual["year"], annual["total_paid"] / 1e9, color="#2196F3", alpha=0.8, width=0.6)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total Spending (Billions $)")
    ax1.set_title("National Medicaid Provider Spending by Year")
    ax1.set_xticks(annual["year"])

    # Add value labels
    for bar, val in zip(bars, annual["total_paid"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 format_currency(val), ha="center", va="bottom", fontsize=9)

    # Claims line on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(annual["year"], annual["total_claims"] / 1e9, "ro-", linewidth=2, markersize=6, label="Total Claims")
    ax2.set_ylabel("Total Claims (Billions)")
    ax2.legend(loc="upper left")

    save_figure(fig, "annual_spending_and_claims", config)
    save_table(annual, "annual_spending_summary", config)
    logger.info("Annual spending plot saved")


def plot_monthly_time_series(ts: pd.DataFrame, config: dict) -> None:
    """Plot monthly spending time series with trend.

    Args:
        ts: Monthly time series DataFrame.
        config: Configuration dictionary.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    # Spending
    axes[0].plot(ts["date"], ts["total_paid"] / 1e9, color="#1976D2", linewidth=1.5)
    axes[0].fill_between(ts["date"], 0, ts["total_paid"] / 1e9, alpha=0.15, color="#1976D2")
    axes[0].set_ylabel("Total Spending ($B)")
    axes[0].set_title("Monthly Medicaid Spending, Claims, and Providers (2018-2024)")

    # Claims
    axes[1].plot(ts["date"], ts["total_claims"] / 1e6, color="#388E3C", linewidth=1.5)
    axes[1].fill_between(ts["date"], 0, ts["total_claims"] / 1e6, alpha=0.15, color="#388E3C")
    axes[1].set_ylabel("Total Claims (M)")

    # Providers
    axes[2].plot(ts["date"], ts["unique_providers"] / 1e3, color="#E64A19", linewidth=1.5)
    axes[2].set_ylabel("Active Providers (K)")
    axes[2].set_xlabel("Date")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"),
                    alpha=0.08, color="red", label="COVID-19 period")

    save_figure(fig, "monthly_time_series", config)
    logger.info("Monthly time series plot saved")


def plot_spending_distribution(config: dict) -> None:
    """Plot distribution of per-row spending amounts (from parquet).

    Args:
        config: Configuration dictionary.
    """
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    # Sample for distribution
    all_paid = []
    for pf in sorted(processed_dir.glob("medicaid_*.parquet")):
        df = pd.read_parquet(pf, columns=["TOTAL_PAID"])
        # Sample 1% for distribution plotting
        sample = df["TOTAL_PAID"].sample(frac=0.01, random_state=42)
        all_paid.append(sample)
        del df

    paid = pd.concat(all_paid)
    paid_positive = paid[paid > 0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Log-scale histogram
    axes[0].hist(np.log10(paid_positive), bins=100, color="#5C6BC0", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("log10(Total Paid $)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Payment Amounts (Log Scale)")
    axes[0].axvline(np.log10(paid_positive.median()), color="red", linestyle="--",
                     label=f"Median: {format_currency(paid_positive.median())}")
    axes[0].axvline(np.log10(paid_positive.mean()), color="orange", linestyle="--",
                     label=f"Mean: {format_currency(paid_positive.mean())}")
    axes[0].legend()

    # Box plot of log payments
    axes[1].boxplot(np.log10(paid_positive), vert=True)
    axes[1].set_ylabel("log10(Total Paid $)")
    axes[1].set_title("Box Plot of Payment Amounts (Log Scale)")

    save_figure(fig, "spending_distribution", config)
    logger.info("Spending distribution plot saved")


def plot_top_hcpcs(config: dict) -> None:
    """Plot top 20 HCPCS codes by total spending.

    Args:
        config: Configuration dictionary.
    """
    root = get_project_root()
    hcpcs_path = root / config["paths"]["processed_dir"] / "hcpcs_features.parquet"
    if not hcpcs_path.exists():
        logger.warning("HCPCS features not yet built, skipping")
        return

    hcpcs = pd.read_parquet(hcpcs_path)
    top20 = hcpcs.nlargest(20, "total_paid")

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(top20)), top20["total_paid"].values / 1e9, color="#26A69A", alpha=0.85)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["hcpcs_code"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Total Spending (Billions $)")
    ax.set_title("Top 20 HCPCS Procedure Codes by Total Medicaid Spending (2018-2024)")

    for bar, val in zip(bars, top20["total_paid"].values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                format_currency(val), va="center", fontsize=8)

    save_figure(fig, "top20_hcpcs_spending", config)
    save_table(top20[["hcpcs_code", "total_paid", "total_claims", "avg_paid_per_claim", "n_providers"]],
               "top20_hcpcs_spending", config)
    logger.info("Top 20 HCPCS plot saved")


def plot_top_providers(config: dict) -> None:
    """Plot top 20 providers by total spending.

    Args:
        config: Configuration dictionary.
    """
    root = get_project_root()
    prov_path = root / config["paths"]["processed_dir"] / "provider_features.parquet"
    if not prov_path.exists():
        logger.warning("Provider features not yet built, skipping")
        return

    providers = pd.read_parquet(prov_path)
    top20 = providers.nlargest(20, "total_paid")

    # Anonymize NPIs for display
    top20 = top20.copy()
    top20["label"] = [f"Provider #{i+1}" for i in range(len(top20))]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(range(len(top20)), top20["total_paid"].values / 1e9, color="#EF5350", alpha=0.85)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["label"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Total Spending (Billions $)")
    ax.set_title("Top 20 Billing Providers by Total Medicaid Spending (2018-2024)")

    for bar, val in zip(bars, top20["total_paid"].values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                format_currency(val), va="center", fontsize=8)

    save_figure(fig, "top20_providers_spending", config)
    save_table(top20[["billing_npi", "total_paid", "total_claims", "n_unique_hcpcs", "n_years_active"]],
               "top20_providers_spending", config)
    logger.info("Top 20 providers plot saved")


def plot_provider_concentration(config: dict) -> None:
    """Plot provider spending concentration (Lorenz curve).

    Args:
        config: Configuration dictionary.
    """
    root = get_project_root()
    prov_path = root / config["paths"]["processed_dir"] / "provider_features.parquet"
    if not prov_path.exists():
        return

    providers = pd.read_parquet(prov_path, columns=["total_paid"])
    providers = providers[providers["total_paid"] > 0].sort_values("total_paid")

    spending = providers["total_paid"].values
    cumulative = np.cumsum(spending) / spending.sum()
    n = len(spending)
    pct_providers = np.arange(1, n + 1) / n

    # Find key thresholds
    p50 = np.searchsorted(cumulative, 0.50) / n
    p80 = np.searchsorted(cumulative, 0.80) / n
    p90 = np.searchsorted(cumulative, 0.90) / n

    # Gini coefficient
    gini = 1 - 2 * np.trapz(cumulative, pct_providers)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(pct_providers * 100, cumulative * 100, color="#1565C0", linewidth=2, label="Lorenz Curve")
    ax.plot([0, 100], [0, 100], "k--", alpha=0.5, label="Perfect Equality")
    ax.fill_between(pct_providers * 100, cumulative * 100, pct_providers * 100, alpha=0.1, color="#1565C0")

    ax.axhline(50, color="gray", alpha=0.3, linestyle=":")
    ax.axhline(80, color="gray", alpha=0.3, linestyle=":")
    ax.axhline(90, color="gray", alpha=0.3, linestyle=":")

    ax.set_xlabel("Cumulative % of Providers (sorted by spending)")
    ax.set_ylabel("Cumulative % of Total Spending")
    ax.set_title(f"Provider Spending Concentration (Gini = {gini:.3f})")
    ax.text(20, 85, f"Top {(1-p50)*100:.1f}% providers = 50% spending\n"
                     f"Top {(1-p80)*100:.1f}% providers = 80% spending\n"
                     f"Top {(1-p90)*100:.1f}% providers = 90% spending",
            fontsize=11, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    save_figure(fig, "provider_concentration_lorenz", config)
    logger.info(f"Provider concentration: Gini={gini:.3f}")


def plot_hcpcs_frequency_vs_cost(config: dict) -> None:
    """Scatter plot of procedure code frequency vs average cost.

    Args:
        config: Configuration dictionary.
    """
    root = get_project_root()
    hcpcs_path = root / config["paths"]["processed_dir"] / "hcpcs_features.parquet"
    if not hcpcs_path.exists():
        return

    hcpcs = pd.read_parquet(hcpcs_path)
    hcpcs = hcpcs[hcpcs["total_claims"] > 0]

    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        hcpcs["total_claims"],
        hcpcs["avg_paid_per_claim"],
        c=np.log10(hcpcs["total_paid"].clip(lower=1)),
        s=10, alpha=0.5, cmap="viridis",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total Claims (log scale)")
    ax.set_ylabel("Average Paid per Claim (log scale)")
    ax.set_title("HCPCS Code: Frequency vs. Average Cost")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("log10(Total Spending)")

    # Annotate top codes
    top = hcpcs.nlargest(10, "total_paid")
    for _, row in top.iterrows():
        ax.annotate(row["hcpcs_code"],
                     (row["total_claims"], row["avg_paid_per_claim"]),
                     fontsize=7, alpha=0.8,
                     xytext=(5, 5), textcoords="offset points")

    save_figure(fig, "hcpcs_frequency_vs_cost", config)
    logger.info("HCPCS frequency vs cost plot saved")


def plot_provider_hcpcs_diversity(config: dict) -> None:
    """Plot distribution of HCPCS code diversity per provider.

    Args:
        config: Configuration dictionary.
    """
    root = get_project_root()
    prov_path = root / config["paths"]["processed_dir"] / "provider_features.parquet"
    if not prov_path.exists():
        return

    providers = pd.read_parquet(prov_path, columns=["n_unique_hcpcs", "total_paid"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram of unique HCPCS per provider
    axes[0].hist(providers["n_unique_hcpcs"], bins=100, color="#7B1FA2", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Number of Unique HCPCS Codes")
    axes[0].set_ylabel("Number of Providers")
    axes[0].set_title("Provider Procedure Code Diversity")
    axes[0].set_yscale("log")
    med = providers["n_unique_hcpcs"].median()
    axes[0].axvline(med, color="red", linestyle="--", label=f"Median: {med:.0f}")
    axes[0].legend()

    # HCPCS diversity vs spending
    axes[1].scatter(providers["n_unique_hcpcs"], providers["total_paid"] / 1e6,
                     s=2, alpha=0.1, color="#7B1FA2")
    axes[1].set_xlabel("Number of Unique HCPCS Codes")
    axes[1].set_ylabel("Total Spending (Millions $)")
    axes[1].set_title("Procedure Diversity vs. Total Spending")
    axes[1].set_yscale("log")

    save_figure(fig, "provider_hcpcs_diversity", config)
    logger.info("Provider HCPCS diversity plot saved")


def plot_paid_per_claim_by_year(config: dict) -> None:
    """Plot average paid per claim over time.

    Args:
        config: Configuration dictionary.
    """
    root = get_project_root()
    ts_path = root / config["paths"]["processed_dir"] / "monthly_time_series.parquet"
    if not ts_path.exists():
        return

    ts = pd.read_parquet(ts_path)
    ts["date"] = pd.to_datetime(ts["date"])

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts["date"], ts["paid_per_claim"], color="#00897B", linewidth=1.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Paid per Claim ($)")
    ax.set_title("Average Medicaid Payment per Claim Over Time")
    ax.grid(True, alpha=0.3)

    # COVID shading
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"),
               alpha=0.08, color="red", label="COVID-19 period")
    ax.legend()

    save_figure(fig, "paid_per_claim_time_series", config)
    logger.info("Paid per claim time series saved")


def plot_seasonality(ts: pd.DataFrame, config: dict) -> None:
    """Plot month-of-year seasonality patterns.

    Args:
        ts: Monthly time series DataFrame.
        config: Configuration dictionary.
    """
    ts["month_num"] = ts["date"].dt.month
    ts["year"] = ts["date"].dt.year
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Spending by month of year
    monthly_avg = ts.groupby("month_num")["total_paid"].mean() / 1e9
    axes[0].bar(range(1, 13), monthly_avg.values, color="#FF7043", alpha=0.8)
    axes[0].set_xticks(range(1, 13))
    axes[0].set_xticklabels(month_names)
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Average Monthly Spending ($B)")
    axes[0].set_title("Average Spending by Month of Year")

    # Year-over-year by month (heatmap)
    pivot = ts.pivot_table(values="total_paid", index="year", columns="month_num", aggfunc="sum") / 1e9
    sns.heatmap(pivot, ax=axes[1], cmap="YlOrRd", annot=True, fmt=".1f",
                xticklabels=month_names, cbar_kws={"label": "Spending ($B)"})
    axes[1].set_title("Monthly Spending Heatmap by Year ($B)")
    axes[1].set_ylabel("Year")

    save_figure(fig, "seasonality_patterns", config)
    logger.info("Seasonality plot saved")


def plot_yoy_growth(ts: pd.DataFrame, config: dict) -> None:
    """Plot year-over-year spending growth rates.

    Args:
        ts: Monthly time series DataFrame.
        config: Configuration dictionary.
    """
    ts["year"] = ts["date"].dt.year
    annual = ts.groupby("year")["total_paid"].sum().reset_index()
    annual["yoy_growth"] = annual["total_paid"].pct_change() * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4CAF50" if g >= 0 else "#F44336" for g in annual["yoy_growth"].fillna(0)]
    bars = ax.bar(annual["year"].iloc[1:], annual["yoy_growth"].iloc[1:], color=colors[1:], alpha=0.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("Year-over-Year Growth (%)")
    ax.set_title("Annual Medicaid Spending Growth Rate")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(annual["year"].iloc[1:])

    for bar, val in zip(bars, annual["yoy_growth"].iloc[1:]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    save_figure(fig, "yoy_spending_growth", config)
    logger.info("YoY growth plot saved")


def run_eda(config: Optional[dict] = None) -> None:
    """Run the full EDA pipeline.

    Args:
        config: Configuration dictionary.
    """
    if config is None:
        config = load_config()

    setup_plotting(config)
    root = get_project_root()

    # Load time series
    ts_path = root / config["paths"]["processed_dir"] / "monthly_time_series.parquet"
    if ts_path.exists():
        ts = pd.read_parquet(ts_path)
        ts["date"] = pd.to_datetime(ts["date"])

        plot_annual_spending(ts, config)
        plot_monthly_time_series(ts, config)
        plot_seasonality(ts.copy(), config)
        plot_yoy_growth(ts.copy(), config)
        plot_paid_per_claim_by_year(config)
    else:
        logger.warning("Monthly time series not found. Run feature_engineering.py --timeseries first.")

    # Spending distribution (from parquet files)
    plot_spending_distribution(config)

    # HCPCS analyses
    plot_top_hcpcs(config)
    plot_hcpcs_frequency_vs_cost(config)

    # Provider analyses
    plot_top_providers(config)
    plot_provider_concentration(config)
    plot_provider_hcpcs_diversity(config)

    logger.info("EDA complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run exploratory data analysis")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_eda(config)
