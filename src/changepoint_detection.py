"""Phase 8D: Per-Provider Changepoint Detection.

Detects spending regime changes at the individual provider level using
the PELT algorithm, going beyond the aggregate Phase 4C time series analysis.

Methods:
    - PELT (Pruned Exact Linear Time) per-provider changepoint detection
    - Changepoint type classification (mean-shift, variance-shift, trend-break)
    - Aggregate timing analysis: when do regime changes cluster?
    - COVID-period enrichment test
    - Top-N provider changepoint visualization
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
import ruptures as rpt
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


# ── Per-Provider Changepoints ────────────────────────────────────────


def detect_provider_changepoints(config: dict, top_n: int = 5000,
                                  min_months: int = 24) -> pd.DataFrame:
    """Run PELT on individual provider monthly spending time series.

    Returns DataFrame with one row per detected changepoint.
    """
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    logger.info("Building provider monthly time series...")

    # Collect monthly spending per provider across all years
    monthly_records = []
    for year in range(2018, 2025):
        path = processed_dir / f"medicaid_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(
            path, columns=["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH",
                           "TOTAL_PAID", "TOTAL_CLAIMS"]
        )
        agg = df.groupby(["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH"]).agg(
            total_paid=("TOTAL_PAID", "sum"),
            total_claims=("TOTAL_CLAIMS", "sum"),
        ).reset_index()
        monthly_records.append(agg)
        del df
        gc.collect()

    monthly = pd.concat(monthly_records, ignore_index=True)
    del monthly_records
    gc.collect()

    # Select top-N providers by total spending
    provider_totals = monthly.groupby("BILLING_PROVIDER_NPI_NUM")["total_paid"].sum()
    top_providers = provider_totals.nlargest(top_n).index
    monthly = monthly[monthly["BILLING_PROVIDER_NPI_NUM"].isin(top_providers)]

    # Pivot to create dense time series per provider
    monthly["month_date"] = pd.to_datetime(monthly["CLAIM_FROM_MONTH"], format="%Y-%m")
    monthly = monthly.sort_values("month_date")

    all_months = pd.date_range(
        monthly["month_date"].min(), monthly["month_date"].max(), freq="MS"
    )

    logger.info(f"Running PELT on {len(top_providers)} providers "
                f"({len(all_months)} months)...")

    changepoint_rows = []
    provider_summary = []
    n_processed = 0

    for npi in top_providers:
        prov_data = monthly[monthly["BILLING_PROVIDER_NPI_NUM"] == npi]
        ts = prov_data.set_index("month_date")["total_paid"].reindex(
            all_months, fill_value=0
        ).values

        # Skip short or zero series
        nonzero_months = np.sum(ts > 0)
        if nonzero_months < min_months:
            continue

        # PELT on log-transformed spending (stabilizes variance)
        try:
            ts_log = np.log1p(np.maximum(ts, 0))
            algo = rpt.Pelt(model="l2", min_size=3, jump=1).fit(ts_log)
            # BIC-like penalty calibrated for log-scale
            pen = 2 * np.log(len(ts_log)) * np.var(ts_log[ts_log > 0]) if np.any(ts_log > 0) else 1.0
            result = algo.predict(pen=pen)
            # Remove the last element (always == len(ts))
            cps = [cp for cp in result if cp < len(ts)]
        except Exception:
            cps = []

        # Classify changepoints
        for cp_idx in cps:
            month_date = all_months[cp_idx]
            before = ts[max(0, cp_idx - 6):cp_idx]
            after = ts[cp_idx:min(len(ts), cp_idx + 6)]

            if len(before) < 2 or len(after) < 2:
                continue

            mean_before = before.mean()
            mean_after = after.mean()
            std_before = before.std() + 1e-6
            std_after = after.std() + 1e-6

            # Classify type
            mean_change = abs(mean_after - mean_before) / (mean_before + 1e-6)
            var_change = abs(std_after - std_before) / (std_before + 1e-6)

            if mean_change > 0.5:
                cp_type = "mean_shift"
            elif var_change > 0.5:
                cp_type = "variance_shift"
            else:
                cp_type = "trend_break"

            # Direction
            direction = "increase" if mean_after > mean_before else "decrease"

            changepoint_rows.append({
                "billing_npi": npi,
                "changepoint_month": str(month_date.strftime("%Y-%m")),
                "changepoint_idx": cp_idx,
                "type": cp_type,
                "direction": direction,
                "mean_before": float(mean_before),
                "mean_after": float(mean_after),
                "pct_change": float(mean_change) * (1 if direction == "increase" else -1),
                "std_before": float(std_before),
                "std_after": float(std_after),
                "is_covid_period": month_date >= pd.Timestamp("2020-03-01") and
                                   month_date <= pd.Timestamp("2021-12-31"),
            })

        provider_summary.append({
            "billing_npi": npi,
            "n_changepoints": len(cps),
            "nonzero_months": int(nonzero_months),
            "total_spending": float(ts.sum()),
        })

        n_processed += 1
        if n_processed % 1000 == 0:
            logger.info(f"  Processed {n_processed}/{len(top_providers)} providers, "
                        f"{len(changepoint_rows)} changepoints found")

    cp_df = pd.DataFrame(changepoint_rows)
    summary_df = pd.DataFrame(provider_summary)

    logger.info(f"Total changepoints: {len(cp_df)} across {n_processed} providers")
    return cp_df, summary_df, all_months


# ── COVID Enrichment Test ────────────────────────────────────────────


def test_covid_enrichment(cp_df: pd.DataFrame) -> dict:
    """Test whether changepoints are enriched during the COVID period."""
    if cp_df.empty:
        return {"error": "No changepoints"}

    n_total = len(cp_df)
    n_covid = cp_df["is_covid_period"].sum()
    # COVID period is roughly 22 months out of ~84 total months
    expected_pct = 22 / 84
    observed_pct = n_covid / n_total

    # Binomial test
    binom_p = sp_stats.binomtest(n_covid, n_total, expected_pct,
                                  alternative="greater").pvalue

    result = {
        "n_total_changepoints": int(n_total),
        "n_covid_changepoints": int(n_covid),
        "expected_pct": float(expected_pct),
        "observed_pct": float(observed_pct),
        "enrichment_ratio": float(observed_pct / expected_pct) if expected_pct > 0 else 0,
        "binomial_p_value": float(binom_p),
        "significant": binom_p < 0.05,
    }
    logger.info(f"COVID enrichment: observed={observed_pct:.1%}, "
                f"expected={expected_pct:.1%}, p={binom_p:.4e}")
    return result


# ── Visualization ────────────────────────────────────────────────────


def plot_example_changepoints(config: dict, cp_df: pd.DataFrame,
                               all_months: pd.DatetimeIndex,
                               n_examples: int = 9) -> None:
    """Plot individual provider time series with detected changepoints."""
    setup_plotting(config)

    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    if cp_df.empty:
        return

    # Select providers with interesting changepoint patterns
    providers_by_cp = cp_df.groupby("billing_npi").size().sort_values(ascending=False)
    example_npis = providers_by_cp.head(n_examples).index

    # Rebuild time series for examples
    monthly_records = []
    for year in range(2018, 2025):
        path = processed_dir / f"medicaid_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(
            path, columns=["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH", "TOTAL_PAID"]
        )
        df = df[df["BILLING_PROVIDER_NPI_NUM"].isin(example_npis)]
        agg = df.groupby(["BILLING_PROVIDER_NPI_NUM", "CLAIM_FROM_MONTH"])["TOTAL_PAID"].sum().reset_index()
        monthly_records.append(agg)
        del df

    monthly = pd.concat(monthly_records, ignore_index=True)
    monthly["month_date"] = pd.to_datetime(monthly["CLAIM_FROM_MONTH"], format="%Y-%m")

    n_cols = 3
    n_rows = (n_examples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, npi in enumerate(example_npis):
        ax = axes[i // n_cols, i % n_cols]
        prov = monthly[monthly["BILLING_PROVIDER_NPI_NUM"] == npi].sort_values("month_date")
        ts = prov.set_index("month_date")["TOTAL_PAID"].reindex(
            all_months, fill_value=0
        )
        ax.plot(ts.index, ts.values / 1e3, color="#2196F3", lw=1.5)

        # Mark changepoints
        npi_cps = cp_df[cp_df["billing_npi"] == npi]
        for _, cp_row in npi_cps.iterrows():
            cp_date = pd.Timestamp(cp_row["changepoint_month"])
            color = "#F44336" if cp_row["direction"] == "increase" else "#4CAF50"
            ax.axvline(cp_date, color=color, ls="--", alpha=0.7, lw=1.5)

        # COVID shading
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"),
                   alpha=0.1, color="red")

        npi_str = str(npi)
        ax.set_title(f"NPI ...{npi_str[-4:]}: {len(npi_cps)} changepoints",
                     fontsize=10)
        ax.set_ylabel("Spending ($K)")
        ax.tick_params(axis="x", rotation=45)

    # Hide empty subplots
    for j in range(len(example_npis), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].set_visible(False)

    fig.suptitle("Provider Spending Changepoints (PELT Detection)",
                 fontsize=14, fontweight="bold")
    save_figure(fig, "changepoint_examples", config)
    del monthly
    gc.collect()


def plot_timing_histogram(cp_df: pd.DataFrame, config: dict) -> None:
    """Histogram of when changepoints occur across time."""
    if cp_df.empty:
        return
    setup_plotting(config)

    cp_dates = pd.to_datetime(cp_df["changepoint_month"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Monthly histogram
    axes[0, 0].hist(cp_dates, bins=84, color="#2196F3", alpha=0.7,
                    edgecolor="white")
    axes[0, 0].axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"),
                       alpha=0.2, color="red", label="COVID period")
    axes[0, 0].set_xlabel("Month")
    axes[0, 0].set_ylabel("Number of Changepoints")
    axes[0, 0].set_title("Changepoint Timing Distribution")
    axes[0, 0].legend()

    # By type
    types = cp_df["type"].value_counts()
    axes[0, 1].bar(types.index, types.values,
                   color=["#F44336", "#4CAF50", "#FF9800"], alpha=0.8)
    axes[0, 1].set_xlabel("Changepoint Type")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Changepoint Types")
    for i, (t, v) in enumerate(types.items()):
        axes[0, 1].text(i, v + types.max() * 0.01, f"{v:,}", ha="center")

    # Direction
    dirs = cp_df["direction"].value_counts()
    axes[1, 0].pie(dirs.values,
                   labels=[f"{d} ({v:,})" for d, v in dirs.items()],
                   colors=["#F44336", "#4CAF50"], autopct="%1.1f%%",
                   startangle=90)
    axes[1, 0].set_title("Changepoint Directions")

    # COVID vs non-COVID
    covid_counts = cp_df["is_covid_period"].value_counts()
    labels = ["Non-COVID", "COVID Period"]
    axes[1, 1].bar(labels, [covid_counts.get(False, 0), covid_counts.get(True, 0)],
                   color=["#2196F3", "#F44336"], alpha=0.8)
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("COVID vs Non-COVID Changepoints")

    fig.suptitle("Changepoint Analysis Summary", fontsize=14, fontweight="bold")
    save_figure(fig, "changepoint_timing", config)


def plot_changepoint_magnitude(cp_df: pd.DataFrame, config: dict) -> None:
    """Plot the magnitude of spending changes at changepoints."""
    if cp_df.empty:
        return
    setup_plotting(config)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Distribution of pct change
    pct = cp_df["pct_change"].clip(-5, 5)
    axes[0].hist(pct, bins=100, color="#9C27B0", alpha=0.7, density=True)
    axes[0].axvline(0, color="gray", ls="-")
    axes[0].set_xlabel("Percentage Change at Changepoint")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Magnitude of Spending Changes")

    # Mean before vs after scatter
    sample = cp_df.sample(min(5000, len(cp_df)), random_state=42)
    axes[1].scatter(np.log10(sample["mean_before"].clip(lower=1)),
                    np.log10(sample["mean_after"].clip(lower=1)),
                    alpha=0.2, s=5, c=sample["is_covid_period"].map(
                        {True: "#F44336", False: "#2196F3"}))
    lims = axes[1].get_xlim()
    axes[1].plot(lims, lims, "k--", alpha=0.5)
    axes[1].set_xlabel("Log10(Mean Before)")
    axes[1].set_ylabel("Log10(Mean After)")
    axes[1].set_title("Before vs After Spending")

    fig.suptitle("Changepoint Magnitude Analysis", fontsize=14, fontweight="bold")
    save_figure(fig, "changepoint_magnitude", config)


def plot_changepoints_per_provider(summary_df: pd.DataFrame, config: dict) -> None:
    """Distribution of changepoints per provider."""
    if summary_df.empty:
        return
    setup_plotting(config)

    fig, ax = plt.subplots(figsize=(10, 6))
    cp_counts = summary_df["n_changepoints"].value_counts().sort_index()
    ax.bar(cp_counts.index, cp_counts.values, color="#2196F3", alpha=0.8)
    ax.set_xlabel("Number of Changepoints")
    ax.set_ylabel("Number of Providers")
    ax.set_title("Changepoints per Provider Distribution",
                 fontsize=14, fontweight="bold")
    ax.text(0.95, 0.95,
            f"Mean: {summary_df['n_changepoints'].mean():.1f}\n"
            f"Median: {summary_df['n_changepoints'].median():.0f}\n"
            f"Max: {summary_df['n_changepoints'].max():.0f}",
            transform=ax.transAxes, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    save_figure(fig, "changepoint_per_provider", config)


# ── Main Pipeline ────────────────────────────────────────────────────


def run_changepoint_detection(config: dict) -> None:
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    tables_dir = root / config["paths"]["tables_dir"]

    cfg = config.get("changepoint", {})
    top_n = cfg.get("top_n_providers", 5000)
    min_months = cfg.get("min_months", 24)

    # Detect changepoints
    logger.info("=== Running per-provider changepoint detection ===")
    cp_df, summary_df, all_months = detect_provider_changepoints(
        config, top_n=top_n, min_months=min_months
    )

    # Save results
    if not cp_df.empty:
        cp_df.to_parquet(processed_dir / "provider_changepoints.parquet", index=False)
    save_table(cp_df, "changepoint_results", config)
    save_table(summary_df, "changepoint_provider_summary", config)

    # COVID enrichment test
    logger.info("=== COVID enrichment test ===")
    covid_result = test_covid_enrichment(cp_df)

    # Visualizations
    logger.info("=== Generating changepoint plots ===")
    plot_example_changepoints(config, cp_df, all_months)
    plot_timing_histogram(cp_df, config)
    plot_changepoint_magnitude(cp_df, config)
    plot_changepoints_per_provider(summary_df, config)

    # Summary
    summary = {
        "n_providers_analyzed": len(summary_df),
        "n_changepoints_total": len(cp_df),
        "mean_changepoints_per_provider": float(summary_df["n_changepoints"].mean()),
        "providers_with_0_cp": int((summary_df["n_changepoints"] == 0).sum()),
        "providers_with_3plus_cp": int((summary_df["n_changepoints"] >= 3).sum()),
        "type_distribution": cp_df["type"].value_counts().to_dict() if not cp_df.empty else {},
        "direction_distribution": cp_df["direction"].value_counts().to_dict() if not cp_df.empty else {},
        "covid_enrichment": covid_result,
    }

    # Ensure all values are JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        return obj

    with open(tables_dir / "changepoint_summary.json", "w") as f:
        json.dump(make_serializable(summary), f, indent=2)

    logger.info(f"Changepoint summary: {len(cp_df)} changepoints, "
                f"COVID enrichment p={covid_result.get('binomial_p_value', 'N/A')}")
    gc.collect()
    logger.info("Phase 8D: Changepoint Detection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8D: Per-Provider Changepoint Detection")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_changepoint_detection(config)
