"""Phase 8B: Extreme Value Theory Analysis of Medicaid spending.

Models the heavy tails of the provider spending distribution using
Generalized Pareto Distribution (GPD) to quantify extreme spending risks.

Methods:
    - GPD fitting via peaks-over-threshold (POT)
    - Value at Risk (VaR) and Conditional VaR at 95%/99%
    - Yearly tail evolution and return level estimation
    - Mean excess function for threshold selection
    - Q-Q diagnostic plots
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
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


# ── GPD Fitting ──────────────────────────────────────────────────────


def fit_gpd(data: np.ndarray, threshold_quantile: float = 0.95) -> dict:
    """Fit Generalized Pareto Distribution to exceedances over threshold.

    Returns:
        Dictionary with GPD parameters, VaR, CVaR, and diagnostics.
    """
    threshold = np.quantile(data, threshold_quantile)
    exceedances = data[data > threshold] - threshold

    if len(exceedances) < 20:
        return {"error": "Too few exceedances", "n_exceedances": len(exceedances)}

    # Fit GPD (shape, loc=0, scale)
    shape, loc, scale = sp_stats.genpareto.fit(exceedances, floc=0)
    n_total = len(data)
    n_exceed = len(exceedances)
    exceed_prob = n_exceed / n_total

    # VaR and CVaR at various confidence levels
    var_cvar = {}
    for alpha in [0.95, 0.99, 0.995]:
        p = 1 - alpha
        if shape != 0:
            var = threshold + (scale / shape) * ((p / exceed_prob) ** (-shape) - 1)
        else:
            var = threshold + scale * np.log(exceed_prob / p)
        # CVaR (Expected Shortfall)
        if shape < 1:
            cvar = var / (1 - shape) + (scale - shape * threshold) / (1 - shape)
        else:
            cvar = np.inf
        var_cvar[f"VaR_{alpha}"] = float(var)
        var_cvar[f"CVaR_{alpha}"] = float(cvar)

    # Return levels
    return_levels = {}
    for period in [10, 50, 100]:
        p = 1 / period
        if shape != 0:
            rl = threshold + (scale / shape) * ((period * exceed_prob) ** shape - 1)
        else:
            rl = threshold + scale * np.log(period * exceed_prob)
        return_levels[f"return_{period}"] = float(rl)

    # KS test for goodness of fit
    ks_stat, ks_p = sp_stats.kstest(exceedances, "genpareto", args=(shape, 0, scale))

    return {
        "threshold": float(threshold),
        "threshold_quantile": threshold_quantile,
        "n_total": n_total,
        "n_exceedances": n_exceed,
        "exceed_prob": float(exceed_prob),
        "shape_xi": float(shape),
        "scale_sigma": float(scale),
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(ks_p),
        **var_cvar,
        **return_levels,
    }


def mean_excess_function(data: np.ndarray, n_thresholds: int = 100) -> pd.DataFrame:
    """Compute the mean excess (mean residual life) function.

    For threshold u: e(u) = E[X - u | X > u].
    A linear mean excess function suggests GPD is appropriate.
    """
    sorted_data = np.sort(data)
    thresholds = np.linspace(
        np.quantile(data, 0.5), np.quantile(data, 0.995), n_thresholds
    )
    rows = []
    for u in thresholds:
        exceedances = sorted_data[sorted_data > u] - u
        if len(exceedances) < 5:
            break
        rows.append({
            "threshold": u,
            "mean_excess": float(exceedances.mean()),
            "n_exceedances": len(exceedances),
        })
    return pd.DataFrame(rows)


# ── Yearly Tail Evolution ────────────────────────────────────────────


def fit_yearly_gpd(config: dict) -> pd.DataFrame:
    """Fit GPD to each year's provider spending distribution."""
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]

    yearly_results = []
    for year in range(2018, 2025):
        path = processed_dir / f"medicaid_{year}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["BILLING_PROVIDER_NPI_NUM", "TOTAL_PAID"])
        provider_spending = df.groupby("BILLING_PROVIDER_NPI_NUM")["TOTAL_PAID"].sum().values
        provider_spending = provider_spending[provider_spending > 0]

        for q in [0.95, 0.99]:
            result = fit_gpd(provider_spending, threshold_quantile=q)
            result["year"] = year
            result["threshold_quantile"] = q
            yearly_results.append(result)

        del df
        gc.collect()
        logger.info(f"  {year}: GPD fit on {len(provider_spending)} providers")

    return pd.DataFrame(yearly_results)


# ── Visualization ────────────────────────────────────────────────────


def plot_gpd_fit(data: np.ndarray, gpd_result: dict, config: dict) -> None:
    """Plot GPD fit diagnostics."""
    setup_plotting(config)

    threshold = gpd_result["threshold"]
    exceedances = data[data > threshold] - threshold
    shape = gpd_result["shape_xi"]
    scale = gpd_result["scale_sigma"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Excess distribution histogram + GPD fit
    ax = axes[0, 0]
    ax.hist(exceedances, bins=100, density=True, alpha=0.6, color="#2196F3",
            label="Observed exceedances")
    x = np.linspace(0, np.quantile(exceedances, 0.99), 500)
    pdf = sp_stats.genpareto.pdf(x, shape, 0, scale)
    ax.plot(x, pdf, "r-", lw=2, label=f"GPD(xi={shape:.3f}, sigma={scale:.1f})")
    ax.set_xlabel("Excess over threshold")
    ax.set_ylabel("Density")
    ax.set_title("GPD Fit to Exceedances")
    ax.legend()

    # 2. Q-Q plot
    ax = axes[0, 1]
    theoretical = sp_stats.genpareto.ppf(
        np.linspace(0.01, 0.99, len(exceedances)), shape, 0, scale
    )
    empirical = np.sort(exceedances)
    # Match sizes
    n = min(len(theoretical), len(empirical))
    theoretical = np.linspace(theoretical.min(), theoretical.max(), n)
    empirical_interp = np.sort(empirical)[-n:] if len(empirical) > n else empirical
    ax.scatter(theoretical[:n], empirical_interp[:n], alpha=0.3, s=10, color="#4CAF50")
    lims = [0, max(theoretical.max(), empirical_interp.max()) * 1.05]
    ax.plot(lims, lims, "r--", lw=1.5, label="45-degree line")
    ax.set_xlabel("Theoretical Quantiles (GPD)")
    ax.set_ylabel("Empirical Quantiles")
    ax.set_title("Q-Q Plot")
    ax.legend()

    # 3. Log survival function
    ax = axes[1, 0]
    sorted_exc = np.sort(exceedances)
    survival_emp = 1 - np.arange(1, len(sorted_exc) + 1) / (len(sorted_exc) + 1)
    ax.semilogy(sorted_exc, survival_emp, ".", alpha=0.3, color="#2196F3",
                label="Empirical", markersize=3)
    x_fit = np.linspace(0, sorted_exc.max(), 500)
    survival_fit = sp_stats.genpareto.sf(x_fit, shape, 0, scale)
    ax.semilogy(x_fit, survival_fit, "r-", lw=2, label="GPD fit")
    ax.set_xlabel("Excess over threshold")
    ax.set_ylabel("Survival Probability (log scale)")
    ax.set_title("Log Survival Function")
    ax.legend()

    # 4. Tail distribution
    ax = axes[1, 1]
    log_spending = np.log10(data[data > 0])
    ax.hist(log_spending, bins=200, density=True, alpha=0.6, color="#9C27B0")
    ax.axvline(np.log10(threshold), color="red", ls="--", lw=2,
               label=f"Threshold: {format_currency(threshold)}")
    ax.set_xlabel("Log10(Total Spending)")
    ax.set_ylabel("Density")
    ax.set_title("Spending Distribution with Threshold")
    ax.legend()

    fig.suptitle("Extreme Value Theory: GPD Tail Analysis",
                 fontsize=14, fontweight="bold")
    save_figure(fig, "evt_gpd_diagnostics", config)


def plot_mean_excess(mef_df: pd.DataFrame, config: dict) -> None:
    """Plot mean excess function."""
    setup_plotting(config)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(mef_df["threshold"], mef_df["mean_excess"], "-", color="#2196F3", lw=1.5)
    ax.fill_between(mef_df["threshold"], mef_df["mean_excess"], alpha=0.2, color="#2196F3")
    ax.set_xlabel("Threshold (Total Spending $)")
    ax.set_ylabel("Mean Excess")
    ax.set_title("Mean Excess Function (Mean Residual Life Plot)",
                 fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    save_figure(fig, "evt_mean_excess", config)


def plot_yearly_tail_evolution(yearly_df: pd.DataFrame, config: dict) -> None:
    """Plot how tail parameters evolve over time."""
    setup_plotting(config)

    df95 = yearly_df[yearly_df["threshold_quantile"] == 0.95].copy()
    if df95.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Shape parameter evolution
    axes[0, 0].plot(df95["year"], df95["shape_xi"], "o-", color="#2196F3", lw=2)
    axes[0, 0].set_xlabel("Year")
    axes[0, 0].set_ylabel("Shape (xi)")
    axes[0, 0].set_title("Tail Shape Parameter Over Time")
    axes[0, 0].axhline(0, color="gray", ls=":", alpha=0.5)
    axes[0, 0].axvline(2020, color="red", ls="--", alpha=0.5, label="COVID")
    axes[0, 0].legend()

    # VaR evolution
    axes[0, 1].plot(df95["year"], df95["VaR_0.99"] / 1e6, "o-", color="#F44336", lw=2,
                    label="VaR 99%")
    axes[0, 1].plot(df95["year"], df95["VaR_0.95"] / 1e6, "s-", color="#FF9800", lw=2,
                    label="VaR 95%")
    axes[0, 1].set_xlabel("Year")
    axes[0, 1].set_ylabel("Value at Risk ($M)")
    axes[0, 1].set_title("Value at Risk Over Time")
    axes[0, 1].legend()

    # CVaR evolution
    if "CVaR_0.99" in df95.columns:
        cvar99 = df95["CVaR_0.99"].replace([np.inf, -np.inf], np.nan).dropna()
        if not cvar99.empty:
            axes[1, 0].plot(df95["year"].iloc[:len(cvar99)], cvar99 / 1e6,
                            "o-", color="#9C27B0", lw=2, label="CVaR 99%")
    cvar95 = df95["CVaR_0.95"].replace([np.inf, -np.inf], np.nan).dropna()
    if not cvar95.empty:
        axes[1, 0].plot(df95["year"].iloc[:len(cvar95)], cvar95 / 1e6,
                        "s-", color="#E91E63", lw=2, label="CVaR 95%")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].set_ylabel("Conditional VaR ($M)")
    axes[1, 0].set_title("Expected Shortfall Over Time")
    axes[1, 0].legend()

    # Threshold evolution
    axes[1, 1].plot(df95["year"], df95["threshold"] / 1e6, "o-", color="#4CAF50", lw=2)
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Threshold ($M)")
    axes[1, 1].set_title("95th Percentile Threshold Over Time")

    fig.suptitle("Yearly Tail Risk Evolution (EVT)", fontsize=14, fontweight="bold")
    save_figure(fig, "evt_yearly_evolution", config)


def plot_return_levels(yearly_df: pd.DataFrame, config: dict) -> None:
    """Plot return level curves."""
    setup_plotting(config)

    df95 = yearly_df[yearly_df["threshold_quantile"] == 0.95].copy()
    if df95.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for col, period, color in [("return_10", "10-year", "#4CAF50"),
                                ("return_50", "50-year", "#FF9800"),
                                ("return_100", "100-year", "#F44336")]:
        if col in df95.columns:
            vals = df95[col].replace([np.inf, -np.inf], np.nan).dropna()
            if not vals.empty:
                ax.plot(df95["year"].iloc[:len(vals)], vals / 1e6,
                        "o-", color=color, lw=2, label=f"{period} return level")

    ax.set_xlabel("Year")
    ax.set_ylabel("Return Level ($M)")
    ax.set_title("Estimated Return Levels by Year",
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.axvline(2020, color="gray", ls=":", alpha=0.5, label="COVID")
    save_figure(fig, "evt_return_levels", config)


def plot_tail_comparison(data: np.ndarray, config: dict) -> None:
    """Compare empirical tail vs fitted distributions."""
    setup_plotting(config)
    log_data = np.log10(data[data > 0])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # CCDF (complementary CDF) on log-log
    sorted_vals = np.sort(data[data > 0])
    ccdf = 1 - np.arange(1, len(sorted_vals) + 1) / (len(sorted_vals) + 1)
    axes[0].loglog(sorted_vals, ccdf, ".", alpha=0.1, markersize=2, color="#2196F3")
    axes[0].set_xlabel("Total Spending ($)")
    axes[0].set_ylabel("P(X > x)")
    axes[0].set_title("Complementary CDF (Log-Log Scale)")

    # Hill estimator for different k values
    n = len(sorted_vals)
    k_values = np.arange(50, min(n // 2, 5000), 50)
    hill_estimates = []
    for k in k_values:
        top_k = sorted_vals[-(k + 1):]
        log_diff = np.log(top_k[1:]) - np.log(top_k[0])
        hill_estimates.append(1 / np.mean(log_diff))

    axes[1].plot(k_values, hill_estimates, "-", color="#FF9800", lw=1.5)
    axes[1].set_xlabel("k (number of upper order statistics)")
    axes[1].set_ylabel("Hill Estimator (tail index)")
    axes[1].set_title("Hill Plot for Tail Index Estimation")
    axes[1].axhline(y=np.median(hill_estimates[-20:]), color="red", ls="--",
                    label=f"Stable estimate: {np.median(hill_estimates[-20:]):.2f}")
    axes[1].legend()

    fig.suptitle("Heavy Tail Characterization", fontsize=14, fontweight="bold")
    save_figure(fig, "evt_tail_comparison", config)


# ── Main Pipeline ────────────────────────────────────────────────────


def run_extreme_value_analysis(config: dict) -> None:
    root = get_project_root()
    processed_dir = root / config["paths"]["processed_dir"]
    tables_dir = root / config["paths"]["tables_dir"]

    # Load overall provider spending
    logger.info("=== Loading provider spending data ===")
    feats = pd.read_parquet(processed_dir / "provider_features.parquet")
    spending = feats["total_paid"].values
    spending_pos = spending[spending > 0]
    logger.info(f"Providers with positive spending: {len(spending_pos)}")

    # Mean excess function
    logger.info("=== Computing mean excess function ===")
    mef_df = mean_excess_function(spending_pos)
    save_table(mef_df, "evt_mean_excess", config)
    plot_mean_excess(mef_df, config)

    # GPD fit at 95th percentile
    logger.info("=== Fitting GPD at 95th percentile ===")
    gpd_95 = fit_gpd(spending_pos, 0.95)
    logger.info(f"GPD(95%): xi={gpd_95.get('shape_xi', 'N/A'):.3f}, "
                f"sigma={gpd_95.get('scale_sigma', 'N/A'):.1f}, "
                f"KS p={gpd_95.get('ks_p_value', 'N/A'):.4f}")

    # GPD fit at 99th percentile
    gpd_99 = fit_gpd(spending_pos, 0.99)

    # Save GPD results
    gpd_df = pd.DataFrame([gpd_95, gpd_99])
    save_table(gpd_df, "evt_gpd_parameters", config)

    # Diagnostic plots
    plot_gpd_fit(spending_pos, gpd_95, config)
    plot_tail_comparison(spending_pos, config)

    # Yearly evolution
    logger.info("=== Fitting yearly GPD ===")
    yearly_df = fit_yearly_gpd(config)
    save_table(yearly_df, "evt_yearly_tail_params", config)
    plot_yearly_tail_evolution(yearly_df, config)
    plot_return_levels(yearly_df, config)

    # Summary
    summary = {
        "n_providers": len(spending_pos),
        "gpd_95th": {
            "shape_xi": gpd_95.get("shape_xi"),
            "scale_sigma": gpd_95.get("scale_sigma"),
            "VaR_95": gpd_95.get("VaR_0.95"),
            "VaR_99": gpd_95.get("VaR_0.99"),
            "CVaR_95": gpd_95.get("CVaR_0.95"),
            "CVaR_99": gpd_95.get("CVaR_0.99"),
            "ks_p_value": gpd_95.get("ks_p_value"),
        },
        "gpd_99th": {
            "shape_xi": gpd_99.get("shape_xi"),
            "scale_sigma": gpd_99.get("scale_sigma"),
            "VaR_99": gpd_99.get("VaR_0.99"),
        },
        "heavy_tail": gpd_95.get("shape_xi", 0) > 0,
        "tail_interpretation": (
            "Heavy-tailed (Frechet-type)" if gpd_95.get("shape_xi", 0) > 0
            else "Light-tailed (Weibull-type)" if gpd_95.get("shape_xi", 0) < 0
            else "Exponential tail"
        ),
    }

    with open(tables_dir / "evt_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"EVT summary: shape={gpd_95.get('shape_xi', 'N/A'):.3f} → "
                f"{summary['tail_interpretation']}")
    gc.collect()
    logger.info("Phase 8B: Extreme Value Theory Analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 8B: Extreme Value Theory")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_extreme_value_analysis(config)
