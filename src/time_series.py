"""Temporal trend analysis, seasonality decomposition, and changepoint detection."""

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
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import (
    load_config, get_project_root, setup_logging, setup_plotting,
    save_figure, save_table, format_currency,
)

logger = setup_logging()


def stl_decomposition(ts: pd.DataFrame, config: dict) -> dict:
    """Perform STL decomposition on monthly spending.

    Args:
        ts: Monthly time series DataFrame with 'date' and 'total_paid'.
        config: Configuration dictionary.

    Returns:
        Dictionary with decomposition components.
    """
    setup_plotting(config)

    ts = ts.sort_values("date").copy()
    ts = ts.set_index("date")
    ts = ts.asfreq("MS")

    # STL decomposition
    stl = STL(ts["total_paid"], period=12, robust=True)
    result = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    axes[0].plot(result.observed / 1e9, color="#1565C0", linewidth=1.5)
    axes[0].set_ylabel("Observed ($B)")
    axes[0].set_title("STL Decomposition of Monthly Medicaid Spending")

    axes[1].plot(result.trend / 1e9, color="#2E7D32", linewidth=2)
    axes[1].set_ylabel("Trend ($B)")

    axes[2].plot(result.seasonal / 1e9, color="#E65100", linewidth=1.5)
    axes[2].set_ylabel("Seasonal ($B)")

    axes[3].plot(result.resid / 1e9, color="#6A1B9A", linewidth=1, alpha=0.7)
    axes[3].set_ylabel("Residual ($B)")
    axes[3].set_xlabel("Date")

    # Shade COVID period
    for ax in axes:
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"),
                   alpha=0.08, color="red")
        ax.grid(True, alpha=0.3)

    save_figure(fig, "stl_decomposition", config)
    logger.info("STL decomposition complete")

    return {
        "trend": result.trend,
        "seasonal": result.seasonal,
        "resid": result.resid,
    }


def adf_stationarity_test(ts: pd.DataFrame) -> dict:
    """Perform Augmented Dickey-Fuller test for stationarity.

    Args:
        ts: Time series DataFrame with 'total_paid'.

    Returns:
        Dictionary with ADF test results.
    """
    result = adfuller(ts["total_paid"].dropna(), autolag="AIC")

    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }


def changepoint_detection(ts: pd.DataFrame, config: dict) -> list:
    """Detect structural breaks in spending trends.

    Args:
        ts: Monthly time series DataFrame.
        config: Configuration dictionary.

    Returns:
        List of changepoint indices.
    """
    import ruptures

    setup_plotting(config)

    signal = ts["total_paid"].values / 1e9

    # PELT algorithm
    algo = ruptures.Pelt(model="rbf", min_size=3, jump=1).fit(signal)
    breakpoints = algo.predict(pen=3)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(ts["date"].values, signal, color="#1565C0", linewidth=1.5)

    for bp in breakpoints[:-1]:  # last one is the end
        if bp < len(ts):
            ax.axvline(ts["date"].iloc[bp], color="red", linestyle="--", alpha=0.7)
            ax.text(ts["date"].iloc[bp], signal.max() * 0.95,
                    ts["date"].iloc[bp].strftime("%Y-%m"), rotation=45,
                    fontsize=8, color="red")

    ax.set_xlabel("Date")
    ax.set_ylabel("Total Spending ($B)")
    ax.set_title(f"Changepoint Detection (PELT, {len(breakpoints)-1} breakpoints)")
    ax.grid(True, alpha=0.3)

    save_figure(fig, "changepoint_detection", config)
    logger.info(f"Detected {len(breakpoints)-1} changepoints")

    return breakpoints


def granger_causality_analysis(ts: pd.DataFrame, config: dict) -> dict:
    """Test Granger causality between spending and claims volume.

    Args:
        ts: Monthly time series DataFrame.
        config: Configuration dictionary.

    Returns:
        Dictionary of Granger causality test results.
    """
    ts_clean = ts[["total_paid", "total_claims"]].dropna()

    # Difference to make stationary if needed
    ts_diff = ts_clean.diff().dropna()

    results = {}
    try:
        # Does claims Granger-cause spending?
        gc1 = grangercausalitytests(
            ts_diff[["total_paid", "total_claims"]].values,
            maxlag=6, verbose=False,
        )
        min_p_1 = min(gc1[lag][0]["ssr_ftest"][1] for lag in gc1)
        results["claims_causes_spending"] = {
            "min_p_value": float(min_p_1),
            "significant": min_p_1 < 0.05,
        }

        # Does spending Granger-cause claims?
        gc2 = grangercausalitytests(
            ts_diff[["total_claims", "total_paid"]].values,
            maxlag=6, verbose=False,
        )
        min_p_2 = min(gc2[lag][0]["ssr_ftest"][1] for lag in gc2)
        results["spending_causes_claims"] = {
            "min_p_value": float(min_p_2),
            "significant": min_p_2 < 0.05,
        }
    except Exception as e:
        logger.warning(f"Granger causality test failed: {e}")
        results["error"] = str(e)

    logger.info(f"Granger causality: {results}")
    return results


def spending_forecast(ts: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Forecast future spending using SARIMA.

    Args:
        ts: Monthly time series DataFrame.
        config: Configuration dictionary.

    Returns:
        DataFrame with forecast values and confidence intervals.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    setup_plotting(config)

    ts_indexed = ts.sort_values("date").set_index("date")
    ts_indexed = ts_indexed.asfreq("MS")

    y = ts_indexed["total_paid"]

    # Fit SARIMA
    model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                     enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False, maxiter=500)

    # Forecast 12 months
    forecast = result.get_forecast(steps=12)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    # Plot
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(y.index, y.values / 1e9, color="#1565C0", linewidth=1.5, label="Historical")
    ax.plot(forecast_mean.index, forecast_mean.values / 1e9, "r--", linewidth=2, label="Forecast")
    ax.fill_between(forecast_ci.index,
                     forecast_ci.iloc[:, 0] / 1e9,
                     forecast_ci.iloc[:, 1] / 1e9,
                     alpha=0.2, color="red", label="95% CI")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Spending ($B)")
    ax.set_title("SARIMA Spending Forecast (12-month horizon)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, "spending_forecast_sarima", config)

    forecast_df = pd.DataFrame({
        "date": forecast_mean.index,
        "forecast": forecast_mean.values,
        "ci_lower": forecast_ci.iloc[:, 0].values,
        "ci_upper": forecast_ci.iloc[:, 1].values,
    })
    save_table(forecast_df, "spending_forecast", config)

    logger.info(f"SARIMA forecast: AIC={result.aic:.0f}")
    return forecast_df


def run_time_series_analysis(config: Optional[dict] = None) -> None:
    """Run the full time series analysis pipeline.

    Args:
        config: Configuration dictionary.
    """
    if config is None:
        config = load_config()

    root = get_project_root()
    ts = pd.read_parquet(root / config["paths"]["processed_dir"] / "monthly_time_series.parquet")
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.sort_values("date").reset_index(drop=True)

    # STL Decomposition
    stl_results = stl_decomposition(ts.copy(), config)

    # Stationarity test
    adf_result = adf_stationarity_test(ts)
    logger.info(f"ADF test: statistic={adf_result['adf_statistic']:.3f}, "
                f"p={adf_result['p_value']:.4f}, stationary={adf_result['is_stationary']}")

    # Changepoint detection
    changepoints = changepoint_detection(ts.copy(), config)

    # Granger causality
    granger = granger_causality_analysis(ts.copy(), config)

    # Forecast
    try:
        forecast = spending_forecast(ts.copy(), config)
    except Exception as e:
        logger.error(f"Forecast failed: {e}")

    logger.info("Time series analysis complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run time series analysis")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_time_series_analysis(config)
